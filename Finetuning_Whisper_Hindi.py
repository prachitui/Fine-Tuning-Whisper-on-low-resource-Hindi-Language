# Importing necessary libraries

from transformers import Seq2SeqTrainer
from datasets import load_dataset, DatasetDict
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Trainer, TrainingArguments

# Using hugging face to download and prepare data
# Common voice is a multilingual dataset available for ASR

common_voice = DatasetDict()

# Using Hindi as our language. Since,  Hindi is very low-resource, we'll combine the train and validation
# splits to give approximately 8 hours of training data. We'll use the 4 hours of test data as our held-out
# test set:

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", trust_remote_code=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", trust_remote_code=True)

print(common_voice)

# Since, Common Voice contains additional metadata information, such as accent and locale, we can disregard it
# to keep the work simple.

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])


# A feature extractor which pre-processes the raw audio-inputs
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# A tokenizer which post-processes the model outputs to text format
# For Hindi, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications.
# We simply have to specify the target language and the task.

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

# Printing the first sentence
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


# To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class.
# This processor object inherits from the WhisperFeatureExtractor and WhisperProcessor, and can be used on the
# audio inputs and model predictions as required. In doing so, we only need to keep track of two objects during
# training: the processor and the model.
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

#  Printing the first example of the Common Voice dataset to see what form the data is in.
print(common_voice["train"][0])


# Since common voice's input audio is sampled at 48kHz, we need to downsample it to 16kHz prior to passing it to the
# Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
# Re-loading the first audio sample in the Common Voice dataset will resample it to the desired sampling rate.
print(common_voice["train"][0])

#  Function to prepare our data to be ready for the model.
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
# Here if num_proc > 1, it will enable multiprocessing.
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)


# Starting our fine-tuning run from the pre-trained Whisper small checkpoint,
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Here we disable the automatic language detection task performed during inference, and force the model to
# generate in Hindi. To do so, we set the language and task arguments to the generation config. We'll also set
# any forced_decoder_ids to None, since this was the legacy way of setting the language and task arguments.
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

# The data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Initialising the data collator

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-prachi",
    per_device_train_batch_size=2,      #4,#8,#16,  # Increase the batch size # you can decrease it by factor of 2 if you run out of memory in your GP
    gradient_accumulation_steps=16,     #8,#4,#2,  # Use gradient accumulation # Increase it by factor of 2 for every decrease above
    num_train_epochs=3,                 # Set number of epochs
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=1000,
    fp16=True,  # Use mixed precision training

)

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# Start training
trainer.train()
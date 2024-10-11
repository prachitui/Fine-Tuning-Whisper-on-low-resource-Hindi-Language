from transformers import Seq2SeqTrainer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Audio
import numpy as np
import evaluate
from transformers import Seq2SeqTrainingArguments

# Load the test dataset
common_voice = DatasetDict()
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", trust_remote_code=True)
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Load the processor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

# Load the fine-tuned model
model = WhisperForConditionalGeneration.from_pretrained("./whisper-finetuned-prachi")
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Prepare the test dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=1)

# Define the data collator
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define the metric
wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}



# Define training arguments for evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-prachi",
    per_device_eval_batch_size=2,
    fp16=True,
)


# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

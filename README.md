# Whisper Fine-Tuning and Evaluation on Hindi Subset of Common Voice Dataset
This repository demonstrates how to fine-tune and evaluate OpenAI's Whisper model on the Hindi subset of the Common Voice dataset using Hugging Face's transformers 
library.

## Table of Contents
 - [Overview](#overview)
 - [Setup and Requirements](#setup-and-requirements)
 - [Dataset](#dataset)
 - [Model Components](#model-components)
 - [Training](#training)
 - [Evaluation](#evaluation)
 - [Usage](#usage)
 - [Acknowledgements](#acknowledgements)

## Overview
Whisper is a powerful ASR model developed by OpenAI. Here, we leverage the Common Voice dataset (version 11.0) with Hindi language to fine-tune and evaluate Whisper for speech-to-text tasks. 
The training script (Finetuning_Whisper_Hindi.py):
- Prepares and preprocesses the audio data.
- Fine-tunes the Whisper model.
- Outputs the model checkpoints and logs to the whisper-finetuned-prachi directory.
  
The evaluation script (Finetuned_Whisper_Evaluation.py)
- Evaluates the trained model with WER set as metric.

## Setup and Requirements
To get started, ensure you have the following installed:
- Python>=3.7
- ```bash
     pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard
  ```
## Dataset
The Common Voice [dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) is a multilingual dataset available for automatic speech recognition. We specifically use the Hindi language subset of the dataset.

## Model Components
- WhisperFeatureExtractor: Extracts the log-Mel spectrograms from the input audio.
- WhisperTokenizer: Tokenizes the transcriptions into token IDs for text-to-speech tasks.
- WhisperProcessor: Combines the feature extractor and tokenizer for efficient processing.

## Training
We use Hugging Face's Trainer API to fine-tune the model. You can also use 'Seq2SeqTrainer' in place of Trainer. Some important training configurations:

- Mixed Precision (fp16): Speeds up training on modern GPUs.
- Gradient Accumulation: Helps when training on smaller GPUs by simulating larger batch sizes.
- The fine-tuned model is saved to the whisper-finetuned-prachi directory.

## Evaluation

The evaluation code loads the fine-tuned model from the whisper-finetuned-prachi directory, preprocesses the test dataset, and evaluates the model using the Seq2SeqTrainer class. 
The WER score is computed and printed to the console.

## Usage

- Clone the repository:
  ```bash
   git clone https://github.com/prachitui/Fine-Tuning-Whisper-on-Common-Voice-Hindi-Dataset.git
   cd Fine-Tuning-Whisper-on-Common-Voice-Hindi-Dataset
  ```
- Run the fine-tuning script:
  ```bash
  python Finetuning_Whisper_Hindi.py
   ```
- Run the evaluation script
  ```bash
  python Finetuned_Whisper_Evaluation.py
  ```
## Acknowledgements
The code is based on common voice [dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) and hugging face whisper examples [here](https://huggingface.co/openai/whisper-small).

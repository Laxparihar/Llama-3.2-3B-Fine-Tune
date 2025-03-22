# Llama-3.2-3B-Fine-Tune

This repository contains code to fine-tune the **Llama 3B model** using **LoRA** (Low-Rank Adaptation) and **QLoRA** for instruction-following tasks. The approach leverages **Hugging Face Transformers** and **PEFT** (Parameter-Efficient Fine-Tuning) for memory-efficient training, making it suitable for large language model fine-tuning. The dataset used consists of instruction-output pairs, ideal for conversational tasks.

## Project Structure:
1. **trainer.py**: Script for training the fine-tuned model using LoRA and QLoRA configurations.
2. **optimise_model.py**: Script to load the fine-tuned model, merge the LoRA parameters, and save the final model.
3. **llama_test.py**: Example script for running inference with the fine-tuned model on text (e.g., bank statement extraction).
4. **Dataset**: A CSV file with columns `instruction`, `input`, and `output` used for training the model.

## Setup and Usage:

### Pre-requisites:
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
Fine-Tuning the Model:
Prepare your dataset: Ensure the dataset contains columns instruction, input, and output. Update the dataset path in trainer.py.

### Run the training script:
- Run the training script:
  ```bash
  python trainer.py


### Optimizing the Fine-Tuned Model:
- Merge LoRA with the base model: After training, use optimise_model.py to merge the LoRA weights into the base model.
- Run the optimise_model script:
  ```bash
  python optimise_model.py


### Uploading the Model:
- The fine-tuned model can be uploaded to Hugging Face using the optimise_model.py script. You can modify the Hugging Face repo name and token in the script to push the model to the Model Hub.


## Files Overview:
- trainer.py
  Contains the code for training the Llama model using LoRA, including:

- Model and tokenizer loading.

- Data preprocessing.

- Training loop with PEFT and QLoRA configurations.

- optimise_model.py
After training, this script merges the LoRA model with the base Llama model and uploads it to Hugging Face.
- Run the Test script:
  ```bash
  llama_test.py
This script allows for inference on your fine-tuned model. In the example, it extracts information from a bank statement, but you can customize it for other use cases.


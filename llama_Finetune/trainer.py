import numpy as np
import pandas as pd
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

# Define model paths
new_model = "llama_Finetune3B"
base_model = "meta-llama/Llama-3.2-3B-Instructt"

# Setup device and quantization configurations
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.chat_template = None

# Load dataset and preprocess
data_df = pd.read_csv('../llama/Dataset_llama_formet.csv')

# Function to prepare training data
def prepare_train_datav2(data_df):
    # Ensure that the required columns exist
    required_columns = ['instruction', 'input', 'output']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {', '.join(missing_columns)}")

    # Format the "text" column as per the instruction
    data_df["text"] = data_df[["instruction", "input", "output"]].apply(
        lambda x: "<|im_start|>system\n" + x["instruction"] + "input\n" + x["input"] +
                  " <|im_end|>\n<|im_start|>assistant\n" + x["output"] + "<|im_end|>\n",
        axis=1
    )
    
    # Create a new Dataset from the DataFrame
    data = Dataset.from_pandas(data_df)
    return data

processed_data = prepare_train_datav2(data_df)

# Function to find all linear layers for LoRA
import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Find all modules for LoRA
modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)

# Setup chat format and PEFT model
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

# Hyperparameters
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=3,
    logging_steps=1,
    warmup_steps=10,
    max_steps=300,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
)

# Setting SFT parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_data,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments
)

# Optional: Enable gradient checkpointing if needed (for large models)
model.gradient_checkpointing_enable()

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

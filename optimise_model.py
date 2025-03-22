from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_name = "meta-llama/Llama-3.2-3B-Instructt"  # Change to your base model
lora_model_path = "../llama/llama_Finetune3B" # Change to your save model

# Load tokenizer from fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# Load base model and resize embeddings
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match tokenizer

# Load fine-tuned LoRA model
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge LoRA into base model
merged_model = lora_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained(lora_model_path)
tokenizer.save_pretrained(lora_model_path)  # Save tokenizer as well

print("✅ LoRA merged and model saved successfully!")



from huggingface_hub import login, HfApi

# Login to Hugging Face
login(token="Your_Hugging_Face_Token")  # Replace with your Hugging Face token

repo_name = "Your/Finetune_Llama-3.2-3B"  # Change to your repo name

# Create repo (if not exists)
api = HfApi()
api.create_repo(repo_name, private=False, exist_ok=True)

# Upload model
merged_model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✅ Model uploaded successfully: https://huggingface.co/{repo_name}")

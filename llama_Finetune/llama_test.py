from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model from Hugging Face Model Hub


# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Your/Finetune_Llama-3.2-3B" )
model = AutoModelForCausalLM.from_pretrained("Your/Finetune_Llama-3.2-3B" )


model = model.to("cuda:1")  # Move model to GPU 1

torch.cuda.empty_cache()

# Bank statement text (you can replace this with any input text)
text = '''
bank passbook test here...........
'''

# Prepare the prompt
instruction = f"""
You are an expert in text extraction for bank statements and passbooks. Your task is to extract only the following relevant information:
- Account Holder Name: The name of the person associated with the account.
- Account Number / Account no: The full account number (integer, length from 9 to 18 digits).
- IFSC Code: The IFSC code related to the account (11 characters).

Please ignore all other information such as balance, bank name, transactions, etc.

**Text of bank statement and passbook:**
{text}

**Expected Output:**
Only output the following keys and their corresponding values:
- Account Holder Name
- Account Number
- IFSC Code

Example output:
{{'Account Holder Name': ' ', 'Account Number': ' ', 'IFSC Code': ' '}}
"""

# Create the final input string
final_input = instruction

# Tokenize the prompt
inputs = tokenizer(final_input, return_tensors='pt', padding=True, truncation=True).to("cuda:1")

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=250, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

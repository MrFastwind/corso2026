import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Device Selection 
# (Set device = "cpu" manually if you want to test the non-CUDA execution path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-> Active Execution Device: {device}")

if device == "cuda":
    print(f"-> Detected GPU: {torch.cuda.get_device_name(0)}")

# 2. Load Model and Tokenizer Locally
model_name = "distilgpt2"
print(f"-> Loading model '{model_name}' into memory...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Move the Model to the Target Hardware (CPU or CUDA)
model.to(device)

# 4. Prepare Input Prompt
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 5. Generate Response
print("-> Generating text completion...")
outputs = model.generate(
    **inputs, 
    max_length=60, 
    pad_token_id=tokenizer.eos_token_id
)

# 6. Decode and Display Results
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Generated Output ---")
print(result)


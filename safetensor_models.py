# Dependencies
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Output file path
OUTPUT_FILE = "target_file.md"
# Model folder path
model_path = "Qwen3.5 9B"

MAX_CTX = 8192
MAX_NEW = 4096

# Check cuda
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0))

# Load tokenizer
print("Load Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    model_max_length=MAX_CTX,
)
print("Tokenizer loaded!")

# 4-bit quantization
# Quantization: reduces model size and memory usage by storing weights with fewer bits
# (e.g. 4-bit NF4 instead of 16-bit bf16). Weights are grouped into small blocks, each with
# its own scale factor, and are dequantized on the fly during inference. This lets a 9B model
# fit in ~6 GB of VRAM at the cost of a small drop in accuracy.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model on GPU
print("Load Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": 0},  # All layers on GPU
    attn_implementation="sdpa",
    trust_remote_code=True,
)
model.eval()
print("Model loaded!")

print(f"VRAM Usage: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

# Prompt and system role
messages = [
    {"role": "system", "content": "Type your role here"},
    {"role": "user", "content": "Prompt"},
]

# Tokenize prompt
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_CTX - MAX_NEW,
).to(device)

print("Input tokens:", inputs["input_ids"].shape[1])

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

# Create file of answer
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(answer)

print(f"File saved on {OUTPUT_FILE} !")

n_new = len(new_tokens)
print(f"Token usage: {n_new} / {MAX_NEW} | If Limited?: {n_new >= MAX_NEW}")
print(f"Maximum VRAM usage: {torch.cuda.max_memory_allocated(0) / 1024**3:.1f} GB")
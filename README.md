# llm-4bit-local-inference

A minimal Python script for running a Hugging Face causal language model locally on a single GPU using 4-bit (NF4) quantization via `bitsandbytes`. The model's response is written to a file instead of streamed to the terminal.

Works with most `transformers`-compatible chat models (Qwen, Llama, Mistral, Gemma, etc.). Just change the model path.

## Features

- 4-bit NF4 quantization with double quantization to reduce VRAM usage
- Entire model loaded on a single GPU
- Chat-template based prompting (system + user roles)
- Output saved to a file (UTF-8, so non-English text works)
- Prints token usage, truncation status, and peak VRAM usage

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA

```bash
pip install -U torch transformers accelerate bitsandbytes
```

## Usage

1. Download a model (local folder or Hugging Face repo ID).
2. Edit the config at the top of the script:

```python
model_path = "path/to/your-model"   # local folder or HF repo ID
OUTPUT_FILE = "target_file.md"      # where the answer is saved
MAX_CTX = 8192                      # total context (prompt + output)
MAX_NEW = 4096                      # max generated tokens
```

3. Set your prompts:

```python
messages = [
    {"role": "system", "content": "Type your role here"},
    {"role": "user", "content": "Prompt"},
]
```

4. Run:

```bash
python main.py
```

## Notes

- **Truncated output?** If the log shows `If Limited?: True`, the model hit `MAX_NEW`. Increase it or shorten the prompt.
- **Out of memory?** Lower `MAX_CTX` and `MAX_NEW`, or close other GPU-heavy apps.
- **Thinking models:** `enable_thinking=False` disables reasoning mode for models that support it. Remove this argument if your chat template does not.
- **Older GPUs:** if bf16 is unsupported, set `bnb_4bit_compute_dtype=torch.float16`.
- **Quantization trade-off:** 4-bit cuts memory use by roughly 3-4x compared to 16-bit, with a small drop in accuracy.
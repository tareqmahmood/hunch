# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "sentencepiece",
#   "safetensors",
# ]
# ///
"""PyTorch fp32 CPU baseline for `eg2 bench`: same text, same lengths."""
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).eval()
base = tok("The agent called read_file and then retried the same call. " * 200, return_tensors="pt")["input_ids"]
print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
with torch.inference_mode():
    for n in [int(a) for a in sys.argv[1:]] or [16, 128, 512, 1024]:
        ids = base[:, :n]
        model(input_ids=ids)
        iters = max(3, min(50, 2048 // n))
        t = time.perf_counter()
        for _ in range(iters):
            model(input_ids=ids)
        dt = (time.perf_counter() - t) / iters
        print(f"{n:>5} tokens: {dt * 1e3:>8.2f} ms/text  {n / dt:>8.0f} tokens/s")
    for n in [32, 128]:
        ids = base[:, :n].repeat(32, 1)  # batches of 32 identical-length texts (no padding)
        model(input_ids=ids)
        t = time.perf_counter()
        for _ in range(8):
            model(input_ids=ids)
        dt = time.perf_counter() - t
        print(f"8 batches of 32 x {n:>4}: {256 / dt:>8.1f} texts/s  {256 * n / dt:>8.0f} tokens/s")

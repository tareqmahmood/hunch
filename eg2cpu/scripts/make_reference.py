# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "sentencepiece",
#   "safetensors",
#   "numpy",
# ]
# ///
"""Dump PyTorch fp32 CPU reference outputs for the Rust port to test against.

    uv run scripts/make_reference.py tests/fixtures/reference.json

For each text: token ids, the mean-pooled state after every layer (24 x 512),
the per-token output of layer 0 for the first 8 tokens, and the final L2-normalized 768-d embedding.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

out = Path(sys.argv[1])
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).eval()

para = ("The agent called read_file on src/main.rs, got a permission error, then retried the same call "
        "with identical arguments before finally running ls -la to inspect the directory. ")
texts = [
    "hello world",
    "bash: git status\nOn branch main\nnothing to commit, working tree clean",
    "Tool call failed: FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml' — émoji 🚀 and 中文 too.",
    para * 18,   # ~ 700 tokens: exercises the 512 sliding window
    para * 60,   # truncated to 1024
]
layers = list(model.language_model.layers)
cases = []
with torch.no_grad():
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=1024)
        caught = []
        hooks = [m.register_forward_hook(lambda mod, i, o: caught.append(o[0] if isinstance(o, tuple) else o)) for m in layers]
        try:
            o = model(**enc)
        finally:
            for h in hooks:
                h.remove()
        cases.append({
            "text": t,
            "ids": enc["input_ids"][0].tolist(),
            "layer_means": [c[0].mean(0).tolist() for c in caught],
            "layer0_tokens": caught[0][0][:8].tolist(),
            "embedding": F.normalize(o.last_hidden_state[0].mean(0), dim=-1).tolist(),
        })
        print(len(cases[-1]["ids"]), "tokens")
out.write_text(json.dumps(cases))

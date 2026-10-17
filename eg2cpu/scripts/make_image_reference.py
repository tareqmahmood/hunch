# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "sentencepiece",
#   "safetensors",
#   "numpy",
#   "pillow",
#   "scikit-image",
# ]
# ///
"""Dump PyTorch fp32 CPU reference outputs for images, for the Rust port to test against.

    uv run scripts/make_image_reference.py tests/fixtures

Writes small test images to <dir>/images/ (downscaled scikit-image sample photos, which are public domain or CC0,
plus a synthetic RGBA gradient) and <dir>/image_reference.json. Images go through the checkpoint's own processor
(`AutoProcessor`: torchvision antialiased bicubic resize, patches of 16, 3x3 pooling) and the full model, as
`model(**processor(images=[img]))`. For each case: token ids, the resized image's size and a sample of its pixels,
the soft tokens the vision tower hands the text tower, all 24 layer means, and the final embedding. A few captions
are embedded too, for a text-image similarity check.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import skimage
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

out = Path(sys.argv[1])
(out / "images").mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
proc = AutoProcessor.from_pretrained("google/embeddinggemma-2")
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).eval()

data = Path(os.path.dirname(skimage.data.__file__))


def sample(name, size, fmt="PNG"):
    im = Image.open(data / name).convert("RGB")
    im = im.resize(size, Image.LANCZOS)
    path = out / "images" / f"{Path(name).stem}.{fmt.lower().replace('jpeg', 'jpg')}"
    im.save(path, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return path


def gradient():
    y, x = np.mgrid[0:24, 0:40]
    rgba = np.stack([x * 6, y * 10, (x + y) * 4, 128 + x * 3], -1).clip(0, 255).astype(np.uint8)
    path = out / "images" / "gradient_rgba.png"
    Image.fromarray(rgba, "RGBA").save(path)
    return path


files = {
    "cat": sample("chelsea.png", (300, 200)),
    "coffee": sample("coffee.png", (240, 160)),
    "astronaut": sample("astronaut.png", (192, 192)),
    "rocket": sample("rocket.jpg", (160, 240), "JPEG"),
    "gradient": gradient(),
}
cases_spec = [("cat", 280), ("coffee", 280), ("astronaut", 280), ("rocket", 280), ("coffee", 70), ("gradient", 70)]
captions = ["a photo of a cat", "a cup of coffee", "an astronaut in a space suit", "a rocket on the launch pad"]

layers = list(model.language_model.layers)


def run(**inputs):
    caught = []
    hooks = [m.register_forward_hook(lambda mod, i, o: caught.append(o[0] if isinstance(o, tuple) else o)) for m in layers]
    try:
        o = model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return o, caught


rng = np.random.default_rng(0)
cases = []
with torch.no_grad():
    for key, n in cases_spec:
        path = files[key]
        img = Image.open(path).convert("RGB")
        inputs = proc(images=[img], return_tensors="pt", max_soft_tokens=n)
        pos = inputs["image_position_ids"][0]
        valid = (pos != -1).all(-1)
        pw, ph = int(pos[valid][:, 0].max()) + 1, int(pos[valid][:, 1].max()) + 1
        # The resized image, recovered from the patches: [ph*16, pw*16, 3] in [0, 1].
        patches = inputs["pixel_values"][0][valid].reshape(ph, pw, 16, 16, 3)
        pixels = (patches.permute(0, 2, 1, 3, 4).reshape(ph * 16, pw * 16, 3) * 255).round().to(torch.int64).numpy()
        idx = rng.integers(0, pixels.size, 512)
        o, caught = run(**inputs)
        feats = o.image_hidden_states
        cases.append({
            "file": f"images/{path.name}",
            "max_soft_tokens": n,
            "ids": inputs["input_ids"][0].tolist(),
            "size": [ph * 16, pw * 16],
            "pixel_sum": int(pixels.sum()),
            "pixel_index": idx.tolist(),
            "pixel_values": pixels.reshape(-1)[idx].tolist(),
            "soft_tokens_head": feats[:2].tolist(),
            "soft_tokens_mean": feats.mean(0).tolist(),
            "layer_means": [c[0].mean(0).tolist() for c in caught],
            "embedding": F.normalize(o.last_hidden_state[0].mean(0), dim=-1).tolist(),
        })
        print(path.name, n, "soft tokens:", feats.shape[0], "size:", cases[-1]["size"])

    text_embs = []
    for c in captions:
        enc = tok(c, return_tensors="pt")
        o, _ = run(**enc)
        text_embs.append(F.normalize(o.last_hidden_state[0].mean(0), dim=-1))
    sims = [[float(torch.tensor(case["embedding"]) @ t) for t in text_embs] for case in cases]
    for case, row in zip(cases, sims):
        print(f"{case['file']:<28} {case['max_soft_tokens']:>4}", " ".join(f"{s:.3f}" for s in row))

(out / "image_reference.json").write_text(json.dumps({"cases": cases, "captions": captions, "similarity": sims}))

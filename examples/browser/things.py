"""Embed the object demo's text prompts with eg2cpu once, for the browser: the page only runs EG2's image path, so
the things it knows zero-shot arrive as one vector each in things.json.

    uv run examples/browser/things.py

The things and their prompts are the ones examples/objects/objects.py uses.
"""

import ast
import json
from pathlib import Path

import numpy as np

from eg2cpu import Model

HERE = Path(__file__).parent
source = (HERE.parent / "objects" / "objects.py").read_text()


def constant(name):
    """A literal assigned at the top level of objects.py."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def a_or_an(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


model = Model()
things = [("nothing", "🫥", constant("NOTHING_PROMPTS")), ("hand", "✋", constant("HAND_PROMPTS"))]
things += [(name, emoji, [f"a person holding up {a_or_an(name)} to a webcam", f"a close-up photo of {a_or_an(name)}"])
           for name, emoji in constant("KNOWN").items()]
out = []
for name, emoji, prompts in things:
    P = np.stack([np.asarray(model.embed(p)[0], np.float32) for p in prompts])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    out.append({"name": name, "emoji": emoji, "prompt": [round(float(v), 6) for v in P.mean(0)]})
phrases = HERE.parent / "objects" / "frames" / "phrases.json"  # Clef's phrasing, when the objects demo has made it
(HERE / "things.json").write_text(json.dumps({"things": out, "phrases": json.loads(phrases.read_text()) if phrases.exists() else {}}))
print(f"{len(out)} things -> {HERE / 'things.json'}")

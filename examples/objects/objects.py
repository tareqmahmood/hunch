"""A webcam that names what you hold up, out loud, and learns what it doesn't know. EG2 embeds each frame once on the
CPU; about forty everyday things are known from text prompts alone, and when it's unsure it asks "what's that?",
listens, and learns the thing from the next two seconds of frames. With --teacher, Clef watches too: a vision model,
local on the GPU or from OpenRouter, that answers in under a second, labels the frames it's sure about for the
student, and says when it sees something that isn't any of the known things.

    uv run --with faster-whisper --with kokoro-onnx examples/objects/objects.py   # then open http://localhost:8766
    uv run --with faster-whisper --with kokoro-onnx --with "mlx-vlm>=0.7.4,<0.8" --with pillow \\
        examples/objects/objects.py --teacher path/to/clef-flash-4bit        # Clef as the teacher (Apple silicon)
    OPENROUTER_API_KEY=... uv run --with faster-whisper --with kokoro-onnx \\
        examples/objects/objects.py --teacher openrouter:cloudflare/clef-flash   # Clef from OpenRouter
    ... --tokens 140                                              # smaller things, about half the frame rate

How it answers:
  empty hands  the first two seconds after start (or holding 0) are labelled "nothing": your face and room, empty
               handed. A frame's distance from them says whether anything is being shown, and what differs from them
               says how much a frame resembles a taught thing.
  zero-shot    every thing has two text prompts ("a person holding up a banana to a webcam"); a frame scores each
               thing by its similarity to them, standardised across the things, and the nearest wins. "Nothing" and
               "hand" (a bare hand or a gesture) have prompts too, and are never named out loud.
  taught       saying a thing's name labels the next two seconds of frames with it (a new thing gets prompts and an
               emoji from its name), and the teacher labels the frames it's sure about. A logistic regression on the
               labelled frames (standardised, classes weighted equally) decides between the taught things, and it
               answers as far as the frame resembles the taught frames: what differs from the empty-handed frames is
               compared with the same for each taught frame. Things never taught keep their zero-shot share.
  the teacher  Clef looks at the newest frame when something is being shown and the student hasn't just been
               confirmed on it (and every few seconds otherwise), one look at a time with a rest between, and picks
               one of the things or "something else". Sure of a thing, it labels that frame for the student; sure of
               something else, it makes the page ask. It also says how each thing is named out loud: "that's an egg",
               "those are scissors", "that's rice".

Speech: Kokoro speaks on the CPU (82M parameters, about 0.2 s a line, downloaded once; --voice picks one, and
"--voice browser" or no kokoro-onnx uses the browser's own voices). What you say is sent here as 16 kHz samples and
transcribed by faster-whisper on the CPU (small.en, downloaded once, told the names of the things it knows). Hold
space to talk at any time ("that's a lemon"); Enter types a name instead.

Labelled frames are saved in frames/ (JPEG, embedding and labels.jsonl, with who labelled each) and taught again at
start; "forget" sets the labels aside as labels.<time>.jsonl and goes back to the things known from prompts alone.
"""

import argparse
import base64
import collections
import io
import json
import os
import re
import struct
import sys
import threading
import time
import unicodedata
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eg2cpu import Model

NOTHING, HAND = "nothing", "hand"  # quiet: never named out loud, never asked about
ELSE = "something else"  # the teacher's answer for a thing that isn't any of the known ones
KNOWN = {  # thing: emoji, known from text prompts alone
    "banana": "🍌", "apple": "🍎", "orange": "🍊", "lemon": "🍋", "avocado": "🥑", "strawberry": "🍓", "grapes": "🍇",
    "carrot": "🥕", "broccoli": "🥦", "egg": "🥚", "donut": "🍩", "cookie": "🍪", "coffee mug": "☕",
    "wine glass": "🍷", "water bottle": "💧", "soda can": "🥫", "spoon": "🥄", "fork": "🍴", "book": "📕",
    "phone": "📱", "headphones": "🎧", "sunglasses": "🕶️", "watch": "⌚", "keys": "🔑", "scissors": "✂️", "pen": "🖊️",
    "pencil": "✏️", "cat": "🐱", "dog": "🐶", "teddy bear": "🧸", "tennis ball": "🎾", "game controller": "🎮",
    "remote control": "📺", "light bulb": "💡", "potted plant": "🪴", "flower": "🌸", "toothbrush": "🪥",
    "baseball cap": "🧢", "shoe": "👟", "sock": "🧦", "camera": "📷", "rubber duck": "🦆", "toy car": "🚗",
    "candle": "🕯️", "laptop": "💻",
}
NOTHING_PROMPTS = ["a person at a desk looking at a webcam with empty hands", "a webcam photo of a face, nothing held up",
                   "a person sitting in a room, holding nothing"]
HAND_PROMPTS = ["an empty hand held up to a webcam", "a person making a hand gesture",
                "a person waving their hand at the camera"]
TEACHER_SAYS = {NOTHING: "Nothing is held up; just a person or a room",
                HAND: "A bare hand or a hand gesture, with nothing in it",
                ELSE: "Something is held up, but none of the other options"}
TEMPERATURE = 2.5  # on standardised similarities
SAY = 0.5  # a thing this likely, held steady, is named out loud
SHOWN = 3.0  # a frame this many empty-handed spreads from the empty-handed frames is showing something
RESEMBLE, RESEMBLE_SCALE = 0.6, 0.05  # likeness to the taught frames at which the student and the prompts weigh the same
TEACHER_SURE = 0.5  # the teacher labels a frame (or asks about it) when its answer is this likely
TEACHER_CAP = 60  # frames the teacher labels per thing; after that it only checks
TEACHER_FRESH = 1.5  # seconds a teacher's answer stands for the frames after it
TEACHER_REST = 0.3  # seconds between the teacher's looks, so EG2 has the machine to itself between them
CONFIRMED = 3.0  # seconds the teacher doesn't look again at a thing it agreed with the student on
IDLE_LOOK = 3.0  # seconds between the teacher's looks when nothing (or a bare hand) is being shown

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--tokens", type=int, default=70, help="soft tokens per frame (70, 140, 280)")
ap.add_argument("--voice", default="af_heart", help="a Kokoro voice (af_heart, af_bella, am_michael, bf_emma...) or browser")
ap.add_argument("--teacher", help="Clef, to teach the student as it watches: a local MLX model directory "
                                   "(clef-flash-4bit), or openrouter:cloudflare/clef-flash (OPENROUTER_API_KEY)")
args = ap.parse_args()

model = Model()
lock = threading.Lock()  # one forward pass at a time; frames queue in the browser, not here


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def a_or_an(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


PHRASES = {"a": ("That's", "a {}"), "an": ("That's", "an {}"), "plural": ("Those are", "{}"), "mass": ("That's", "{}")}


def naming(name):
    """How to say a thing out loud: Clef's phrasing once it has one, "a" or "an" before that."""
    kind = teacher.phrases.get(name) if teacher is not None else None
    verb, noun = PHRASES[kind] if kind else ("That's", a_or_an(name))
    return {"sentence": f"{verb} {noun.format(name)}!", "noun": noun.format(name)}


def prompts_for(name):
    return [f"a person holding up {a_or_an(name)} to a webcam", f"a close-up photo of {a_or_an(name)}"]


def embed_texts(texts):
    return unit(np.stack([model.embed(t)[0] for t in texts]))


class Emojis:
    """Every single-character emoji, by its Unicode name, so a spoken name finds the nearest one."""

    def __init__(self, cache):
        chars = [chr(c) for r in ((0x1F300, 0x1F650), (0x1F680, 0x1F700), (0x1F900, 0x1FB00), (0x2600, 0x2800))
                 for c in range(*r)]
        self.chars = [c for c in chars if unicodedata.name(c, "") and unicodedata.category(c) == "So"]
        self.names = [unicodedata.name(c).lower() for c in self.chars]
        if cache.exists() and len(np.load(cache)) == len(self.chars):
            self.vectors = np.load(cache)
        else:
            self.vectors = embed_texts([f"an emoji of {n}" for n in self.names])
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, self.vectors)

    def find(self, name):
        for words in (name, name.rstrip("s")):
            if words in self.names:
                return self.chars[self.names.index(words)]
        return self.chars[int(np.argmax(self.vectors @ embed_texts([f"an emoji of {name}"])[0]))]


class Things:
    """Everything it can name: prompts for each, labelled frames for some. Labels refit the student in the
    background, so teaching at the frame rate never waits on it."""

    def __init__(self):
        self.names, self.emoji, self.prompts = [], [], []
        self.labelled: list[tuple[np.ndarray, int, str]] = []
        self.student = self.empty = self.residuals = None
        self.changed = threading.Event()
        self.adding = threading.Lock()
        self.add(NOTHING, "🫥", NOTHING_PROMPTS)
        self.add(HAND, "✋", HAND_PROMPTS)
        for name, emoji in KNOWN.items():
            self.add(name, emoji, prompts_for(name))
        self.known = len(self.names)  # the things known from prompts alone; names taught later come after
        threading.Thread(target=self.refitting, daemon=True).start()

    def add(self, name, emoji, prompts):
        self.prompts.append(embed_texts(prompts))
        self.emoji.append(emoji)
        self.names.append(name)  # last: a name is only answered once its prompts are in place

    def find(self, name):
        for n in (name, name + "s", name.rstrip("s"), name.removesuffix("es")):
            if n in self.names:
                return self.names.index(n)
        return None

    def label(self, x, name, by="you"):
        with self.adding:
            i = self.find(name)
            if i is None:
                self.add(name, emojis.find(name), prompts_for(name))
                i = len(self.names) - 1
        self.labelled.append((x, i, by))
        self.changed.set()
        return self.names[i]

    def forget(self):
        """Back to the things known from prompts alone: the taught frames go, and so do the taught names."""
        with self.adding:
            self.labelled = []
            self.student = self.empty = self.residuals = None
            k = self.known
            self.names, self.emoji, self.prompts = self.names[:k], self.emoji[:k], self.prompts[:k]
        self.changed.set()

    def counts(self):
        out = collections.defaultdict(lambda: {"you": 0, "clef": 0})
        names = self.names
        for _, i, by in self.labelled:
            if i < len(names):
                out[names[i]][by] += 1
        return dict(out)

    def refitting(self):
        while True:
            self.changed.wait()
            self.changed.clear()
            self.refit()
            time.sleep(0.2)  # labels arriving meanwhile are taken together

    def refit(self):
        labelled = list(self.labelled)
        ys = np.array([y for _, y, _ in labelled])
        X = np.stack([x for x, _, _ in labelled]) if labelled else None
        student = empty = residuals = None
        if (ys == 0).sum() >= 2:  # the empty-handed frames: where they sit, and how far apart they spread
            E = X[ys == 0]
            centre = E.mean(0)
            d = np.linalg.norm(E - centre, axis=1)
            empty = {"centre": centre, "d": float(np.median(d)), "spread": float(d.std() + 1e-4)}
            residuals = unit(X[ys > 1] - centre) if (ys > 1).any() else None  # things, not nothing or a hand
        if len(set(ys.tolist())) >= 2:
            student = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
            student.fit(X, ys)
        self.student, self.empty, self.residuals = student, empty, residuals

    def answer(self, x):
        names, emoji, prompts = self.names, self.emoji, self.prompts  # forget() swaps these for shorter lists
        n = min(len(names), len(emoji), len(prompts))
        student, empty, residuals = self.student, self.empty, self.residuals
        if student is not None and student.classes_.max() >= n:  # fitted before a forget
            student = None
        s = np.array([float((P @ x).mean()) for P in prompts[:n]])
        p = zero = softmax(TEMPERATURE * (s - s.mean()) / (s.std() + 1e-6))
        shown, resemble = None, 0.0
        if empty is not None:
            shown = (float(np.linalg.norm(x - empty["centre"])) - empty["d"]) / empty["spread"]
            if residuals is not None:
                resemble = float(np.max(residuals @ unit(x - empty["centre"])))
        if student is not None:
            taught = np.zeros(n)
            taught[student.classes_] = student.predict_proba(x[None])[0]
            # the student answers as far as the frame looks like the things it was taught, or like empty hands
            w = float(sigmoid((resemble - RESEMBLE) / RESEMBLE_SCALE)) if residuals is not None else 0.0
            if shown is not None:
                w = max(w, float(sigmoid(2 * (SHOWN - shown))))
            p = (1 - w) * zero + w * taught
        best = int(np.argmax(p))
        if names[best] in (NOTHING, HAND):
            state = "nothing"
        elif p[best] >= SAY:
            state = "known"
        else:  # no thing stands out: worth asking about only if something is being shown
            state = "unsure" if shown is None or shown >= SHOWN else "nothing"
        top = np.argsort(-p)[:5]
        return {"state": state, "name": names[best], "emoji": emoji[best], "p": round(float(p[best]), 3),
                "top": [{"name": names[i], "emoji": emoji[i], "p": round(float(p[i]), 3),
                         "zero": round(float(zero[i]), 3)} for i in top],
                "shown": None if shown is None else round(shown, 2), "resemble": round(resemble, 3),
                "labelled": self.counts(), "ready": empty is not None}


class LocalClef:
    """Clef from a local MLX model directory, on the GPU."""

    def __init__(self, path):
        sys.path.insert(0, str(Path(path).resolve()))
        import clef_mlx
        from PIL import Image
        self.Image = Image
        self.model = clef_mlx.load(path)
        self.spent = 0.0

    def predict(self, text, questions, image=None):
        record = {"state": {"task": text}, "questions": questions}
        if image is not None:
            record["images"] = [self.Image.open(io.BytesIO(image)).convert("RGB")]
        return self.model.predict(record)


class OpenRouterClef:
    """Clef from OpenRouter's decisions endpoint (OPENROUTER_API_KEY), adding up what it costs."""

    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model):
        self.model, self.key = model, os.environ["OPENROUTER_API_KEY"]
        self.spent = 0.0

    def predict(self, text, questions, image=None):
        state = [{"type": "text", "text": text}]
        if image is not None:
            kind = "png" if image.startswith(b"\x89PNG") else "jpeg"
            url = f"data:image/{kind};base64," + base64.b64encode(image).decode()
            state.append({"type": "image_url", "image_url": {"url": url}})
        request = urllib.request.Request(self.URL, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                                         data=json.dumps({"model": self.model, "state": state, "questions": questions}).encode())
        out = json.load(urllib.request.urlopen(request, timeout=30))
        self.spent += out.get("usage", {}).get("cost", 0.0)
        return {q: a["probabilities"] for q, a in out["answers"].items()}


class Teacher:
    """Clef looks at the newest frame when it's worth it. Sure of a thing, it labels the frame for the student."""

    def __init__(self, where):
        self.clef = OpenRouterClef(where.removeprefix("openrouter:")) if where.startswith("openrouter:") else LocalClef(where)
        self.newest = None
        self.wake = threading.Event()
        self.last = None
        self.agree = collections.deque(maxlen=50)
        self.agreed = {}  # thing: when the teacher last agreed with the student on it
        self.looked = collections.deque(maxlen=200)  # when the teacher looked
        self.busy = threading.Lock()  # one question to Clef at a time
        self.phrases = json.loads(PHRASED.read_text()) if PHRASED.exists() else {}
        self.ask(grey_png())  # loads the vision tower before the first frame
        threading.Thread(target=self.watching, daemon=True).start()
        threading.Thread(target=lambda: [self.phrase(n) for n in things.names[2:]], daemon=True).start()

    def predict(self, text, questions, image=None):
        with self.busy:
            return self.clef.predict(text, questions, image)

    def phrase(self, name):
        if name not in self.phrases:
            question = {"type": "choice", "criteria": {k: f"{v} {n.format(name)}." for k, (v, n) in PHRASES.items()},
                        "instructions": f"Someone points at a {name!r} (or some) and names it out loud. "
                                        "Which sentence is correct English?"}
            out = self.predict(json.dumps({"thing": name}), {"phrase": question})["phrase"]
            self.phrases[name] = max(out, key=out.get)
            FRAMES.mkdir(exist_ok=True)
            PHRASED.write_text(json.dumps(self.phrases, indent=0))
        return self.phrases[name]

    def forget(self):
        self.agree.clear()
        self.agreed.clear()
        self.last = None

    def look(self, jpeg, x, student):
        """Worth a look: a thing the teacher hasn't just agreed with the student on, or now and then otherwise."""
        now = time.time()
        if student["state"] == "nothing":  # idle: an occasional look, in case a held thing reads as a bare hand
            since = now - self.looked[-1] if self.looked else IDLE_LOOK
            worth = (student["shown"] or 0) >= SHOWN and since >= IDLE_LOOK
        else:
            worth = now - self.agreed.get(student["name"], 0) > CONFIRMED
        if worth:
            self.newest = (jpeg, x)
            self.wake.set()

    def ask(self, image):
        names = things.names[:]
        criteria = {n: TEACHER_SAYS.get(n, naming(n)["noun"].capitalize()) for n in names} | {ELSE: TEACHER_SAYS[ELSE]}
        question = {"type": "choice", "instructions": "What is the person holding up to the camera?", "criteria": criteria}
        return self.predict("A webcam frame.", {"thing": question}, image)["thing"]

    def watching(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            jpeg, x = self.newest
            self.looked.append(time.time())
            t0 = time.perf_counter()
            out = self.ask(jpeg)
            ms = (time.perf_counter() - t0) * 1000
            best = max(out, key=out.get)
            p = out[best]
            i = things.find(best) if best != ELSE else None
            self.last = {"name": best, "emoji": "❓" if i is None else things.emoji[i], "p": round(p, 3),
                         "ms": round(ms), "at": time.time()}
            if p >= TEACHER_SURE and i is not None:
                agreed = things.answer(x)["name"] == best
                self.agree.append(agreed)
                if agreed:
                    self.agreed[best] = time.time()
                if things.counts().get(best, {}).get("clef", 0) < TEACHER_CAP:
                    things.label(x, best, by="clef")
                    save(jpeg, x, best, "clef")
            time.sleep(TEACHER_REST)

    def describe(self):
        if self.last is None:
            return None
        age = time.time() - self.last["at"]
        minute = sum(1 for t in self.looked if time.time() - t < 60)
        return self.last | {"at": None, "age": round(age, 2), "fresh": age < TEACHER_FRESH, "per_minute": minute,
                            "spent": round(self.clef.spent, 6),
                            "agree": round(float(np.mean(self.agree)), 3) if self.agree else None, "checked": len(self.agree)}


FILLER = re.compile(r"^(?:(?:no|nope|wrong|actually|um+|uh+|so|well|okay|ok|oh|hmm+)\b[,.!]?\s*)*"
                    r"(?:(?:it'?s|it is|that'?s|that is|this is|this'?s|these are|those are|i'?ve got|i have)\s+)?"
                    r"(?:(?:a|an|the|my|some|one)\s+)?")


def name_from(text):
    words = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    words = FILLER.sub("", words).strip()
    words = " ".join(words.split()[:3])
    return words if len(words) > 1 else ""


class Ears:
    """faster-whisper on the CPU (small.en), loaded on first use, told the names of the things it may hear."""

    def __init__(self):
        self.model = None

    def hear(self, samples):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("small.en", device="cpu", compute_type="int8")
        hint = "Naming things held up to a camera: " + ", ".join(things.names[2:]) + "."
        segments, _ = self.model.transcribe(samples, language="en", beam_size=1, vad_filter=True, initial_prompt=hint)
        return " ".join(s.text for s in segments).strip()


class Voice:
    """Kokoro on the CPU, its model downloaded on first start; each line is made once and kept."""

    FILES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

    def __init__(self, name, home):
        from kokoro_onnx import Kokoro
        home.mkdir(parents=True, exist_ok=True)
        for f in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"):
            if not (home / f).exists():
                print(f"downloading {f}", flush=True)
                urllib.request.urlretrieve(self.FILES + f, home / f".{f}")
                (home / f".{f}").rename(home / f)
        self.kokoro = Kokoro(str(home / "kokoro-v1.0.int8.onnx"), str(home / "voices-v1.0.bin"))
        self.name = name
        self.lines = {}
        self.busy = threading.Lock()

    def say(self, text):
        if text not in self.lines:
            with self.busy:
                samples, rate = self.kokoro.create(text, voice=self.name, speed=1.05, lang="en-us")
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            self.lines[text] = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                                + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                                + b"data" + struct.pack("<I", len(pcm)) + pcm)
        return self.lines[text]


def voice_for(name):
    if name == "browser":
        return None
    try:
        return Voice(name, FRAMES / "kokoro")
    except ImportError:
        print("kokoro-onnx isn't installed: the browser's voices will speak", flush=True)
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply((Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/frame":  # JPEG bytes in, the answer out; X-Label: <thing> labels the frame
            with lock:  # timed inside: the meter shows EG2's time, not the wait for another frame
                t0 = time.perf_counter()
                x = unit(model.embed_image(body, max_soft_tokens=args.tokens)[0])
                ms = (time.perf_counter() - t0) * 1000
            name = (self.headers.get("X-Label") or "").strip().lower()
            if name:
                save(body, x, things.label(x, name), "you")
            out = {"ms": round(ms, 1), "tokens": args.tokens} | things.answer(x)
            out["say"] = naming(out["name"])["sentence"]
            if teacher is not None:
                teacher.look(body, x, out)
                out["teacher"] = said = teacher.describe()
                if said and said["fresh"] and said["name"] == ELSE and said["p"] >= TEACHER_SURE and out["state"] != "nothing":
                    out["state"] = "unsure"  # the teacher sees a thing that isn't any of the known ones
            self.reply(out)
        elif self.path == "/hear":  # speech in (16 kHz mono float32 samples), the words and the thing they name out
            t0 = time.perf_counter()
            text = ears.hear(np.frombuffer(body, dtype=np.float32))
            name = name_from(text)
            i = things.find(name) if name else None
            emoji = things.emoji[i] if i is not None else (emojis.find(name) if name else "")
            name = things.names[i] if i is not None else name
            if name and teacher is not None:
                teacher.phrase(name)
            self.reply({"text": text, "name": name, "emoji": emoji, "known": i is not None,
                        "noun": naming(name)["noun"] if name else "", "ms": round((time.perf_counter() - t0) * 1000)})
        elif self.path == "/say":  # a line of text in, the voice saying it out (WAV), or 404 for the browser's voices
            if voice is None:
                self.send_error(404)
            else:
                self.reply(voice.say(body.decode()), "audio/wav")
        elif self.path == "/forget":
            things.forget()
            if teacher is not None:
                teacher.forget()
            if LABELS.exists():
                LABELS.rename(FRAMES / f"labels.{int(time.time())}.jsonl")
            self.reply({})
        else:
            self.send_error(404)


def grey_png(side=64):
    rows = b"".join(b"\0" + b"\x80" * 3 * side for _ in range(side))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


FRAMES = Path(__file__).parent / "frames"
LABELS = FRAMES / "labels.jsonl"
PHRASED = FRAMES / "phrases.json"  # how Clef said to name each thing, kept across restarts


def save(jpeg, x, name, by):
    FRAMES.mkdir(exist_ok=True)
    path = FRAMES / f"{time.time_ns()}.jpg"
    path.write_bytes(jpeg)
    np.save(path.with_suffix(f".{args.tokens}.npy"), x)
    with LABELS.open("a") as f:
        f.write(json.dumps({"file": path.name, "thing": name, "by": by}) + "\n")


def saved_embedding(name):
    cached = (FRAMES / name).with_suffix(f".{args.tokens}.npy")
    if cached.exists():
        return np.load(cached)
    x = unit(model.embed_image((FRAMES / name).read_bytes(), max_soft_tokens=args.tokens)[0])
    np.save(cached, x)
    return x


emojis = Emojis(FRAMES / "emoji-names.npy")
things = Things()
ears = Ears()
voice = voice_for(args.voice)
if LABELS.exists():  # teach again what was taught before the restart
    rows = [json.loads(line) for line in LABELS.read_text().splitlines()]
    for r in rows:
        things.label(saved_embedding(r["file"]), r["thing"], r.get("by", "you"))
    print(f"taught again from {len(rows)} saved frames", flush=True)
model.embed_image(grey_png(), max_soft_tokens=args.tokens)  # loads the vision tower before the first frame
teacher = None  # naming() reads it while Clef loads
teacher = Teacher(args.teacher) if args.teacher else None
print(f"objects: http://localhost:{args.port}  ({args.tokens} soft tokens per frame, {len(things.names)} things"
      + (", Clef teaching)" if teacher else ")"), flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

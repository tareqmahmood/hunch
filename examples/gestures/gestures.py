"""A live webcam recognizer: EG2 embeds each frame once, and every question answers from that one embedding. Two
questions pick one answer (what's the hand doing? what's the face showing?), and three detectors say yes or no (fire?
a cat? your thing?). Text prompts answer zero-shot, and frames you label teach each question as it goes.

    uv run gestures/gestures.py            # then open http://localhost:8765 and allow the camera
    uv run gestures/gestures.py --tokens 140   # sharper (smaller hands, farther away) at about half the frame rate

How a question that picks one answer answers:
  zero-shot  every answer is a few text prompts ("a hand giving a thumbs up"); a frame scores each answer by its
             similarity to that answer's prompts, standardised across the question's answers, so the nearest one wins
  taught     holding an answer's key labels frames with it; a logistic regression on the question's labelled frames
             (standardised, so it learns what differs between your frames rather than the room and face they share;
             classes weighted equally, however long each key was held) answers once two of its answers have frames.
             It only reshares the probability the prompts give the answers it was taught; answers without frames keep
             their zero-shot share. Its weight grows with the frames you label (half and half at 8 frames).

How a detector answers:
  zero-shot  a frame's similarity to the "yes" prompts minus its similarity to the "no" prompts, against a threshold
             set between webcam frames and frames with a fire or cat pasted in
  taught     one frame with the thing and one without, in the same scene, is enough: the direction from the "no"
             frames to the "yes" frames is added to the prompts' direction, and the threshold moves halfway between
             them. A detector without prompts ("your thing") is that direction alone, so it only answers once taught.

Labelled frames are saved in frames/ (JPEG, embedding and labels.jsonl) and taught again at start; "forget" sets the
labels aside as labels.<time>.jsonl.
"""

import argparse
import json
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eg2cpu import Model

QUESTIONS = {  # question: (title, {answer: (emoji, key, prompts)}, the answer that means nothing to show)
    "gesture": ("What's the hand doing?", {
        "none": ("🙈", "0", ["a photo of a person with no hands visible", "a webcam photo of a room with nobody in it",
                             "a webcam photo of a face, hands out of view"]),
        "thumbs_up": ("👍", "1", ["a hand giving a thumbs up", "a photo of a thumbs up gesture"]),
        "thumbs_down": ("👎", "2", ["a hand giving a thumbs down", "a photo of a thumbs down gesture"]),
        "open_palm": ("✋", "3", ["an open hand with the palm facing the camera", "a hand raised with all five fingers spread"]),
        "fist": ("✊", "4", ["a closed fist", "a hand clenched into a fist"]),
        "peace": ("✌️", "5", ["a hand making a peace sign with two fingers", "a hand showing the V sign"]),
        "point": ("☝️", "6", ["a hand pointing up with the index finger", "one finger raised"]),
        "ok": ("👌", "7", ["a hand making the OK sign", "thumb and index finger touching in a circle"]),
        "rock": ("🤘", "8", ["a hand making the rock and roll horns sign", "index and little finger raised, devil horns"]),
    }, "none"),
    "emotion": ("What's the face showing?", {
        "no_face": ("🫥", "x", ["a webcam photo of an empty room", "a photo with no face in it"]),
        "neutral": ("😐", "n", ["a person with a neutral, expressionless face", "a calm face looking at the camera"]),
        "happy": ("😄", "h", ["a person smiling happily", "a big grin, a happy laughing face"]),
        "sad": ("😢", "s", ["a person with a sad face", "a frowning, unhappy face about to cry"]),
        "surprised": ("😮", "u", ["a surprised face with mouth wide open", "a shocked person with raised eyebrows"]),
        "angry": ("😠", "a", ["an angry face with furrowed brows", "a person scowling in anger"]),
        "silly": ("😜", "t", ["a person sticking their tongue out", "a silly goofy face pulling a funny expression"]),
    }, "no_face"),
}
DETECTORS = {  # detector: (title, emoji, key for yes (shift for no), no prompts, yes prompts, zero-shot threshold)
    "fire": ("Fire?", "🔥", "f", ["a normal indoor webcam photo", "a room with no fire or flames"],
             ["a photo of fire with bright flames", "a burning flame from a lighter or candle", "a fire burning"], -0.04),
    "cat": ("A cat?", "🐱", "c", ["a webcam photo with no animals", "a person in a room, no cat"],
            ["a photo of a cat", "a cat in front of the camera", "a kitten"], -0.03),
    "thing": ("Your thing? (no prompts: teach it once)", "⭐", "m", [], [], None),
}
TEMPERATURE = 2.5  # on standardised similarities
HALF_WEIGHT_AT = 8  # labelled frames at which the student and the prompts weigh the same (about a second of teaching)
ZERO_SCALE = 0.01  # detector scores this far past the threshold are p = 0.73 (negatives sit ~0.05 below it)
NUDGE = 0.5  # how far the taught direction turns a detector's prompt direction

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--tokens", type=int, default=70, help="soft tokens per frame (70, 140, 280)")
args = ap.parse_args()

model = Model()
lock = threading.Lock()  # one forward pass at a time; frames queue in the browser, not here


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z):
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


def mean_prompt(prompts):
    return np.stack([unit(model.embed(p)[0]) for p in prompts]).mean(0)


class Taught:
    """Labelled frames for one question."""

    def __init__(self, names):
        self.names = names
        self.labelled: list[tuple[np.ndarray, int]] = []

    def label(self, x, name, refit=True):
        self.labelled.append((x, self.names.index(name)))
        if refit:
            self.refit()

    def forget(self):
        self.labelled.clear()
        self.refit()

    def counts(self):
        return {n: sum(1 for _, y in self.labelled if y == i) for i, n in enumerate(self.names)}

    def named(self, v):
        return None if v is None else dict(zip(self.names, np.round(v, 4).tolist()))


class Question(Taught):
    """Picks one answer."""

    def __init__(self, title, answers, quiet):
        super().__init__(list(answers))
        self.title, self.answers, self.quiet = title, answers, quiet
        self.prompts = [np.stack([unit(model.embed(p)[0]) for p in answers[a][2]]) for a in self.names]
        self.student = None

    def describe(self):
        return {"title": self.title, "answers": [{"name": n, "emoji": e, "key": k, "quiet": n == self.quiet}
                                                 for n, (e, k, _) in self.answers.items()]}

    def refit(self):
        ys = [y for _, y in self.labelled]
        self.student = None
        if len(set(ys)) >= 2:
            self.student = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
            self.student.fit(np.stack([x for x, _ in self.labelled]), ys)

    def answer(self, x):
        s = np.array([float((P @ x).mean()) for P in self.prompts])
        zero = softmax(TEMPERATURE * (s - s.mean()) / (s.std() + 1e-6))
        p, mode = zero, "zero-shot: text prompts only"
        if self.labelled:
            mode = f"zero-shot until two answers have frames ({len(self.labelled)} labelled)"
        if self.student is not None:
            known = self.student.classes_
            taught = np.zeros(len(self.names))
            taught[known] = self.student.predict_proba(x[None])[0]
            w = len(self.labelled) / (len(self.labelled) + HALF_WEIGHT_AT)
            share = zero[known] / zero[known].sum()
            mixed = np.exp((1 - w) * np.log(share) + w * np.log(np.maximum(taught[known], 1e-6)))  # geometric blend
            p = zero.copy()
            p[known] = zero[known].sum() * mixed / mixed.sum()
            mode = f"taught: {len(self.labelled)} labelled frames, student weight {w:.0%}"
        return {"p": self.named(p), "zero": self.named(zero), "mode": mode, "labelled": self.counts()}


class Detector(Taught):
    """Says yes or no."""

    def __init__(self, title, emoji, key, no, yes, threshold):
        super().__init__(["no", "yes"])
        self.title, self.emoji, self.key, self.threshold = title, emoji, key, threshold
        self.v0 = mean_prompt(yes) - mean_prompt(no) if yes else None
        self.v = self.cut = self.scale = None

    def describe(self):
        return {"title": self.title, "answers": [
            {"name": "no", "emoji": "✖️", "key": "⇧" + self.key, "press": self.key.upper(), "quiet": True},
            {"name": "yes", "emoji": self.emoji, "key": self.key, "press": self.key, "alert": True}]}

    def refit(self):
        no = [x for x, y in self.labelled if y == 0]
        yes = [x for x, y in self.labelled if y == 1]
        self.v = None
        if no and yes:
            mn, my = np.mean(no, 0), np.mean(yes, 0)
            d = unit(my - mn)
            self.v = d if self.v0 is None else unit(self.v0) + NUDGE * d
            a, b = float(mn @ self.v), float(my @ self.v)
            self.cut, self.scale = (a + b) / 2, max(abs(b - a) / 8, 1e-4)  # the taught frames sit at p ~ 0.02 and 0.98

    def answer(self, x):
        zero = None if self.v0 is None else sigmoid((float(x @ self.v0) - self.threshold) / ZERO_SCALE)
        n = self.counts()
        if self.v is not None:
            p, mode = sigmoid((float(x @ self.v) - self.cut) / self.scale), f"taught: {n['yes']} frames with, {n['no']} without"
        elif zero is not None:
            p, mode = zero, "zero-shot: text prompts only" + (" (teach a frame with and one without)" if self.labelled else "")
        else:
            p, mode = 0.0, f"show it once: hold {self.key} with your thing in view, shift+{self.key} without"
        return {"p": self.named([1 - p, p]), "zero": None if zero is None else self.named([1 - zero, zero]),
                "mode": mode, "labelled": n}


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
        elif self.path == "/questions":
            self.reply({"tokens": args.tokens, "questions": {q: question.describe() for q, question in questions.items()}})
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/frame":  # JPEG bytes in, every question's answer out; X-Label: question/answer labels the frame
            t0 = time.perf_counter()
            with lock:
                x = unit(model.embed_image(body, max_soft_tokens=args.tokens)[0])
            ms = (time.perf_counter() - t0) * 1000
            q, _, name = (self.headers.get("X-Label") or "").partition("/")
            if q in questions and name in questions[q].names:
                questions[q].label(x, name)
                save(body, x, q, name)
            self.reply({"ms": round(ms, 1), "questions": {q: question.answer(x) for q, question in questions.items()}})
        elif self.path == "/forget":
            for question in questions.values():
                question.forget()
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


def save(jpeg, x, q, name):
    FRAMES.mkdir(exist_ok=True)
    path = FRAMES / f"{time.time_ns()}.jpg"
    path.write_bytes(jpeg)
    np.save(path.with_suffix(f".{args.tokens}.npy"), x)
    with LABELS.open("a") as f:
        f.write(json.dumps({"file": path.name, "question": q, "answer": name}) + "\n")


def saved_embedding(name):
    cached = (FRAMES / name).with_suffix(f".{args.tokens}.npy")
    if cached.exists():
        return np.load(cached)
    x = unit(model.embed_image((FRAMES / name).read_bytes(), max_soft_tokens=args.tokens)[0])
    np.save(cached, x)
    return x


questions = {q: Question(*spec) for q, spec in QUESTIONS.items()} | {d: Detector(*spec) for d, spec in DETECTORS.items()}
if LABELS.exists():  # teach again what was taught before the restart
    rows = [json.loads(line) for line in LABELS.read_text().splitlines()]
    for r in rows:
        if r["question"] in questions:
            questions[r["question"]].label(saved_embedding(r["file"]), r["answer"], refit=False)
    for question in questions.values():
        question.refit()
    print(f"taught again from {len(rows)} saved frames", flush=True)
model.embed_image(grey_png(), max_soft_tokens=args.tokens)  # loads the vision tower before the first frame
print(f"gestures: http://localhost:{args.port}  ({args.tokens} soft tokens per frame)", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

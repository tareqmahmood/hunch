"""`kind: levels` on a public ordinal dataset: SST-5 (sentence sentiment, very negative .. very positive).

Contrasts: 200 sets of 5 train sentences, one per level (not matched: different sentences, so each set is only
centred, not paired). Calibration: 60 dev sentences, 12 per level. Test: the 2,210 test sentences.
Compared with: the same contrasts as an unordered 5-way `choice`, `layer: auto`, and, as a reference for what the
embedding can do with far more labels, multinomial logistic regression trained on all 8,544 train sentences.

    uv run levels_sst5/levels_sst5.py
"""

import json
import random
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

import eg2cpu
from eg2cpu import decisions

from hunch_examples import OUT, SST5, TEACHER_QUESTIONS, Labels, find, teacher_compare
LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
K = 5


def load(split):
    return [json.loads(l) for l in open(find(SST5, f"{split}.jsonl"))]


train, dev, test = load("train"), load("dev"), load("test")
rng = random.Random(0)
by = {k: [r["text"] for r in train if r["label"] == k] for k in range(K)}
for k in by:
    rng.shuffle(by[k])
contrasts = [{LEVELS[k]: by[k][i] for k in range(K)} for i in range(200)]
dby = {k: [r["text"] for r in dev if r["label"] == k] for k in range(K)}
cal = [{"span": {"text": t}, "label": LEVELS[k]} for k in range(K) for t in rng.sample(dby[k], 12)]
yt = np.array([r["label"] for r in test])
spec = {"decision": "sst5_sentiment", "kind": "levels", "levels": LEVELS, "unit": "text",
        "readout": {"layer": "final", "method": "ridge"}, "contrasts": contrasts,
        "escalate": {"when": "confidence < 0.4", "prompt": "How positive is this review sentence?"}}

model = eg2cpu.Model()
cache = {}
results = {}
probs = {}  # name -> class probabilities on test, for the teacher comparison


def report(name, P, score, status=None):
    probs[name] = P if status is None else (P, np.array(status))
    m = decisions.ordinal_metrics(yt, P, score, K)
    r = {k: round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] for k, v in m.items()}
    extra = ""
    if status is not None:
        st = np.array(status)
        ans = st == "answer"
        r |= {"escalate": round(float(np.mean(st == "escalate")), 3),
              "accuracy_answered": round(float(np.mean(np.argmax(P, 1)[ans] == yt[ans])), 3)}
        extra = f"  escalate {r['escalate']:.0%}  acc answered {r['accuracy_answered']:.3f}"
    # calibration: mean confidence vs accuracy
    r["mean_confidence"] = round(float(P.max(1).mean()), 3)
    print(f"  {name:<44} spearman {m['spearman']:.3f}  QWK {m['qwk']:.3f}  acc {m['accuracy']:.3f} "
          f"({m['accuracy_ci'][0]:.2f}-{m['accuracy_ci'][1]:.2f})  within-one {m['within_one']:.3f}  "
          f"mean conf {r['mean_confidence']:.2f}{extra}")
    results[name] = r


def apply(d):
    [vs] = zip(*model.apply([d.handle], [{"text": r["text"]} for r in test]))
    return vs


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lev = decisions.compile(spec, model, cal, _cache=cache)
    print("contrast CV:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lev.report["contrast_cv"].items()
                                      if k not in ("spearman_ci",)}))
    auto = decisions.compile(spec | {"readout": {"layer": "auto", "method": "auto"}}, model, cal, _cache=cache)
    ch = decisions.compile(spec | {"kind": "choice", "options": LEVELS, "levels": None, "escalate": None,
                                   "readout": {"layer": "final", "method": "lda_shrinkage"}}, model, cal, _cache=cache)

print(f"\n== SST-5 test ({len(test)} sentences; majority class {np.bincount(yt).max() / len(yt):.3f})")
for name, d in (("levels (final, ridge)", lev), (f"levels, auto -> layer {auto.report['layer']}, {auto.report['method']}", auto)):
    vs = apply(d)
    P = np.array([[v["p"][str(i)] for i in range(K)] for v in vs])
    report(name, P, np.array([v["score"] for v in vs]), [v["status"] for v in vs])
vs = apply(ch)
P = np.array([[v["p"][l] for l in LEVELS] for v in vs])
report("choice (same 1,000 contrasts, unordered)", P, P @ np.arange(K))

# reference: what the final embedding supports with all 8,544 labelled train sentences
E = decisions._Encoder(model, "final", cache)
Xtr, Xte = E([r["text"] for r in train]), E([r["text"] for r in test])
clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, [r["label"] for r in train])
P = clf.predict_proba(Xte)
report("reference: logistic on all 8,544 train", P, P @ np.arange(K))
(OUT / "levels_sst5.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {OUT / 'levels_sst5.json'}")

# Comparison with the teacher, asked the same 5-level question zero-shot (levels_sst5/teacher.jsonl).
# With labels: the 60 calibration sentences plus up to 200 labelled test sentences from other folds (cross-fitted).
Xcal = E([c["span"]["text"] for c in cal])
ycal = np.array([LEVELS.index(c["label"]) for c in cal])
teacher_compare("levels_sst5", "SST-5 test", TEACHER_QUESTIONS["sentiment"], [r["text"] for r in test], yt, probs,
                labels=Labels(Xte, base_X=Xcal, base_y=ycal), spec="levels (final, ridge)")

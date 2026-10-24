"""Does `readout.layer: auto` pick the layer that transfers? A retrospective check on two decisions.

For each decision, every candidate layer is compiled explicitly and scored on held-out target data (the
ground truth). Then `layer: auto` runs without seeing that data: on grouped cross-validation over the contrasts,
with and without contrast groups, and (for contains_pii) with a small transfer set kept apart from the rows that
score it. For each pick, a paired bootstrap gives the AUC difference between the best layer and the picked one, with
its 95% interval: the pick is level with the best when the interval includes zero.

    tool_call_failed  56 synthetic pairs; calibrate on 60 pi results, truth = 580 other pi results
    contains_pii      100 natural + 100 matched gretel contrasts; calibrate on 60 gretel windows from unseen
                      document types; truth = the other gretel windows of those types. The transfer set is 50 gretel
                      windows of the unseen types with their typed twins, kept out of the truth.

    uv run layer_auto/layer_auto.py
"""

import json
import random
import struct
import time
import warnings
from collections import defaultdict

import numpy as np

import eg2cpu
from eg2cpu import decisions

import hunch_examples as data
from hunch_examples import pii_data
from hunch_examples import TEACHER_QUESTIONS, DECISIONS, OUT, teacher_probs, tool_state

warnings.simplefilter("ignore")
model = eg2cpu.Model()
CACHE: dict = {}  # shared by every compile and by scoring, so each text is encoded once
CANDS = decisions.LAYER_CANDIDATES


def directions(d: decisions.Decision) -> np.ndarray:
    n = struct.unpack("<Q", d.data[:8])[0]
    v = json.loads(d.data[8:8 + n])["directions"]
    s, e = v["data_offsets"]
    return np.frombuffer(d.data[8 + n + s:8 + n + e], dtype="<f4").reshape(v["shape"]).astype(np.float64)


def readout(texts, layer):
    return decisions._Encoder(model, layer, CACHE)(texts)


SCORES: dict = {}  # (target, layer) -> each candidate's scores on that target, for the paired bootstrap


def truth(spec, cal, targets):
    """Each candidate layer compiled explicitly; AUC of its direction on each target (texts, labels, groups)."""
    out = {}
    for c in CANDS:
        d = decisions.compile(dict(spec, readout=dict(spec["readout"], layer=c)), model, cal, _cache=CACHE)
        w = directions(d)[0]
        for name, (t, y, _) in targets.items():
            SCORES[(name, str(c))] = readout(t, c) @ w
        out[str(c)] = {name: decisions.auc(y, SCORES[(name, str(c))]) for name, (_, y, _) in targets.items()}
    return out


def gap(targets, tr, picks, reps=1000):
    """Per pick and target: the AUC of the best layer minus the picked layer's, with a 95% paired bootstrap interval
    (the same resampled rows, or groups, score both)."""
    rng = np.random.default_rng(0)
    out = {}
    for name, (_, y, groups) in targets.items():
        best = max((c for c in tr if c != "teacher"), key=lambda c: tr[c][name])
        by = defaultdict(list)
        for i, g in enumerate(groups):
            by[g].append(i)
        parts = [np.array(v) for v in by.values()]
        samples = [np.concatenate([parts[j] for j in rng.integers(0, len(parts), len(parts))]) for _ in range(reps)]
        for k, layer in picks.items():
            c = str(layer)
            d = [decisions.auc(y[i], SCORES[(name, best)][i]) - decisions.auc(y[i], SCORES[(name, c)][i]) for i in samples]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.setdefault(k, {})[name] = {"best": best, "picked": c, "difference": tr[best][name] - tr[c][name],
                                           "ci95": [float(lo), float(hi)], "level": bool(lo <= 0)}
            print(f"   {k}, {name}: best layer {best} - picked {c} = {tr[best][name] - tr[c][name]:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] -> {'level' if lo <= 0 else 'worse'}")
    return out


def show(title, tr, picks):
    names = list(next(iter(tr.values())))
    print(f"\n== {title}\n   {'layer':>6}" + "".join(f"{n:>22}" for n in names) + "   picked by")
    best = {n: max((c for c in tr if c != "teacher"), key=lambda c: tr[c][n]) for n in names}  # best layer
    for c, row in tr.items():
        mark = ", ".join(k for k, v in picks.items() if str(v) == c)
        print(f"   {c:>6}" + "".join(f"{row[n]:>21.3f}{'*' if best[n] == c else ' '}" for n in names) + f"   {mark}")
    print("   (* best on that target)")


def auto(spec, cal, transfer=None):
    d = decisions.compile(dict(spec, readout=dict(spec["readout"], layer="auto")), model, cal, transfer=transfer,
                          _cache=CACHE)
    return d.report["layer"], d.report["layer_reason"]


results = {}
t0 = time.time()

# ---------------------------------------------------------------- tool_call_failed
spec, _ = decisions.load_spec(DECISIONS / "tool_call_failed.yaml")
spec = dict(spec, readout={"layer": "final", "method": "auto"})
syn = data.synthetic(random.Random(11))
assert [c["call"] for c in spec["contrasts"]] == [c for _, c, _, _ in syn], "spec contrasts differ from synthetic()"
grouped = dict(spec, contrasts=[dict(c, group=g) for c, (g, *_) in zip(spec["contrasts"], syn)])
rows = []
for tool, s, y in data.pi_real({"bash": 150, "edit": 100, "read": 70}, random.Random(11)):
    call, _, output = s.partition("\n")
    rows.append({"span": {"kind": "tool_result", "tool": tool, "call": call, "output": output}, "label": bool(y)})
random.Random(5).shuffle(rows)
cal, test = rows[:60], rows[60:]
targets = {"pi test (580)": ([f"{r['span']['call']}\n{r['span']['output']}" for r in test],
                              np.array([int(r["label"]) for r in test]), list(range(len(test))))}
tr = truth(spec, cal, targets)
# reference: the teacher zero-shot on the same units (layer_auto/teacher.jsonl, by hash)
C = teacher_probs("layer_auto", TEACHER_QUESTIONS["tool_failed"], [tool_state(r["span"]) for r in test])
if C is not None:
    tr["teacher"] = {"pi test (580)": decisions.auc(targets["pi test (580)"][1], C[:, 1])}
picks = {"auto (no groups)": auto(spec, cal), "auto (groups = template)": auto(grouped, cal)}
show("tool_call_failed", tr, {k: v[0] for k, v in picks.items()})
for k, (layer, why) in picks.items():
    print(f"   {k}: {layer} | {why}")
results["tool_call_failed"] = {"truth": tr, "picks": picks, "gap": gap(targets, tr, {k: v[0] for k, v in picks.items()})}

# ---------------------------------------------------------------- contains_pii
rng = random.Random(1)
g = pii_data.load_gretel()
pos = [r for r in g if r["n_pii"]]
neg = [r for r in g if not r["n_pii"] and not pii_data.LOOKS_LIKE_PII.search(r["text"])]
types = sorted({r["group"] for r in pos + neg})
rng.shuffle(types)
dev = set(types[: len(types) // 2])
con = rng.sample([r for r in pos if r["group"] in dev], 200)
neg_by = defaultdict(list)
for r in neg:
    if r["group"] in dev:
        neg_by[r["doc"]].append(r)
        neg_by[r["group"]].append(r)
contrasts = []
for r in con[:100]:  # natural: a PII window and a no-PII window from the same document (else the same type)
    pool = neg_by[r["doc"]] or neg_by[r["group"]]
    if pool:
        contrasts.append({"yes": r["text"], "no": rng.choice(pool)["text"], "group": r["group"]})
contrasts += [{"yes": r["text"], "no": r["typed"], "group": r["group"]} for r in con[100:]]  # matched
ev_pos = [r for r in pos if r["group"] not in dev]
ev_neg = [r for r in neg if r["group"] not in dev]
rng.shuffle(ev_pos)
rng.shuffle(ev_neg)
cal = [{"span": {"text": r["text"]}, "label": "yes"} for r in ev_pos[:30]] + \
      [{"span": {"text": r["text"]}, "label": "no"} for r in ev_neg[:30]]
transfer = [{"span": {"text": r[k]}, "label": lab} for r in ev_pos[30:80] for k, lab in (("text", "yes"), ("typed", "no"))]
ev_pos = ev_pos[:30] + ev_pos[80:]  # the transfer windows are kept out of the truth
test = ev_pos[30:] + ev_neg[30:]
spec = {"decision": "contains_pii", "kind": "boolean", "unit": "text", "readout": {"layer": "final", "method": "auto"},
        "hard_negatives": pii_data.HARD_NEGATIVES, "contrasts": contrasts}
ungrouped = dict(spec, contrasts=[{k: v for k, v in c.items() if k != "group"} for c in contrasts])
targets = {
    "gretel unseen types": ([r["text"] for r in test], np.array([1] * (len(ev_pos) - 30) + [0] * (len(ev_neg) - 30)),
                            [r["doc"] for r in test]),
}
print(f"\ncontains_pii: {len(contrasts)} contrasts from {len(dev)} document types, 60 calibration, "
      f"{len(test)} gretel test windows, 50 gretel transfer pairs")
tr = truth(spec, cal, targets)
C = [teacher_probs("layer_auto", TEACHER_QUESTIONS["pii"], t) for t, _, _ in targets.values()]
if all(c is not None for c in C):
    tr["teacher"] = {n: decisions.auc(yv, c[:, 1]) for (n, (_, yv, _)), c in zip(targets.items(), C)}
picks = {"auto (no groups)": auto(ungrouped, cal), "auto (groups = document type)": auto(spec, cal),
         "auto (+ 50 gretel transfer pairs)": auto(spec, cal, transfer)}
show("contains_pii", tr, {k: v[0] for k, v in picks.items()})
for k, (layer, why) in picks.items():
    print(f"   {k}: {layer} | {why}")
results["contains_pii"] = {"truth": tr, "picks": picks, "gap": gap(targets, tr, {k: v[0] for k, v in picks.items()})}

out = OUT / "layer_auto.json"
out.write_text(json.dumps(results, indent=1))
print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")

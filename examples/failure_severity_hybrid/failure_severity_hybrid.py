"""Failure severity as a hybrid program vs single handles, over 5 random calibration/test splits.

  hybrid      code first: a result the harness didn't flag is `none`; a flagged one is graded minor / recoverable /
              blocking by decisions/failure_severity_graded.yaml (kind: levels), calibrated on 60 flagged errors.
  single      decisions/failure_severity.yaml (one levels handle over all four levels), calibrated on 60 random rows;
              with and without its code/doc hard negatives, and with layer/method auto.
  flag only   not flagged -> none, flagged -> recoverable.

Labels: data/failure_severity_labels.jsonl (320 pi results, labelled by Claude, an LLM). Each split calibrates
on its own rows and tests on the rows neither calibration used (about 210). Also the 69-result keyword stress set.

    uv run failure_severity_hybrid/failure_severity_hybrid.py
"""

import json
import random
import warnings

import numpy as np

import eg2cpu
from eg2cpu import decisions

from hunch_examples import shipped_rows, TEACHER_QUESTIONS, DECISIONS, OUT, teacher_probs, tool_state
LEVELS = ["none", "minor", "recoverable", "blocking"]
K = 4
SEEDS = range(5)


rows = shipped_rows("failure_severity_labels.jsonl")
keyword = shipped_rows("failure_severity_keyword_set.jsonl")
model = eg2cpu.Model()
single_spec, _ = decisions.load_spec(DECISIONS / "failure_severity.yaml")
graded_spec, _ = decisions.load_spec(DECISIONS / "failure_severity_graded.yaml")
cache: dict = {}


def cal_rows(rs, levels):
    return [{"span": r["span"], "label": r["label"]} for r in rs if r["label"] in levels]


def probs(h, rs, labels):
    [vs] = zip(*model.apply([h], [r["span"] for r in rs]))
    return np.array([[v["p"][str(i)] for i in range(len(labels))] if v["p"] else [np.nan] * len(labels) for v in vs])


def single(spec, cal, rs):
    P = probs(decisions.compile(spec, model, cal_rows(cal, LEVELS), _cache=cache).handle, rs, LEVELS)
    return np.argmax(P, 1), P @ np.arange(K)


def hybrid(spec, cal, rs):
    """Code first, then the graded handle on flagged results only."""
    d = decisions.compile(spec, model, cal_rows(cal, LEVELS[1:]), _cache=cache)
    flagged = np.array([r["is_error"] == 1 for r in rs])
    pred, score = np.zeros(len(rs), dtype=int), np.zeros(len(rs))
    if flagged.any():
        P = probs(d.handle, [r for r, f in zip(rs, flagged) if f], LEVELS[1:])
        pred[flagged] = 1 + np.argmax(P, 1)
        score[flagged] = 1 + P @ np.arange(K - 1)
    return pred, score, d.report


def metrics(y, pred, score, err):
    e = err == 1
    rec = {LEVELS[k]: float(np.mean(pred[y == k] == k)) if (y == k).any() else None for k in range(K)}
    return {"qwk": decisions.qwk(y, pred, K), "spearman": decisions.spearman(score, y),
            "accuracy": float(np.mean(pred == y)), "within_one": float(np.mean(np.abs(pred - y) <= 1)),
            "spearman_within_errors": decisions.spearman(score[e], y[e]), "recall": rec}


methods = {
    "single, no hard negatives": lambda cal, rs: single(dict(single_spec, hard_negatives=None), cal, rs),
    "single, with hard negatives (final)": lambda cal, rs: single(single_spec, cal, rs),
    "single, with hard negatives, auto": lambda cal, rs: single(single_spec | {"readout": {"layer": "auto", "method": "auto"}}, cal, rs),
    "flag only (none / recoverable)": lambda cal, rs: (np.array([2 if r["is_error"] else 0 for r in rs]),
                                                       np.array([2.0 if r["is_error"] else 0.0 for r in rs])),
    "hybrid: flag + graded (final, ridge)": lambda cal, rs: hybrid(graded_spec | {"readout": {"layer": "final", "method": "ridge"}}, cal, rs)[:2],
    "hybrid: flag + graded (auto)": lambda cal, rs: hybrid(graded_spec, cal, rs)[:2],
}

# The teacher on the same rows, from failure_severity_hybrid/teacher.jsonl (hashed): one 4-level question on
# every result, and the 3-level question on flagged results for the teacher's own hybrid.
everything = rows + keyword
C4 = teacher_probs("failure_severity_hybrid", TEACHER_QUESTIONS["severity"], [tool_state(r["span"]) for r in everything])
flagged_rows = [r for r in everything if r["is_error"] == 1]
C3 = teacher_probs("failure_severity_hybrid", TEACHER_QUESTIONS["severity_flagged"], [tool_state(r["span"]) for r in flagged_rows])
if C4 is not None and C3 is not None:
    c4 = {id(r): P for r, P in zip(everything, C4)}
    c3 = {id(r): P for r, P in zip(flagged_rows, C3)}

    def teacher_single(cal, rs):
        P = np.array([c4[id(r)] for r in rs])
        return np.argmax(P, 1), P @ np.arange(K)

    def teacher_hybrid(cal, rs):
        pred, score = np.zeros(len(rs), dtype=int), np.zeros(len(rs))
        for i, r in enumerate(rs):
            if r["is_error"] == 1:
                pred[i] = 1 + int(np.argmax(c3[id(r)]))
                score[i] = 1 + float(c3[id(r)] @ np.arange(K - 1))
        return pred, score

    def cascade(spec, cal, rs):
        """The EG2 hybrid, except flagged results the graded handle doesn't answer (escalate, abstain) go to the
        teacher."""
        d = decisions.compile(spec, model, cal_rows(cal, LEVELS[1:]), _cache=cache)
        pred, score = np.zeros(len(rs), dtype=int), np.zeros(len(rs))
        fl = [i for i, r in enumerate(rs) if r["is_error"] == 1]
        if fl:
            [vs] = zip(*model.apply([d.handle], [rs[i]["span"] for i in fl]))
            for i, v in zip(fl, vs):
                P = (np.array([v["p"][str(j)] for j in range(K - 1)]) if v["status"] == "answer" and v["p"]
                     else c3[id(rs[i])])
                pred[i], score[i] = 1 + int(np.argmax(P)), 1 + float(P @ np.arange(K - 1))
        return pred, score

    methods["teacher: one 4-level question"] = teacher_single
    methods["teacher hybrid: flag + teacher graded"] = teacher_hybrid
    methods["cascade: EG2 hybrid, teacher when passed up"] = lambda cal, rs: cascade(
        graded_spec | {"readout": {"layer": "final", "method": "ridge"}}, cal, rs)
runs = {m: [] for m in methods}
kw = {m: [] for m in methods}
picked = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for seed in SEEDS:
        rng = random.Random(seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cal_s = [rows[i] for i in idx[:60]]
        flagged = [i for i in idx if rows[i]["is_error"] == 1]
        cal_h = [rows[i] for i in rng.sample(flagged, 60)]
        used = {id(r) for r in cal_s + cal_h}
        test = [r for r in rows if id(r) not in used]
        y = np.array([LEVELS.index(r["label"]) for r in test])
        err = np.array([r["is_error"] for r in test])
        for name, fn in methods.items():
            cal = cal_h if name.startswith(("hybrid", "cascade")) else cal_s
            pred, score = fn(cal, test)
            runs[name].append(metrics(y, pred, score, err))
            kpred, _ = fn(cal, keyword)
            ky = np.array([LEVELS.index(r["label"]) for r in keyword])
            kw[name].append(float(np.mean(kpred[ky == 0] == 0)))
        picked.append(hybrid(graded_spec, cal_h, test[:1])[2]["layer"])
        print(f"split {seed}: test {len(test)} {np.bincount(y, minlength=K).tolist()}  "
              f"calibration: single {np.bincount([LEVELS.index(r['label']) for r in cal_s], minlength=K).tolist()}, "
              f"hybrid {np.bincount([LEVELS.index(r['label']) for r in cal_h], minlength=K).tolist()}")


def ms(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "n/a"


print(f"\ngraded handle, layer picked by auto per split: {picked}")
print(f"\n{'method':<38}{'QWK':>16}{'spearman':>16}{'accuracy':>16}{'within errors':>16}"
      f"{'minor recall':>16}{'blocking recall':>17}{'keyword: none':>16}")
out = {}
for name in methods:
    r = runs[name]
    row = {k: ms([x[k] for x in r]) for k in ("qwk", "spearman", "accuracy", "spearman_within_errors")}
    row["minor_recall"] = ms([x["recall"]["minor"] for x in r])
    row["blocking_recall"] = ms([x["recall"]["blocking"] for x in r])
    row["keyword_none_read_as_none"] = ms(kw[name])
    out[name] = row
    print(f"{name:<38}{row['qwk']:>16}{row['spearman']:>16}{row['accuracy']:>16}{row['spearman_within_errors']:>16}"
          f"{row['minor_recall']:>16}{row['blocking_recall']:>17}{row['keyword_none_read_as_none']:>16}")
out["graded_layers_picked"] = picked
(OUT / "failure_severity_hybrid.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {OUT / 'failure_severity_hybrid.json'}")

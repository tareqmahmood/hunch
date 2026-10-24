"""Validate decisions/failure_severity.yaml (kind: levels) on real pi coding-agent tool results.

Labels: data/failure_severity_labels.jsonl, 320 results sampled from the pi traces and labelled none / minor /
recoverable / blocking by an LLM (its model and prompt were not recorded), plus data/failure_severity_keyword_set.jsonl:
69 results that matched blocking-type keywords (rate limit, 403, credentials, permission ...), mostly source code
that merely mentions them. Calibrate on 60 of the 320 and test on the other 260, split by session (no session is
on both sides), and on the keyword set.

Baselines on the same test rows:
  is_error         the harness's error flag (0/1)
  tool_call_failed the existing boolean handle's p(yes), as a severity score
  choice           the same contrasts compiled as an unordered 4-way choice

    uv run failure_severity/failure_severity.py
"""

import json
import sys
import random
import time
import warnings

import numpy as np

import eg2cpu
from eg2cpu import decisions

from hunch.proxy import state_text
from hunch_examples import (shipped_rows, stream_rows, load_jsonl, resolve_pi, DECISIONS, OUT, SHIPPED, STREAM_QUESTIONS,
                            TEACHER_QUESTIONS, Labels, linked_groups, teacher_compare, tool_state)
LEVELS = ["none", "minor", "recoverable", "blocking"]
K = len(LEVELS)


rows = shipped_rows("failure_severity_labels.jsonl")
keyword = shipped_rows("failure_severity_keyword_set.jsonl")
session = {h: t["session"] for h, t in resolve_pi({r["state"] for r in rows}).items()}
sessions = sorted({session[r["state"]] for r in rows})
random.Random(17).shuffle(sessions)
cal_sessions = set()
for sid in sessions:  # whole sessions to calibration until it holds 60 rows; the first 60 of them calibrate
    if sum(session[r["state"]] in cal_sessions for r in rows) >= 60:
        break
    cal_sessions.add(sid)
cal = [r for r in rows if session[r["state"]] in cal_sessions][:60]
test = [r for r in rows if session[r["state"]] not in cal_sessions]
y = np.array([LEVELS.index(r["label"]) for r in test])
print(f"calibration 60 {np.bincount([LEVELS.index(r['label']) for r in cal], minlength=K).tolist()}  "
      f"test {len(test)} {np.bincount(y, minlength=K).tolist()}  keyword set {len(keyword)}")

model = eg2cpu.Model()
spec, _ = decisions.load_spec(DECISIONS / "failure_severity.yaml")
cache = {}


def run(d, rs):
    t0 = time.perf_counter()
    [vs] = zip(*model.apply([d.handle], [r["span"] for r in rs]))
    return list(vs), (time.perf_counter() - t0) / len(rs) * 1000


def summary(name, yt, P, score, status=None):
    m = decisions.ordinal_metrics(yt, P, score, K)
    pred = np.argmax(P, 1)
    out = {k: round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] for k, v in m.items()}
    if status is not None:
        st = np.array(status)
        ans = st == "answer"
        out |= {"escalate": round(float(np.mean(st == "escalate")), 3), "abstain": round(float(np.mean(st == "abstain")), 3),
                "accuracy_answered": round(float(np.mean(pred[ans] == yt[ans])), 3) if ans.any() else None,
                "mean_confidence": round(float(P.max(1).mean()), 3)}
    print(f"  {name:<34} spearman {m['spearman']:.3f} ({m['spearman_ci'][0]:.2f}-{m['spearman_ci'][1]:.2f})  "
          f"QWK {m['qwk']:.3f}  acc {m['accuracy']:.3f}  within-one {m['within_one']:.3f}"
          + (f"  escalate {out['escalate']:.0%}  acc answered {out['accuracy_answered']}" if status is not None else ""))
    return out


def confusion(yt, pred):
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(yt, pred):
        C[a, b] += 1
    print("    confusion (rows = label, cols = predicted):  " + "  ".join(f"{l[:5]:>5}" for l in LEVELS))
    for i, row in enumerate(C):
        print(f"    {LEVELS[i]:>11}" + "".join(f"{v:>7}" for v in row))
    return C.tolist()


results = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    lev = decisions.compile(spec, model, [{"span": r["span"], "label": r["label"]} for r in cal], _cache=cache)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; contrast CV: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lev.report['contrast_cv'].items() if k != 'spearman_ci'})}")
    lev.save(OUT / "failure_severity.eg2h")
    auto = decisions.compile(spec | {"readout": {"layer": "auto", "method": "auto"}}, model,
                             [{"span": r["span"], "label": r["label"]} for r in cal], _cache=cache)
    choice = decisions.compile(spec | {"kind": "choice", "options": LEVELS, "levels": None,
                                       "readout": {"layer": "final", "method": "lda_shrinkage"},
                                       "escalate": None, "calibration": {"method": "multinomial"}}, model,
                               [{"span": r["span"], "label": r["label"]} for r in cal], _cache=cache)

print(f"\n== test: {len(test)} labelled pi results")
for name, d in (("levels (spec: final, ridge)", lev), (f"levels, layer/method auto -> {auto.report['layer']}/{auto.report['method']}", auto)):
    vs, ms = run(d, test)
    P = np.array([[v["p"][str(i)] for i in range(K)] for v in vs])
    results[name] = summary(name, y, P, np.array([v["score"] for v in vs]), [v["status"] for v in vs]) | {"ms_per_span": round(ms, 1)}
    results[name]["confusion"] = confusion(y, np.argmax(P, 1))
vs, _ = run(choice, test)
P = np.array([[v["p"][l] for l in LEVELS] for v in vs])
results["choice (same contrasts, unordered)"] = summary("choice (same contrasts, unordered)", y, P, P @ np.arange(K))
if not (OUT / "tool_call_failed.eg2h").exists():
    sys.exit(f"no {OUT / 'tool_call_failed.eg2h'}: run tool_call_failed.py to compile it")
tcf = eg2cpu.Handle(str(OUT / "tool_call_failed.eg2h"))
[vs] = zip(*model.apply([tcf], [r["span"] for r in test]))
p = np.array([v["p"] for v in vs])
rho = decisions.spearman(p, y)
err = np.array([r["is_error"] for r in test])
results["baselines"] = {"tool_call_failed_p_spearman": round(rho, 3), "is_error_spearman": round(decisions.spearman(err, y), 3)}
print(f"  {'tool_call_failed p(yes) as a score':<34} spearman {rho:.3f}")
print(f"  {'harness is_error flag':<34} spearman {decisions.spearman(err, y):.3f}")
print("  blocking items in test:", [(r["span"]["output"][:60].replace(chr(10), ' '), ) for r in test if r["label"] == "blocking"])

print("\n== keyword stress set (69 results full of 'credentials', '403', 'rate limit' ...)")
vs, _ = run(lev, keyword)
yk = np.array([LEVELS.index(r["label"]) for r in keyword])
pk = np.array([np.argmax([v["p"][str(i)] for i in range(K)]) for v in vs])
none = yk == 0
print(f"  labelled none: {none.sum()}, read as none {np.mean(pk[none] == 0):.2f}, as blocking {np.mean(pk[none] == 3):.2f}")
for r, v, lab in zip(keyword, vs, yk):
    if lab == 3:
        print(f"  blocking item -> {v['answer']} (p_blocking {v['p']['3']:.2f}): {r['span']['output'][:80]!r}")
results["keyword_set"] = {"n": len(keyword), "none_read_as_none": round(float(np.mean(pk[none] == 0)), 3),
                          "none_read_as_blocking": round(float(np.mean(pk[none] == 3)), 3),
                          "blocking_predictions": [LEVELS[int(x)] for x in pk[yk == 3]]}
results["compile_report"] = {k: v for k, v in lev.report.items() if k not in ("methods",)}
(OUT / "failure_severity_validation.json").write_text(json.dumps(results, indent=1, default=str))
print(f"\nwrote {OUT / 'failure_severity_validation.json'}")


# Comparison with the teacher on the same 4-level question (answers in failure_severity/teacher.jsonl).
def level_probs(d, rs, kind="levels"):
    vs, _ = run(d, rs)
    return (np.array([[v["p"][str(i)] for i in range(K)] if kind == "levels" else [v["p"][l] for l in LEVELS] for v in vs]),
            np.array([v["status"] for v in vs]))


# With labels (the 260-result test only): the 60 calibration rows plus up to 200 labelled test rows from other folds,
# grouped by session.
E = decisions._Encoder(model, "final", cache)
unit = lambda rs: [state_text(tool_state(r["span"])) for r in rs]  # noqa: E731
with_labels = Labels(E(unit(test)), groups=[session[r["state"]] for r in test], base_X=E(unit(cal)),
                     base_y=np.array([LEVELS.index(r["label"]) for r in cal]))
for title, rs, lab in (("260 labelled pi results", test, with_labels), ("keyword stress set", keyword, None)):
    teacher_compare("failure_severity", title, TEACHER_QUESTIONS["severity"], [tool_state(r["span"]) for r in rs],
                    [LEVELS.index(r["label"]) for r in rs],
                    {"harness flag (none / recoverable)": np.eye(K)[[2 if r["is_error"] else 0 for r in rs]],
                     "levels (spec)": level_probs(lev, rs), "levels (auto)": level_probs(auto, rs),
                     "choice": level_probs(choice, rs, "choice")}, labels=lab, spec="levels (spec)")


# The pi stream as it arrives: 2,000 results in time order, 5% flagged by the harness, every one labelled
# (data/failure_severity_stream_labels.jsonl). Most problems here are unflagged: searches that found nothing, warnings,
# and silent failures, where the command exits cleanly but the output shows it failed. The flag can't see those, so
# besides the whole stream each model is scored on the unflagged rows: recall of problems (minor and up) and of
# failures (recoverable and up), and precision of what it calls a problem. The teacher's answers are the stream's
# (its severity question has the same four levels). The stream repeats itself (120 of its 2,000 results are exact
# repeats of an earlier state), so each distinct state is scored once. Labels: the 320 labelled results plus up to 200
# stream rows from other folds, grouped by session.
seen, stream = set(), []
for r in stream_rows():  # the first occurrence of each state
    if r["state"] not in seen:
        seen.add(r["state"])
        stream.append(r)
by_state = {r["state"]: r["label"] for r in load_jsonl(SHIPPED / "failure_severity_stream_labels.jsonl")}
ys = np.array([LEVELS.index(by_state[r["state"]]) for r in stream])
flag = np.array([r["is_error"] for r in stream])
print(f"\n== pi stream: {len(stream)} distinct results, {flag.sum()} flagged, labels {np.bincount(ys, minlength=K).tolist()}, "
      f"unflagged problems {int(((ys > 0) & (flag == 0)).sum())} ({int(((ys > 1) & (flag == 0)).sum())} failures)")


def silent_recall(least):
    def fn(y, pred, rows):
        quiet = flag[rows] == 0
        return np.mean(pred[quiet & (y >= least)] >= least)
    return fn


def silent_precision(y, pred, rows):
    quiet = flag[rows] == 0
    return np.mean(y[quiet & (pred > 0)] > 0)


fresh = [r for r in rows if r["state"] not in by_state]
stream_labels = Labels(E(unit(stream)), groups=[r["session"] for r in stream], base_X=E(unit(fresh)),
                       base_y=np.array([LEVELS.index(r["label"]) for r in fresh]))
teacher_compare("failure_severity", "pi stream", STREAM_QUESTIONS["severity"], [tool_state(r["span"]) for r in stream], ys,
                {"harness flag (none / recoverable)": np.eye(K)[np.where(flag == 1, 2, 0)],
                 "levels (spec)": level_probs(lev, stream)}, labels=stream_labels, answers="stream",
                extra={"unflagged: problem recall": silent_recall(1), "unflagged: failure recall": silent_recall(2),
                       "unflagged: problem precision": silent_precision}, spec="levels (spec)")


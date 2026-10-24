"""Can the teacher detect indirect prompt injection in tool outputs, and is it a teacher worth distilling?

Units are built from InjecAgent and the pi traces (hunch_examples.injection_units, cached in results/), and the
teacher's answers are in injection_teacher/teacher.jsonl, keyed by question and state hashes (units without one
are asked of TEACHER_URL when it is set):
  400 InjecAgent tool responses across all 17 user tools: 200 with an attacker instruction in the slot (base and
      "enhanced" wording, both attack types), 200 benign twins filling the same slot with content from the
      hunch_examples/injection_fillers.py pool (about half of it imperative but addressed to people)
  300 ordinary pi coding-agent tool results (bash / edit / read), assumed clean
The teacher answers one noul question per unit. Scored against the ground truth: AUC, accuracy at 0.5, recall, false
positives on benign twins (imperative / descriptive) and on real pi outputs. It writes the two teacher files for
preflight.py (results/injection_teacher_{answers,truth}.jsonl), and trains the proxy on each to show rule 9 of
docs/question-design.md: ground-truth labels beat a teacher's answers where the teacher is wrong.

    uv run injection_teacher/injection_teacher.py
    uv run preflight/preflight.py results/injection_teacher_truth.jsonl --type noul --name injection
"""

import json
import random
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from hunch_examples import OUT, TEACHER_QUESTIONS, injection_units, teacher_probs, tool_state

QUESTION = TEACHER_QUESTIONS["injection"]
units = injection_units()
print(f"{len(units)} units ({sum(u['label'] for u in units)} injected)", flush=True)
C = teacher_probs("injection_teacher", QUESTION, [tool_state(u["span"]) for u in units])
if C is None:
    sys.exit(1)
p = C[:, 1]
y = np.array([u["label"] for u in units])
src = np.array([u["source"] for u in units])
print("\n== the teacher as an injection detector (zero-shot)")
print(f"   InjecAgent units: AUC {roc_auc_score(y[src != 'pi'], p[src != 'pi']):.3f}; all units incl. pi: AUC {roc_auc_score(y, p):.3f}")
for s in ("injecagent-base", "injecagent-enhanced", "benign-imperative", "benign-descriptive", "pi"):
    m = src == s
    print(f"   {s:<22} n={m.sum():>3}  flagged (p >= 0.5) {np.mean(p[m] >= 0.5):.0%}  mean p {p[m].mean():.2f}")
# Below 0.5: is the polite miss the teacher's ranking or its cut-off? (cut-offs read off the truth, not cross-fitted)
for cut in (0.2, 0.1, 0.05, 0.03, 0.02):
    print(f"   cut {cut:<4}  polite caught {np.mean(p[src == 'injecagent-base'] >= cut):.0%}  "
          f"benign flagged {np.mean(p[y == 0] >= cut):.1%}  pi flagged {np.mean(p[src == 'pi'] >= cut):.1%}")
# teacher files for the pre-flight: the teacher's answers, and the ground truth
for name, ans in (("answers", lambda i: [float(p[i])]), ("truth", lambda i: int(y[i]))):
    with open(OUT / f"injection_teacher_{name}.jsonl", "w") as fh:
        for i, u in enumerate(units):
            fh.write(json.dumps({"span": u["span"], "answer": ans(i), "group": u["group"]}) + "\n")
print(f"\nwrote {OUT / 'injection_teacher_answers.jsonl'} and {OUT / 'injection_teacher_truth.jsonl'}")

# Rule 9 through the proxy's training endpoint: the same student (layer 8, logistic regression) trained once on the
# teacher's answers given as "sampled" (as if it had answered live traffic) and once on the InjecAgent labels given as
# "labelled" (ground truth), then scored against the ground truth on held-out units, with the teacher on the same
# units. The split halves the tools, the attack texts, the filler strings and the pi sessions: a held-out InjecAgent
# unit has a held-out tool and a held-out attack (or filler); units that mix the halves are used for neither.
from eg2cpu import Model  # noqa: E402
from hunch import Proxy  # noqa: E402

rng = random.Random(3)


def held_half(values):
    vs = sorted(set(values))
    rng.shuffle(vs)
    return set(vs[len(vs) // 2:])


inj = [u for u in units if u["source"] != "pi"]
held = {"tool": held_half(u["tool"] for u in inj), "attack": held_half(u["attack"] for u in inj if "attack" in u),
        "filler": held_half(u["filler"] for u in inj if "filler" in u),
        "session": held_half(u["group"] for u in units if u["source"] == "pi")}


def side(u):
    if u["source"] == "pi":
        return u["group"] in held["session"]
    keys = [u["tool"] in held["tool"], u["attack"] in held["attack"] if "attack" in u else u["filler"] in held["filler"]]
    return all(keys) if all(keys) or not any(keys) else None


sides = [side(u) for u in units]
tr = [i for i, h in enumerate(sides) if h is False]
te = [i for i, h in enumerate(sides) if h is True]
encoder = Model()
yt, st_ = y[te], src[te]
print(f"\n== the proxy trained on {len(tr)} units, scored on {len(te)} held-out units "
      f"({sum(sides[i] is None for i in range(len(units)))} units mix the halves and are left out); held out: "
      + ", ".join(f"{s} {int((st_ == s).sum())}" for s in ("injecagent-base", "injecagent-enhanced", "benign-imperative",
                                                           "benign-descriptive", "pi")))
caught = lambda got, s_: np.mean(got[st_ == s_] >= 0.5)  # noqa: E731
result = {"n_train": len(tr), "n_held_out": len(te), "held_out": {s: int((st_ == s).sum()) for s in set(st_.tolist())}}
rows = {"teacher (zero-shot)": p[te]}
for name, kind, answer in (("trained on the teacher's answers (sampled)", "sampled", lambda i: [float(p[i])]),
                           ("trained on the labels (labelled)", "labelled", lambda i: bool(y[i]))):
    db = OUT / f"injection_proxy_{kind}.db"
    db.unlink(missing_ok=True)
    proxy = Proxy(encoder, None, db, min_examples=60)
    s = proxy.train(QUESTION, [{"state": units[i]["span"], "answer": answer(i), "kind": kind} for i in tr])
    rows[f"{name}, graduated {s['graduated']} ({s['mode']})"] = np.array(
        [a["noul"] for a in proxy.predict(QUESTION, [units[i]["span"] for i in te])])
for name, got in rows.items():
    r = {"polite_caught": float(caught(got, "injecagent-base")), "enhanced_caught": float(caught(got, "injecagent-enhanced")),
         "benign_and_pi_flagged": float(np.mean(got[yt == 0] >= 0.5))}
    result[name] = r
    print(f"   {name:<62} polite caught {r['polite_caught']:.0%}  enhanced caught {r['enhanced_caught']:.0%}  "
          f"benign + pi flagged {r['benign_and_pi_flagged']:.1%}")
(OUT / "injection_teacher.json").write_text(json.dumps(result, indent=1))

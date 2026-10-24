"""Pre-flight check for a proxy question: will a cheap EmbeddingGemma 2 student be able to take it over from the
teacher (an LLM judge or a dedicated decision model), and how? See docs/question-design.md.

Input: a JSONL file of real units with the teacher's answers, 150-300 rows is enough:
    {"span": {"call": "...", "output": "..."}, "answer": <teacher answer>, "group": "<session id, optional>"}
`answer` is the teacher's probabilities (a list, one per option; for noul [p_yes] or a single number) or a label
(an option index, or for noul true/false). `group` keeps one session's units in the same CV fold, as a proxy would
meet them.

For each readout (layer 4, 8, 12, final; costed at its measured exit time on these units) a student is
cross-validated against the teacher with the proxy's graduation rules (docs/question-design.md):
  symmetric   agreement 95% lower bound >= --target and recall on the teacher's non-majority answers >= --minority
  one-sided   (noul) answer only a confident "no": recall of forwarded "yes" >= 0.85 and local miss rate <= 2%
  levels      (score) Spearman, quadratic-weighted kappa, within-one, and whether held-out level means are in order
Also: the teacher's own uncertainty, how many units exceed the 1,024-token window, optional hard negatives (units
with a required answer), and an estimate of coverage and compute per item at the best readout.

    uv run preflight/preflight.py answers.jsonl --type noul --name my_question
    uv run preflight/preflight.py answers.jsonl --type choice --options source,test_run,build,...
    uv run preflight/preflight.py answers.jsonl --type score --options none,minor,recoverable,blocking
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

import eg2cpu
from hunch_examples import kappa
from hunch.proxy import wilson

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("answers")
ap.add_argument("--type", choices=["noul", "choice", "score"], required=True)
ap.add_argument("--options", help="comma-separated option names (choice/score), in order")
ap.add_argument("--name", default=None)
ap.add_argument("--template", default="{call}\n{output}", help="how a span becomes text (default: call + output)")
ap.add_argument("--hard", help="JSONL of hard negatives: {span, answer} units that must read that answer")
ap.add_argument("--target", type=float, default=0.90)
ap.add_argument("--minority", type=float, default=0.80)
ap.add_argument("--backend-ms", type=float, default=494.0, help="teacher cost per item in ms (default: the shipped stream's teacher, a local LLM)")
ap.add_argument("--audit", type=float, default=0.10)
args = ap.parse_args()
name = args.name or Path(args.answers).stem


def render(span):
    return args.template.format(**{k: v for k, v in span.items()})


def label_of(a, k):
    if isinstance(a, (list, tuple)):
        return int(a[0] >= 0.5) if args.type == "noul" and len(a) == 1 else int(np.argmax(a))
    if isinstance(a, bool):
        return int(a)
    if isinstance(a, float) and args.type == "noul":
        return int(a >= 0.5)
    return int(a)


def conf_of(a):
    if isinstance(a, (list, tuple)):
        return max(a[0], 1 - a[0]) if len(a) == 1 else max(a)
    return max(a, 1 - a) if isinstance(a, float) else 1.0


rows = [json.loads(l) for l in open(args.answers) if l.strip()]
k = 2 if args.type == "noul" else len(args.options.split(",")) if args.options else None
opts = ["no", "yes"] if args.type == "noul" else (args.options.split(",") if args.options else None)
y = np.array([label_of(r["answer"], k) for r in rows])
if opts is None:
    opts = [str(i) for i in range(int(y.max()) + 1)]
K = len(opts)
w = np.array([conf_of(r["answer"]) for r in rows])
groups = [str(r.get("group", i)) for i, r in enumerate(rows)]
texts = [render(r["span"]) for r in rows]
n = len(rows)
counts = np.bincount(y, minlength=K)
maj = int(counts.argmax())
print(f"== pre-flight: {name} ({args.type}), {n} units, {len(set(groups))} groups")
print("   teacher answers: " + ", ".join(f"{o} {c}" for o, c in zip(opts, counts)) + f"  (majority {counts.max() / n:.0%})")
print(f"   teacher unsure (top probability < 0.6): {np.mean(w < 0.6):.0%}")

model = eg2cpu.Model()
toks = [len(model.tokenize(t)) for t in texts]
print(f"   unit length: median {int(np.median(toks))} tokens, {np.mean(np.array(toks) >= 1024):.0%} hit the 1,024 window")

# features: one full pass gives every layer; time the exits on these units for the cost estimate
outs = model.embed_batch(texts)
EMB = np.array([e for e, _ in outs], dtype=np.float64)
MEANS = np.array([m for _, m in outs], dtype=np.float64)
sample = texts[: min(200, n)]
cost = {}
for layer in (4, 8, 12):
    t0 = time.perf_counter()
    model.embed_batch_partial(sample, layer)
    cost[layer] = (time.perf_counter() - t0) / len(sample) * 1000
t0 = time.perf_counter()
model.embed_batch(sample)
cost["final"] = (time.perf_counter() - t0) / len(sample) * 1000


def z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


READOUTS = {"layer 4": (z(MEANS[:, 4]), cost[4]), "layer 8": (z(MEANS[:, 8]), cost[8]),
            "layer 12": (z(MEANS[:, 12]), cost[12]), "final": (z(EMB), cost["final"])}


def cv_proba(F):
    folds = (GroupKFold(5).split(F, y, groups) if len(set(groups)) >= 5 else KFold(5, shuffle=True, random_state=0).split(F))
    P = np.zeros((n, K))
    for tr, te in folds:
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(F[tr], y[tr], sample_weight=w[tr])
        P[np.ix_(te, clf.classes_)] = clf.predict_proba(F[te])
    return P


def qwk(a, b):
    O = np.zeros((K, K))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)]) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


rare_yes = args.type == "noul" and counts[1] / n < 0.2
print(f"\n   {'readout':<9}{'ms':>6}{'kappa':>8}{'minority recall':>17}{'symmetric: coverage':>21}" +
      (f"{'one-sided: coverage (misses)':>30}" if args.type == "noul" else "") +
      (f"{'spearman':>10}{'QWK':>7}{'within-1':>10}{'levels in order':>17}" if args.type == "score" else ""))
best = None
results = {}
for rname, (F, ms) in READOUTS.items():
    P = cv_proba(F)
    pred, conf = P.argmax(1), P.max(1)
    agree = pred == y
    minority = y != maj
    r = {"ms": ms, "kappa": kappa(y, pred),
         "minority_recall": float(agree[minority].mean()) if minority.any() else None}
    # symmetric graduation: the lowest confidence threshold that passes both bounds; coverage above it
    r["symmetric_coverage"] = 0.0
    for t in np.arange(0.5, 0.995, 0.01):
        sel = conf >= t
        ms_ = sel & minority
        if (sel.sum() >= 30 and wilson(int(agree[sel].sum()), int(sel.sum()))[0] >= args.target
                and ms_.sum() >= 15 and wilson(int(agree[ms_].sum()), int(ms_.sum()))[0] >= args.minority):
            r["symmetric_coverage"] = float(sel.mean())
            break
    line = f"   {rname:<9}{ms:>6.0f}{r['kappa']:>8.3f}{(r['minority_recall'] or 0):>17.3f}{r['symmetric_coverage']:>21.0%}"
    if args.type == "noul":
        p_yes = P[:, 1]
        r["one_sided_coverage"], r["one_sided_misses"] = 0.0, None
        if counts[1] >= 15:
            for tau in np.unique(np.round(p_yes, 4))[::-1]:
                local = p_yes < tau
                if (local.sum() >= 30 and wilson(int((y[~local] == 1).sum()), int(counts[1]))[0] >= 0.85
                        and wilson(int((y[local] == 1).sum()), int(local.sum()))[1] <= 0.02):
                    r["one_sided_coverage"], r["one_sided_misses"] = float(local.mean()), int((y[local] == 1).sum())
                    break
        line += f"{r['one_sided_coverage']:>23.0%} ({r['one_sided_misses'] if r['one_sided_misses'] is not None else '-'})"
    if args.type == "score":
        score = P @ np.arange(K)
        means = [score[y == c].mean() for c in range(K) if (y == c).any()]
        r |= {"spearman": spearman(score, y), "qwk": qwk(y, pred), "within_one": float(np.mean(np.abs(pred - y) <= 1)),
              "levels_in_order": bool(all(b > a for a, b in zip(means, means[1:])))}
        line += f"{r['spearman']:>10.3f}{r['qwk']:>7.3f}{r['within_one']:>10.3f}{str(r['levels_in_order']):>17}"
    print(line)
    cov = max(r["symmetric_coverage"], r.get("one_sided_coverage", 0.0))
    r["coverage"] = cov
    r["ms_per_item"] = ms + (1 - cov * (1 - args.audit)) * args.backend_ms
    results[rname] = r
    if best is None or r["ms_per_item"] < results[best]["ms_per_item"]:
        best = rname

if args.hard:
    hard = [json.loads(l) for l in open(args.hard) if l.strip()]
    hy = np.array([label_of(h["answer"], k) for h in hard])
    F_best = READOUTS[best][0]
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(F_best, y, sample_weight=w)
    hout = model.embed_batch([render(h["span"]) for h in hard])
    raw = np.array([e for e, _ in hout]) if best == "final" else np.array([m[int(best.split()[1])] for _, m in hout])
    src = EMB if best == "final" else MEANS[:, int(best.split()[1])]
    Hf = (raw - src.mean(0)) / (src.std(0) + 1e-6)
    print(f"\n   hard negatives at {best}: {int(np.sum(clf.predict(Hf) != hy))} of {len(hard)} misread")

def needed(bar):
    """Fewest rare answers for which even perfect recall can clear `bar` as a 95% lower bound."""
    m = 1
    while wilson(m, m)[0] < bar:
        m += 1
    return m


b = results[best]
rare_n = int(counts[1]) if args.type == "noul" else int(n - counts[maj])
rare_rate = rare_n / n
need = needed(0.85 if rare_yes else args.minority)
# The minority-recall bound can only be met with enough rare answers above the threshold; with fewer, no readout
# can pass however good it is, so the honest verdict is "collect more", not "backend only".
from sklearn.metrics import roc_auc_score  # noqa: E402

signal = None
if args.type == "noul" and 0 < counts[1] < n:
    signal = roc_auc_score(y, cv_proba(READOUTS[best][0])[:, 1])
print(f"\n== verdict for {name}")
if b["symmetric_coverage"] > 0:
    print(f"   GRADUATES symmetrically at {best}: ~{b['symmetric_coverage']:.0%} of units answered locally at kappa "
          f"{b['kappa']:.2f}; ~{b['ms_per_item']:.0f} ms/item vs {args.backend_ms:.0f} backend-only")
elif b.get("one_sided_coverage", 0) > 0:
    print(f"   GRADUATES ONE-SIDED at {best}: ~{b['one_sided_coverage']:.0%} answered 'no' locally, "
          f"{b['one_sided_misses']} of {counts[1]} teacher 'yes' missed; ~{b['ms_per_item']:.0f} ms/item vs "
          f"{args.backend_ms:.0f} backend-only. Route it one-sided.")
elif rare_n < need * 1.5:
    more = int(math.ceil((need * 1.5 - rare_n) / max(rare_rate, 1e-9)))
    print(f"   NOT ENOUGH DATA: the teacher gave only {rare_n} rare answers ({rare_rate:.1%} of units). Proving recall "
          f"to the graduation bar needs at least {need} even if every one is caught, and in practice ~{int(need * 1.5)}. "
          f"Collect about {more} more units" + (f" (the signal is there: CV AUC {signal:.2f} at {best})" if signal else "") + ".")
    if rare_yes:
        print("   Route it one-sided once it graduates (guide rule 3).")
else:
    print("   BACKEND ONLY: no readout passes the graduation rules on these units.")
    hints = []
    if rare_yes and counts[1] < 15:
        hints.append(f"only {counts[1]} teacher 'yes' answers: collect more units, or rephrase so the rare case is common enough")
    if b["minority_recall"] is not None and b["minority_recall"] < 0.6:
        hints.append("the student misses most of the teacher's non-majority answers: split rare classes into their own "
                     "yes/no questions, or decompose the judgement into observable sub-questions (guide rules 2-3)")
    if args.type == "score" and not b.get("levels_in_order", True):
        hints.append("held-out levels are out of order: the levels are probably categories, not intensities; use choice")
    if np.mean(w < 0.6) > 0.15:
        hints.append("the teacher itself is unsure on many units: tighten the instructions (guide rule 5)")
    if np.mean(np.array(toks) >= 1024) > 0.1:
        hints.append("many units hit the 1,024-token window: use shorter units or windows (guide rule 7)")
    for h in hints or ["the answer may need reasoning or context the unit doesn't contain (guide: what phrasing can't fix)"]:
        print(f"   - {h}")
out = Path(args.answers).with_suffix(".preflight.json")
out.write_text(json.dumps({"name": name, "type": args.type, "n": n, "teacher_counts": counts.tolist(), "best": best,
                           "readouts": results}, indent=1, default=float))
print(f"\nwrote {out}")

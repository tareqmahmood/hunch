"""Guided question optimization: use the units where the teacher is wrong and EG2 is right to rewrite the teacher's question.

    uv run question_pilot/question_pilot.py mine injection          # teach / protect sets on dev, failure clusters, minimal pairs
    TEACHER_URL=... uv run question_pilot/question_pilot.py eval injection question_pilot/recorded/injection_candidates.json
    TEACHER_URL=... uv run question_pilot/question_pilot.py test injection question_pilot/recorded/injection_candidates.json
    (the same for pii)

Units and truth:
  injection  hunch_examples.injection_units: 400 InjecAgent responses (200 with the dataset's attacker instruction, base
             and enhanced wording, 200 benign twins) and 300 pi tool results (clean). Groups: tool, or pi session.
  pii        gretel windows from every document type (PII / no PII). Groups: document type.
             Only units whose original-question teacher answer is already cached are used, so v0 costs no teacher time.
Splits: whole groups, half dev and half test (seeded). Everything below is chosen on dev; test is read once, by `test`.
An injection test unit whose attack text or filler string also appears in a dev unit is left out, so test attacks and
fillers are unseen as well as test tools.

EG2: logistic regression on layer-8 means trained on the labels (the proxy's labelled student). On dev its
predictions are out-of-fold (grouped 5-fold); on test it is trained on all of dev.

Sets on dev, against the truth: teach = teacher wrong, EG2 right; protect = teacher right, EG2 wrong; both wrong; both right.
`mine` clusters the teach set (k-means on EG2 features) and pairs each teach unit with its nearest dev unit of the
other label that the teacher gets right (a minimal pair), and writes them to results/pilot/ for writing candidate questions.

`eval` asks the teacher each candidate question on dev and scores it against v0 (the question the examples use):
accuracy (Wilson interval), AUC, teach set fixed, protect set kept, accuracy per source, a paired bootstrap of the
accuracy gain, and the cascade (EG2 answers outside |p - 0.5| < 0.2, the rest go to the teacher). A candidate qualifies if
the gain's 95% lower bound is above 0 and no source loses more than 5 points; the best qualifying one is selected.
`test` scores v0 and the selected candidate once on the held-out test groups.

The recorded run (logs, candidates with their rationale, selections and scores) is in question_pilot/recorded/, the teacher's answers to
every question version in question_pilot/teacher.jsonl, and the write-up in question_pilot/README.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from hunch_examples import OUT, TEACHER_QUESTIONS, injection_units, teacher_files, teacher_probs, tool_state
from eg2cpu import Model, decisions
from hunch.proxy import AnswerCache, question_key, state_hash

PILOT = OUT / "pilot"
RECORDED = Path(__file__).parent / "recorded"  # the recorded run
PILOT.mkdir(exist_ok=True)
BAND = 0.2  # EG2 passes a unit up when |p - 0.5| < BAND, as the compiled decisions do


def injection_task():
    units = []
    for u in injection_units():
        units.append({"state": tool_state(u["span"]), "text": f"{u['span']['call']}\n{u['span']['output']}",
                      "y": u["label"], "group": u["group"], "source": u["source"],
                      "content": u.get("attack") or u.get("filler")})
    return units


def pii_units():
    from hunch_examples import pii_data as pii

    cached = AnswerCache(*teacher_files())
    qk = question_key(TEACHER_QUESTIONS["pii"])
    have = lambda t: (qk, state_hash(t)) in cached.answers  # noqa: E731
    units = []
    for r in pii.load_gretel():
        if r["n_pii"]:
            units.append({"text": r["text"], "y": 1, "group": r["group"], "source": "gretel-pii"})
        elif not pii.LOOKS_LIKE_PII.search(r["text"]):
            units.append({"text": r["text"], "y": 0, "group": r["group"], "source": "gretel-none"})
    units = [u for u in units if have(u["text"])]
    for u in units:
        u["state"] = u["text"]
    # keep it pilot-sized: up to 400 gretel windows per label
    rng = random.Random(0)
    g = {s: [u for u in units if u["source"] == s] for s in ("gretel-pii", "gretel-none")}
    return [u for s in g for u in rng.sample(g[s], min(400, len(g[s])))]


TASKS = {"injection": (injection_task, "injection"), "pii": (pii_units, "pii")}


def split(units, seed=7):
    """Half the groups to dev, half to test. Gretel document types and InjecAgent tools are few, so they are split
    within each source family; pi sessions are many. An injection test unit with an attack text or filler string that
    a dev unit also has is left out (split "unused")."""
    fam = lambda u: u["source"].split("-")[0]  # noqa: E731
    test_groups = set()
    for f in sorted({fam(u) for u in units}):
        gs = sorted({u["group"] for u in units if fam(u) == f})
        random.Random(seed).shuffle(gs)
        test_groups |= set(gs[len(gs) // 2:])
    for u in units:
        u["split"] = "test" if u["group"] in test_groups else "dev"
    seen = {u.get("content") for u in units if u["split"] == "dev"} - {None}
    for u in units:
        if u["split"] == "test" and u.get("content") in seen:
            u["split"] = "unused"
    return units


def features(model, units):
    parts = model.embed_batch_partial([u["text"] for u in units], 8)
    return np.array([p.layer_means[8] for p in parts], dtype=np.float64)


def eg2_probs(X, units):
    """p(yes): out-of-fold on dev (grouped 5-fold), and from a model trained on all of dev for test."""
    y = np.array([u["y"] for u in units])
    dev = np.array([u["split"] == "dev" for u in units])
    mu, sd = X[dev].mean(0), X[dev].std(0) + 1e-6
    Z = (X - mu) / sd
    fit = lambda idx: LogisticRegression(max_iter=3000, C=0.5).fit(Z[idx], y[idx])  # noqa: E731
    p = np.zeros(len(units))
    di = np.flatnonzero(dev)
    groups = [units[i]["group"] for i in di]
    for tr, te in GroupKFold(5).split(di, groups=groups):
        p[di[te]] = fit(di[tr]).predict_proba(Z[di[te]])[:, 1]
    p[~dev] = fit(di).predict_proba(Z[~dev])[:, 1]
    return p


def load(task):
    units = split(TASKS[task][0]())
    model = Model()
    X = features(model, units)
    p_eg2 = eg2_probs(X, units)
    C = teacher_probs("question_pilot", TEACHER_QUESTIONS[TASKS[task][1]], [u["state"] for u in units])
    if C is None:
        sys.exit("v0 teacher answers missing")
    for u, pe, pc in zip(units, p_eg2, C[:, 1]):
        u["eg2"], u["teacher0"] = float(pe), float(pc)
    return units, X


def mine(task):
    units, X = load(task)
    dev = [i for i, u in enumerate(units) if u["split"] == "dev"]
    ce = np.array([(units[i]["teacher0"] >= 0.5) == bool(units[i]["y"]) for i in dev])
    ee = np.array([(units[i]["eg2"] >= 0.5) == bool(units[i]["y"]) for i in dev])
    sets = {"teach": ~ce & ee, "protect": ce & ~ee, "both_wrong": ~ce & ~ee, "both_right": ce & ee}
    n_test = sum(u["split"] == "test" for u in units)
    print(f"{task}: {len(units)} units, dev {len(dev)} ({len({units[i]['group'] for i in dev})} groups), test {n_test}")
    print(f"  dev accuracy: teacher v0 {ce.mean():.3f}, EG2 (out-of-fold) {ee.mean():.3f}")
    for name, m in sets.items():
        print(f"  {name:<11} {m.sum():>4}  " + "  ".join(
            f"{s} {sum(1 for i, k in zip(dev, m) if k and units[i]['source'] == s)}" for s in sorted({u['source'] for u in units})))
    teach = [dev[j] for j in np.flatnonzero(sets["teach"])]
    k = max(1, min(6, len(teach) // 8))
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xn[teach]) if len(teach) >= k > 1 else np.zeros(len(teach), int)
    right_other = {c: [i for i in dev if units[i]["y"] == c and (units[i]["teacher0"] >= 0.5) == bool(c)] for c in (0, 1)}
    out = []
    for c in range(k):
        members = [t for t, l in zip(teach, lab) if l == c]
        srcs = {s: sum(units[i]["source"] == s for i in members) for s in {units[i]["source"] for i in members}}
        print(f"\n  cluster {c}: {len(members)} units {srcs}")
        for i in members[:3]:
            pool = right_other[1 - units[i]["y"]]
            j = pool[int(np.argmax(Xn[pool] @ Xn[i]))] if pool else None
            out.append({"cluster": c, "unit": units[i]["text"], "truth": units[i]["y"], "teacher_p": units[i]["teacher0"],
                        "eg2_p": units[i]["eg2"], "minimal_pair": units[j]["text"] if j is not None else None,
                        "pair_truth": units[j]["y"] if j is not None else None})
            print(f"    truth {units[i]['y']}  teacher {units[i]['teacher0']:.2f}  EG2 {units[i]['eg2']:.2f}: "
                  f"{units[i]['text'][-220:]!r}")
            if j is not None:
                print(f"      pair (truth {units[j]['y']}, teacher right): {units[j]['text'][-160:]!r}")
    with open(PILOT / f"{task}_teach.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {PILOT / f'{task}_teach.jsonl'}")


def score(units, P, p_eg2):
    y = np.array([u["y"] for u in units])
    pred = P >= 0.5
    ok = pred == y.astype(bool)
    local = np.abs(p_eg2 - 0.5) >= BAND
    casc = np.where(local, p_eg2 >= 0.5, pred) == y.astype(bool)
    by = {s: float(ok[[u["source"] == s for u in units]].mean()) for s in sorted({u["source"] for u in units})}
    return {"accuracy": float(ok.mean()), "accuracy_ci": decisions.proportion_ci(int(ok.sum()), len(ok)),
            "auc": decisions.auc(y, P), "by_source": by, "cascade_accuracy": float(casc.mean()),
            "cascade_eg2_share": float(local.mean())}, ok


def evaluate(task, cand_path, which):
    units, _ = load(task)
    sub = [u for u in units if u["split"] == which]
    cands = json.load(open(cand_path))
    v0 = TEACHER_QUESTIONS[TASKS[task][1]]
    if which == "test":
        f = PILOT / f"{task}_selected.json"
        sel = json.load(open(f if f.exists() else RECORDED / f.name))  # this run's dev pick, else the recorded one
        cands = [c for c in cands if c["name"] == sel["selected"]] if sel["selected"] else []
    p_eg2 = np.array([u["eg2"] for u in sub])
    y = np.array([u["y"] for u in sub])
    print(f"{which}: {len(sub)} units " + str({s_: sum(u['source'] == s_ for u in sub) for s_ in sorted({u['source'] for u in sub})}))
    ce0 = np.array([(u["teacher0"] >= 0.5) == bool(u["y"]) for u in sub])
    ee = (p_eg2 >= 0.5) == y.astype(bool)
    teach, protect = ~ce0 & ee, ce0 & ~ee
    base, ok0 = score(sub, np.array([u["teacher0"] for u in sub]), p_eg2)
    print(f"{task} {which}: {len(sub)} units; teach {teach.sum()}, protect {protect.sum()}; EG2 accuracy {ee.mean():.3f}")
    show = lambda name, r, extra="": print(  # noqa: E731
        f"  {name:<22} acc {r['accuracy']:.3f} ({r['accuracy_ci'][0]:.3f}-{r['accuracy_ci'][1]:.3f})  AUC {r['auc']:.3f}  "
        f"cascade {r['cascade_accuracy']:.3f} (EG2 {r['cascade_eg2_share']:.0%}){extra}\n"
        f"  {'':<22} " + "  ".join(f"{s} {v:.2f}" for s, v in r["by_source"].items()))
    show("v0 (current)", base)
    results = {"v0": base}
    rng = np.random.default_rng(0)
    groups = sorted({u["group"] for u in sub})
    gidx = {g: [i for i, u in enumerate(sub) if u["group"] == g] for g in groups}
    for c in cands:
        q = dict(v0, instructions=c["instructions"])
        P = teacher_probs("question_pilot", q, [u["state"] for u in sub])
        if P is None:
            sys.exit("set TEACHER_URL to ask the teacher the candidate questions")
        r, ok = score(sub, P[:, 1], p_eg2)
        # paired bootstrap over groups of the accuracy gain
        gains = []
        for _ in range(2000):
            idx = np.concatenate([gidx[g] for g in rng.choice(groups, len(groups))])
            gains.append(ok[idx].mean() - ok0[idx].mean())
        lo, hi = np.quantile(gains, [0.025, 0.975])
        worst = min(r["by_source"][s] - base["by_source"][s] for s in base["by_source"])
        # each source's share of the gain: its share of the units times its accuracy change
        r["gain_by_source"] = {s_: float(np.mean([u["source"] == s_ for u in sub]) * (r["by_source"][s_] - base["by_source"][s_]))
                               for s_ in base["by_source"]}
        r |= {"gain": float(ok.mean() - ok0.mean()), "gain_ci": [float(lo), float(hi)],
              "teach_fixed": float(ok[teach].mean()) if teach.any() else None,
              "protect_kept": float(ok[protect].mean()) if protect.any() else None, "worst_source_change": float(worst),
              "qualifies": bool(lo > 0 and worst > -0.05), "instructions": c["instructions"]}
        results[c["name"]] = r
        show(c["name"], r, f"  gain {r['gain']:+.3f} ({lo:+.3f} to {hi:+.3f})  teach fixed "
             f"{r['teach_fixed'] if r['teach_fixed'] is None else round(r['teach_fixed'], 3)}  protect kept "
             f"{r['protect_kept'] if r['protect_kept'] is None else round(r['protect_kept'], 3)}  "
             f"worst source {worst:+.3f}{'  QUALIFIES' if r['qualifies'] else ''}")
        print(f"  {'':<22} gain by source: " + "  ".join(f"{s_} {v:+.3f}" for s_, v in r["gain_by_source"].items()))
    if which == "dev":
        ok_c = [n for n, r in results.items() if n != "v0" and r["qualifies"]]
        best = max(ok_c, key=lambda n: results[n]["accuracy"]) if ok_c else None
        (PILOT / f"{task}_selected.json").write_text(json.dumps({"selected": best}))
        print(f"\nselected on dev: {best}")
    (PILOT / f"{task}_{which}.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {PILOT / f'{task}_{which}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "eval", "test"])
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("candidates", nargs="?")
    a = ap.parse_args()
    if a.cmd == "mine":
        mine(a.task)
    else:
        evaluate(a.task, a.candidates, "dev" if a.cmd == "eval" else "test")

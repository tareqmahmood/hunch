"""Early exit for the proxy: which encoder layers do the students need, and can easy items stop early?

Same stream, teacher answers and policy as proxy_replay.py: output_kind graduates on
agreement + recall on the teacher's minority answers; tool_failed is routed one-sided (only a confident "not failed" is
answered locally). Severity is left out (it never graduates). Encoder cost is measured, not assumed:
proxy_early_exit/early_exit_timing.json has eg2cpu's batched ms/item for exits after layers 4, 8, 12, the full pass,
and resuming an exit at layer 8 to the end (Model.embed_batch_partial / resume_batch, bit-identical to a full pass).

  feature sets  each question's student on: final + layer 8 (the replay's default), final only, layer 4, 8 or 12
                alone; each costed at the exit it needs.
  cascade       stage A: layer-8 students, paid at the layer-8 exit. A request is served there if every question
                is confident at A. Otherwise resume to the end (paid) and use the full-feature students (stage B);
                still not confident: the teacher. Both stages learn only from teacher-answered items; an audited item pays
                the resume too, so stage B sees it.

    uv run proxy_early_exit/proxy_early_exit.py
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import eg2cpu

from hunch_examples import OUT, kappa, stream_rows
from hunch.proxy import wilson
WARMUP, REFIT, AUDIT_WINDOW, AUDIT, TARGET = 150, 100, 40, 0.10, 0.90
MINORITY_TARGET, MINORITY_MIN = 0.80, 15
ONE_SIDED_RECALL, ONE_SIDED_MISS = 0.85, 0.02
QS_ALL = ["tool_failed", "output_kind"]
ONE_SIDED = {"tool_failed"}

rows = stream_rows()  # the teacher's answers from data/teacher_stream.jsonl
n = len(rows)
timing = json.load(open(Path(__file__).parent / "early_exit_timing.json"))
TEACHER_MS = float(np.median([r["seconds"] for r in rows])) * 1000
T_FULL, T_RESUME8 = timing["full pass"], timing["resume layer 8 -> end"]
T_EXIT = {4: timing["exit after layer 4"], 8: timing["exit after layer 8"], 12: timing["exit after layer 12"]}


def teacher(r, q):
    p = r["teacher"][q]
    return int(p[0] >= 0.5) if q == "tool_failed" else int(np.argmax(p))


Y = {q: np.array([teacher(r, q) for r in rows]) for q in QS_ALL}
WGT = {q: np.array([max(r["teacher"][q][0], 1 - r["teacher"][q][0]) if q == "tool_failed" else max(r["teacher"][q]) for r in rows])
       for q in QS_ALL}
MAJ = {q: int(np.bincount(Y[q]).argmax()) for q in QS_ALL}

model = eg2cpu.Model()
outs = model.embed_batch([f"{r['call']}\n{r['output']}" for r in rows])
EMB = np.array([e for e, _ in outs], dtype=np.float64)
MEANS = np.array([m for _, m in outs], dtype=np.float64)  # [n, 24, 512]


def z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


FEATURES = {  # name -> (matrix, encoder ms/item it needs)
    "final + layer 8": (z(np.hstack([EMB, MEANS[:, 8]])), T_FULL),
    "final only": (z(EMB), T_FULL),
    "layer 12": (z(MEANS[:, 12]), T_EXIT[12]),
    "layer 8": (z(MEANS[:, 8]), T_EXIT[8]),
    "layer 4": (z(MEANS[:, 4]), T_EXIT[4]),
}
print(f"{n} items; teacher {TEACHER_MS:.0f} ms/item; eg2cpu ms/item: " + ", ".join(f"exit@{k} {v:.1f}" for k, v in T_EXIT.items())
      + f", full {T_FULL:.1f}, resume 8->end {T_RESUME8:.1f}")


class Student:
    """One question's student on one feature set: fit, graduation threshold, and the local decision."""

    def __init__(self, q, F):
        self.q, self.F, self.clf, self.thr = q, F, None, None

    def fit_on(self, idx):
        y = Y[self.q][idx]
        if len(set(y.tolist())) < 2:
            return None
        return LogisticRegression(max_iter=2000, C=0.5).fit(self.F[idx], y, sample_weight=WGT[self.q][idx])

    def p_yes(self, clf, idx):
        return clf.predict_proba(self.F[idx])[:, list(clf.classes_).index(1)]

    def refit(self, idx):
        idx = np.array(idx)
        self.clf = self.fit_on(idx)
        self.thr = None if self.clf is None or len(idx) < 60 else self.pick(idx)

    def pick(self, idx):
        y = Y[self.q][idx]
        folds = list(KFold(5, shuffle=True, random_state=0).split(idx))
        if self.q in ONE_SIDED:
            if y.sum() < MINORITY_MIN:
                return None
            prob = np.zeros(len(idx))
            for tr, te in folds:
                c = self.fit_on(idx[tr])
                if c is None:
                    return None
                prob[te] = self.p_yes(c, idx[te])
            best = None
            for tau in np.unique(np.round(prob, 4)):
                local = prob < tau
                if (local.sum() >= 30 and wilson(int((y[~local] == 1).sum()), int(y.sum()))[0] >= ONE_SIDED_RECALL
                        and wilson(int((y[local] == 1).sum()), int(local.sum()))[1] <= ONE_SIDED_MISS):
                    best = float(tau)
            return best
        pred, conf = np.zeros(len(idx), dtype=int), np.zeros(len(idx))
        for tr, te in folds:
            c = self.fit_on(idx[tr])
            if c is None:
                return None
            P = c.predict_proba(self.F[idx[te]])
            pred[te], conf[te] = c.classes_[np.argmax(P, 1)], P.max(1)
        agree, minority = pred == y, y != MAJ[self.q]
        for t in np.arange(0.5, 0.995, 0.01):
            sel = conf >= t
            ms = sel & minority
            if (sel.sum() >= 30 and wilson(int(agree[sel].sum()), int(sel.sum()))[0] >= TARGET
                    and ms.sum() >= MINORITY_MIN and wilson(int(agree[ms].sum()), int(ms.sum()))[0] >= MINORITY_TARGET):
                return float(t)
        return None

    def local(self, i):
        """(answer, confident) for item i; not confident if not graduated."""
        if self.clf is None or self.thr is None:
            return None, False
        if self.q in ONE_SIDED:
            return 0, bool(self.p_yes(self.clf, [i])[0] < self.thr)
        P = self.clf.predict_proba(self.F[[i]])[0]
        return int(self.clf.classes_[np.argmax(P)]), bool(P.max() >= self.thr)


def replay(qs, stages, seed=0):
    """stages: list of (feature-set name, encoder ms to reach it from the previous stage). A request is served at the
    first stage where every question is confident (each question may use any stage reached so far)."""
    rng = random.Random(seed)
    students = [{q: Student(q, FEATURES[name][0]) for q in qs} for name, _ in stages]
    labelled, since, audits = [], 0, {q: [] for q in qs}
    demoted = {q: False for q in qs}
    stage_served = np.full(n, -1)
    ans = {q: np.full(n, -1) for q in qs}
    enc_ms = np.zeros(n)
    called = np.zeros(n, dtype=bool)
    Xn = FEATURES[stages[0][0]][0] / np.linalg.norm(FEATURES[stages[0][0]][0], axis=1, keepdims=True)
    nn_floor = None
    for i in range(n):
        enc_ms[i] = stages[0][1]
        decided = {}
        served_at = -1
        novel = nn_floor is not None and float((Xn[np.array(labelled)] @ Xn[i]).max()) < nn_floor
        if i >= WARMUP and not novel:
            for s, (name, cost) in enumerate(stages):
                if s > 0:
                    enc_ms[i] += cost
                for q in qs:
                    if q not in decided and not demoted[q]:
                        a, ok = students[s][q].local(i)
                        if ok:
                            decided[q] = a
                if len(decided) == len(qs):
                    served_at = s
                    break
        audited = served_at >= 0 and rng.random() < AUDIT
        if served_at >= 0:
            for q in qs:
                ans[q][i] = decided[q]
        stage_served[i] = served_at
        if served_at < 0 or audited:
            called[i] = True
            # stage B students train on every teacher-answered item, so finish the pass if it stopped early
            enc_ms[i] = sum(c for _, c in stages)
            labelled.append(i)
            since += 1
        if audited:
            for q in qs:
                audits[q].append(int(ans[q][i] == Y[q][i]))
                a = audits[q][-AUDIT_WINDOW:]
                demoted[q] = len(a) >= 20 and wilson(sum(a), len(a))[1] < TARGET
        if (i + 1 == WARMUP) or (i >= WARMUP and since >= REFIT):
            for st in students:
                for q in qs:
                    st[q].refit(labelled)
            if any(demoted.values()):
                demoted = {q: False for q in qs}  # re-qualified at the refit
            S = Xn[np.array(labelled)] @ Xn[np.array(labelled)].T
            np.fill_diagonal(S, -1)
            nn_floor = float(np.quantile(S.max(1), 0.05))
            since = 0
    half = slice(n // 2, n)
    served = stage_served[half] >= 0
    out = {"served_local": float(served.mean()),
           "served_at_stage": [float(np.mean(stage_served[half] == s)) for s in range(len(stages))],
           "teacher_calls": float(called[half].mean()),
           "ms_per_item": float(enc_ms[half].mean() + called[half].mean() * TEACHER_MS)}
    for q in qs:
        yq, aq = Y[q][half][served], ans[q][half][served]
        if q in ONE_SIDED:
            out[q] = {"teacher_failed_answered_no_locally": int((yq == 1).sum()), "teacher_failed_total": int((Y[q][half] == 1).sum())}
        elif served.any():
            po = np.mean(aq == yq)
            mi = yq != MAJ[q]
            out[q] = {"agreement": float(po), "kappa": kappa(yq, aq),
                      "minority_recall": float(np.mean(aq[mi] == yq[mi])) if mi.any() else None}
    return out


def show(label, r):
    parts = [f"served locally {r['served_local']:.0%}"]
    if len(r["served_at_stage"]) > 1:
        parts.append("at layer 8 " + f"{r['served_at_stage'][0]:.0%}" + ", after resume " + f"{r['served_at_stage'][1]:.0%}")
    parts += [f"teacher calls/item {r['teacher_calls']:.2f}", f"{r['ms_per_item']:.0f} ms/item vs {TEACHER_MS:.0f} teacher-only "
              f"({TEACHER_MS / r['ms_per_item']:.2f}x)"]
    if "output_kind" in r:
        k = r["output_kind"]
        parts.append(f"output_kind kappa {k['kappa']:.3f}, minority recall {k['minority_recall']:.3f}")
    if "tool_failed" in r:
        t = r["tool_failed"]
        parts.append(f"tool_failed misses {t['teacher_failed_answered_no_locally']}/{t['teacher_failed_total']}")
    print(f"  {label:<46} " + " | ".join(parts))


results = {"timing": timing, "teacher_ms": TEACHER_MS}
for qs in (["output_kind"], ["tool_failed"], ["tool_failed", "output_kind"]):
    key = " + ".join(qs)
    print(f"\n== {key} (second half of the stream)")
    results[key] = {}
    for name, (_, cost) in FEATURES.items():
        r = replay(qs, [(name, cost)])
        results[key][name] = r
        show(f"{name} ({cost:.0f} ms)", r)
    r = replay(qs, [("layer 8", T_EXIT[8]), ("final + layer 8", T_RESUME8)])
    results[key]["cascade: layer 8 -> resume -> teacher"] = r
    show(f"cascade layer 8 ({T_EXIT[8]:.0f} ms) -> full (+{T_RESUME8:.0f} ms)", r)
(OUT / "proxy_early_exit.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {OUT / 'proxy_early_exit.json'}")

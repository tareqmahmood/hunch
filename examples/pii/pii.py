"""Can EmbeddingGemma 2 tell whether a text contains PII, and what kind? (eg2cpu, mean-pooled readouts)

Limit up front: eg2cpu returns the normalized embedding and per-layer *means*, not per-token states, so this asks
whether a window of text contains a PII value and of which type, not where the span is.

Data: gretelai/synthetic_pii_finance_multilingual, English test split (2,891 synthetic finance documents, 60 document
types, NER-labelled spans; Apache-2.0), for training and held-out evaluation.

Units are windows of whole lines/sentences, 40-320 chars, cut from each document; a window with a span crossing
its edge is dropped. PII categories (types outside these, like company, date, time, swift_bic_code, are not PII):
  person      name, first_name, last_name
  contact     email, phone_number
  address     street_address
  financial   iban, bban, credit_card_number, credit_card_security_code, bank_routing_number, account_pin
  gov_id      ssn, passport_number, driver_license_number
  credential  password, api_key, user_name, ipv4, ipv6
Windows holding only customer_id, employee_id, date_of_birth or local_latlng are left out of every task.

Tasks
  natural   windows with >=1 PII span vs windows from the same documents with none (and no email/phone-looking
            string, since the NER labels miss some). Balanced.
  pairs     the same PII window with each value replaced by a phrase naming its type ("the customer's email
            address"): topic and wording kept, value gone. Scored as AUC over both sides and paired accuracy.
  choice    windows with exactly one PII category: which one. Control: the same windows with every value masked
            as "***" (field labels kept), read by the probe fitted on the unmasked ones.

Probes: diff of means, LDA (Ledoit-Wolf shrinkage), L2 logistic regression; readouts: final embedding and layer
means 4/8/12/16/20/23. Gretel folds are grouped by document_type (5-fold GroupKFold), so test windows come from
document formats the probe never saw. 95% intervals are bootstrap over source documents. Then decision handles
(eg2cpu.decisions.compile; layer 8 or final x matched-only or natural+matched contrasts), contrasts from half the
document types, calibrated on 60 natural windows from the other half, tested on the rest of that half. The spec written (results/contains_pii.yaml) is layer 8, natural+matched.

    uv run pii/pii.py
"""

import hashlib
import json
import random
from collections import Counter, defaultdict

import eg2cpu
import numpy as np
import yaml
from eg2cpu import decisions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import hunch_examples as data
from hunch_examples import OUT
from hunch_examples.pii_data import HARD_NEGATIVES, LOOKS_LIKE_PII, load_gretel

CACHE = OUT / "pii_probe_emb.npz"
LAYERS = [4, 8, 12, 16, 20, 23]
READOUTS = ["final"] + LAYERS
CATS = ["person", "contact", "address", "financial", "gov_id", "credential"]


# ---------------------------------------------------------------- embeddings (cached by text)
def embed_all(texts, cache_path=CACHE, layers=LAYERS, dtype=np.float32, chunk=2000):
    """{readout: rows} for "final" + `layers`, cached by text in `cache_path`; embeds in chunks to bound memory."""
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    cache = dict(np.load(cache_path, allow_pickle=False)) if cache_path.exists() else {}
    if cache and not {f"l{l}" for l in layers} <= set(cache):
        print(f"{cache_path.name} lacks some of layers {list(layers)}: rebuilding it")
        cache = {}  # the cache only stores texts, not their sources, so it is rebuilt from what is asked for
    layers = sorted(set(layers) | {int(k[1:]) for k in cache if k.startswith("l")})  # keep every cached layer
    have = set(cache["keys"].tolist()) if cache else set()
    todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in have})
    if todo:
        print(f"embedding {len(todo)} texts")
        model = eg2cpu.Model()
        for c in range(0, len(todo), chunk):
            part = todo[c: c + chunk]
            res = model.embed_batch([t for _, t in part])
            new = {"keys": np.array([k for k, _ in part]), "final": np.stack([e for e, _ in res]).astype(dtype),
                   **{f"l{l}": np.stack([m[l] for _, m in res]).astype(dtype) for l in layers}}
            cache = new if not cache else {k: np.concatenate([cache[k], new[k]]) for k in new}
        np.savez(cache_path, **cache)
    idx = {k: i for i, k in enumerate(cache["keys"].tolist())}
    rows = np.array([idx[k] for k in keys])
    return {r: cache["final" if r == "final" else f"l{r}"][rows].astype(np.float32) for r in ["final"] + list(layers)}


# ---------------------------------------------------------------- probes
def fit_binary(method, X, y):
    if method == "diff_of_means":
        w = X[y == 1].mean(0) - X[y == 0].mean(0)
        c = (X[y == 1] @ w).mean() / 2 + (X[y == 0] @ w).mean() / 2
        return lambda Z: Z @ w - c
    if method == "lda_shrinkage":
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        return m.decision_function
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))


def fit_choice(method, X, y, k):
    if method == "diff_of_means":  # nearest class mean: the choice analogue of diff of means
        mus = np.array([X[y == j].mean(0) for j in range(k)])
        return lambda Z: -((Z[:, None, :] - mus[None]) ** 2).sum(-1)
    if method == "lda_shrinkage":
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        return m.decision_function
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))


METHODS = ["diff_of_means", "lda_shrinkage", "logistic"]


def boot(stat, clusters, reps=500, seed=0):
    """Point estimate and 95% interval, resampling source documents."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    keys = list(by)
    vals = []
    for _ in range(reps):
        idx = np.concatenate([by[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        try:
            vals.append(stat(idx))
        except ValueError:
            pass
    return [round(float(stat(np.arange(len(clusters)))), 3), round(float(np.percentile(vals, 2.5)), 3),
            round(float(np.percentile(vals, 97.5)), 3)]


def binary_metrics(y, s, clusters):
    return {"auc": boot(lambda i: roc_auc_score(y[i], s[i]), clusters),
            "acc": boot(lambda i: np.mean((s[i] > 0) == y[i]), clusters)}


def fmt(m):
    return "  ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in m.items())


def cv_scores(E, y, groups, method, fit=fit_binary, extra=None):
    """Out-of-fold scores; `extra` = (E2, groups2): also score E2 rows whose group is held out in that fold."""
    s = np.zeros(len(y)) if fit is fit_binary else None
    s2 = None
    for tr, te in GroupKFold(5).split(E, y, groups):
        f = fit(method, E[tr], y[tr])
        out = f(E[te])
        if s is None:
            s = np.zeros((len(y), out.shape[1]))
        s[te] = out
        if extra is not None:
            E2, g2 = extra
            held = np.isin(g2, np.unique(groups[te]))
            o2 = f(E2[held])
            if s2 is None:
                s2 = np.zeros((len(g2),) + o2.shape[1:])
            s2[held] = o2
    return s, s2


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_gretel()
    pos = [r for r in g if r["n_pii"]]
    neg = [r for r in g if not r["n_pii"] and not LOOKS_LIKE_PII.search(r["text"])]
    dropped = sum(1 for r in g if not r["n_pii"] and LOOKS_LIKE_PII.search(r["text"]))
    # keep every window of the rarer categories, subsample the person/address-heavy rest
    rare = [r for r in pos if set(r["cats"]) - {"person", "address"}]
    common = [r for r in pos if not set(r["cats"]) - {"person", "address"}]
    pos = rare + rng.sample(common, 500)
    neg = rng.sample(neg, len(pos))
    rng.sample(range(6655), 400)  # fixes the random state the test split, and the shipped teacher answers, depend on
    print(f"gretel windows: {len(g)}  pii {len(pos)} (rare-category {len(rare)})  no-pii {len(neg)} "
          f"(dropped {dropped} unlabelled windows that look like email/phone)")
    print("gretel categories:", Counter(c for r in pos for c in r["cats"]).most_common())

    nat = pos + neg
    E_nat = embed_all([r["text"] for r in nat])
    E_typ = embed_all([r["typed"] for r in pos])
    E_neu = embed_all([r["neutral"] for r in pos])

    y_nat = np.array([1] * len(pos) + [0] * len(neg))
    g_nat = np.array([r["group"] for r in nat])
    d_nat = [r["doc"] for r in nat]
    g_pos = np.array([r["group"] for r in pos])
    d_pairs = [r["doc"] for r in pos] * 2
    y_pairs = np.array([1] * len(pos) + [0] * len(pos))
    results = {"n": {"natural": len(nat), "pairs": len(pos), "dropped_lookalike": dropped}}

    # -------- natural, pairs, cross-task
    print("\n== contains PII (held out by document_type)")
    for ro in READOUTS:
        for m in METHODS:
            s_nat, s_typ = cv_scores(E_nat[ro], y_nat, g_nat, m, extra=(E_typ[ro], g_pos))
            s_pos = s_nat[: len(pos)]
            pair = {"auc": boot(lambda i: roc_auc_score(y_pairs[i], np.concatenate([s_pos, s_typ])[i]), d_pairs),
                    "paired_acc": boot(lambda i: np.mean(s_pos[i] > s_typ[i]), [r["doc"] for r in pos])}
            # trained on matched pairs only, tested on held-out natural windows
            X_pairs = E_typ[ro]
            s_x = np.zeros(len(nat))
            for tr, te in GroupKFold(5).split(E_nat[ro], y_nat, g_nat):
                tr_pos = np.isin(g_pos, np.unique(g_nat[tr]))
                f = fit_binary(m, np.vstack([E_nat[ro][: len(pos)][tr_pos], X_pairs[tr_pos]]),
                               np.array([1] * tr_pos.sum() + [0] * tr_pos.sum()))
                s_x[te] = f(E_nat[ro][te])
            r = {"natural": binary_metrics(y_nat, s_nat, d_nat), "pairs_from_natural": pair,
                 "natural_from_pairs": {"auc": boot(lambda i: roc_auc_score(y_nat[i], s_x[i]), d_nat)}}
            results.setdefault("binary", {}).setdefault(str(ro), {})[m] = r
            print(f"{str(ro):>5} {m:<14} natural {fmt(r['natural'])}\n{'':20} pairs   {fmt(pair)}  "
                  f"| natural<-pairs auc {r['natural_from_pairs']['auc'][0]:.3f}")

    # -------- which category (single-category windows)
    one = [i for i, r in enumerate(pos) if len(r["cats"]) == 1]
    yc = np.array([CATS.index(pos[i]["cats"][0]) for i in one])
    gc = g_pos[one]
    dc = [pos[i]["doc"] for i in one]
    print(f"\n== which PII category ({len(one)} single-category windows: {dict(Counter(CATS[j] for j in yc))})")
    print("   chance = majority share", round(np.bincount(yc).max() / len(yc), 3))
    for ro in READOUTS:
        X = E_nat[ro][: len(pos)][one]
        Xn = E_neu[ro][one]
        for m in METHODS:
            fit = lambda mm, A, b: fit_choice(mm, A, b, len(CATS))  # noqa: E731
            s, s_n = np.zeros((len(one), len(CATS))), np.zeros((len(one), len(CATS)))
            for tr, te in GroupKFold(5).split(X, yc, gc):
                f = fit(m, X[tr], yc[tr])
                s[te], s_n[te] = f(X[te]), f(Xn[te])
            r = {"acc": boot(lambda i: np.mean(s[i].argmax(1) == yc[i]), dc),
                 "masked_acc": boot(lambda i: np.mean(s_n[i].argmax(1) == yc[i]), dc)}
            results.setdefault("choice", {}).setdefault(str(ro), {})[m] = r
            print(f"{str(ro):>5} {m:<14} {fmt(r)}")
        if ro == "final":
            f = fit_choice("logistic", X, yc, len(CATS))
            cm = np.zeros((len(CATS), len(CATS)), int)
            s = np.zeros((len(one), len(CATS)))
            for tr, te in GroupKFold(5).split(X, yc, gc):
                s[te] = fit_choice("logistic", X[tr], yc[tr], len(CATS))(X[te])
            for t, p in zip(yc, s.argmax(1)):
                cm[t, p] += 1
            results["choice_confusion_final_logistic"] = {"labels": CATS, "matrix": cm.tolist()}
            print("      confusion (final, logistic; rows = true):", CATS, cm.tolist())

    # -------- top-scoring "no PII" windows: are they unlabelled PII?
    s_nat, _ = cv_scores(E_nat["final"], y_nat, g_nat, "logistic")
    top = sorted(range(len(pos), len(nat)), key=lambda i: -s_nat[i])[:8]
    results["top_false_positives"] = [nat[i]["text"][:200] for i in top]
    print("\n== highest-scoring no-PII windows (final, logistic)")
    for i in top:
        print("   ", repr(nat[i]["text"][:160]))

    # -------- decision handle
    results["decision"] = decision(pos, neg, rng)
    (OUT / "pii_probe.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'pii_probe.json'}")


# The spec ships layer 8 with natural + matched contrasts; the other three are the 2x2 around it.
VARIANTS = [
    ("contains_pii", 8, "natural+pairs"),
    ("contains_pii_l8_pairs", 8, "pairs"),
    ("contains_pii_final_natural+pairs", "final", "natural+pairs"),
    ("contains_pii_final_pairs", "final", "pairs"),
]


def decision(pos, neg, rng):
    """Contrasts from half the document types; calibration (60) and test from the other half."""
    types = sorted({r["group"] for r in pos + neg})
    rng.shuffle(types)
    dev = set(types[: len(types) // 2])
    con = rng.sample([r for r in pos if r["group"] in dev], 200)
    pairs = [{"yes": r["text"], "no": r["typed"]} for r in con[:200]]
    # natural contrasts: a PII window and a no-PII window from the same document (else the same document type)
    neg_by = defaultdict(list)
    for r in neg:
        if r["group"] in dev:
            neg_by[r["doc"]].append(r), neg_by[r["group"]].append(r)
    natural = []
    for r in con[:100]:
        pool = neg_by[r["doc"]] or neg_by[r["group"]]
        if pool:
            natural.append({"yes": r["text"], "no": rng.choice(pool)["text"]})
    ev_pos = [r for r in pos if r["group"] not in dev]
    ev_neg = [r for r in neg if r["group"] not in dev]
    rng.shuffle(ev_pos), rng.shuffle(ev_neg)
    cal = [{"span": {"text": r["text"]}, "label": "yes"} for r in ev_pos[:30]] + \
          [{"span": {"text": r["text"]}, "label": "no"} for r in ev_neg[:30]]
    test = [(r["text"], 1, r["doc"]) for r in ev_pos[30:]] + [(r["text"], 0, r["doc"]) for r in ev_neg[30:]]
    test_types = [r["group"] for r in ev_pos[30:]] + [r["group"] for r in ev_neg[30:]]
    y = np.array([t[1] for t in test])
    docs = [t[2] for t in test]
    model = eg2cpu.Model()
    out = {}
    collected = {}  # name -> (test probabilities, statuses), for the teacher comparison
    for name, layer, kind in VARIANTS:
        contrasts = natural + pairs[:100] if kind == "natural+pairs" else pairs
        spec = {
            "decision": "contains_pii", "kind": "boolean", "unit": "text",
            "encoder": "google/embeddinggemma-2@<rev>", "readout": {"layer": layer, "method": "auto"},
            "hard_negatives": HARD_NEGATIVES,
            "calibration": {"method": "platt", "sample": 60},
            "abstain": {"mahalanobis_quantile": 0.99},
            "escalate": {"when": "|p - 0.5| < 0.2", "prompt": "Does this text contain a personal data value (a "
                         "person's name, contact details, address, account or card number, government ID, or "
                         "credential)? Mentioning a kind of data without giving the value does not count."},
            "contrasts": contrasts,
        }
        d = decisions.compile(spec, model, cal)
        if name == "contains_pii":
            (OUT / "contains_pii.yaml").write_text(
                f"# Generated by examples/pii/pii.py: {len(natural)} same-document (PII window, no-PII window) pairs "
                f"+ 100 matched pairs (value vs a phrase naming its type),\n# from {len(dev)} of {len(types)} gretel "
                "document types. Compile with 60 calibration windows from the other types.\n"
                + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120))
            d.save(OUT / "contains_pii.eg2h")

        def run(texts):
            [vs] = zip(*model.apply([d.handle], [{"text": t} for t in texts]))
            return np.array([v["p"] for v in vs]), np.array([v["status"] for v in vs])

        p, st = run([t[0] for t in test])
        ans = st == "answer"
        res = {"layer": layer, "contrasts": kind, "n_contrasts": len(contrasts), "method": d.report["method"],
               "method_reason": d.report.get("method_reason"),
               "calibration_auc": [round(d.report["calibration_auc"], 3)] + [round(x, 3) for x in d.report["calibration_auc_ci"]],
               "hard_negatives_misread_held_out": d.report["hard_negatives_violated"], "test_n": len(test),
               "test": {"auc": boot(lambda i: roc_auc_score(y[i], p[i]), docs),
                        "acc": boot(lambda i: np.mean((p[i] >= 0.5) == y[i]), docs)},
               "escalate": round(float(np.mean(st == "escalate")), 3), "abstain": round(float(np.mean(st == "abstain")), 3),
               "acc_answered": round(float(np.mean((p[ans] >= 0.5) == y[ans])), 3),
               "hard_negatives_p": [round(float(x), 3) for x in run(HARD_NEGATIVES)[0]]}
        out[name] = res
        collected[name] = (p, st)
        print(f"\n== decision {name}: layer {layer}, {len(contrasts)} {kind} contrasts -> {res['method']}")
        print("   " + json.dumps({k: v for k, v in res.items() if k not in ("layer", "contrasts")}))
    # Comparison with the teacher, asked whether each text holds a personal data value (pii/teacher.jsonl)
    two = lambda x: np.stack([1 - x, x], 1)
    q = data.TEACHER_QUESTIONS["pii"]
    # With labels: layer-8 features, the 60 calibration windows plus up to 200
    # labelled test windows from other document types (cross-fitted, grouped by document type).
    feats = lambda texts: embed_all(texts)[8]  # noqa: E731
    labels = data.Labels(feats([t[0] for t in test]), groups=test_types, base_X=feats([c["span"]["text"] for c in cal]),
                           base_y=np.array([int(c["label"] == "yes") for c in cal]))
    data.teacher_compare("pii", "gretel windows from unseen document types", q, [t[0] for t in test], y,
                           {n: (two(pt), stt) for n, (pt, stt) in collected.items()}, labels=labels,
                           spec="contains_pii")
    return out


if __name__ == "__main__":
    main()

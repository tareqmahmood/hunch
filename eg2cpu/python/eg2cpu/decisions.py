"""Compile contrastive decision specs into handles that eg2cpu applies in Rust.

A spec (YAML or a dict) names a question about a span of a trace, the matched contrasts that define it, how to
read it out of EmbeddingGemma 2, how to calibrate it on a labelled target-domain sample, when to abstain, and
when to escalate to an LLM. `compile` turns it into a handle: one or more directions in the encoder's space plus
the constants to standardize, calibrate, abstain and escalate. Applying any number of handles to a span costs one
encoder pass (shared by every handle that embeds the same text), a dot product per direction, and a dim x dim
whitening product per handle that can abstain.

    import eg2cpu
    from eg2cpu import decisions

    model = eg2cpu.Model()
    d = decisions.compile("tool_call_failed.yaml", model, calibration)  # [{"span": {...}, "label": "yes"}, ...]
    d.save("tool_call_failed.eg2h")
    h = eg2cpu.Handle("tool_call_failed.eg2h")
    [[v]] = model.apply([h], [{"kind": "tool_result", "call": "bash: make", "output": "make: *** Error 2"}])
    v["status"], v["answer"], v["p"]                             # "answer", "yes", 0.97

Spec keys (the eg2cpu README's "Decisions" section is the full reference):

    decision        name
    kind            boolean | choice | score | levels
    options         choice only: the labels
    levels          levels only: the ordered levels, lowest first; answers carry per-level
                    probabilities, the expected level as `score`, `confidence` and a `legend`
    monotonicity_check  error | warn (default) | off: levels whose held-out means are out of order
    unit            tool_result | tool_call | text, or {template: "...", varies: field}
    applies_when    gate over span fields, e.g. span.kind == "tool_result"; a span.<field> on the right is that
                    field's value, and true / false match in any case
    encoder         google/embeddinggemma-2@<rev>  (<rev> or no @ = the loaded model's revision)
    readout         {layer: final | <int> | auto, layers: [...] (candidates for auto),
                     method: diff_of_means | lda_shrinkage | ridge (score) | auto}
                    layer auto compiles each candidate and picks by transfer score (if given) or grouped CV on
                    the contrasts, keeping final within one standard error of the best
    transfer        {data: file.jsonl}: optional labelled rows from a second domain; scored in the report
                    every method is fitted and compared in the report; auto picks the default unless it
                    misreads a hard negative and another method doesn't
    hard_negative_check  error | warn (default) | off: what to do when the chosen method misreads one,
                    each checked held out (refit without it)
    contrasts       boolean: {<base fields>, yes, no}; choice: {<base fields>, <option>: text, ...};
                    score: {<base fields>, levels: [{<varies>: text, value: x}, ...]};
                    levels: {<base fields>, <level>: text, ...}, at least two levels per set
    hard_negatives  boolean: text or {<fields>} that must read "no"; choice, levels: {..., label}; score:
                    {..., value} (anchors the axis; not checked, since a value has no wrong side)
    siblings        names of neighbouring decisions (recorded)
    calibration     {method, sample: n, data: file.jsonl}; data rows are {"span": {...}, "label": ...}. method is
                    optional and must match the kind: platt (boolean), multinomial (choice), linear (score),
                    ordinal (levels). Labels, options, levels and contrast keys are compared as text (`no` and
                    False are "no", 1 is "1"). A choice sample needs rows of every option
    abstain         {mahalanobis_quantile: 0.99}
    escalate        {when: "|p - 0.5| < 0.2" (boolean) | "|value - c| < w" (score, levels) |
                     "confidence < t" or "p_max < t" | "margin < t" (choice, levels),
                     to: "model-a | model-b", prompt: "..."}
                    the prompt may name {unit} (the rendered unit) and the unit template's fields; without
                    {unit} the unit text comes first, then the prompt. A score verdict can read "escalate", but
                    `escalate()` resolves only boolean, choice and levels decisions
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._eg2cpu import HANDLE_FORMAT, Handle, Model
from ._eg2cpu import gate_eval as gate_eval, render_template as render_template

UNITS = {
    "tool_result": {"template": "{call}\n{output}", "varies": "output"},
    "tool_call": {"template": "{call}", "varies": "call"},
    "text": {"template": "{text}", "varies": "text"},
}
METHODS = {
    "boolean": ("diff_of_means", "lda_shrinkage"),
    "choice": ("diff_of_means", "lda_shrinkage"),
    "score": ("diff_of_means", "ridge"),
    "levels": ("diff_of_means", "ridge"),
}


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------- spans and specs


def span(d: dict[str, Any]) -> dict[str, str]:
    """Spans are str -> str maps; numbers and booleans are stringified, nested values JSON-encoded."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = v if isinstance(v, str) else json.dumps(v) if isinstance(v, (dict, list)) else str(v).lower() if isinstance(v, bool) else str(v)
    return out


def load_spec(spec: str | Path | dict) -> tuple[dict, Path]:
    """Parse a YAML file (or take a dict); returns the spec and the directory relative paths resolve against."""
    if isinstance(spec, dict):
        return spec, Path.cwd()
    import yaml

    path = Path(spec)
    return yaml.safe_load(path.read_text()), path.resolve().parent


def _label_name(x: Any) -> str:
    """A label, option, level or contrast key as text. YAML 1.1 reads unquoted yes / no as booleans and digits as
    numbers; every label goes through this one conversion, so `no`, `"no"` and False all name the level "no", and
    1 and "1" both name the level "1"."""
    return "yes" if x is True else "no" if x is False else str(x)


def _label_keys(d: dict) -> dict:
    return {_label_name(k): v for k, v in d.items()}


def _unit(spec: dict) -> dict:
    u = spec.get("unit", "tool_result")
    if isinstance(u, str):
        if u not in UNITS:
            raise SpecError(f"unit {u!r}: expected one of {sorted(UNITS)} or {{template, varies}}")
        return {"name": u, **UNITS[u]}
    if "template" not in u or "varies" not in u:
        raise SpecError("a custom unit needs 'template' and 'varies'")
    return {"name": u.get("name", "custom"), "template": u["template"], "varies": u["varies"]}


def _escalate_rule(when: str, kind: str) -> dict:
    num = r"(-?\d+(?:\.\d+)?)"
    if m := re.fullmatch(rf"\s*\|\s*(p|value|score)\s*-\s*{num}\s*\|\s*<\s*{num}\s*", when):
        var, center, width = m.group(1), float(m.group(2)), float(m.group(3))
        if (var == "p") != (kind == "boolean") or kind == "choice":
            raise SpecError(f"escalate.when {when!r}: use |p - c| < w for boolean and |value - c| < w for score "
                            "or levels (|score - c| < w also works for levels)")
        return {"type": "band", "center": center, "width": width}
    if m := re.fullmatch(rf"\s*(p_max|confidence|margin)\s*<\s*{num}\s*", when):
        if kind not in ("choice", "levels"):
            raise SpecError(f"escalate.when {when!r}: p_max / confidence / margin rules are for choice and levels")
        return {"type": "margin_below" if m.group(1) == "margin" else "p_max_below", "value": float(m.group(2))}
    raise SpecError(f"escalate.when {when!r}: expected '|p - c| < w', '|value - c| < w', 'confidence < t' "
                    "(or 'p_max < t') or 'margin < t'")


# ---------------------------------------------------------------- numerics


def ledoit_wolf(Xc: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance of already-centred rows (toward a scaled identity)."""
    n, p = Xc.shape
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    b2 = ((np.einsum("ij,ij->i", Xc, Xc) ** 2).sum() / n - (S**2).sum()) / n / p
    shrink = 1.0 if d2 == 0 else min(max(b2, 0.0), d2) / d2
    return shrink * mu * np.eye(p) + (1 - shrink) * S


def whitener(Xc: np.ndarray) -> np.ndarray:
    """W with |W x|^2 = x^T Sigma^-1 x for the shrunk covariance of centred rows."""
    return np.linalg.cholesky(np.linalg.inv(ledoit_wolf(Xc))).T


def heldout_distances(R: np.ndarray, groups: list[str], folds: int = 5) -> np.ndarray:
    """Each row's Mahalanobis distance under a mean and covariance fitted without its group (grouped k-fold).
    In-sample distances are far too small when rows are few relative to dimensions, and rows that share a
    call (a matched contrast set, calibration rows of one call) must be held out together or the threshold
    leaks; either would make every new span look out of distribution."""
    names = sorted(set(groups))
    fold_of = dict(zip(np.random.default_rng(0).permutation(names), np.arange(len(names)) % folds))
    f_row = np.array([fold_of[g] for g in groups])
    d = np.empty(len(R))
    for f in range(min(folds, len(names))):
        te, tr = np.flatnonzero(f_row == f), np.flatnonzero(f_row != f)
        mu = R[tr].mean(0)
        d[te] = np.linalg.norm((R[te] - mu) @ whitener(R[tr] - mu).T, axis=1)
    return d


def _unit_norm(w: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(w)
    if not np.isfinite(n) or n == 0:
        raise SpecError("the contrasts do not separate: the estimated direction is zero")
    return w / n


def _ridge_direction(Xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Unit ridge-regression direction for centred rows and targets, penalty 0.1 x the mean eigenvalue of Xc^T Xc."""
    G = Xc.T @ Xc
    return _unit_norm(np.linalg.solve(G + 0.1 * np.trace(G) / len(G) * np.eye(len(G)), Xc.T @ yc))


def fit_logistic(z: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> tuple[float, float]:
    """1-D logistic regression by Newton's method (Platt scaling on standardized scores)."""
    a, b = 1.0, 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([((p - y) * z).sum() + l2 * a, (p - y).sum()])
        w = p * (1 - p)
        H = np.array([[(w * z * z).sum() + l2, (w * z).sum()], [(w * z).sum(), w.sum() + 1e-9]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def fit_softmax(Z: np.ndarray, y: np.ndarray, k: int, l2: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression on standardized scores, logits = Z A^T + b, by damped Newton."""
    n, m = Z.shape
    X = np.hstack([Z, np.ones((n, 1))])
    Y = np.eye(k)[y]
    theta = np.zeros((k, m + 1))
    theta[:, :m] = np.eye(k, m)  # start from "the readout score is the logit"
    reg = np.full((k, m + 1), l2)
    reg[:, m] = 1e-9

    def loss(t):
        L = X @ t.T
        L -= L.max(1, keepdims=True)
        return -(Y * (L - np.log(np.exp(L).sum(1, keepdims=True)))).sum() + 0.5 * (reg * t * t).sum()

    for _ in range(100):
        L = X @ theta.T
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = (P - Y).T @ X + reg * theta
        H = np.zeros((k * (m + 1), k * (m + 1)))
        for i in range(n):
            H += np.kron(np.diag(P[i]) - np.outer(P[i], P[i]), np.outer(X[i], X[i]))
        H += np.diag(reg.ravel())
        step = np.linalg.lstsq(H, G.ravel(), rcond=None)[0].reshape(k, m + 1)
        t, f0 = 1.0, loss(theta)
        while loss(theta - t * step) > f0 and t > 1e-6:
            t /= 2
        theta = theta - t * step
        if np.abs(t * step).max() < 1e-9:
            break
    return theta[:, :m], theta[:, m]


def auc_ci(a: float, n_pos: int, n_neg: int) -> list[float]:
    """95% interval for an AUC (Hanley & McNeil 1982)."""
    if not (n_pos and n_neg) or not np.isfinite(a):
        return [float("nan"), float("nan")]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt(max(a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a), 0) / (n_pos * n_neg))
    return [max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)]


def proportion_ci(k: int, n: int) -> list[float]:
    """95% Wilson interval for k successes in n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p, z = k / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def corr_ci(r: float, n: int) -> list[float]:
    """95% interval for a Pearson correlation (Fisher z)."""
    if n < 4 or not np.isfinite(r):
        return [float("nan"), float("nan")]
    z, se = math.atanh(max(min(r, 0.999999), -0.999999)), 1 / math.sqrt(n - 3)
    return [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)]


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        idx = x == v
        if idx.sum() > 1:
            r[idx] = r[idx].mean()
    return r


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC of scores `s` for 0/1 labels `y` (Mann-Whitney, ties count half); nan if either class is absent."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _ranks(np.r_[pos, neg])  # 0-based
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(np.asarray(a, float)), _ranks(np.asarray(b, float)))[0, 1])


def qwk(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Quadratic-weighted Cohen's kappa between two level assignments."""
    O = np.zeros((k, k))
    for a, b in zip(y, pred):
        O[int(a), int(b)] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E_ = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E_).sum()
    return float(1 - (w * O).sum() / den) if den else float("nan")


def ordinal_probs(z: np.ndarray, a: float, theta: np.ndarray) -> np.ndarray:
    """Proportional odds: P(level <= k) = sigmoid(theta_k - a z); rows are per-level probabilities."""
    cum = 1 / (1 + np.exp(-(theta[None, :] - a * z[:, None])))
    cum = np.hstack([cum, np.ones((len(z), 1))])
    return np.diff(np.hstack([np.zeros((len(z), 1)), cum]), axis=1)


def fit_ordinal(z: np.ndarray, y: np.ndarray, k: int, prior: float = 1e-2) -> tuple[float, np.ndarray]:
    """Slope and increasing cut-points of a proportional-odds model on standardized scores, by damped Newton
    (numerical derivatives; there are only k parameters). Cut-point gaps are parameterized on a log scale and
    pulled gently toward their starting values, so a level missing from a small sample can't collapse."""
    y = y.astype(int)
    counts = np.bincount(y, minlength=k) + 0.5
    cum = np.cumsum(counts)[:-1] / counts.sum()
    theta0 = np.log(cum / (1 - cum))
    gaps0 = np.log(np.maximum(np.diff(theta0), 1e-3))

    def unpack(v):
        return v[0], np.concatenate([[v[1]], v[1] + np.cumsum(np.exp(v[2:]))])

    def nll(v):
        a, th = unpack(v)
        P = ordinal_probs(z, a, th)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).sum() + prior * ((v[2:] - gaps0) ** 2).sum() \
            + 1e-4 * v[0] ** 2

    v = np.concatenate([[1.0, theta0[0]], gaps0])
    h = 1e-4
    for _ in range(100):
        n = len(v)
        E = np.eye(n) * h
        g = np.array([(nll(v + E[i]) - nll(v - E[i])) / (2 * h) for i in range(n)])
        H = np.array([[(nll(v + E[i] + E[j]) - nll(v + E[i] - E[j]) - nll(v - E[i] + E[j]) + nll(v - E[i] - E[j]))
                       / (4 * h * h) for j in range(n)] for i in range(n)])
        H = (H + H.T) / 2 + np.eye(n) * 1e-6
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = g
        t, f0 = 1.0, nll(v)
        while nll(v - t * step) > f0 and t > 1e-6:
            t /= 2
        v = v - t * step
        if np.abs(t * step).max() < 1e-8:
            break
    a, th = unpack(v)
    return float(a), th


def ordinal_metrics(y: np.ndarray, P: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Spearman of the readout with the level, quadratic-weighted kappa of the most likely level, exact and
    within-one accuracy, each with a 95% interval where one is cheap."""
    pred = np.argmax(P, 1)
    exact, near = int(np.sum(pred == y)), int(np.sum(np.abs(pred - y) <= 1))
    rho = spearman(scores, y)
    return {"spearman": rho, "spearman_ci": corr_ci(rho, len(y)), "qwk": qwk(y, pred, k),
            "accuracy": exact / len(y), "accuracy_ci": proportion_ci(exact, len(y)),
            "within_one": near / len(y), "within_one_ci": proportion_ci(near, len(y))}


def _fit_generic(kind: str, method: str, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Directions from labelled rows alone (no matched structure), for cross-validating a readout."""
    if kind in ("score", "levels"):
        return _axis_direction(method, X - X.mean(0), y - y.mean())
    classes = [1, 0] if kind == "boolean" else list(range(k))
    present = [c for c in classes if (y == c).any()]
    mus = {c: X[y == c].mean(0) for c in present}
    if kind == "boolean":
        d = mus[1] - mus[0]
        if method == "diff_of_means":
            return _unit_norm(d)[None]
        Sigma = ledoit_wolf(np.vstack([X[y == c] - mus[c] for c in present]))
        return _unit_norm(np.linalg.solve(Sigma, d))[None]
    grand = np.mean([mus[c] for c in present], 0)
    Sigma = ledoit_wolf(np.vstack([X[y == c] - mus[c] for c in present])) if method != "diff_of_means" else None
    W = []
    for c in classes:
        if c not in mus:  # no training rows of this class in the fold: it can't be predicted
            W.append(np.zeros(X.shape[1]))
            continue
        d = mus[c] - grand
        W.append(_unit_norm(d if Sigma is None else np.linalg.solve(Sigma, d)))
    return np.array(W)


def contrast_cv(kind: str, method: str, X: np.ndarray, y: np.ndarray, groups: list[str], pairs: list[tuple[int, int]],
                k: int, folds: int = 5) -> dict | None:
    """Grouped k-fold over the contrasts: whole groups (a contrast, or the spec's `group`, e.g. a document type)
    are held out, the readout is refit on the rest, and the held-out rows are scored. This estimates how a
    readout generalizes to formats it hasn't seen, which is what transfer needs, from the contrasts alone."""
    names = sorted(set(groups))
    if len(names) < 2:
        return None
    fold_of = dict(zip(np.random.default_rng(0).permutation(names), np.arange(len(names)) % folds))
    f_row = np.array([fold_of[g] for g in groups])
    S = np.zeros((len(X), 1 if kind != "choice" else k))
    for f in range(min(folds, len(names))):
        te, tr = np.flatnonzero(f_row == f), np.flatnonzero(f_row != f)
        if kind == "boolean" and len(set(y[tr].tolist())) < 2:
            return None
        W = _fit_generic(kind, method, X[tr], y[tr], k)
        Str = X[tr] @ W.T
        S[te] = (X[te] @ W.T - Str.mean(0)) / (Str.std(0) + 1e-12)
    if kind == "boolean":
        a = auc(y.astype(int), S[:, 0])
        r = {"auc": a, "auc_ci": auc_ci(a, int(y.sum()), int((1 - y).sum()))}
        if pairs:
            r["paired_accuracy"] = float(np.mean([S[i, 0] > S[j, 0] for i, j in pairs]))
        return r
    if kind == "choice":
        hits = int(np.sum(np.argmax(S, 1) == y))
        return {"accuracy": hits / len(y), "accuracy_ci": proportion_ci(hits, len(y))}
    if kind == "levels":
        # Monotonicity, held out: the mean score of each level must rise with the level, and within a held-out
        # set each higher level should outscore the next lower one.
        present = sorted(set(y.astype(int).tolist()))
        means = [float(S[y == lv, 0].mean()) for lv in present]
        rho = spearman(S[:, 0], y)
        r = {"spearman": rho, "spearman_ci": corr_ci(rho, len(y)), "level_means": dict(zip(map(str, present), means)),
             "monotone": bool(all(b > a for a, b in zip(means, means[1:])))}
        if pairs:
            r["order_accuracy"] = float(np.mean([S[j, 0] > S[i, 0] for i, j in pairs]))
        return r
    c = float(np.corrcoef(S[:, 0], y)[0, 1])
    return {"corr": c, "corr_ci": corr_ci(c, len(y))}


# ---------------------------------------------------------------- compiled decisions


@dataclass
class Decision:
    """A compiled decision: the handle file's bytes, its metadata, and a loaded Rust `Handle`."""

    data: bytes
    metadata: dict
    handle: Handle = field(repr=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.data)
        return path

    @property
    def report(self) -> dict:
        return self.metadata.get("report", {})


def _json_safe(x: Any) -> Any:
    """`x` with every NaN or infinite float as None: JSON has neither, and the Rust loader rejects them. Metrics
    that can't be computed (an AUC on one class, an interval from three rows) are null in the handle."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def safetensors_bytes(tensors: dict[str, np.ndarray], meta: dict[str, str]) -> bytes:
    """A safetensors file (float32 tensors, `meta` as `__metadata__`) as bytes: the handle file format."""
    header: dict[str, Any] = {"__metadata__": meta}
    blobs, off = [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


class _Encoder:
    """Readout vectors for texts at one layer. One forward pass yields every layer, so the cache keeps them all
    and can be shared: `layer: auto` scores each candidate layer without encoding anything twice."""

    def __init__(self, model: Model, layer, cache: dict | None = None):
        self.model, self.layer = model, layer
        self.cache = {} if cache is None else cache

    def __call__(self, texts: list[str]) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for t, out in zip(todo, self.model.embed_batch(todo)):
            self.cache[t] = out
        pick = (lambda o: o[0]) if self.layer == "final" else (lambda o: o[1][self.layer])
        return np.array([pick(self.cache[t]) for t in texts], dtype=np.float64).reshape(len(texts), -1)


def _template_fields(template: str) -> list[str]:
    """The span fields a `{field}` template names (`{{` and `}}` are literal braces); raises on a malformed one."""
    render_template(template, {}, unit="")
    names = re.findall(r"\{([^{}]*)\}", re.sub(r"\{\{|\}\}", "", template))
    return [n.strip().removeprefix("span.") for n in names]


def _render(unit: dict, fields: dict, what: str) -> str:
    text = render_template(unit["template"], span(fields))
    if text is None:
        raise SpecError(f"{what}: missing a field the unit template {unit['template']!r} needs ({sorted(fields)} given)")
    return text


def _label_index(kind: str, labels: list[str], label: Any, what: str) -> int | float:
    if kind == "boolean":
        if isinstance(label, str) and label.lower() in ("yes", "no", "true", "false", "1", "0"):
            return int(label.lower() in ("yes", "true", "1"))
        if isinstance(label, (bool, int)) and int(label) in (0, 1):
            return int(label)
        raise SpecError(f"{what}: boolean label must be yes/no or true/false, got {label!r}")
    if kind in ("choice", "levels"):
        if label is None or _label_name(label) not in labels:
            raise SpecError(f"{what}: {'label' if kind == 'choice' else 'level'} {label!r} is not one of {labels}")
        return labels.index(_label_name(label))
    return float(label)


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _spec_data(section: dict | None, base: Path) -> list[dict] | None:
    """The rows of a spec section's `data` jsonl (a relative path resolves against `base`); None without one."""
    if not (section or {}).get("data"):
        return None
    path = Path(section["data"])
    return _read_jsonl(path if path.is_absolute() else base / path)


@dataclass
class _Spec:
    """The validated parts of a spec that every compile step reads."""

    name: str
    kind: str
    unit: dict
    gate: str
    labels: list[str]
    encoder: str
    monotonicity_check: str


def _parse_spec(spec: dict, model: Model) -> _Spec:
    kind = spec.get("kind", "boolean")
    if kind not in METHODS:
        raise SpecError(f"kind {kind!r}: expected boolean, choice, score or levels")
    unit = _unit(spec)
    gate = spec.get("applies_when") or ""
    if gate:
        gate_eval(gate, {})  # raises on a syntax error
    labels = (["no", "yes"] if kind == "boolean" else [_label_name(x) for x in spec.get("options") or []]
              if kind == "choice" else [_label_name(x) for x in spec.get("levels") or []] if kind == "levels" else [])
    if len(set(labels)) < len(labels):
        raise SpecError(f"{'options' if kind == 'choice' else 'levels'} {labels}: names must be distinct")
    if kind == "choice" and len(labels) < 2:
        raise SpecError("a choice decision needs at least two options")
    if kind == "levels" and len(labels) < 2:
        raise SpecError("a levels decision needs `levels`: at least two, in order from lowest to highest")
    mono_check = spec.get("monotonicity_check", "warn")
    if mono_check not in ("error", "warn", "off"):
        raise SpecError(f"monotonicity_check {mono_check!r}: expected error, warn or off")
    enc_name, _, rev = str(spec.get("encoder", "google/embeddinggemma-2")).partition("@")
    if rev and not (rev.startswith("<") and rev.endswith(">")) and rev != model.revision:
        raise SpecError(f"spec pins encoder revision {rev}, but the loaded model is {model.revision}")
    return _Spec(name=spec.get("decision") or "decision", kind=kind, unit=unit, gate=gate, labels=labels,
                 encoder=enc_name, monotonicity_check=mono_check)


def _fit_options(spec: dict, d: _Spec, layer, model: Model) -> tuple[str, str, dict | None]:
    """The readout method, the hard negative check and the escalation rule, validated for a fixed layer."""
    if layer != "final" and not (isinstance(layer, int) and 0 <= layer < model.num_layers):
        raise SpecError(f"readout.layer {layer!r}: expected 'final', auto or 0..{model.num_layers - 1}")
    method = (spec.get("readout") or {}).get("method", METHODS[d.kind][0])
    if method != "auto" and method not in METHODS[d.kind]:
        raise SpecError(f"readout.method {method!r} for {d.kind}: expected auto or one of {METHODS[d.kind]}")
    check = spec.get("hard_negative_check", "warn")
    if check not in ("error", "warn", "off"):
        raise SpecError(f"hard_negative_check {check!r}: expected error, warn or off")
    rule = _escalate_rule(str(spec["escalate"].get("when", "")), d.kind) if spec.get("escalate") else None
    return method, check, rule


# ---- contrasts -> training examples


@dataclass
class _Training:
    """The rows a readout is fitted on: the contrasts, then the hard negatives.

    `directions(method, keep)` fits the readout with the hard negatives the boolean mask `keep` selects. `groups`
    names each row's contrast (abstain's held-out distances), `cv_groups` and `cv_pairs` feed `contrast_cv`, and
    `hard_X`, `hard_y`, `hard_texts` are the hard negatives a fit must read as labelled."""

    directions: Callable[[str, np.ndarray], np.ndarray]
    X: np.ndarray
    y: np.ndarray
    groups: list[str]
    cv_groups: list[str]
    cv_pairs: list[tuple[int, int]]
    hard_X: np.ndarray
    hard_y: np.ndarray
    hard_texts: list[str]


def _contrast_fields(c: dict, value: str, d: _Spec) -> dict:
    return {k: v for k, v in c.items() if k not in ("yes", "no", "levels", "group", *d.labels)} | {d.unit["varies"]: value}


def _hard_fields(h, unit: dict) -> dict:
    # A bare string is the varying field; other template fields the unit needs are left empty.
    blank = {f: "" for f in re.findall(r"\{(\w+)\}", unit["template"])}
    if isinstance(h, str):
        return blank | {unit["varies"]: h}
    return blank | {k: v for k, v in h.items() if k not in ("label", "value")}


def _contrast_sets(d: _Spec, contrasts: list[dict]) -> list[tuple[list[str], Any]]:
    """(texts, targets) per matched contrast set: the same situation under each label (choice, levels) or at each
    value (score) it gives. Targets are label indices, or a float array of values for a score."""
    sets = []
    for i, c in enumerate(contrasts):
        if d.kind == "score":
            levels = c.get("levels") or []
            if len(levels) < 2:
                raise SpecError(f"contrast {i}: a score contrast needs at least two levels")
            texts = [_render(d.unit, _contrast_fields(c, lv.get(d.unit["varies"], lv.get("text")), d), f"contrast {i}")
                     for lv in levels]
            sets.append((texts, np.array([float(lv["value"]) for lv in levels])))
            continue
        present = [lab for lab in d.labels if lab in c]
        if len(present) < 2:
            raise SpecError(f"contrast {i}: a {d.kind} contrast needs at least two of {d.labels}")
        sets.append(([_render(d.unit, _contrast_fields(c, c[lab], d), f"contrast {i}") for lab in present],
                     [d.labels.index(lab) for lab in present]))
    return sets


def _hard_rows(d: _Spec, hard: list) -> list[tuple[str, int | float]]:
    """(text, target) per hard negative of a choice, levels or score decision."""
    rows = []
    for i, h in enumerate(hard):
        text = _render(d.unit, _hard_fields(h, d.unit), f"hard negative {i}")
        target = (float(h["value"]) if d.kind == "score" else
                  _label_index(d.kind, d.labels, h.get("label") if isinstance(h, dict) else None, f"hard negative {i}"))
        rows.append((text, target))
    return rows


def _set_rows(sets: list[tuple[list[str], Any]]) -> list[int]:
    """The contrast index of each row when the sets are stacked."""
    return [i for i, (ts, _) in enumerate(sets) for _ in ts]


def _row_groups(contrasts: list[dict], rows: list[int], n_hard: int) -> tuple[list[str], list[str]]:
    """Per-row groups: `c<i>` (the row's contrast) and the cross-validation group (the contrast's `group`, else
    the same); each hard negative `h<j>` is its own group in both."""
    hard = [f"h{j}" for j in range(n_hard)]
    return [f"c{i}" for i in rows] + hard, [str(contrasts[i].get("group", f"c{i}")) for i in rows] + hard


def _adjacent_level_pairs(sets: list[tuple[list[str], list[int]]]) -> list[tuple[int, int]]:
    """Within-set adjacent pairs (lower level, higher level) as row indices, for the held-out order accuracy."""
    pairs, off = [], 0
    for ts, c in sets:
        o = sorted(range(len(c)), key=lambda q: c[q])
        pairs += [(off + o[q], off + o[q + 1]) for q in range(len(o) - 1) if c[o[q]] < c[o[q + 1]]]
        off += len(ts)
    return pairs


def _axis_direction(method: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One direction along centred targets: score, levels."""
    if method == "diff_of_means":
        return _unit_norm(X.T @ y)[None]
    return _ridge_direction(X, y)[None]


def _boolean_directions(method: str, Xp: np.ndarray, Xn: np.ndarray) -> np.ndarray:
    dm = Xp.mean(0) - Xn.mean(0)
    if method == "diff_of_means":
        return _unit_norm(dm)[None]
    return _unit_norm(np.linalg.solve(ledoit_wolf(np.vstack([Xp - Xp.mean(0), Xn - Xn.mean(0)])), dm))[None]


def _choice_directions(method: str, Xs: list[np.ndarray], allX: np.ndarray, ys: np.ndarray, Hk: np.ndarray,
                       hk: np.ndarray, k: int) -> np.ndarray:
    if method == "diff_of_means":
        # Matched: centre each contrast set, so what differs between options is all that is left.
        Xc = np.vstack([X - X.mean(0) for X in Xs] + ([Hk - allX.mean(0)] if len(Hk) else []))
        yc = np.r_[ys, hk].astype(int)
        return np.array([_unit_norm(Xc[yc == j].mean(0)) for j in range(k)])
    X, y = np.vstack([allX, Hk]), np.r_[ys, hk].astype(int)
    mus = np.array([X[y == j].mean(0) for j in range(k)])
    Sigma = ledoit_wolf(X - mus[y])
    return np.array([_unit_norm(np.linalg.solve(Sigma, mus[j] - mus.mean(0))) for j in range(k)])


def _levels_directions(method: str, Xc: np.ndarray, yc: np.ndarray, Hk: np.ndarray, hk: np.ndarray,
                       centre: np.ndarray, level_mean: float) -> np.ndarray:
    """One axis for levels and score. Hard negatives anchor it too: each adds (x - mean, target - mean target)
    like a one-row set."""
    if len(hk):
        Xc, yc = np.vstack([Xc, Hk - centre]), np.r_[yc, hk - level_mean]
    return _axis_direction(method, Xc, yc)


def _boolean_training(d: _Spec, contrasts: list[dict], hard: list, E: _Encoder) -> _Training:
    P = [_render(d.unit, _contrast_fields(c, c["yes"], d), f"contrast {i}") for i, c in enumerate(contrasts)]
    N = [_render(d.unit, _contrast_fields(c, c["no"], d), f"contrast {i}") for i, c in enumerate(contrasts)]
    H = [_render(d.unit, _hard_fields(h, d.unit), f"hard negative {i}") for i, h in enumerate(hard)]
    Xp, Xn, Xh = E(P), E(N), E(H) if H else np.zeros((0, len(E(P[:1])[0])))
    Xneg = np.vstack([Xn, Xh])
    groups, cv_groups = _row_groups(contrasts, list(range(len(P))) * 2, len(H))
    return _Training(directions=lambda m, keep: _boolean_directions(m, Xp, np.vstack([Xn, Xh[keep]])),
                     X=np.vstack([Xp, Xneg]), y=np.r_[np.ones(len(Xp)), np.zeros(len(Xneg))],
                     groups=groups, cv_groups=cv_groups, cv_pairs=[(i, len(P) + i) for i in range(len(P))],
                     hard_X=Xh, hard_y=np.zeros(len(Xh), dtype=int), hard_texts=H)


def _set_training(d: _Spec, contrasts: list[dict], hard: list, E: _Encoder) -> _Training:
    """Choice and levels: matched sets of the same situation under different labels."""
    sets = _contrast_sets(d, contrasts)
    Hx = _hard_rows(d, hard)
    Xs = [E(t) for t, _ in sets]
    allX = np.vstack(Xs)
    ys = np.concatenate([np.array(c) for _, c in sets])
    Xh = E([t for t, _ in Hx]) if Hx else np.zeros((0, allX.shape[1]))
    yh = np.array([c for _, c in Hx], dtype=int)
    if d.kind == "choice":
        def directions(m, keep):
            return _choice_directions(m, Xs, allX, ys, Xh[keep], yh[keep], len(d.labels))
        cv_pairs = []
    else:
        # One ordinal axis: centre each set (only the grading is left), fit against centred level indices.
        Xc = np.vstack([X - X.mean(0) for X in Xs])
        yc = np.concatenate([np.array(c, float) - np.mean(c) for _, c in sets])
        level_mean = np.mean([v for _, c in sets for v in c])

        def directions(m, keep):
            return _levels_directions(m, Xc, yc, Xh[keep], yh[keep], allX.mean(0), level_mean)
        cv_pairs = _adjacent_level_pairs(sets)
    groups, cv_groups = _row_groups(contrasts, _set_rows(sets), len(Hx))
    return _Training(directions=directions, X=np.vstack([allX, Xh]), y=np.r_[ys, yh].astype(int), groups=groups,
                     cv_groups=cv_groups, cv_pairs=cv_pairs, hard_X=Xh, hard_y=yh, hard_texts=[t for t, _ in Hx])


def _score_training(d: _Spec, contrasts: list[dict], hard: list, E: _Encoder) -> _Training:
    """Score: matched sets at several values. Hard negatives anchor the axis as they do for levels, but aren't
    checked: a value has no "wrong side" to read them on."""
    sets = _contrast_sets(d, contrasts)
    Xg = [E(t) for t, _ in sets]
    allX = np.vstack(Xg)
    Xc = np.vstack([X - X.mean(0) for X in Xg])
    yc = np.concatenate([v - v.mean() for _, v in sets])
    value_mean = np.concatenate([v for _, v in sets]).mean()
    Hx = _hard_rows(d, hard)
    Xh = E([t for t, _ in Hx]) if Hx else np.zeros((0, allX.shape[1]))
    yh = np.array([v for _, v in Hx], dtype=float)
    groups, cv_groups = _row_groups(contrasts, _set_rows(sets), len(Hx))
    return _Training(directions=lambda m, keep: _levels_directions(m, Xc, yc, Xh, yh, allX.mean(0), value_mean),
                     X=np.vstack([allX, Xh]), y=np.concatenate([v for _, v in sets] + [yh]), groups=groups,
                     cv_groups=cv_groups, cv_pairs=[],
                     hard_X=np.zeros((0, allX.shape[1])), hard_y=np.zeros(0), hard_texts=[])


def _training(spec: dict, d: _Spec, E: _Encoder) -> tuple[_Training, int, int]:
    """The training rows from the spec's contrasts and hard negatives, and how many of each there are."""
    contrasts = [_label_keys(c) for c in spec.get("contrasts") or []]
    if not contrasts:
        raise SpecError("no contrasts")
    hard = spec.get("hard_negatives") or []
    build = {"boolean": _boolean_training, "choice": _set_training, "levels": _set_training,
             "score": _score_training}[d.kind]
    return build(d, contrasts, hard, E), len(contrasts), len(hard)


# ---- labelled rows from the target domain


def _labelled_rows(d: _Spec, rows: list[dict], what: str) -> list[tuple[str, int | float, dict]]:
    """(text, label index, span) per {"span", "label"} row the decision applies to."""
    out = []
    for i, row in enumerate(rows):
        s = span(row["span"])
        if d.gate and not gate_eval(d.gate, s):
            continue  # the decision does not apply to this span
        text = render_template(d.unit["template"], s)
        if text is None:
            raise SpecError(f"{what} {i}: span lacks a field the unit template needs")
        out.append((text, _label_index(d.kind, d.labels, row["label"], f"{what} {i}"), s))
    return out


def _calibration_rows(spec: dict, d: _Spec, calibration: list[dict] | None,
                      allow_uncalibrated: bool) -> list[tuple[str, int | float, dict]]:
    cal_spec = dict(spec.get("calibration") or {})
    cal_method = {"boolean": "platt", "choice": "multinomial", "score": "linear", "levels": "ordinal"}[d.kind]
    if cal_spec.get("method", cal_method) != cal_method:
        raise SpecError(f"calibration.method {cal_spec['method']!r}: a {d.kind} decision is calibrated with {cal_method}")
    cal_rows = _labelled_rows(d, calibration or [], "calibration row")
    want = cal_spec.get("sample")
    if cal_rows and want and len(cal_rows) != int(want):
        warnings.warn(f"{d.name}: calibration.sample says {want}, got {len(cal_rows)} applicable rows", stacklevel=3)
    if not cal_rows and not allow_uncalibrated:
        raise SpecError(f"{d.name}: calibration is mandatory (rankings transfer, cut-offs don't); give calibration "
                        "data or pass allow_uncalibrated=True")
    return cal_rows


# ---- fit and calibrate a readout


def _sigmoid(v):
    return 1 / (1 + np.exp(-v))


def _calibrate(d: _Spec, rule: dict | None, t: _Training, Xcal: np.ndarray, ycal: np.ndarray, W: np.ndarray) -> dict:
    """Calibrate one candidate direction set and measure it on the calibration sample (with 95% intervals,
    which are wide at n=60); `violations` lists the hard negatives it misreads."""
    kind, labels = d.kind, d.labels
    S = Xcal @ W.T
    mean, std = S.mean(0), S.std(0) + 1e-12
    Z = (S - mean) / std
    Zh = (t.hard_X @ W.T - mean) / std
    r: dict[str, Any] = {}
    if kind == "boolean":
        a, b = fit_logistic(Z[:, 0], ycal.astype(float))
        cal = {"method": "platt", "weights": [[a]], "bias": [b]}
        p = _sigmoid(a * Z[:, 0] + b)
        A = auc(ycal, S[:, 0])
        r |= {"train_auc": auc(t.y, t.X @ W[0]), "calibration_auc": A,
              "calibration_auc_ci": auc_ci(A, int(ycal.sum()), int((1 - ycal).sum())),
              "calibration_accuracy": float(np.mean((p >= 0.5) == ycal))}
        bad = np.flatnonzero(_sigmoid(a * Zh[:, 0] + b) >= 0.5)
        unsure = np.abs(p - rule["center"]) < rule["width"] if rule else None
    elif kind == "choice":
        yc = ycal.astype(int)
        A_, bias = fit_softmax(Z, yc, len(labels))
        cal = {"method": "multinomial", "weights": A_.tolist(), "bias": bias.tolist()}
        L = Z @ A_.T + bias
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        k = int(np.sum(np.argmax(L, 1) == yc))
        r |= {"train_accuracy": float(np.mean(np.argmax(t.X @ W.T, 1) == t.y)),
              "calibration_accuracy": k / len(yc), "calibration_accuracy_ci": proportion_ci(k, len(yc))}
        bad = np.flatnonzero(np.argmax(Zh @ A_.T + bias, 1) != t.hard_y) if len(Zh) else np.zeros(0, dtype=int)
        Ps = np.sort(P, 1)[:, ::-1]
        unsure = None if not rule else Ps[:, 0] < rule["value"] if rule["type"] == "p_max_below" else Ps[:, 0] - Ps[:, 1] < rule["value"]
    elif kind == "levels":
        yc_ = ycal.astype(int)
        a, theta = fit_ordinal(Z[:, 0], yc_, len(labels))
        cal = {"method": "ordinal", "weights": [[a]], "bias": theta.tolist()}
        P = ordinal_probs(Z[:, 0], a, theta)
        r |= {"train_spearman": spearman(t.X @ W[0], t.y),
              **{f"calibration_{k_}": v_ for k_, v_ in ordinal_metrics(yc_, P, S[:, 0], len(labels)).items()}}
        bad = (np.flatnonzero(np.argmax(ordinal_probs(Zh[:, 0], a, theta), 1) != t.hard_y) if len(Zh)
               else np.zeros(0, dtype=int))
        Ps = np.sort(P, 1)[:, ::-1]
        if not rule:
            unsure = None
        elif rule["type"] == "band":
            unsure = np.abs(P @ np.arange(len(labels)) - rule["center"]) < rule["width"]
        else:
            unsure = Ps[:, 0] < rule["value"] if rule["type"] == "p_max_below" else Ps[:, 0] - Ps[:, 1] < rule["value"]
    else:
        a, b = np.polyfit(Z[:, 0], ycal.astype(float), 1)
        cal = {"method": "linear", "weights": [[float(a)]], "bias": [float(b)]}
        v = a * Z[:, 0] + b
        c = float(np.corrcoef(S[:, 0], ycal)[0, 1])
        r |= {"train_corr": float(np.corrcoef(t.X @ W[0], t.y)[0, 1]), "calibration_corr": c,
              "calibration_corr_ci": corr_ci(c, len(ycal)), "calibration_rmse": float(np.sqrt(np.mean((v - ycal) ** 2)))}
        bad = np.zeros(0, dtype=int)
        unsure = np.abs(v - rule["center"]) < rule["width"] if rule else None
    if unsure is not None:
        r["calibration_escalate"] = float(np.mean(unsure))
    return {"W": W, "mean": mean, "std": std, "cal": cal, "report": r, "violations": [int(i) for i in bad]}


def _fit_methods(d: _Spec, rule: dict | None, t: _Training, Xcal: np.ndarray, ycal: np.ndarray,
                 calibrated: bool) -> dict[str, dict]:
    """Fit and calibrate every method for the kind.

    Hard negatives are also training points, and in 768 dimensions a method can carve almost any training point
    onto its labelled side, so "keeps every hard negative" in-sample says little. Each one is checked held out:
    refit without it, then read it. Uncalibrated, the calibration rows are the training rows, so the held-out
    calibration leaves that hard negative out as well. Score hard negatives aren't checked (None in the report)."""
    n_hard = len(t.hard_X)
    everything = np.ones(n_hard, dtype=bool)
    fits = {}
    for m in METHODS[d.kind]:
        fits[m] = _calibrate(d, rule, t, Xcal, ycal, t.directions(m, everything))
        in_sample = fits[m]["violations"]
        held_out = []
        for i in range(n_hard):
            row = len(t.X) - n_hard + i  # the hard negatives are the last training rows
            Xc_, yc_ = (Xcal, ycal) if calibrated else (np.delete(Xcal, row, 0), np.delete(ycal, row))
            W = t.directions(m, everything & (np.arange(n_hard) != i))
            if i in _calibrate(d, rule, t, Xc_, yc_, W)["violations"]:
                held_out.append(i)
        fits[m]["violations"] = held_out
        checked = d.kind != "score"
        fits[m]["report"] |= {"hard_negatives_violated": len(held_out) if checked else None,
                              "hard_negatives_violated_in_sample": len(in_sample) if checked else None}
        if d.kind == "boolean":
            fits[m]["report"]["hard_negatives_read_yes"] = len(held_out)
    return fits


def _choose_method(method: str, kind: str, fits: dict[str, dict], n_hard: int) -> tuple[str, str]:
    """The method and why: as the spec says, or by a fixed rule for auto (never by calibration score, which at
    n=60 is mostly noise and would also bias the numbers reported)."""
    order = list(METHODS[kind])
    if method != "auto":
        return method, "set in the spec"
    if not n_hard:
        return order[0], f"auto: no hard negatives to decide by, so the default ({order[0]})"
    if ok := [m for m in order if not fits[m]["violations"]]:
        chosen = ok[0]
        return chosen, (f"auto: the default ({order[0]}) keeps every hard negative" if chosen == order[0] else
                        f"auto: {order[0]} misreads {len(fits[order[0]]['violations'])} of {n_hard} hard negatives; "
                        f"{chosen} keeps all")
    chosen = min(order, key=lambda m: len(fits[m]["violations"]))
    return chosen, f"auto: no method keeps every hard negative; {chosen} misreads the fewest"


def _check_hard_negatives(d: _Spec, check: str, chosen: str, fits: dict[str, dict], t: _Training) -> None:
    violations = fits[chosen]["violations"]
    if not violations or check == "off":
        return
    shown = "; ".join(repr(t.hard_texts[i][:60]) for i in violations[:3])
    others = [m for m in METHODS[d.kind] if m != chosen and not fits[m]["violations"]]
    msg = (f"{d.name}: {len(violations)} of {len(t.hard_X)} hard negatives read wrong with {chosen}: {shown}"
           + (f" ({', '.join(others)} keeps them all; or use method: auto)" if others else ""))
    if check == "error":
        raise SpecError(msg)
    warnings.warn(msg, stacklevel=3)


def _check_monotone(d: _Spec, cv: dict | None) -> None:
    if d.kind != "levels" or not cv or cv["monotone"] or d.monotonicity_check == "off":
        return
    means = ", ".join(f"{d.labels[int(k_)]} {v_:+.2f}" for k_, v_ in cv["level_means"].items())
    msg = (f"{d.name}: held-out level means are not in order ({means}); the levels may not lie on one axis. "
           "Consider kind: choice, or check the contrasts.")
    if d.monotonicity_check == "error":
        raise SpecError(msg)
    warnings.warn(msg, stacklevel=3)


def _transfer_report(d: _Spec, transfer: list[dict], E: _Encoder, f: dict) -> dict:
    """The chosen fit `f` scored on a labelled set from a second domain."""
    rows = _labelled_rows(d, transfer, "transfer row")
    Xt, yt = E([t for t, _, _ in rows]), np.array([y_ for _, y_, _ in rows])
    St = Xt @ f["W"].T
    mean, std, cal = f["mean"], f["std"], f["cal"]
    if d.kind == "boolean":
        a_ = auc(yt.astype(int), St[:, 0])
        return {"n": len(yt), "auc": a_, "auc_ci": auc_ci(a_, int(yt.sum()), int((1 - yt).sum()))}
    if d.kind == "choice":
        L_ = ((St - mean) / std) @ np.array(cal["weights"]).T + np.array(cal["bias"])
        hits = int(np.sum(np.argmax(L_, 1) == yt))
        return {"n": len(yt), "accuracy": hits / len(yt), "accuracy_ci": proportion_ci(hits, len(yt))}
    if d.kind == "levels":
        Pt = ordinal_probs(((St - mean) / std)[:, 0], cal["weights"][0][0], np.array(cal["bias"]))
        return {"n": len(yt), **ordinal_metrics(yt.astype(int), Pt, St[:, 0], len(d.labels))}
    c_ = float(np.corrcoef(St[:, 0], yt)[0, 1])
    return {"n": len(yt), "corr": c_, "corr_ci": corr_ci(c_, len(yt))}


# ---- abstain and escalate


def _abstain(spec: dict, d: _Spec, t: _Training, Xcal: np.ndarray,
             cal_rows: list[tuple[str, int | float, dict]]) -> tuple[dict, dict[str, np.ndarray]]:
    """Mahalanobis distance from everything compiled or calibrated on: the metadata and its tensors."""
    calibrated = bool(cal_rows)
    q = float(spec["abstain"].get("mahalanobis_quantile", 0.99))
    R = np.vstack([t.X] + ([Xcal] if calibrated else []))
    mu = R.mean(0)
    whiten = whitener(R - mu)
    # Threshold from held-out distances, over the target-domain sample when there is one.
    # Calibration rows that share everything but the varying field (e.g. one call) form a group.
    varies = d.unit["varies"]
    cal_groups = ["k" + json.dumps({k: v for k, v in s.items() if k != varies}, sort_keys=True) for _, _, s in cal_rows]
    dist = heldout_distances(R, t.groups + (cal_groups if calibrated else []))
    ref = dist[len(t.X):] if calibrated else dist
    meta = {"mahalanobis_quantile": q, "threshold": float(np.quantile(ref, q)), "reference_n": len(R)}
    return meta, {"abstain_mean": mu, "abstain_whiten": whiten}


def _escalation(spec: dict, d: _Spec, rule: dict | None, cal_rows: list[tuple[str, int | float, dict]],
                ycal: np.ndarray, llm: str | None, llm_model: str) -> dict:
    """The escalation metadata; with `llm`, a boolean decision's LLM answers are Platt-calibrated on the sample."""
    e = spec["escalate"]
    prompt = (e.get("prompt") or "").strip()
    if not prompt:
        raise SpecError("escalate needs a prompt")
    unit_fields = set(_template_fields(d.unit["template"]))
    fields = _template_fields(prompt)  # raises on a malformed template
    if extra := sorted(set(fields) - unit_fields - {"unit"}):
        raise SpecError(f"escalate.prompt names {', '.join('{' + f + '}' for f in extra)}: a prompt may name {{unit}} "
                        f"and the unit template's fields ({', '.join(sorted(unit_fields))}), which every span the "
                        "decision reads has")
    if "unit" not in fields:
        prompt = "{unit}\n\n" + prompt  # the escalation model sees the same unit the encoder did
    to = e.get("to") or []
    meta = {"rule": rule, "prompt": prompt,
            "to": [t.strip() for t in to.split("|")] if isinstance(to, str) else list(to), "llm_calibration": None}
    if llm and cal_rows:
        if d.kind != "boolean":
            raise SpecError("compile-time LLM calibration is implemented for boolean decisions")
        pl = np.array([ask_yes_no(llm, render_template(prompt, s, unit=t), model=llm_model) for t, _, s in cal_rows])
        lg = np.log(np.clip(pl, 1e-6, 1 - 1e-6) / np.clip(1 - pl, 1e-6, 1))
        m, sd = float(lg.mean()), float(lg.std() + 1e-12)
        a2, b2 = fit_logistic((lg - m) / sd, ycal.astype(float))
        meta["llm_calibration"] = {"endpoint_model": llm_model, "mean": m, "std": sd, "a": a2, "b": b2,
                                   "auc": auc(ycal, pl)}
    return meta


def compile(
    spec: str | Path | dict,
    model: Model | None = None,
    calibration: list[dict] | None = None,
    *,
    allow_uncalibrated: bool = False,
    llm: str | None = None,
    llm_model: str = "default_model",
    transfer: list[dict] | None = None,
    _cache: dict | None = None,
    _selection: dict | None = None,
) -> Decision:
    """Compile a spec into a handle.

    `calibration` is a list of {"span": {...}, "label": ...} rows from the target domain; it defaults to the
    file named by the spec's `calibration.data`. Calibration is mandatory (rankings transfer, cut-offs don't)
    unless `allow_uncalibrated`, which fits the cut-off on the contrasts themselves and marks the handle so.
    With `llm` (an OpenAI-compatible chat endpoint) a boolean decision's escalation answers are also
    Platt-calibrated on the same sample. `transfer` (or the spec's `transfer.data`) is an optional labelled set
    from a second domain, same row format; the report scores the readout on it, and `layer: auto` uses it.
    """
    spec, base = load_spec(spec)
    model = model or Model()
    d = _parse_spec(spec, model)
    layer = (spec.get("readout") or {}).get("layer", "final")
    if calibration is None:
        calibration = _spec_data(spec.get("calibration"), base)
    if transfer is None:
        transfer = _spec_data(spec.get("transfer"), base)
    if layer == "auto":
        return _compile_layer_auto(spec, model, calibration, transfer, allow_uncalibrated=allow_uncalibrated,
                                   llm=llm, llm_model=llm_model, cache=_cache if _cache is not None else {})
    method, check, rule = _fit_options(spec, d, layer, model)
    E = _Encoder(model, layer, _cache)
    t, n_contrasts, n_hard = _training(spec, d, E)

    cal_rows = _calibration_rows(spec, d, calibration, allow_uncalibrated)
    calibrated = bool(cal_rows)
    if calibrated:
        Xcal = E([text for text, _, _ in cal_rows])
        ycal = np.array([y for _, y, _ in cal_rows])
    else:
        Xcal, ycal = t.X, t.y
    if d.kind == "boolean" and len(set(ycal.tolist())) < 2:
        raise SpecError(f"{d.name}: calibration labels are all one class")
    if d.kind == "choice" and (missing := [lab for j, lab in enumerate(d.labels) if not (ycal == j).any()]):
        raise SpecError(f"{d.name}: no calibration rows are labelled {', '.join(map(repr, missing))}; the multinomial "
                        "calibration would make an option it never sees unreachable, so give it rows of every option")

    fits = _fit_methods(d, rule, t, Xcal, ycal, calibrated)
    chosen, reason = _choose_method(method, d.kind, fits, len(t.hard_X))
    _check_hard_negatives(d, check, chosen, fits, t)
    f = fits[chosen]

    readout = {"layer": layer, "method": chosen, "requested": method}
    report: dict[str, Any] = {"contrasts": n_contrasts, "hard_negatives": n_hard, "calibration_n": len(cal_rows),
                              "layer": layer, "method": chosen, "method_reason": reason, **f["report"],
                              "methods": {m: fits[m]["report"] for m in METHODS[d.kind]}}
    report["contrast_cv"] = contrast_cv(d.kind, chosen, t.X, np.asarray(t.y), t.cv_groups, t.cv_pairs, len(d.labels))
    _check_monotone(d, report["contrast_cv"])
    if transfer:
        report["transfer"] = _transfer_report(d, transfer, E, f)
    if _selection:
        readout["layer_requested"] = "auto"
        report |= _selection
    cal = f["cal"] | {"calibrated": calibrated, "n": len(cal_rows) if calibrated else len(t.y)}

    tensors = {"directions": f["W"]}
    abstain = None
    if spec.get("abstain"):
        abstain, abstain_tensors = _abstain(spec, d, t, Xcal, cal_rows)
        tensors |= abstain_tensors
    escalation = _escalation(spec, d, rule, cal_rows, ycal, llm, llm_model) if spec.get("escalate") else None
    meta = {
        "format": HANDLE_FORMAT,
        "decision": d.name,
        "kind": d.kind,
        "labels": d.labels,
        "unit": d.unit,
        "applies_when": d.gate,
        "encoder": {"model": d.encoder, "revision": model.revision},
        "readout": readout,
        "standardize": {"mean": f["mean"].tolist(), "std": f["std"].tolist()},
        "calibration": cal,
        "abstain": abstain,
        "escalate": escalation,
        "siblings": list(spec.get("siblings") or []),
        "report": report,
    }
    meta = _json_safe(meta)
    data = safetensors_bytes(tensors, {"handle": json.dumps(meta, allow_nan=False)})
    return Decision(data=data, metadata=meta, handle=Handle.from_bytes(data))


LAYER_CANDIDATES = ["final", 4, 8, 12, 16, 20, 23]


def _primary(kind: str, r: dict, use_transfer: bool) -> tuple[float, float] | None:
    """(score, standard error) a layer is ranked by: transfer when given, else grouped CV on the contrasts."""
    src = r.get("transfer") if use_transfer else r.get("contrast_cv")
    key = {"boolean": "auc", "choice": "accuracy", "score": "corr", "levels": "spearman"}[kind]
    if not src or src.get(key) is None:
        return None
    lo, hi = _nan_for_none(src[f"{key}_ci"])
    return src[key], (hi - lo) / 3.92


def _calibration_interval(kind: str, r: dict) -> list[float]:
    key = {"boolean": "calibration_auc_ci", "choice": "calibration_accuracy_ci", "score": "calibration_corr_ci",
           "levels": "calibration_spearman_ci"}[kind]
    return _nan_for_none(r.get(key) or [None, None])


def _nan_for_none(xs: list) -> list[float]:
    return [float("nan") if x is None else x for x in xs]


def _compile_layer_auto(spec, model, calibration, transfer, *, allow_uncalibrated, llm, llm_model, cache) -> Decision:
    """`readout.layer: auto`. Compile each candidate layer (`readout.layers`, default final and every fourth layer)
    on a shared embedding cache, then pick by a fixed rule:

    1. drop layers whose calibration interval lies wholly below the best layer's (the 60-item sample can only
       rule a layer out, never choose between close ones);
    2. rank the rest by the transfer score when a transfer set is given, else by grouped cross-validation on
       the contrasts;
    3. keep the final embedding if it is within one standard error of the best; else take the best.
    """
    kind = spec.get("kind", "boolean")
    readout = dict(spec.get("readout") or {})
    cands = list(readout.get("layers") or LAYER_CANDIDATES)
    for c in cands:
        if c != "final" and not (isinstance(c, int) and 0 <= c < model.num_layers):
            raise SpecError(f"readout.layers: {c!r} is not 'final' or 0..{model.num_layers - 1}")
    use_transfer = bool(transfer)
    rows, reports = {}, {}
    for c in cands:
        trial = dict(spec, readout=dict(readout, layer=c), abstain=None, hard_negative_check="off")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = compile(trial, model, calibration, allow_uncalibrated=allow_uncalibrated, transfer=transfer,
                        _cache=cache).report
        reports[c] = r
        prim = _primary(kind, r, use_transfer)
        rows[str(c)] = {"method": r["method"], "score": None if prim is None else prim[0],
                        "score_se": None if prim is None else prim[1],
                        "calibration_ci": _calibration_interval(kind, r),
                        "contrast_cv": r.get("contrast_cv"), "transfer": r.get("transfer"),
                        "hard_negatives_violated": r.get("hard_negatives_violated")}
    scored = [c for c in cands if rows[str(c)]["score"] is not None]
    if not scored:
        raise SpecError("layer: auto needs at least two contrast groups (or a transfer set) to score layers")
    cal_lo = {c: rows[str(c)]["calibration_ci"][0] for c in scored}
    cal_hi = {c: rows[str(c)]["calibration_ci"][1] for c in scored}
    best_cal = max(scored, key=lambda c: (cal_lo[c] + cal_hi[c]) / 2)
    alive = [c for c in scored if not cal_hi[c] < cal_lo[best_cal]]
    dropped = [c for c in scored if c not in alive]
    best = max(alive, key=lambda c: rows[str(c)]["score"])
    b = rows[str(best)]
    basis = "transfer set" if use_transfer else "grouped cross-validation on the contrasts"
    if "final" in alive and best != "final" and rows["final"]["score"] >= b["score"] - b["score_se"]:
        chosen = "final"
        reason = (f"auto: final is within one standard error of the best layer ({best}: {b['score']:.3f} vs "
                  f"{rows['final']['score']:.3f}) by {basis}")
    else:
        chosen = best
        reason = f"auto: layer {best} scores best by {basis} ({b['score']:.3f})"
    if dropped:
        reason += f"; calibration ruled out {', '.join(map(str, dropped))}"
    selection = {"layer_reason": reason, "layer_basis": basis, "layers": rows,
                 "layer_selection_note": "the chosen layer's score is selected among candidates, so it is optimistic"}
    final = dict(spec, readout=dict(readout, layer=chosen))
    return compile(final, model, calibration, allow_uncalibrated=allow_uncalibrated, llm=llm, llm_model=llm_model,
                   transfer=transfer, _cache=cache, _selection=selection)


def metadata(handle: Handle) -> dict:
    """The metadata `compile` stored in a handle (kind, labels, readout, calibration, report, ...) as a dict."""
    return json.loads(handle.metadata_json)


# ---------------------------------------------------------------- escalation


def _top_logprobs(endpoint: str, body: dict, timeout: float) -> list[dict]:
    """POST `body` to an OpenAI-compatible chat endpoint; the top logprobs of the first generated token."""
    import urllib.request

    req = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


def ask_yes_no(endpoint: str, prompt: str, *, model: str = "default_model", timeout: float = 300, extra: dict | None = None) -> float:
    """P(yes) from one token of an OpenAI-compatible chat endpoint with logprobs. `extra` is merged into the
    request body, for server-specific options (e.g. {"chat_template_kwargs": {"enable_thinking": False}})."""
    top = _top_logprobs(endpoint, {
        "model": model, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
        "messages": [{"role": "user", "content": prompt + "\n\nAnswer yes or no."}],
    } | (extra or {}), timeout)
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else 0.5


def ask_choice(
    endpoint: str, prompt: str, labels: list[str], *, model: str = "default_model", timeout: float = 300, extra: dict | None = None,
) -> dict[str, float]:
    """Per-option probabilities from one token: options are lettered A, B, ... in the prompt. `extra` as in ask_yes_no."""
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    opts = "\n".join(f"{l}. {lab}" for l, lab in zip(letters, labels))
    top = _top_logprobs(endpoint, {
        "model": model, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{opts}\n\nAnswer with the letter only."}],
    } | (extra or {}), timeout)
    mass = {lab: sum(math.exp(t["logprob"]) for t in top if t["token"].strip().upper() == l) for l, lab in zip(letters, labels)}
    z = sum(mass.values())
    return {lab: (v / z if z else 1 / len(labels)) for lab, v in mass.items()}


def escalate(verdict: dict, handle: Handle, endpoint: str, *, model: str | None = None) -> dict:
    """Resolve an `escalate` verdict with the escalation model; returns the verdict updated with its answer.

    Boolean answers are Platt-calibrated when the handle was compiled with `llm=`; choice answers are the
    model's letter probabilities; levels answers are those probabilities per level. Score verdicts can read
    "escalate", but this resolves only boolean, choice and levels decisions: a score handle raises ValueError.
    """
    if verdict.get("status") != "escalate":
        return verdict
    if not verdict.get("prompt"):
        raise ValueError("an escalate verdict without a prompt: the handle has no escalation prompt")
    meta = metadata(handle)
    llm_cal = (meta.get("escalate") or {}).get("llm_calibration")
    model = model or (llm_cal or {}).get("endpoint_model", "default_model")
    out = dict(verdict)
    if meta["kind"] == "boolean":
        p = ask_yes_no(endpoint, verdict["prompt"], model=model)
        if llm_cal:
            lg = math.log(min(max(p, 1e-6), 1 - 1e-6) / min(max(1 - p, 1e-6), 1))
            p = 1 / (1 + math.exp(-(llm_cal["a"] * (lg - llm_cal["mean"]) / llm_cal["std"] + llm_cal["b"])))
        out |= {"status": "escalated", "p": p, "answer": "yes" if p >= 0.5 else "no", "p_encoder": verdict["p"]}
    elif meta["kind"] == "choice":
        p = ask_choice(endpoint, verdict["prompt"], meta["labels"], model=model)
        out |= {"status": "escalated", "p": p, "answer": max(p, key=p.get), "p_encoder": verdict["p"]}
    elif meta["kind"] == "levels":
        p = ask_choice(endpoint, verdict["prompt"], meta["labels"], model=model)
        probs = {str(i): p[lab] for i, lab in enumerate(meta["labels"])}
        best = max(probs, key=probs.get)
        out |= {"status": "escalated", "p": probs, "answer": meta["labels"][int(best)], "confidence": probs[best],
                "score": sum(int(i) * q for i, q in probs.items()), "p_encoder": verdict["p"]}
    else:
        raise ValueError("score decisions do not escalate")
    return out


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eg2cpu-decisions", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile a spec into a handle file")
    c.add_argument("spec")
    c.add_argument("-o", "--out", help="output path (default: <decision>.eg2h next to the spec)")
    c.add_argument("--calibration", help="jsonl of {span, label} rows (default: the spec's calibration.data)")
    c.add_argument("--allow-uncalibrated", action="store_true")
    c.add_argument("--llm", help="OpenAI-compatible chat endpoint to calibrate escalation answers on the sample")
    c.add_argument("--llm-model", default="default_model")
    a = sub.add_parser("apply", help="apply handles to spans (jsonl of span dicts); prints verdicts as jsonl")
    a.add_argument("handles", nargs="+")
    a.add_argument("--spans", required=True)
    args = ap.parse_args(argv)

    model = Model()
    if args.cmd == "compile":
        cal = None
        if args.calibration:
            cal = _read_jsonl(args.calibration)
        d = compile(args.spec, model, cal, allow_uncalibrated=args.allow_uncalibrated, llm=args.llm, llm_model=args.llm_model)
        out = Path(args.out) if args.out else Path(args.spec).with_name(f"{d.metadata['decision']}.eg2h")
        d.save(out)
        print(json.dumps({"handle": str(out), **d.report}, indent=2))
    else:
        handles = [Handle(h) for h in args.handles]
        spans = [span(row) for row in _read_jsonl(args.spans)]
        for row in model.apply(handles, spans):
            print(json.dumps(row))


if __name__ == "__main__":
    main(sys.argv[1:])

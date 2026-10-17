"""Decision handles: Rust `Model.apply` against a numpy re-implementation for every kind, plus the compiler's checks,
`layer: auto` and the ordinal fit. Needs the EmbeddingGemma 2 checkpoint.

    uv run --package eg2cpu pytest -q eg2cpu/tests    # from the repo root
"""

import json
import random
import struct

import numpy as np
import pytest

import eg2cpu
from eg2cpu import decisions

PKGS = ["parser", "checker", "vm", "compiler", "builtin", "optimizer", "conf", "ast", "lexer", "codegen"]


@pytest.fixture(scope="module")
def model():
    return eg2cpu.Model()


def go_pair(rng, p):
    t = rng.choice(["TestParse", "TestCheck", "TestRun"])
    call = f"bash: go test ./{p}/..."
    return call, f"--- FAIL: {t} (0.01s)\nFAIL\tgithub.com/acme/app/{p}", f"ok  \tgithub.com/acme/app/{p}\t0.{rng.randint(1, 9)}s"


CMDS = [
    ("npm install", "npm ERR! code ERESOLVE", "added 214 packages in 6s"),
    ("cargo build", "error[E0308]: mismatched types", "Finished dev [unoptimized] target(s) in 2.31s"),
    ("ls config/", "ls: config/: No such file or directory", "app.yaml  db.yaml"),
    ("make build", "make: *** [build] Error 2", "go build -o bin/app ./cmd/app"),
    ("pytest -q", "1 failed, 12 passed in 0.8s", "13 passed in 0.7s"),
    ("cat README.md", "cat: README.md: No such file or directory", "# Project\nA small app."),
]


def boolean_spec():
    rng = random.Random(0)
    contrasts = []
    for p in PKGS[:6]:
        call, bad, good = go_pair(rng, p)
        contrasts.append({"call": call, "yes": bad, "no": good})
    contrasts += [{"call": f"bash: {c}", "yes": b, "no": g} for c, b, g in CMDS[:4]]
    return {
        "decision": "tool_call_failed",
        "kind": "boolean",
        "unit": "tool_result",
        "applies_when": 'span.kind == "tool_result"',
        "encoder": "google/embeddinggemma-2@<rev>",
        "readout": {"layer": "final", "method": "diff_of_means"},
        "contrasts": contrasts,
        "hard_negatives": ["Speed test failed: No Connection."],
        "abstain": {"mahalanobis_quantile": 0.99},
        "escalate": {"when": "|p - 0.5| < 0.2", "to": "llm", "prompt": "Did the tool call itself fail?"},
    }


def boolean_calibration():
    rng = random.Random(1)
    rows = []
    for p in PKGS[6:]:
        call, bad, good = go_pair(rng, p)
        rows += [{"span": {"kind": "tool_result", "call": call, "output": bad}, "label": "yes"},
                 {"span": {"kind": "tool_result", "call": call, "output": good}, "label": "no"}]
    for c, b, g in CMDS[4:]:
        rows += [{"span": {"kind": "tool_result", "call": f"bash: {c}", "output": b}, "label": True},
                 {"span": {"kind": "tool_result", "call": f"bash: {c}", "output": g}, "label": False}]
    return rows


OUTCOMES = {
    "passed": "ok  \tgithub.com/acme/app/{p}\t0.4s",
    "test_failure": "--- FAIL: TestRun (0.02s)\n    {p}_test.go:41: got 3, want 4\nFAIL",
    "build_error": "# github.com/acme/app/{p}\n./{p}.go:12:5: undefined: foo\nFAIL\tgithub.com/acme/app/{p} [build failed]",
}


def choice_spec():
    # The contrasts are 30 near-identical templates, so this handle's abstain check rightly treats differently
    # phrased outputs as unfamiliar; tests read abstention from the boolean handle.
    return {
        "decision": "go_test_outcome",
        "kind": "choice",
        "options": list(OUTCOMES),
        "unit": "tool_result",
        "readout": {"layer": "final", "method": "lda_shrinkage"},
        "contrasts": [{"call": f"bash: go test ./{p}/...", **{k: v.format(p=p) for k, v in OUTCOMES.items()}} for p in PKGS[:6]],
        "abstain": {"mahalanobis_quantile": 0.95},
        "escalate": {"when": "p_max < 0.6", "to": "llm", "prompt": "Unit:\n{unit}\nWhat was the outcome?"},
    }


def choice_calibration():
    return [{"span": {"call": f"bash: go test ./{p}/...", "output": v.format(p=p)}, "label": k}
            for p in PKGS[6:] for k, v in OUTCOMES.items()]


def score_spec():
    def levels(p):
        return [{"output": f"{n} of 10 tests passed in ./{p}", "value": n / 10} for n in (0, 3, 6, 10)]

    return {
        "decision": "pass_fraction",
        "kind": "score",
        "unit": "tool_result",
        "readout": {"layer": 11, "method": "ridge"},
        "contrasts": [{"call": f"bash: go test ./{p}/...", "levels": levels(p)} for p in PKGS[:6]],
    }


def score_calibration():
    return [{"span": {"call": f"bash: go test ./{p}/...", "output": f"{n} of 10 tests passed in ./{p}"}, "label": n / 10}
            for p in PKGS[6:] for n in (1, 5, 9)]


@pytest.fixture(scope="module")
def compiled(model):
    return {
        "boolean": decisions.compile(boolean_spec(), model, boolean_calibration()),
        "choice": decisions.compile(choice_spec(), model, choice_calibration()),
        "score": decisions.compile(score_spec(), model, score_calibration()),
    }


def read_tensors(data: bytes) -> dict[str, np.ndarray]:
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    out = {}
    for name, v in header.items():
        if name != "__metadata__":
            s, e = v["data_offsets"]
            out[name] = np.frombuffer(data[8 + n + s:8 + n + e], dtype="<f4").reshape(v["shape"]).astype(np.float64)
    return out


def reference(model, d: decisions.Decision, s: dict) -> dict:
    """The handle math, re-implemented in numpy from the file's tensors and metadata."""
    m, t = d.metadata, read_tensors(d.data)
    if m["applies_when"] and not decisions.gate_eval(m["applies_when"], s):
        return {"status": "not_applicable"}
    text = decisions.render_template(m["unit"]["template"], s)
    if text is None:
        return {"status": "not_applicable"}
    emb, means = model.embed(text)
    x = (emb if m["readout"]["layer"] == "final" else means[m["readout"]["layer"]]).astype(np.float64)
    z = (t["directions"] @ x - np.array(m["standardize"]["mean"])) / np.array(m["standardize"]["std"])
    logits = np.array(m["calibration"]["weights"]) @ z + np.array(m["calibration"]["bias"])
    out = {"z": z}
    if m["kind"] == "boolean":
        p = 1 / (1 + np.exp(-logits[0]))
        out |= {"p": p, "answer": "yes" if p >= 0.5 else "no"}
    elif m["kind"] == "choice":
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        out |= {"p": dict(zip(m["labels"], p)), "answer": m["labels"][int(np.argmax(p))]}
    else:
        out |= {"value": logits[0]}
    far = False
    if m["abstain"]:
        out["distance"] = float(np.linalg.norm(t["abstain_whiten"] @ (x - t["abstain_mean"])))
        far = out["distance"] > m["abstain"]["threshold"]
    unsure = False
    if m["escalate"]:
        r = m["escalate"]["rule"]
        if r["type"] == "band":
            unsure = abs((out["p"] if m["kind"] == "boolean" else out["value"]) - r["center"]) < r["width"]
        elif r["type"] == "p_max_below":
            unsure = max(out["p"].values()) < r["value"]
        else:
            ps = sorted(out["p"].values(), reverse=True)
            unsure = ps[0] - ps[1] < r["value"]
    out["status"] = "abstain" if far else "escalate" if unsure else "answer"
    return out


def reference_levels(model, d: decisions.Decision, s: dict) -> np.ndarray:
    """A levels handle's per-level probabilities, re-implemented in numpy like `reference`."""
    m, t = d.metadata, read_tensors(d.data)
    text = decisions.render_template(m["unit"]["template"], s)
    emb, means = model.embed(text)
    x = (emb if m["readout"]["layer"] == "final" else means[m["readout"]["layer"]]).astype(np.float64)
    z = (t["directions"][0] @ x - m["standardize"]["mean"][0]) / m["standardize"]["std"][0]
    return decisions.ordinal_probs(np.array([z]), m["calibration"]["weights"][0][0], np.array(m["calibration"]["bias"]))[0]


PROBES = [
    {"kind": "tool_result", "call": "bash: go test ./router/...", "output": "--- FAIL: TestRoute (0.00s)\nFAIL"},
    {"kind": "tool_result", "call": "bash: go test ./router/...", "output": "ok  \tgithub.com/acme/app/router\t0.2s"},
    {"kind": "tool_result", "call": "bash: go test ./router/...", "output": "7 of 10 tests passed in ./router"},
    {"kind": "tool_result", "call": "bash: go test ./db/...", "output": "./db.go:3:1: syntax error\nFAIL\tgithub.com/acme/app/db [build failed]"},
    {"kind": "tool_result", "call": "bash: npm test", "output": "Tests: 2 failed, 40 passed"},
    {"kind": "tool_result", "call": "read: poem.txt", "output": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
    {"kind": "message", "call": "bash: ls", "output": "a b c"},  # gated out for the boolean decision
    {"kind": "tool_result", "output": "no call field"},  # unit template needs {call}
]


@pytest.mark.parametrize("kind", ["boolean", "choice", "score"])
def test_rust_apply_matches_numpy(model, compiled, kind):
    d = compiled[kind]
    rows = model.apply([d.handle], PROBES)
    for s, [v] in zip(PROBES, rows):
        want = reference(model, d, s)
        assert v["status"] == want["status"], (s, v, want)
        if want["status"] == "not_applicable":
            continue
        np.testing.assert_allclose(v["z"], want["z"], atol=2e-4)
        if kind == "boolean":
            assert abs(v["p"] - want["p"]) < 1e-4 and v["answer"] == want["answer"]
        elif kind == "choice":
            assert v["answer"] == want["answer"]
            for lab, p in want["p"].items():
                assert abs(v["p"][lab] - p) < 1e-4
        else:
            assert abs(v["value"] - want["value"]) < 1e-4
        if "distance" in want:
            assert abs(v["distance"] - want["distance"]) < 1e-3 * max(1.0, want["distance"])
        if v["status"] == "escalate":
            unit = decisions.render_template(d.metadata["unit"]["template"], s)
            assert unit in v["prompt"]


def test_answers_make_sense(model, compiled):
    rows = model.apply([compiled["boolean"].handle, compiled["choice"].handle, compiled["score"].handle], PROBES[:4])
    assert [r[0]["answer"] for r in rows[:2]] == ["yes", "no"]
    assert rows[0][1]["answer"] == "test_failure" and rows[1][1]["answer"] == "passed" and rows[3][1]["answer"] == "build_error"
    lo, hi = (
        {"call": "bash: go test ./router/...", "output": f"{n} of 10 tests passed in ./router"} for n in (2, 8))
    [[vlo], [vhi]] = model.apply([compiled["score"].handle], [lo, hi])
    assert vlo["value"] < vhi["value"]
    assert compiled["boolean"].report["calibration_auc"] > 0.9
    assert compiled["choice"].report["calibration_accuracy"] > 0.9
    assert compiled["score"].report["calibration_corr"] > 0.8


def test_gate_abstain_and_in_domain(model, compiled):
    [[poem], [gated], [missing]] = model.apply([compiled["boolean"].handle], PROBES[5:])
    assert poem["status"] == "abstain"
    # In-domain spans (unseen packages, same kinds of output) must not abstain under the boolean handle.
    for [v] in model.apply([compiled["boolean"].handle], PROBES[:5]):
        assert v["status"] != "abstain", v
    assert gated["status"] == "not_applicable" and missing["status"] == "not_applicable"


def test_save_load_roundtrip(tmp_path, model, compiled):
    path = compiled["choice"].save(tmp_path / "c.eg2h")
    h = eg2cpu.Handle(str(path))
    assert (h.name, h.kind, h.labels) == ("go_test_outcome", "choice", list(OUTCOMES))
    assert model.apply([h], PROBES[:2]) == model.apply([compiled["choice"].handle], PROBES[:2])


def test_revision_mismatch(model, compiled):
    d = compiled["boolean"]
    m = dict(d.metadata, encoder={"model": "google/embeddinggemma-2", "revision": "0" * 40})
    data = decisions.safetensors_bytes(read_tensors(d.data), {"handle": json.dumps(m)})
    with pytest.raises(RuntimeError, match="revision"):
        model.apply([eg2cpu.Handle.from_bytes(data)], PROBES[:1])


def test_spec_errors(model):
    with pytest.raises(decisions.SpecError, match="mandatory"):
        decisions.compile(boolean_spec(), model)
    s = boolean_spec() | {"escalate": {"when": "p_max < 0.5", "prompt": "x"}}
    with pytest.raises(decisions.SpecError, match="choice"):
        decisions.compile(s, model, boolean_calibration())
    with pytest.raises(decisions.SpecError, match="pins encoder revision"):
        decisions.compile(boolean_spec() | {"encoder": "google/embeddinggemma-2@" + "1" * 40}, model, boolean_calibration())
    with pytest.raises(RuntimeError):
        decisions.compile(boolean_spec() | {"applies_when": "span.kind = 'x'"}, model, boolean_calibration())


def test_unquoted_yaml_yes_no(model, tmp_path):
    import yaml

    spec = boolean_spec()
    text = yaml.safe_dump(spec).replace("'yes':", "yes:").replace("'no':", "no:")
    assert "\n  yes:" in text and isinstance(yaml.safe_load(text)["contrasts"][0], dict)
    (tmp_path / "s.yaml").write_text(text)
    d = decisions.compile(tmp_path / "s.yaml", model, boolean_calibration())
    assert d.report["calibration_auc"] > 0.9


def test_report_compares_methods(compiled):
    for kind, d in compiled.items():
        methods = d.report["methods"]
        assert set(methods) == set(decisions.METHODS[kind])
        assert d.report["method"] == d.metadata["readout"]["method"] == d.metadata["readout"]["requested"]
        for r in methods.values():
            key = {"boolean": "calibration_auc_ci", "choice": "calibration_accuracy_ci", "score": "calibration_corr_ci"}[kind]
            lo, hi = r[key]
            assert lo <= hi
            assert "hard_negatives_violated" in r
    assert "calibration_escalate" in compiled["boolean"].report


def test_auto_follows_its_rule(model):
    d = decisions.compile(boolean_spec() | {"readout": {"layer": "final", "method": "auto"}}, model, boolean_calibration())
    m = d.report["methods"]
    # The documented rule: the default unless it misreads a hard negative and another method doesn't; when every
    # method misreads one, the one that misreads the fewest (the default on a tie).
    order = list(decisions.METHODS["boolean"])
    ok = [k for k in order if not m[k]["hard_negatives_violated"]]
    want = ok[0] if ok else min(order, key=lambda k: m[k]["hard_negatives_violated"])
    assert d.report["method"] == want
    assert d.metadata["readout"] == {"layer": "final", "method": d.report["method"], "requested": "auto"}
    assert d.report["method_reason"].startswith("auto:")


def test_hard_negative_check(model):
    # A "hard negative" that is really a failure: every method reads it "yes".
    s = boolean_spec() | {"hard_negatives": ["--- FAIL: TestRoute (0.00s)\nFAIL\tgithub.com/acme/app/router"]}
    with pytest.raises(decisions.SpecError, match="hard negatives read wrong"):
        decisions.compile(s | {"hard_negative_check": "error"}, model, boolean_calibration())
    with pytest.warns(UserWarning, match="hard negatives read wrong"):
        d = decisions.compile(s, model, boolean_calibration())
    assert d.report["hard_negatives_read_yes"] == 1
    import warnings as w

    with w.catch_warnings():
        w.simplefilter("error")
        decisions.compile(s | {"hard_negative_check": "off"}, model, boolean_calibration())
    with pytest.raises(decisions.SpecError, match="hard_negative_check"):
        decisions.compile(s | {"hard_negative_check": "strict"}, model, boolean_calibration())


def test_hard_negatives_are_checked_held_out(model):
    # A copy of a "yes" contrast listed as a hard negative: in-sample LDA can carve it onto the "no" side,
    # but held out it reads "yes", which is what it is.
    c0 = boolean_spec()["contrasts"][0]
    s = boolean_spec() | {"hard_negatives": [{"call": c0["call"], "output": c0["yes"]}], "hard_negative_check": "off"}
    r = decisions.compile(s, model, boolean_calibration()).report["methods"]["lda_shrinkage"]
    assert r["hard_negatives_violated_in_sample"] == 0 and r["hard_negatives_violated"] == 1


class CountingModel:
    """Delegates to a Model and counts the texts it encodes."""

    def __init__(self, model):
        self.model, self.encoded = model, []

    def embed_batch(self, texts):
        self.encoded += list(texts)
        return self.model.embed_batch(texts)

    def __getattr__(self, name):
        return getattr(self.model, name)


def test_layer_auto(model):
    counting = CountingModel(model)
    s = boolean_spec() | {"readout": {"layer": "auto", "method": "auto"}}
    d = decisions.compile(s, counting, boolean_calibration())
    r = d.report
    assert set(r["layers"]) == {str(c) for c in decisions.LAYER_CANDIDATES}
    assert r["layer"] in decisions.LAYER_CANDIDATES
    assert d.metadata["readout"]["layer"] == r["layer"] and d.metadata["readout"]["layer_requested"] == "auto"
    assert r["layer_reason"].startswith("auto:") and r["layer_basis"].startswith("grouped cross-validation")
    for row in r["layers"].values():
        assert row["score"] is not None and row["contrast_cv"]["auc_ci"][0] <= row["contrast_cv"]["auc"]
    # every distinct text is encoded once, however many candidate layers were compiled
    assert len(counting.encoded) == len(set(counting.encoded))
    # the handle applies at the chosen layer
    [[v]] = model.apply([d.handle], PROBES[:1])
    assert v["status"] in ("answer", "escalate")


def test_layer_auto_with_transfer_set(model):
    transfer = [{"span": {"kind": "tool_result", "call": f"bash: {c}", "output": o}, "label": lab}
                for c, b, g in CMDS for o, lab in ((b, "yes"), (g, "no"))]
    s = boolean_spec() | {"readout": {"layer": "auto", "layers": ["final", 8], "method": "diff_of_means"}}
    d = decisions.compile(s, model, boolean_calibration(), transfer=transfer)
    assert d.report["layer_basis"] == "transfer set"
    assert set(d.report["layers"]) == {"final", "8"}
    assert d.report["transfer"]["n"] == len(transfer)


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_layer_auto_other_kinds(model, kind):
    spec, cal = (choice_spec(), choice_calibration()) if kind == "choice" else (score_spec(), score_calibration())
    spec = spec | {"readout": dict(spec["readout"], layer="auto", layers=["final", 8, 16])}
    d = decisions.compile(spec, model, cal)
    assert d.report["layer"] in ("final", 8, 16)
    key = "accuracy" if kind == "choice" else "corr"
    assert all(key in row["contrast_cv"] for row in d.report["layers"].values())


def test_auc_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)  # coarse scores: many ties, within and across classes
    pos, neg = s[y == 1], s[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert decisions.auc(y, s) == pytest.approx(pairwise, abs=1e-12)
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return
    assert decisions.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_ordinal_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    a, theta = 1.7, np.array([-1.5, -0.2, 0.9, 2.0])
    P = decisions.ordinal_probs(z, a, theta)
    y = np.array([rng.choice(5, p=p) for p in P])
    a_hat, th_hat = decisions.fit_ordinal(z, y, 5, prior=0.0)
    assert abs(a_hat - a) < 0.15 and np.abs(th_hat - theta).max() < 0.2
    assert decisions.qwk(y, y, 5) == 1.0
    assert decisions.spearman(np.arange(10.0), np.arange(10.0) ** 3) == pytest.approx(1.0)


PASS_LEVELS = ["none", "few", "most", "all"]


def pass_text(p, n):
    return f"{n} of 10 tests passed in ./{p}"


def levels_spec(order=PASS_LEVELS):
    counts = dict(zip(PASS_LEVELS, (0, 3, 7, 10)))
    return {
        "decision": "tests_passing",
        "kind": "levels",
        "levels": list(order),
        "unit": "tool_result",
        "readout": {"layer": 11, "method": "ridge"},
        "contrasts": [{"call": f"bash: go test ./{p}/...", **{lv: pass_text(p, counts[lv]) for lv in order}} for p in PKGS[:6]],
        "escalate": {"when": "confidence < 0.5", "to": "llm", "prompt": "How many tests passed?"},
    }


def levels_calibration(order=PASS_LEVELS):
    counts = {"none": (0, 1), "few": (2, 3, 4), "most": (6, 7, 8), "all": (10,)}
    return [{"span": {"call": f"bash: go test ./{p}/...", "output": pass_text(p, n)}, "label": lv}
            for p in PKGS[6:] for lv in order for n in counts[lv]]


def test_levels_decision(model):
    d = decisions.compile(levels_spec(), model, levels_calibration())
    r = d.report
    assert r["contrast_cv"]["monotone"] and r["contrast_cv"]["order_accuracy"] > 0.8
    assert r["calibration_spearman"] > 0.8 and r["calibration_within_one"] > 0.9
    theta = np.array(d.metadata["calibration"]["bias"])
    assert len(theta) == 3 and np.all(np.diff(theta) > 0)
    spans = [{"call": "bash: go test ./router/...", "output": pass_text("router", n)} for n in (0, 2, 5, 8, 10)]
    rows = model.apply([d.handle], spans)
    for s, [v] in zip(spans, rows):
        # a levels answer: per-level probabilities, expected level, confidence, legend
        assert set(v["p"]) == {"0", "1", "2", "3"} and v["legend"]["3"] == "all"
        assert abs(sum(v["p"].values()) - 1) < 1e-4
        assert v["score"] == pytest.approx(sum(int(k) * q for k, q in v["p"].items()), abs=1e-4)
        assert v["confidence"] == pytest.approx(max(v["p"].values()))
        assert v["answer"] == PASS_LEVELS[int(max(v["p"], key=v["p"].get))]
        # numpy reference
        want = reference_levels(model, d, s)
        np.testing.assert_allclose([v["p"][str(i)] for i in range(4)], want, atol=1e-4)
        assert (v["status"] == "escalate") == (max(want) < 0.5)
    scores = [v["score"] for [v] in rows]
    assert scores[0] < scores[2] < scores[4]


def test_levels_without_a_consistent_axis_warn(model):
    # Each contrast set maps pass counts to levels by a different permutation, so no single direction orders the
    # levels the same way in every set: held-out level means come out of order. (A consistent relabelling, e.g.
    # all < none < most < few everywhere, is still one axis, and a 768-d readout can learn it.)
    rng = random.Random(3)
    counts = [0, 3, 7, 10]
    contrasts = []
    for p in PKGS[:8]:
        perm = rng.sample(counts, 4)
        contrasts.append({"call": f"bash: go test ./{p}/...", **{lv: pass_text(p, n) for lv, n in zip(PASS_LEVELS, perm)}})
    spec = levels_spec() | {"contrasts": contrasts}
    cal = [{"span": {"call": f"bash: go test ./{p}/...", "output": pass_text(p, n)}, "label": lv}
           for p in PKGS[8:] for lv, n in zip(PASS_LEVELS, rng.sample(counts, 4))]
    with pytest.warns(UserWarning, match="not in order"):
        d = decisions.compile(spec, model, cal)
    assert not d.report["contrast_cv"]["monotone"]
    with pytest.raises(decisions.SpecError, match="not in order"):
        decisions.compile(spec | {"monotonicity_check": "error"}, model, cal)


def quiet_boolean_spec():
    """The boolean spec without the hard negative warning the default spec's diff of means raises."""
    return boolean_spec() | {"hard_negative_check": "off"}


def test_uncomputable_metrics_are_null(model):
    # A one-class transfer set has no AUC. JSON has no NaN, so the handle stores null and still loads.
    transfer = [{"span": {"kind": "tool_result", "call": f"bash: {c}", "output": b}, "label": "yes"} for c, b, _ in CMDS]
    d = decisions.compile(quiet_boolean_spec(), model, boolean_calibration(), transfer=transfer)
    assert d.report["transfer"]["auc"] is None and d.report["transfer"]["auc_ci"] == [None, None]
    assert json.loads(d.handle.metadata_json, parse_constant=lambda c: pytest.fail(f"bare {c} in the handle"))
    assert decisions.metadata(d.handle)["report"]["transfer"]["n"] == len(transfer)


def test_spec_data_resolves_next_to_the_spec(model, tmp_path, monkeypatch):
    import yaml

    (tmp_path / "spec").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "spec" / "cal.jsonl").write_text("\n".join(json.dumps(r) for r in boolean_calibration()))
    monkeypatch.chdir(tmp_path / "elsewhere")
    for readout in ({"layer": "final"}, {"layer": "auto", "layers": ["final", 8]}):
        spec = quiet_boolean_spec() | {"readout": readout, "calibration": {"data": "cal.jsonl"}}
        (tmp_path / "spec" / "s.yaml").write_text(yaml.safe_dump(spec))
        d = decisions.compile(tmp_path / "spec" / "s.yaml", model)
        assert d.report["calibration_n"] == len(boolean_calibration())


def test_choice_calibration_needs_every_option(model):
    cal = [r for r in choice_calibration() if r["label"] != "build_error"]
    with pytest.raises(decisions.SpecError, match="no calibration rows are labelled 'build_error'"):
        decisions.compile(choice_spec(), model, cal)


def test_level_names_convert_one_way(model, tmp_path):
    import yaml

    # The same levels named three ways must compile to the same handle: words, numbers (0, 3, 7, 10 as level names,
    # contrast keys and labels), and words with an unquoted YAML `no` (a boolean to YAML).
    def same_handle(d, names):
        assert d.metadata["labels"] == names
        np.testing.assert_allclose(read_tensors(d.data)["directions"], read_tensors(want.data)["directions"], atol=1e-6)
        np.testing.assert_allclose(d.metadata["calibration"]["bias"], want.metadata["calibration"]["bias"], atol=1e-6)
        lookup = dict(zip(PASS_LEVELS, names))
        assert [v["answer"] for [v] in model.apply([d.handle], spans)] == [lookup[a] for a in answers]

    spec = levels_spec() | {"escalate": None}
    want = decisions.compile(spec, model, levels_calibration())
    spans = [{"call": "bash: go test ./router/...", "output": pass_text("router", n)} for n in (0, 3, 7, 10)]
    answers = [v["answer"] for [v] in model.apply([want.handle], spans)]

    def renamed(names):
        to = dict(zip(PASS_LEVELS, names))
        return (spec | {"levels": names, "contrasts": [{k if k not in to else to[k]: v for k, v in c.items()}
                                                       for c in spec["contrasts"]]},
                [r | {"label": to[r["label"]]} for r in levels_calibration()])

    s, cal = renamed([0, 3, 7, 10])
    same_handle(decisions.compile(s, model, cal), ["0", "3", "7", "10"])
    with pytest.raises(decisions.SpecError, match="is not one of"):
        decisions.compile(s, model, cal + [cal[0] | {"label": 1}])  # an index is not a level name
    with pytest.raises(decisions.SpecError, match="distinct"):
        decisions.compile(s | {"levels": [0, "0", 7, 10]}, model, cal)
    s, cal = renamed(["no", "few", "most", "all"])
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(s).replace("- 'no'", "- no").replace("'no':", "no:"))
    assert yaml.safe_load((tmp_path / "s.yaml").read_text())["levels"][0] is False
    same_handle(decisions.compile(tmp_path / "s.yaml", model, cal), ["no", "few", "most", "all"])


def test_gate_compares_fields_and_booleans(model, compiled):
    assert decisions.gate_eval("span.a == span.b", {"a": "x", "b": "x"})
    assert not decisions.gate_eval("span.a == span.b", {"a": "x", "b": "span.b"})
    assert decisions.gate_eval('span.is_error == "true"', {"is_error": "True"})
    d = decisions.compile(quiet_boolean_spec() | {"applies_when": "span.kind == tool_result and span.is_error == true"},
                          model, [r | {"span": r["span"] | {"is_error": True}} for r in boolean_calibration()])
    [[v]] = model.apply([d.handle], [PROBES[0] | {"is_error": str(True)}])
    assert v["status"] != "not_applicable"


def test_malformed_handle_errors(compiled):
    with pytest.raises(RuntimeError):
        eg2cpu.Handle.from_bytes(b"\xff" * 16)
    d = compiled["boolean"]
    m = dict(d.metadata, standardize={"mean": [0.0], "std": [0.0]})
    with pytest.raises(RuntimeError, match="std must be positive"):
        eg2cpu.Handle.from_bytes(decisions.safetensors_bytes(read_tensors(d.data), {"handle": json.dumps(m)}))


def test_escalation_prompt_fields(model, compiled):
    esc = quiet_boolean_spec()["escalate"]
    with pytest.raises(decisions.SpecError, match=r"\{tool\}"):
        decisions.compile(quiet_boolean_spec() | {"escalate": esc | {"prompt": "Tool {tool}: did it fail?"}}, model,
                          boolean_calibration())
    # A prompt with placeholders but no {unit} still shows the escalation model the unit first.
    s = quiet_boolean_spec() | {"escalate": esc | {"prompt": "The call was {call}. Did it fail?"}}
    d = decisions.compile(s, model, boolean_calibration())
    assert d.metadata["escalate"]["prompt"] == "{unit}\n\nThe call was {call}. Did it fail?"
    with pytest.raises(ValueError, match="without a prompt"):
        decisions.escalate({"status": "escalate", "p": 0.5, "prompt": None}, compiled["boolean"].handle, "http://unused")


def test_uncalibrated_hard_negative_check_is_held_out(model, monkeypatch):
    # Uncalibrated, the calibration rows are the training rows; the held-out check must calibrate without the hard
    # negative it reads, as it fits the direction without it.
    sizes, calibrate = [], decisions._calibrate
    monkeypatch.setattr(decisions, "_calibrate", lambda d, rule, t, Xcal, *a: sizes.append((len(t.X), len(Xcal)))
                        or calibrate(d, rule, t, Xcal, *a))
    decisions.compile(quiet_boolean_spec(), model, allow_uncalibrated=True)
    n = sizes[0][0]
    assert {s for _, s in sizes} == {n, n - 1}


def test_score_hard_negatives_anchor_the_axis(model, compiled):
    hard = [{"call": "bash: go test ./router/...", "output": "no tests to run in ./router", "value": 0.0}]
    d = decisions.compile(score_spec() | {"hard_negatives": hard}, model, score_calibration())
    assert not np.allclose(read_tensors(d.data)["directions"], read_tensors(compiled["score"].data)["directions"])
    assert d.report["hard_negatives"] == 1 and d.report["hard_negatives_violated"] is None

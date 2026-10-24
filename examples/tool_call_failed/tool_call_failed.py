"""Compile decisions/tool_call_failed.yaml and check it on real traces.

Calibrate on 60 labelled pi coding-agent tool results (the harness's isError flag), test on the other 580, and
report AUC, calibrated accuracy, how much the uncertainty band escalates and how accurate the rest is, and how
often it abstains in-domain versus on tau2 customer-support tool outputs (a different domain). Ablations: the
layer-4 readout the spec warns about, and diff of means. Finally, the cost of applying several handles at once.

    uv run tool_call_failed/tool_call_failed.py [out_dir]
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

import eg2cpu
from eg2cpu import decisions

import hunch_examples as data
from hunch.proxy import state_text
from hunch_examples import DECISIONS, OUT

SPEC = DECISIONS / "tool_call_failed.yaml"
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT


def as_row(tool, s, y):
    call, _, output = s.partition("\n")
    return {"span": {"kind": "tool_result", "tool": tool, "call": call, "output": output}, "label": bool(y)}


rows = [as_row(*r) for r in data.pi_real({"bash": 150, "edit": 100, "read": 70}, random.Random(11))]
random.Random(5).shuffle(rows)
cal, test = rows[:60], rows[60:]
y = np.array([r["label"] for r in test])
print(f"pi-real: {len(cal)} calibration + {len(test)} test ({y.sum()} failures in test)")

# Out of domain: tau2 customer-support tool outputs (its "Error:" prefix stripped, so the
# output has to say it failed).
ood = []
for f in data.tau2_files()[:4]:
    for line in open(f):
        for m in json.loads(line)["messages"]:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                body = re.sub(r"^Error:?\s*", "", m["content"]).strip()
                if body:
                    ood.append({"kind": "tool_result", "call": "tool: support_api", "output": data.clip(body, 600)})
ood = random.Random(7).sample(ood, min(200, len(ood)))

model = eg2cpu.Model()
print(f"encoder revision {model.revision}, isa {eg2cpu.isa()}")


def evaluate(name, d):
    t0 = time.perf_counter()
    [vs] = zip(*model.apply([d.handle], [r["span"] for r in test]))
    dt = time.perf_counter() - t0
    p = np.array([v["p"] for v in vs])
    st = np.array([v["status"] for v in vs])
    ans = st == "answer"
    res = {
        "auc": decisions.auc(y.astype(int), p),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "escalate": float(np.mean(st == "escalate")),
        "abstain_in_domain": float(np.mean(st == "abstain")),
        "accuracy_answered": float(np.mean((p[ans] >= 0.5) == y[ans])) if ans.any() else float("nan"),
        "ms_per_span": dt / len(test) * 1000,
    }
    if d.metadata["abstain"]:
        [ov] = zip(*model.apply([d.handle], ood))
        res["abstain_tau2"] = float(np.mean([v["status"] == "abstain" for v in ov]))
    print(f"\n== {name}")
    print("  compile report: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in d.report.items()}))
    print("  test:           " + json.dumps({k: round(v, 3) for k, v in res.items()}))
    return res, vs


spec, _ = decisions.load_spec(SPEC)
t0 = time.perf_counter()
main = decisions.compile(SPEC, model, cal)
print(f"compiled in {time.perf_counter() - t0:.1f}s")
path = main.save(out_dir / "tool_call_failed.eg2h")
print(f"handle: {path} ({path.stat().st_size / 1e6:.1f} MB)")
res, vs = evaluate(f"tool_call_failed (spec: final layer, method auto -> {main.report['method']})", main)
print(f"  method reason: {main.report['method_reason']}")

esc = next((v for v in vs if v["status"] == "escalate"), None)
if esc:
    print("\n  example escalation prompt:\n    " + esc["prompt"][:400].replace("\n", "\n    "))

layer4 = decisions.compile(spec | {"readout": {"layer": 4, "method": "diff_of_means"}}, model, cal)
evaluate("ablation: layer 4 readout", layer4)
diff_means = decisions.compile(spec | {"readout": {"layer": "final", "method": "diff_of_means"}}, model, cal)
evaluate("ablation: diff of means", diff_means)

# Several handles over the same spans share the encoder pass: the marginal cost of a handle is tiny.
spans = [r["span"] for r in test]
for hs in ([main.handle], [main.handle, diff_means.handle], [main.handle, diff_means.handle, layer4.handle]):
    t0 = time.perf_counter()
    model.apply(hs, spans)
    print(f"{len(hs)} handle(s): {(time.perf_counter() - t0) / len(spans) * 1000:.1f} ms/span")

# Comparison with the teacher: the same question (tool_call_failed/teacher.jsonl holds its answers by hash).
q = data.TEACHER_QUESTIONS["tool_failed"]
ours = {"decision (spec)": data.boolean_probs(model, main.handle, spans),
        "ablation: layer 4": data.boolean_probs(model, layer4.handle, spans),
        "ablation: diff of means": data.boolean_probs(model, diff_means.handle, spans)}
# With labels: the 60 calibration rows plus up to 200 labelled test rows from other folds. The test is balanced per
# tool; precision is also given at the share of results the harness flags in the pi stream (data/pi_stream.jsonl).
prevalence = float(np.mean([r["is_error"] for r in data.load_jsonl(data.SHIPPED / "pi_stream.jsonl")]))
E = decisions._Encoder(model, "final")
unit = lambda rs: [state_text(data.tool_state(r["span"])) for r in rs]  # noqa: E731
labels = data.Labels(E(unit(test)), base_X=E(unit(cal)), base_y=np.array([int(r["label"]) for r in cal]))
data.teacher_compare("tool_call_failed", "pi test (harness flag as truth)", q, [data.tool_state(s) for s in spans],
                  y.astype(int), ours, labels=labels, spec="decision (spec)", prevalence=prevalence)
data.teacher_compare("tool_call_failed", "tau2 support-bot outputs (no labels)", q, [data.tool_state(s) for s in ood],
                  None, {"decision (spec)": data.boolean_probs(model, main.handle, ood)})

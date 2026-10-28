# preflight: will a question graduate?

A check to run before deploying a proxy question: given 150–300 real units with the teacher's answers, can a cheap
EmbeddingGemma 2 student take the question over, at which layer, and how? The rules are in
[`../../docs/question-design.md`](../../docs/question-design.md).

From `examples/`:

```bash
uv run preflight/preflight.py answers.jsonl --type noul --name my_question
uv run preflight/preflight.py answers.jsonl --type choice --options source,test_run,build,search,vcs,edit,other
uv run preflight/preflight.py answers.jsonl --type score --options none,minor,recoverable,blocking
```

Input, one JSON object per line:

```json
{"span": {"call": "...", "output": "..."}, "answer": [0.93], "group": "session-1"}
```

`answer` is the teacher's probabilities or a label. The probabilities are one per option. For yes/no, use `[p_yes]`
or a number. `group` keeps the units of one session in the same cross-validation fold.

## What it does

For layers 4, 8 and 12 and the final embedding, each costed at its measured exit time on these units, a student is
cross-validated against the teacher with the proxy's graduation rules:
- **symmetric:** agreement 95% lower bound ≥ `--target`, and recall on the teacher's minority answers ≥ `--minority`;
- **one-sided** (yes/no): only a confident "no" is answered locally;
- **levels** (score): Spearman, QWK, within-one, and whether held-out level means are in order.

It also tells you how unsure the teacher itself was. It tells you which units are longer than the 1,024-token window.
It tells you how it did on any hard negatives you gave it. It also tells you how much coverage and compute per item
to expect at the best readout.

## Findings

On `injection_teacher`'s two answer files, the injection question graduates symmetrically: at layer 8 with kappa 0.87
on the teacher's answers (`results/injection_teacher_answers.jsonl`), and at layer 4 with kappa 0.89 on the ground
truth (`results/injection_teacher_truth.jsonl`).

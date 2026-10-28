# Designing questions hunch can take over

Some questions can be learned by hunch's students and some can't. This guide lists what makes the difference, from
the examples, and gives a check to run before deploying a question. [../hunch/README.md](../hunch/README.md) explains
how hunch works; this page assumes you know what a teacher, a student and graduation are.

Terms used below:
- A **unit** is the text a question is asked about, such as one tool call and its output. hunch calls it a *state*;
  eg2cpu's compiled decisions call it a *unit*.
- **kappa** and **QWK** (quadratic weighted kappa, for ordered levels) measure agreement beyond chance: 1 is perfect,
  0 is chance. **AUC** measures how well scores rank the "yes" cases above the "no" cases: 1 is perfect, 0.5 random.
- A **hard negative** is an example that looks like a "yes" but isn't, such as source code that mentions an error.
- **Symmetric** and **one-sided** are hunch's two graduation modes: a symmetric student answers every question it is
  confident about; a one-sided one answers only a confident "no" and passes every possible "yes" to the teacher.

The evidence comes from these examples, all in [../examples](../examples): `proxy_replay` (hunch's policy against a
teacher on 2,000 tool results from public coding-agent sessions), `pii`, `prompt_injection`, `injection_teacher`,
`failure_severity` and `levels_sst5`.

## What limits a question

| limit | evidence | why |
|---|---|---|
| **The answer must be readable in the unit's text** | Output kind: kappa 0.94 against the teacher. PII present: AUC 0.97 for a probe trained on labelled gretel windows (layer 8, held out by document type), 0.88 for the decision compiled from 200 contrasts. Sentiment intensity: QWK 0.63 (the spec) to 0.70 (`layer: auto`). Failure severity: QWK 0.45 | Embeddings encode what the text says. "Who can fix this?", "what did the user want?" and "is the world in this state?" need context or reasoning. |
| **Topic vs property** | Code that *mentions* errors read as failures (26% correct without hard negatives); PII field names scored like PII values | The encoder reads what the text is about. Without matched contrasts and hard negatives, a question about a property gets answered about the topic. |
| **Rare answers** | `tool_failed` (5% "yes") never graduates symmetrically: recall on "yes" caps at about 0.75. One-sided, it answers about 70% locally, all of it "no", so its agreement with the teacher (0.995) only matches always answering "no". Blocking errors (about 1 in 80) are never learnable | A few positives can't prove recall, and agreement is fooled by base rates: always answering "no" agrees 95% of the time. |
| **Teacher blind spots** | The teacher catches 20% of polite prompt injections. On the same 26 held-out ones (unseen tools, attacks and fillers) it catches 23%, a student trained on its answers 27%, and the same student trained on ground truth 81% | A student copies its teacher's misses. It can't be better than its teacher unless it learns from something else. |
| **Teacher consistency** | The teacher calls 20% of harness-flagged errors "not failed", and is unsure on 35% of severity answers | The student copies the teacher, so an inconsistent teacher sets the ceiling. |
| **Format shift** | Prompt injection: the compiled decision reaches AUC 0.999 on familiar tool formats but 0.90 on new ones, where it flags 37% of imperative benign texts; a student trained on 200 InjecAgent labels flags 47% of real coding-agent tool outputs | Students learn format cues. Novelty checks and audits catch this, at the cost of coverage. |
| **Scales** | Sentiment levels worked; severity levels didn't (held-out levels came out of order) | A scale works when the wording intensifies along it, not when its levels are causal categories. |
| **Whether vs where** | Detecting takes one encoder pass; locating the part of the text responsible takes about 8 (re-encoding fragments) | An embedding of the whole text says whether, not where. |
| **Bundled questions** | One ungraduated question in a request sent every request to the teacher | The teacher reads the state once for all questions, so forwarding part of a request saves little. |
| **Unit length** | The encoder reads at most 1,024 tokens, and a small signal dilutes in a long unit | hunch keeps the head and tail of a long state; otherwise, split long outputs yourself or leave them to the teacher. |
| **Layer** | The compiled `tool_call_failed` direction is best at layer 20 on real traces (AUC 0.92), with the final embedding level with it (0.90) and layers 12 and 23 worst; PII reads best at the final layer on gretel | Questions live in different layers. hunch reads one layer for every question (`--layer`, default 8); the pre-flight shows which layer each question needs. |

## Rules for writing questions

1. **Ask about a unit, not the trace.** Ask "Did this tool call fail?" of each tool result, not "Did the agent
   struggle?" of the trace. Build trace-level answers in code from unit-level ones. Every success above was at the
   unit level.
2. **Ask for observable properties, and compose judgements in code.** Don't ask "How severe is this failure?". Ask
   "did it fail?" (the harness already knows), "does it mention credentials, permissions, quotas or rate limits?" and
   "is it an empty result?", then let a short program assign severity. Each sub-question is readable from the text;
   the judgement lives in code, where it's transparent and free.
3. **Put the rare answer on the "yes" side, and keep it binary.** Ask "Contains a secret?" or "Blocked by something
   outside the agent's reach?". Rare yes/no questions graduate one-sided. Split a rare class out of a multi-class
   question into its own yes/no question, rather than hoping a 2% class survives a 7-way choice.
4. **Use a scale (`score`) only for intensity.** "How frustrated is the user?" fits. "Which kind of failure?" is a
   `choice`.
5. **Say what doesn't count.** For example: "Source code or docs that mention errors are not a failure"; "naming a
   field without its value is not PII". This makes the teacher more consistent, and the teacher's answers are what
   the student learns. If you also use eg2cpu's compiled decisions, add the same cases to them as hard negatives.
6. **Freeze the wording, and name the family.** Graduation and the teacher's answers belong to the exact question
   text, and rewording starts over. Give every question a `family`, so its labelled examples survive a rewording.
   Change the wording deliberately: test the new wording on a dev set and a test set first, as `../examples/question_pilot/`
   does. Ask one general question of every tool, with the tool named in the state, rather than one hand-worded
   question per tool.
7. **Give the unit what the question needs, and nothing more.** Include the call with its output, and keep units
   short. Known facts (exit code, harness flag, tool name) can be extra fields of the state: hunch adds them to the
   text it reads, as `name: value` lines, and your code can use them directly.
8. **Group questions by how well they graduate.** Put questions that graduate, or are likely to, in their own
   requests. Don't bundle a severity or world-state question with them, or every request pays for the teacher.
9. **Check the teacher before learning from it.** Where some ground truth exists (benchmarks, harness flags,
   labelled samples), run the pre-flight twice, once on the teacher's answers and once on the ground truth, and score
   the teacher against the ground truth. If the teacher has a blind spot, train the student on the ground truth
   (`labelled` cases), and keep the teacher for what the student passes on.
10. **Route safety questions one-sided.** For injections, secrets and destructive actions, a miss costs more than a
    forward: prefer one-sided graduation even when symmetric passes.

## What phrasing can't fix

Some questions need reasoning or knowledge the unit doesn't contain: preconditions, causal severity, "should the
agent have done X?". Those stay with the teacher however they're worded. hunch finds this out itself: those questions
never graduate, and audits keep the ones that did graduate honest.

## Pre-flight check

Collect 150–300 teacher answers on real units, and from `examples/` run the check before deploying a question:

```bash
uv run preflight/preflight.py answers.jsonl --type noul --name tool_failed
```

```bash
uv run preflight/preflight.py answers.jsonl --type choice --options source,test_run,build,search,vcs,edit,other
```

```bash
uv run preflight/preflight.py answers.jsonl --type score --options none,minor,recoverable,blocking
```

**Input.** `answers.jsonl` has one line per unit (`span` is the unit, the state hunch would see):

```json
{"span": {"call": "...", "output": "..."}, "answer": [0.03], "group": "<session id>"}
```

- `answer` is the teacher's probabilities or a label. Each option has one probability. For `noul`, the probabilities
  are `[p_yes]`.
- `group` keeps the units of a session in the same cross-validation fold, as hunch meets them.
- `--hard hard.jsonl` adds units that must read a given answer.

**What it does.** It cross-validates a student against the teacher for layers 4, 8 and 12 and the final embedding. It
uses the graduation rules of hunch ([glossary](glossary.md#the-proxy-hunch)). It costs each layer at its measured
encoding time on your units. For `score` questions, it also checks Spearman correlation, QWK, how often the answer is
within one level, and whether the levels come out in order. It reports how unsure the teacher is. It reports how many
units exceed the 1,024-token window. It ends with one verdict:

| verdict | meaning | what to do |
|---|---|---|
| **GRADUATES symmetrically** at layer *k* | the expected coverage, kappa and cost per item | deploy, reading layer *k* |
| **GRADUATES ONE-SIDED** | only a confident "no" would be answered locally | deploy; hunch picks one-sided itself |
| **NOT ENOUGH DATA** | too few rare answers to prove recall; it prints how many more to collect, and the CV AUC (whether there is any signal) | collect more, re-run |
| **BACKEND ONLY** | no layer passes | rephrase (rules 1–5 and 7, with the printed hints), or keep the question on the teacher |

**Does it predict the replay?** On the first 300 units of the shipped stream, the check gave the same verdicts as the
full 2,000-unit replays. Costs are estimated compute per unit, compared with the teacher's 494 ms:

| question | pre-flight on 300 units | replay on 2,000 |
|---|---|---|
| output_kind | graduates at layer 8: about 90% local, kappa 0.85, about 115 ms per unit | graduated at unit 149; 92% local at layer 8, kappa 0.94 |
| tool_failed | not enough data: 16 "yes" answers where about 33 are needed (CV AUC 0.79) | graduated one-sided at unit 549 (lost it at 701, back at 1001) |
| severity | backend only: rare levels missed, levels out of order, teacher unsure on 35% | never graduated |

For `tool_failed`, the check asked for about 320 more units. At 650 units, it asked for about 90 more units.

**Prompt injection** (`injection_teacher`, 700 units grouped by tool): on the teacher's answers the question
graduates at layer 8 (kappa 0.87), and on the ground truth at layer 4 (kappa 0.89). Only the check against ground
truth shows that a student of this teacher would inherit its 80% miss rate on polite injections, which is rule 9.

The pre-flight is a snapshot. hunch re-checks graduation as answers arrive and audits after graduating, so a question
that passes the pre-flight can still be demoted when the traffic shifts.

"""The hunch proxy as a service, end to end over HTTP, with the teacher's cached answers as the upstream model.

    uv run proxy_service/proxy_service.py                     # replay the shipped stream (2,000 pi tool results the teacher answered)
    uv run proxy_service/proxy_service.py --no-seed           # without the labelled seed for tool_failed
    TEACHER_URL=... uv run proxy_service/proxy_service.py --live   # ask a live teacher instead

What happens:
  1. A proxy starts on a free port (`hunch.serve`), backed by the teacher's cached answers: an OfflineBackend over
     data/teacher_stream.jsonl (question hash, state hash -> answer object), so the run costs no teacher time.
  2. POST /v1/train seeds tool_failed with the harness's isError flag on the first 150 items, as "labelled" ground
     truth. (The teacher calls 20% of flagged errors "not failed": labelled cases are how a proxy learns past a teacher's
     blind spot, rule 9 of docs/question-design.md.)
  3. The other 1,850 items stream through POST /v1/answer in order: one request per question, so a question that
     never graduates (severity) doesn't drag the others upstream (rule 8). Forwarded answers become "sampled" examples.
  4. GET /v1/questions shows each question's examples, graduation mode and audits.
Reported per question, over the last 500 items and over the whole run: the share of requests the student answered
without any teacher call (an audited answer called the teacher, so it doesn't count), the share answered from the
store (a state the teacher answered before), and the teacher calls made per request; and, over the same requests,
how often the student's answers agree with the teacher's, beside what always giving the teacher's most common answer
would score.
"""

import argparse
import json
import os
import threading
import time
import urllib.request

import numpy as np

from hunch_examples import OUT, STREAM_QUESTIONS, shipped_rows, teacher_file, tool_state
from eg2cpu import Model
from hunch import AnswerCache, HTTPBackend, OfflineBackend, Proxy, parse_answer, serve
from hunch.proxy import question_key, state_hash

ap = argparse.ArgumentParser()
ap.add_argument("--no-seed", action="store_true", help="skip the labelled seed for tool_failed")
ap.add_argument("--live", action="store_true", help="ask TEACHER_URL instead of the cached answers")
ap.add_argument("--seed-n", type=int, default=150)
args = ap.parse_args()

rows = sorted(shipped_rows("pi_stream.jsonl"), key=lambda r: r["i"])
state = tool_state
answers = AnswerCache(teacher_file("stream"))  # the teacher's answers (TEACHER picks whose), keyed by hashes


class Counted(OfflineBackend):
    """The cached teacher, recording each (question, state) it is asked: forwarded requests and audits."""

    asked: list = []
    calls = property(lambda self: self.hits + self.misses)

    def __call__(self, request):
        self.asked += [(question_key(q), state_hash(request["state"])) for q in request["questions"].values()]
        return super().__call__(request)


if args.live and not os.environ.get("TEACHER_URL"):
    ap.error("--live needs TEACHER_URL")
backend = HTTPBackend(os.environ["TEACHER_URL"]) if args.live else Counted(answers, model="teacher (cached)")
db = OUT / "proxy_service.db"
db.unlink(missing_ok=True)
proxy = Proxy(Model(), backend, db, layer=8, audit=0.10, refit_every=100)
server = serve(proxy, port=0)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"
print(f"proxy at {base}, layer 8, backend: {'live teacher' if args.live else 'cached teacher answers'}")


def call(path, body=None):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


start = 0
if not args.no_seed:
    seed = rows[: args.seed_n]
    s = call("/v1/train", {"question": STREAM_QUESTIONS["tool_failed"],
                           "cases": [{"state": state(r), "answer": bool(r["is_error"]), "kind": "labelled"} for r in seed]})
    start = args.seed_n
    print(f"seeded tool_failed with {len(seed)} labelled cases (harness flag, {sum(r['is_error'] for r in seed)} errors): "
          f"graduated={s['graduated']}")

opts = {q: list(STREAM_QUESTIONS[q]["criteria"]) if q == "output_kind" else None for q in STREAM_QUESTIONS}
log = {q: [] for q in STREAM_QUESTIONS}  # (who answered, agrees with the teacher, the teacher's answer, state hash)
t0 = time.perf_counter()
for n, r in enumerate(rows[start:], 1):
    for qid, q in STREAM_QUESTIONS.items():
        out = call("/v1/answer", {"state": state(r), "questions": {qid: q}})
        a = out["answers"][qid]
        teacher = parse_answer(q, answers.get(q, state(r)))[0]
        log[qid].append((out["answered_by"], parse_answer(q, {k: v for k, v in a.items() if k in ("noul", "probabilities")})[0] == teacher,
                         teacher, state_hash(state(r))))
    if n % 250 == 0:
        share = "  ".join(f"{q} {np.mean([w != 'backend' for w, *_ in v[-250:]]):.0%}" for q, v in log.items())
        print(f"{start + n:>5} items  answered without the teacher, last 250: {share}  ({time.perf_counter() - t0:.0f} s)",
              flush=True)

print(f"\n{len(rows) - start} items x {len(STREAM_QUESTIONS)} questions in {time.perf_counter() - t0:.0f} s; "
      f"upstream calls {getattr(backend, 'calls', '?')} of {(len(rows) - start) * len(STREAM_QUESTIONS)}")
proxy.wait()  # background audits
status = {s["key"]: s for s in call("/v1/questions")["questions"]}
asked = set(getattr(backend, "asked", []))
result = {}
for qid, q in STREAM_QUESTIONS.items():
    who = np.array([w for w, *_ in log[qid]])
    agrees = np.array([g for _, g, _, _ in log[qid]])
    told = np.array([t for _, _, t, _ in log[qid]])
    # a student answer whose state the teacher was asked about is an audit (a state it had answered would be a store hit)
    audited = np.array([w == "proxy" and (question_key(q), h) in asked for w, _, _, h in log[qid]])
    s = next(s for s in status.values() if s["spec"]["instructions"] == q["instructions"])
    res = {"mode": s["mode"], "graduated": s["graduated"], "examples": s["examples"], "audits": s["audits"]}
    for name, window in (("last 500", slice(-500, None)), ("all", slice(None))):
        w, a, t, au = who[window], agrees[window], told[window], audited[window]
        student = w == "proxy"
        common = int(np.bincount(told).argmax())  # the teacher's most common answer over the run
        res[name] = {
            "requests": len(w), "student_without_teacher": float(np.mean(student & ~au)), "audited": float(np.mean(au)),
            "from_store": float(np.mean(w == "cache")), "forwarded": float(np.mean(w == "backend")),
            "teacher_calls_per_request": float(np.mean((w == "backend") | au)),
            "student_agreement": float(a[student].mean()) if student.any() else None,
            "most_common_answer_agreement": float(np.mean(t[student] == common)) if student.any() else None}
        m = res[name]
        print(f"  {qid:<12} {name:<9} student, no teacher call {m['student_without_teacher']:.0%}  audited {m['audited']:.0%}  "
              f"store {m['from_store']:.0%}  forwarded {m['forwarded']:.0%}  teacher calls/request "
              f"{m['teacher_calls_per_request']:.2f}  student agreement "
              + (f"{m['student_agreement']:.3f} (always the teacher's most common answer: "
                 f"{m['most_common_answer_agreement']:.3f})" if student.any() else "-"))
    print(f"  {qid:<12} mode {s['mode']}; examples {s['examples']}; audits {s['audits']}")
    result[qid] = res
(OUT / "proxy_service.json").write_text(json.dumps(result, indent=1))
print(f"\nwrote {OUT / 'proxy_service.json'}; the proxy's database is {db}")
server.shutdown()

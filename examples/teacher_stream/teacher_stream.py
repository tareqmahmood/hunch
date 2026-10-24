"""Build the proxy examples' stream and ask a teacher about it: a chronological stream of real pi tool results, each
answered once by the teacher on three typed questions. proxy_replay.py, proxy_early_exit.py and proxy_service.py
replay the proxy over it. It writes the stream as a manifest (state hashes, no text) to results/pi_stream.jsonl and the
teacher's answers, keyed by question and state hashes, to results/teacher/stream.jsonl; to replay your own, copy them
over data/pi_stream.jsonl and data/teacher_stream.jsonl.

Stream: pi sessions in time order (the timestamp in each file name), up to 8 tool results per session in session
order, 2,000 results in all; outputs clipped to 1,200 characters (hunch_examples.clip).
Questions (hunch_examples.STREAM_QUESTIONS):
  tool_failed  noul    did the call itself fail (not: bad news about the world, an empty search)?
  output_kind  choice  source / test_run / build / search / vcs / edit / other
  severity     score   none / minor / recoverable / blocking
Resumes from results/pi_stream.jsonl.

    TEACHER_URL=http://localhost:8000/v1/answer uv run teacher_stream/teacher_stream.py
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

from hunch_examples import OUT, STREAM_QUESTIONS, clip, pi_files, pi_tool_results, session_id, tool_state
from hunch import AnswerCache, HTTPBackend, state_hash

if not os.environ.get("TEACHER_URL"):
    sys.exit("set TEACHER_URL to a teacher endpoint (POST {state, questions} -> {answers})")
TEACHER = HTTPBackend(os.environ["TEACHER_URL"])
ANSWERS = AnswerCache(OUT / "teacher/stream.jsonl", write_to=OUT / "teacher/stream.jsonl")
STREAM = OUT / "pi_stream.jsonl"
N, PER_SESSION = 2000, 8


def stream():
    stamp = lambda f: (re.search(r"(\d{4}-\d{2}-\d{2}T[\d-]+Z)", f) or [None, f])[1]
    items = []
    for f in sorted(pi_files(), key=stamp):
        got = [{"session": session_id(r.file), "tool": r.tool, "is_error": r.is_error, "call": r.call,
                "output": clip(r.text)} for r in pi_tool_results([f])]
        if got:
            keep = sorted(random.Random(Path(f).name).sample(range(len(got)), min(PER_SESSION, len(got))))
            items += [got[i] for i in keep]
        if len(items) >= N:
            break
    return items[:N]


items = stream()
done = {json.loads(line)["i"] for line in open(STREAM)} if STREAM.exists() else set()
print(f"stream {len(items)} results from {len({x['session'] for x in items})} sessions; {len(done)} already answered",
      flush=True)
t_start = time.monotonic()
with open(STREAM, "a") as f:
    for i, x in enumerate(items):
        if i in done:
            continue
        state = tool_state(x)
        t0 = time.monotonic()
        got = TEACHER({"state": state, "questions": STREAM_QUESTIONS})
        for q, qd in STREAM_QUESTIONS.items():
            ANSWERS.put(qd, state, got["answers"][q], got.get("model", "teacher"))  # the shipped stream's: clef-flash
        f.write(json.dumps({"i": i, "state": state_hash(state), "session": x["session"], "tool": x["tool"],
                            "is_error": x["is_error"], "seconds": round(time.monotonic() - t0, 4)}) + "\n")
        f.flush()
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(items)}  {(time.monotonic() - t_start) / 60:.1f} min elapsed", flush=True)
print("done", flush=True)

"""A teacher endpoint backed by a model on OpenRouter, for TEACHER_URL.

    uv run teacher_openrouter.py deepseek/deepseek-v4.1-flash --port 8101
    TEACHER=deepseek-v4.1-flash TEACHER_URL=http://localhost:8101/v1/answer uv run tool_call_failed/tool_call_failed.py

Takes {"state", "questions"} and returns {"answers"} (hunch.HTTPBackend), one OpenRouter call per question. The model
states its probabilities as JSON (structured output): a hosted model may not return log-probabilities, so every
hosted teacher is asked the same way. Reads OPENROUTER_API_KEY from the environment or from ../.env.
"""

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from hunch.proxy import answer_object, options, state_text

URL = "https://openrouter.ai/api/v1/chat/completions"
STATS, STATS_LOCK = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}, threading.Lock()
SYSTEM = ("You answer one question about a piece of text. Give your probability for each answer: how likely it is to "
          "be the right one. Reply with JSON only.")


def key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("set OPENROUTER_API_KEY (or put it in .env)")


def prompt(q: dict, text: str) -> tuple[str, dict]:
    """The user message and the JSON schema of the reply, for one typed question."""
    if q["type"] == "noul":
        ask = f"{q['instructions']}\n\nReply with p_yes: the probability that the answer is yes."
        schema = {"type": "object", "properties": {"p_yes": {"type": "number"}}, "required": ["p_yes"],
                  "additionalProperties": False}
    else:
        opts = options(q)
        crit = q["criteria"] if isinstance(q["criteria"], list) else list(q["criteria"].values())
        legend = "\n".join(f"{o}: {c}" for o, c in zip(opts, crit))
        ask = f"{q['instructions']}\n\nAnswers:\n{legend}\n\nReply with a probability for each answer, summing to 1."
        schema = {"type": "object", "properties": {o: {"type": "number"} for o in opts}, "required": opts,
                  "additionalProperties": False}
    return f"{ask}\n\n<text>\n{text}\n</text>", schema


def ask(model: str, q: dict, text: str, effort: str, api_key: str, tries: int = 6) -> dict:
    user, schema = prompt(q, text)
    body = {"model": model, "temperature": 0, "max_tokens": 4000, "reasoning": {"effort": effort},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
    last = None
    for t in range(tries):
        body["max_tokens"] = 4000 * 2 ** min(t, 2)  # an empty reply usually means reasoning used up the budget
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {
                "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            content = out["choices"][0]["message"]["content"]
            reply = json.loads(content[content.index("{"):content.rindex("}") + 1])
            if q["type"] == "noul":
                p1 = min(max(float(reply["p_yes"]), 0.0), 1.0)
                p = np.array([1 - p1, p1])
            else:
                p = np.clip(np.array([float(reply[o]) for o in options(q)]), 0, None)
                p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
            usage = out.get("usage", {})
            with STATS_LOCK:
                STATS["calls"] += 1
                STATS["in"] += usage.get("prompt_tokens", 0)
                STATS["out"] += usage.get("completion_tokens", 0)
                STATS["cost"] += float(usage.get("cost") or 0)
            return answer_object(q, p)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, IndexError,
                TypeError, AttributeError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError):
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code in (401, 402, 403):
                    raise RuntimeError(last) from e
            elif "out" in locals():
                last = f"{e!r}; reply {json.dumps(out)[:300]}"
            print(f"retry {t + 1}/{tries}: {last}", flush=True)
            time.sleep(min(60, 2 ** t + random.random()))
    raise RuntimeError(f"{model}: no usable answer after {tries} tries: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="an OpenRouter model id")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--effort", default="low", help="reasoning effort (default low)")
    args = ap.parse_args()
    api_key = key()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = state_text(req["state"])
            try:
                answers = {qid: ask(args.model, q, text, args.effort, api_key) for qid, q in req["questions"].items()}
                body, code = json.dumps({"answers": answers}).encode(), 200
            except Exception as e:  # noqa: BLE001
                print(f"failed: {e!r}", flush=True)
                body, code = json.dumps({"error": repr(e)}).encode(), 502
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            with STATS_LOCK:
                if STATS["calls"] and STATS["calls"] % 500 == 0:
                    print(f"{args.model}: {STATS}", flush=True)

        def do_GET(self):  # usage so far
            body = json.dumps({"model": args.model, **STATS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"{args.model} on http://localhost:{args.port}/v1/answer (reasoning effort {args.effort})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

"""One column of the README demo: a stream of synthetic tool calls answered by hunch, a hosted judge or a frontier
model, with the money spent so far pinned at the top. Every column draws the same requests (same seed), so they
differ only in how fast each answer comes back and what it costs.

    python stream.py hunch --seconds 20

Speed and price per request, for ~200-token tool results and one caller:
  hunch     21 ms p50 for a local answer (EG2 layer-8 exit, M5 Pro); 75% answered locally and 0.25 judge calls per
            request (forwards plus audits), as in examples/proxy_service; encoder $0.0038 per million tokens (Graviton 4,
            layer-8 exit), plus the judge's price for its calls
  judge     a hosted judge at $0.042 per million input tokens and ~500 ms per answer (the judge latency in hunch's
            benchmarks)
  frontier  GPT-6 Luna at ~4.9x the judge's cost per trace (published: $54 vs $11 per 10K traces); its latency is
            assumed, ~1.2 s, not measured here
"""

import argparse
import os
import random
import sys
import time

TOKENS = 200
JUDGE_PER_M, ENCODER_PER_M, FRONTIER_X = 0.042, 0.0038, 54 / 11
LOCAL_SHARE, JUDGE_CALLS = 0.75, 0.25
COLUMNS = {  # title, $ per request, seconds for one answer
    "hunch": ("hunch (EG2 student -> judge)", TOKENS * (ENCODER_PER_M + JUDGE_CALLS * JUDGE_PER_M) / 1e6, None),
    "judge": ("hosted judge", TOKENS * JUDGE_PER_M / 1e6, lambda r: r.gauss(0.50, 0.08)),
    "frontier": ("frontier model (GPT-6 Luna)", FRONTIER_X * TOKENS * JUDGE_PER_M / 1e6, lambda r: r.gauss(1.2, 0.25)),
}
CALLS = ["npm test", "git status", "rg -n TODO src", "ls packages", "cargo build", "pytest -q", "git push",
         "gh pr view 42", "make lint", "tsc --noEmit", "go test ./...", "curl -s api/health"]
FILES = ["src/app.ts", "README.md", "lib/parse.py", "cmd/main.go", "test/api.spec.ts", "Cargo.toml"]
GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def requests(seed: int):
    """The same endless request stream for every column: (tool call, failed?)."""
    r = random.Random(seed)
    while True:
        kind = r.random()
        call = (f"bash: {r.choice(CALLS)}" if kind < 0.6 else
                f"read: {r.choice(FILES)}" if kind < 0.85 else f"edit: {r.choice(FILES)}")
        yield call, r.random() < 0.12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("column", choices=COLUMNS)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--hold", type=float, default=3, help="seconds to keep the last frame")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    title, price, latency = COLUMNS[args.column]
    width, height = os.get_terminal_size()
    timing = random.Random(args.seed * 100 + list(COLUMNS).index(args.column))
    out = sys.stdout

    def header(spent: float, n: int) -> None:
        out.write("\0337\033[1;1H\033[2K" + f"{BOLD}{title}{RESET}"[: width + 8])
        out.write(f"\033[2;1H\033[2K{BOLD}${spent:,.6f}{RESET} spent · {n} requests")
        out.write(f"\033[3;1H\033[2K{DIM}${price * 1e6:,.2f} per million requests{RESET}")
        out.write(f"\033[4;1H\033[2K{DIM}{'─' * width}{RESET}\0338")
        out.flush()

    out.write(f"\033[2J\033[?25l\033[5;{height}r\033[5;1H")  # clear, hide the cursor, scroll below the header
    spent, n, t0 = 0.0, 0, time.monotonic()
    header(spent, n)
    for call, failed in requests(args.seed):
        if args.column == "hunch":
            local = timing.random() < LOCAL_SHARE
            wait = max(0.008, timing.lognormvariate(-3.86, 0.35)) + (0 if local else max(0.2, timing.gauss(0.5, 0.08)))
            who = "student" if local else "-> judge"
        else:
            wait, who = max(0.2, latency(timing)), ""
        if time.monotonic() + wait - t0 > args.seconds:
            break
        time.sleep(wait)
        spent, n = spent + price, n + 1
        verdict = f"{RED}failed{RESET}" if failed else f"{GREEN}ok    {RESET}"
        ms = f"{wait * 1000:5.0f} ms"
        tag = f" {DIM}{who}{RESET}" if who else ""
        room = width - 6 - 1 - len(ms) - 3 - (len("-> judge") + 1 if args.column == "hunch" else 0)
        out.write(f"{call[:room]:<{room}} {verdict} {DIM}{ms}{RESET}{tag}\r\n")
        header(spent, n)
    time.sleep(args.hold)
    out.write("\033[?25h")


if __name__ == "__main__":
    main()

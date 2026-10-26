#!/bin/sh
# Records the README demo: each column's stream on its own, then tmux playing the three back side by side, as a gif.
# Needs asciinema, tmux and agg (brew install asciinema tmux agg).
set -e
cd "$(dirname "$0")"
out=${1:-hunch-demo.gif}
seconds=${SECONDS_SHOWN:-20}
tmp=$(mktemp -d)
export TERM=xterm-256color  # headless recordings have no terminal type, and tmux needs one
trap 'rm -rf "$tmp"' EXIT

for c in hunch judge frontier; do
  asciinema rec --headless --quiet --overwrite --window-size 49x24 -c "python3 stream.py $c --seconds $seconds --hold 0" "$tmp/$c.cast"
done

cat > "$tmp/tmux.conf" <<EOF
set -g status off
set -g pane-border-style fg=colour238
set -g pane-active-border-style fg=colour238
set -g remain-on-exit on
set -g default-terminal xterm-256color
EOF
play() { echo "asciinema play -q $tmp/$1.cast; tput civis; sleep 30"; }  # then hide the pane's cursor
asciinema rec --headless --quiet --overwrite --window-size 149x24 -c "tmux -f $tmp/tmux.conf -L hunch-demo new-session \
  '$(play hunch)' \; split-window -h '$(play judge)' \; split-window -h '$(play frontier)' \; \
  select-layout even-horizontal \; run-shell -b 'sleep $((seconds + 3)); tmux -L hunch-demo kill-server'" "$tmp/demo.cast"
# cut the recording before tmux exits, so it ends on the streams rather than tmux's exit message
python3 - "$tmp/demo.cast" "$((seconds + 2))" <<'PY'
import json, sys
path, stop = sys.argv[1], float(sys.argv[2])
lines = open(path).read().splitlines()
keep, t = [lines[0]], 0.0
for line in lines[1:]:
    t += json.loads(line)[0]  # asciicast v3: each event's time is relative to the one before
    if t > stop:
        break
    keep.append(line)
open(path, "w").write("\n".join(keep) + "\n")
PY
agg --font-size 14 --theme monokai --idle-time-limit 3 --last-frame-duration 4 "$tmp/demo.cast" "$out"
echo "wrote $out"

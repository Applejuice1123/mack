#!/bin/bash
# Repeatable demo for the README GIF: a 300-note vault in a temp folder, tagged
# by Mack while you watch the pace hold.
#
#   scripts/demo.sh           real model
#   MACK_STUB_ENGINE=1 scripts/demo.sh   no model, canned replies (for rehearsing the recording)
#
# Suggested recording: this script in the left pane; Activity Monitor (or
# `mack bench --seconds 20 --paces gentle`) in the right. Partway through, stop
# it with Ctrl-C and run the printed `mack run ... --pace gentle --wait` command
# to show the slowdown drop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEMO="$(mktemp -d)/vault"
$PY tests/fixtures/make_fixtures.py vault "$DEMO" --n 300
echo
echo "Two-minute quick start on 20 notes:"
echo "  mack run --recipe tag --limit 20 --pace full --wait $DEMO"
$PY -m mack.cli run --recipe tag --limit 20 --pace full --wait "$DEMO"
JOB=$($PY -m mack.cli status | awk 'NR==2 {print $1}')
$PY -m mack.cli review "$JOB" --no-open
$PY -m mack.cli apply "$JOB" --dry-run | head -5
echo
echo "Now the whole vault at the gentle pace:"
echo "  mack run --recipe tag --pace gentle --wait $DEMO"

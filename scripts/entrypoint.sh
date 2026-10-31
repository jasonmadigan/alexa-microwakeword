#!/usr/bin/env bash
# bring /env in line with uv.lock (a no-op once installed), then run the command.
# with WAKEWORD_LOG set, all output (sync included) is also appended to that file.
set -euo pipefail

if [[ -n "${WAKEWORD_LOG:-}" ]]; then
  log=$WAKEWORD_LOG
  WAKEWORD_LOG= "$0" "$@" 2>&1 | tee -a "$log"
  exit "${PIPESTATUS[0]}"
fi

uv sync --frozen --quiet
export PYTHONUNBUFFERED=1

# tensorflow 2.21 misses some pip-installed cuda libraries (libcusolver) on its own search path
for dir in /env/lib/python3*/site-packages/nvidia/*/lib; do
  LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

exec "$@"

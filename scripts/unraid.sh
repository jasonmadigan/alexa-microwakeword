#!/usr/bin/env bash
# drive the harness on unraid: source, python env and data live on its nvme cache,
# the workstation only edits code
set -euo pipefail

HOST=${UNRAID_HOST:-unraid}
ROOT=${UNRAID_ROOT:-/mnt/cache/wakeword}
IMAGE=${WAKEWORD_IMAGE:-alexa-wakeword:latest}
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-3000}
cd "$(dirname "$0")/.."

ARGS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
--user 99:100 --shm-size=8g -v $ROOT/src:/app -v $ROOT/env:/env -v $ROOT/data:/data"

check_gpu() {
  # the gpu may be shared with other services; refuse rather than starve them or oom
  local free
  free=$(ssh "$HOST" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1)
  if (( free < MIN_FREE_VRAM_MB )); then
    echo "only ${free} MiB vram free (need ${MIN_FREE_VRAM_MB}); see what holds it with: ssh $HOST nvidia-smi" >&2
    exit 1
  fi
}

case "${1:-}" in
  sync)
    ssh "$HOST" "mkdir -p $ROOT/src $ROOT/env $ROOT/data/logs $ROOT/data/household \
      && chown 99:100 $ROOT/env $ROOT/data $ROOT/data/logs $ROOT/data/household"
    rsync -rlpt --delete --exclude-from=.dockerignore ./ "$HOST:$ROOT/src/"
    # editable installs write egg-info into the source tree, as the container user
    ssh "$HOST" "chown -R 99:100 $ROOT/src"
    ;;
  build)
    ssh "$HOST" "docker build -t $IMAGE $ROOT/src"
    ;;
  test)
    ssh "$HOST" "docker run --rm $ARGS $IMAGE pytest -q"
    ;;
  run)
    # detached so it survives ssh drops; output kept in data/logs
    shift
    check_gpu
    name="wakeword-$(date +%Y%m%d-%H%M%S)"
    ssh "$HOST" "docker run -d --rm --name $name -e WAKEWORD_LOG=/data/logs/$name.log $ARGS $IMAGE \
      wakeword $(printf '%q ' "$@")"
    echo "$name"
    ;;
  logs)
    name=${2:-$(ssh "$HOST" "ls -t $ROOT/data/logs | head -1" | sed 's/\.log$//')}
    ssh "$HOST" "tail -n 200 -F $ROOT/data/logs/$name.log"
    ;;
  status)
    ssh "$HOST" "docker ps --filter name=wakeword- --format '{{.Names}} {{.Status}}'; \
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv"
    ;;
  pull)
    # exported model files back to dist/
    mkdir -p dist
    rsync -a "$HOST:$ROOT/data/dist/" dist/
    ;;
  *)
    echo "usage: $0 sync|build|test|run <wakeword args>|logs [name]|status|pull" >&2
    exit 2
    ;;
esac

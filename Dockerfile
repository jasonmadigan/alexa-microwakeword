# syntax=docker/dockerfile:1
# runtime only: python, uv and ffmpeg. source, the uv-managed environment and data
# are bind-mounted from nvme (see scripts/unraid.sh); unraid's docker.img sits on
# an array disk, far too slow to hold ~5 GB of tensorflow and cuda wheels.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# runs as 99:100 on unraid, so every writable path is a mount or /tmp. tensorflow has no
# blackwell kernels, so ptx is jit-compiled on first use; the cache keeps that one-off
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/env \
    UV_CACHE_DIR=/data/cache/uv \
    PATH=/env/bin:$PATH \
    WAKEWORD_DATA=/data \
    HOME=/tmp \
    HF_HOME=/data/cache/huggingface \
    NUMBA_CACHE_DIR=/data/cache/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=1 \
    CUDA_CACHE_PATH=/data/cache/nv \
    CUDA_CACHE_MAXSIZE=4294967296

WORKDIR /app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["wakeword", "--help"]

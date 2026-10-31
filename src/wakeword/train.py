"""microwakeword training run from the harness config."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .config import Config

log = logging.getLogger(__name__)

MODEL_RELPATH = Path("tflite_stream_state_internal_quant") / "stream_state_internal_quant.tflite"
TRAINING_KEYS = (
    "training_steps", "learning_rates", "positive_class_weight", "negative_class_weight", "batch_size",
    "time_mask_max_size", "time_mask_count", "freq_mask_max_size", "freq_mask_count", "eval_step_interval",
    "clip_duration_ms", "target_minimization", "minimization_metric", "maximization_metric",
)


def training_parameters(cfg: Config) -> dict:
    t = dict(cfg["training"])
    t["training_steps"] = t.pop("steps")
    lay = cfg.layout
    sets = t.pop("feature_sets")
    features = []
    for name, spec in sets.items():
        if name in ("positive", "hard_negative"):
            path, truth = lay.features / name, name == "positive"
        elif name in cfg["fetch"]["negatives"]:
            path, truth = lay.negatives / name, False
        else:
            raise KeyError(f"feature set {name} is neither generated nor fetched")
        features.append({"features_dir": str(path), "truth": truth, "type": "mmap", **spec})
    params = {k: t[k] for k in TRAINING_KEYS if k in t}
    params.update({"window_step_ms": 10, "train_dir": str(lay.train), "features": features})
    lengths = {len(params[k]) for k in params if isinstance(params[k], list) and k != "features"}
    if len(lengths) != 1:
        raise ValueError("training list settings must all have one entry per training phase")
    return params


def command(cfg: Config, params_path: Path, train: bool = True) -> list[str]:
    m = cfg["model"]
    args = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={params_path}",
        f"--train={int(train)}",
        "--restore_checkpoint=1",
        "--test_tf_nonstreaming=0",
        "--test_tflite_nonstreaming=0",
        "--test_tflite_nonstreaming_quantized=0",
        "--test_tflite_streaming=0",
        "--test_tflite_streaming_quantized=1",
        "--use_weights=best_weights",
        m["architecture"],
    ]
    for key, value in m["args"].items():
        args += [f"--{key}", str(value)]
    return args


def fingerprint(cfg: Config, params: dict) -> str:
    """training inputs: parameters, model shape and every feature shard's input digest."""
    markers = sorted(
        (str(p.relative_to(cfg.layout.features)), p.read_text().strip())
        for p in cfg.layout.features.glob("*/*/*_mmap/.done")
    )
    blob = json.dumps([params, cfg["model"], markers], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _fresh_or_resumable(train_dir: Path, fp: str) -> None:
    # resuming a checkpoint trained on other inputs would silently mix runs
    marker = train_dir / "inputs.sha256"
    if train_dir.exists() and any(train_dir.iterdir()):
        if marker.exists() and marker.read_text().strip() == fp:
            return
        old = marker.read_text().strip()[:8] if marker.exists() else "unknown"
        aside = train_dir.with_name(f"{train_dir.name}.{old}.{int(time.time())}")
        train_dir.rename(aside)
        log.warning("training inputs changed; previous run moved to %s", aside)
    train_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(fp + "\n")


def run(cfg: Config, train: bool = True) -> Path:
    lay = cfg.layout
    params = training_parameters(cfg)
    _fresh_or_resumable(lay.train, fingerprint(cfg, params))
    params_path = lay.work / "training_parameters.yaml"
    params_path.write_text(yaml.safe_dump(params, sort_keys=False))
    env = {**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")}
    cmd = command(cfg, params_path, train)
    log.info("running %s", " ".join(cmd))
    with (lay.work / "train.log").open("a") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        if proc.wait() != 0:
            raise RuntimeError(f"training failed with exit code {proc.returncode}; see {lay.work / 'train.log'}")
    model = lay.train / MODEL_RELPATH
    if not model.exists():
        raise FileNotFoundError(model)
    return model

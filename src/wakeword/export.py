"""esphome micro_wake_word manifest and model files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import Config
from .streaming import StreamingModel

# firmware maps features 0..26 onto int8 -128..127 without reading the model's quant params
FIRMWARE_INPUT_SCALE = 26.0 / 255.0


def check_model(path: Path) -> dict:
    model = StreamingModel(path)
    scale, zero = model.input_quantization
    if model.stride != 3:
        raise ValueError(f"{path}: input stride {model.stride}, firmware expects 3 with 10 ms features")
    if zero != -128 or not np.isclose(scale, FIRMWARE_INPUT_SCALE, rtol=0.02):
        raise ValueError(f"{path}: input quantisation ({scale}, {zero}) does not match the firmware's 0..26 mapping")
    return {"stride": model.stride, "input_scale": float(scale), "input_zero_point": int(zero)}


def manifest(cfg: Config, model_file: str, cutoff: float, window: int) -> dict:
    e = cfg["export"]
    return {
        "type": "micro",
        "wake_word": cfg["wake_word"],
        "author": e["author"],
        "website": e["website"],
        "model": model_file,
        "trained_languages": e["trained_languages"],
        "version": 2,
        "micro": {
            "probability_cutoff": cutoff,
            "sliding_window_size": window,
            "feature_step_size": 10,
            "tensor_arena_size": e["tensor_arena_size"],
            "minimum_esphome_version": e["minimum_esphome_version"],
        },
    }


ESPHOME_SNIPPET = """\
# copy {name}.json and {name}.tflite next to the device yaml in the esphome config dir
micro_wake_word:
  models:
    - model: {name}.json
      id: alexa
"""


def run(cfg: Config, model_path: Path, report: dict) -> Path:
    best = report["models"]["candidate"]["best"]
    if not best["meets_target"]:
        raise RuntimeError("no operating point met the false-accept target; not exporting")
    check_model(model_path)
    out = cfg.layout.dist
    out.mkdir(parents=True, exist_ok=True)
    name = cfg.name
    shutil.copyfile(model_path, out / f"{name}.tflite")
    data = manifest(cfg, f"{name}.tflite", best["probability_cutoff"], best["sliding_window_size"])
    (out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "esphome.yaml").write_text(ESPHOME_SNIPPET.format(name=name))
    report_md = cfg.layout.eval / "report.md"
    if report_md.exists():
        shutil.copyfile(report_md, out / "report.md")
    return out

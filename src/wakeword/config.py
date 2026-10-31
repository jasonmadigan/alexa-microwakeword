from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            # null in an overlay removes the key from the base
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    base = data.pop("base", None)
    if base is None:
        return data
    return deep_merge(_read((path.parent / base).resolve()), data)


@dataclass(frozen=True)
class Layout:
    root: Path
    name: str

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def voices(self) -> Path:
        return self.root / "voices"

    @property
    def backgrounds(self) -> Path:
        return self.root / "backgrounds"

    @property
    def negatives(self) -> Path:
        return self.root / "negatives"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"

    @property
    def household(self) -> Path:
        return self.root / "household"

    @property
    def work(self) -> Path:
        return self.root / "work" / self.name

    @property
    def clips(self) -> Path:
        return self.work / "clips"

    @property
    def features(self) -> Path:
        return self.work / "features"

    @property
    def train(self) -> Path:
        return self.work / "train"

    @property
    def eval(self) -> Path:
        return self.work / "eval"

    @property
    def dist(self) -> Path:
        return self.root / "dist" / self.name


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def workers(self) -> int:
        env = os.environ.get("WAKEWORD_WORKERS")
        if env:
            return int(env)
        return int(self.raw.get("workers") or max(1, (os.cpu_count() or 2) // 2))

    @property
    def layout(self) -> Layout:
        root = os.environ.get("WAKEWORD_DATA") or self.raw.get("data_root") or REPO_ROOT / "data"
        return Layout(Path(root).resolve(), self.name)

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    return Config(_read(path), path)

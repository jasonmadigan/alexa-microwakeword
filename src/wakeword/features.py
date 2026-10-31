"""augmented spectrogram features in microwakeword's ragged mmap layout."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .catalogue import SPLITS
from .config import Config

log = logging.getLogger(__name__)

# feature set name per clip kind; the training config refers to these
FEATURE_SETS = {"positive": "positive", "negative": "hard_negative"}


def background_dirs(cfg: Config) -> tuple[list[str], list[str]]:
    root = cfg.layout.backgrounds
    noise = [str(root / d) for d in ("fma_16k", "audioset_16k") if any((root / d).glob("*.wav"))]
    rirs = [str(root / "mit_rirs")] if any((root / "mit_rirs").glob("*.wav")) else []
    return noise, rirs


def make_augmenter(aug: dict, noise: list[str], rirs: list[str]):
    from microwakeword.audio.augmentation import Augmentation

    return Augmentation(
        augmentation_duration_s=aug["duration_s"],
        augmentation_probabilities=aug["probabilities"],
        impulse_paths=rirs,
        background_paths=noise,
        background_min_snr_db=aug["background_min_snr_db"],
        background_max_snr_db=aug["background_max_snr_db"],
        min_jitter_s=aug["min_jitter_s"],
        max_jitter_s=aug["max_jitter_s"],
    )


class FileClips:
    """stands in for microwakeword's datasets-backed Clips in SpectrogramGeneration."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    @staticmethod
    def _load(path: str) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32")
        if sr != 16000:
            raise ValueError(f"{path}: expected 16 khz, got {sr}")
        return audio

    def audio_generator(self, split: str | None = None, repeat: int = 1):
        for _ in range(repeat):
            for path in self.paths:
                yield self._load(path)

    def get_random_clip(self) -> np.ndarray:
        return self._load(random.choice(self.paths))

    def random_audio_generator(self, max_clips: float = math.inf):
        while max_clips > 0:
            max_clips -= 1
            yield self.get_random_clip()


def as_uint16(spectrograms):
    # raw microfrontend values; exact, half the size of float32, and microwakeword
    # rescales uint16 sets on load
    for s in spectrograms:
        yield np.rint(s * 25.6).astype(np.uint16)


def stable_seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _shard(job: dict) -> int:
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    random.seed(job["seed"])
    np.random.seed(job["seed"])
    out = Path(job["out"])
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generator = SpectrogramGeneration(
        clips=FileClips(job["paths"]),
        augmenter=make_augmenter(job["augmentation"], job["noise"], job["rirs"]),
        step_ms=10,
        slide_frames=job["slide_frames"],
    )
    RaggedMmap.from_generator(
        out_dir=str(out),
        sample_generator=as_uint16(generator.spectrogram_generator(repeat=job["repeat"])),
        batch_size=100,
        verbose=False,
    )
    (out / ".done").write_text(job["digest"] + "\n")
    return len(job["paths"])


def inputs_digest(paths: list[str], spec: dict) -> str:
    return hashlib.sha256(json.dumps([paths, spec], sort_keys=True).encode()).hexdigest()[:16]


def build(cfg: Config, manifest: list[dict]) -> None:
    noise, rirs = background_dirs(cfg)
    if not noise or not rirs:
        raise RuntimeError("background audio missing; run fetch first")
    feats = cfg["features"]
    jobs = []
    current: set[Path] = set()
    for kind, set_name in FEATURE_SETS.items():
        for split in SPLITS:
            paths = sorted(r["path"] for r in manifest if r["kind"] == kind and r["split"] == split)
            if not paths:
                continue
            n_shards = min(len(paths), cfg.workers * 2)
            for i in range(n_shards):
                out = cfg.layout.features / set_name / split / f"{set_name}_{i:03d}_mmap"
                current.add(out)
                part = paths[i::n_shards]
                spec = {"augmentation": cfg["augmentation"], "repeat": feats["repeat"][split], "slide": feats["slide_frames"][split], "seed": cfg.seed}
                digest = inputs_digest(part, spec)
                if (out / ".done").exists() and (out / ".done").read_text().strip() == digest:
                    continue
                jobs.append(
                    {
                        "digest": digest,
                        "paths": part,
                        "out": str(out),
                        "augmentation": cfg["augmentation"],
                        "noise": noise,
                        "rirs": rirs,
                        "slide_frames": feats["slide_frames"][split],
                        "repeat": feats["repeat"][split],
                        "seed": stable_seed(cfg.seed, kind, split, i),
                    }
                )
    # shards from an older layout would otherwise still be globbed in by training
    for stale in cfg.layout.features.glob("*/*/*_mmap"):
        if stale not in current:
            shutil.rmtree(stale)
    log.info("feature shards to build: %d", len(jobs))
    with ProcessPoolExecutor(cfg.workers) as pool:
        for n, _ in enumerate(pool.map(_shard, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                log.info("feature shards %d/%d", n, len(jobs))

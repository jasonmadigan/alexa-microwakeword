"""downloads: piper voices, negative feature sets, background audio, baseline models."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from .audio import decode_bytes, load_audio, save_wav
from .config import Config

log = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path}: sha256 {actual}, expected {expected}")


def _done(marker: Path, value: str) -> bool:
    return marker.exists() and marker.read_text().strip() == value


def _mark(marker: Path, value: str) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(value + "\n")


def voices_index(cfg: Config) -> dict:
    pin = cfg["pins"]["piper_voices"]
    path = hf_hub_download(pin["repo"], "voices.json", revision=pin["revision"], local_dir=cfg.layout.voices)
    return json.loads(Path(path).read_text())


def catalogue_voices(cfg: Config) -> list[str]:
    return list(yaml.safe_load(cfg.resolve(cfg["voices"]).read_text())["voices"])


def voice_files(cfg: Config, index: dict, voice: str) -> tuple[Path, Path]:
    files = [f for f in index[voice]["files"] if f.endswith((".onnx", ".onnx.json"))]
    model = next(f for f in files if f.endswith(".onnx"))
    return cfg.layout.voices / model, cfg.layout.voices / f"{model}.json"


def fetch_voices(cfg: Config) -> None:
    pin = cfg["pins"]["piper_voices"]
    index = voices_index(cfg)
    for voice in catalogue_voices(cfg):
        if voice not in index:
            raise KeyError(f"{voice} not in piper-voices@{pin['revision']}")
        for f in index[voice]["files"]:
            if f.endswith((".onnx", ".onnx.json")):
                hf_hub_download(pin["repo"], f, revision=pin["revision"], local_dir=cfg.layout.voices)
        log.info("voice %s ready", voice)


def fetch_negatives(cfg: Config) -> None:
    pin = cfg["pins"]["negatives"]
    for name, digest in cfg["fetch"]["negatives"].items():
        marker = cfg.layout.negatives / f".{name}.done"
        if _done(marker, digest):
            continue
        zpath = Path(
            hf_hub_download(
                pin["repo"], f"{name}.zip", repo_type="dataset", revision=pin["revision"],
                local_dir=cfg.layout.downloads / "negatives",
            )
        )
        _verify(zpath, digest)
        target = cfg.layout.negatives / name
        if target.exists():
            shutil.rmtree(target)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cfg.layout.negatives)
        if not target.is_dir():
            raise RuntimeError(f"{zpath} did not contain {name}/")
        zpath.unlink()
        _mark(marker, digest)
        log.info("negative set %s ready", name)


def _convert(job: tuple[str, str]) -> None:
    src, dst = job
    save_wav(dst, load_audio(src))


def _convert_all(jobs: list[tuple[str, str]], workers: int) -> None:
    jobs = [j for j in jobs if not Path(j[1]).exists()]
    if not jobs:
        return
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(_convert, jobs, chunksize=32):
            pass


def _audioset_rows(job: tuple[str, int, str]) -> int:
    path, group, out_dir = job
    table = pq.ParquetFile(path).read_row_group(group, columns=["video_id", "audio"])
    written = 0
    for vid, audio in zip(table.column("video_id").to_pylist(), table.column("audio").to_pylist()):
        dst = Path(out_dir) / f"{vid}.wav"
        if dst.exists() or not audio or not audio.get("bytes"):
            continue
        try:
            save_wav(dst, decode_bytes(audio["bytes"]))
            written += 1
        except Exception as exc:  # a few audioset clips are corrupt upstream
            log.warning("skipping audioset %s: %s", vid, exc)
    return written


def fetch_backgrounds(cfg: Config) -> None:
    lay = cfg.layout
    pins = cfg["pins"]

    rir_dir = lay.backgrounds / "mit_rirs"
    if not _done(lay.backgrounds / ".mit_rirs.done", pins["rirs"]["revision"]):
        src = Path(
            snapshot_download(
                pins["rirs"]["repo"], repo_type="dataset", revision=pins["rirs"]["revision"],
                allow_patterns=["16khz/*.wav"], local_dir=lay.downloads / "rirs",
            )
        )
        _convert_all([(str(p), str(rir_dir / p.name)) for p in sorted((src / "16khz").glob("*.wav"))], cfg.workers)
        _mark(lay.backgrounds / ".mit_rirs.done", pins["rirs"]["revision"])
        log.info("rirs ready: %d", len(list(rir_dir.glob("*.wav"))))

    fma_dir = lay.backgrounds / "fma_16k"
    if not _done(lay.backgrounds / ".fma.done", pins["fma"]["revision"]):
        zpath = Path(
            hf_hub_download(
                pins["fma"]["repo"], "fma_xs.zip", repo_type="dataset", revision=pins["fma"]["revision"],
                local_dir=lay.downloads / "fma",
            )
        )
        extract = lay.downloads / "fma" / "extracted"
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract)
        mp3s = sorted(extract.rglob("*.mp3"))
        _convert_all([(str(p), str(fma_dir / f"{p.stem}.wav")) for p in mp3s], cfg.workers)
        shutil.rmtree(extract)
        _mark(lay.backgrounds / ".fma.done", pins["fma"]["revision"])
        log.info("fma ready: %d", len(list(fma_dir.glob("*.wav"))))

    audioset_dir = lay.backgrounds / "audioset_16k"
    for shard in cfg["fetch"].get("audioset_shards") or []:
        marker = lay.backgrounds / f".audioset-{Path(shard).stem}.done"
        if _done(marker, pins["audioset"]["revision"]):
            continue
        path = hf_hub_download(
            pins["audioset"]["repo"], shard, repo_type="dataset", revision=pins["audioset"]["revision"],
            local_dir=lay.downloads / "audioset",
        )
        audioset_dir.mkdir(parents=True, exist_ok=True)
        groups = pq.ParquetFile(path).num_row_groups
        with ProcessPoolExecutor(cfg.workers) as pool:
            written = sum(pool.map(_audioset_rows, [(path, g, str(audioset_dir)) for g in range(groups)]))
        Path(path).unlink()
        _mark(marker, pins["audioset"]["revision"])
        log.info("audioset %s ready: %d clips", shard, written)


def fetch_baselines(cfg: Config) -> None:
    pin = cfg["pins"]["esphome_models"]
    base = f"https://raw.githubusercontent.com/{pin['repo']}/{pin['revision']}/models/v2"
    out = cfg.layout.baselines
    out.mkdir(parents=True, exist_ok=True)
    for name, digests in cfg["fetch"]["baselines"].items():
        for ext in ("tflite", "json"):
            dst = out / f"{name}.{ext}"
            if dst.exists() and sha256(dst) == digests[ext]:
                continue
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(f"{base}/{name}.{ext}", timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            _verify(tmp, digests[ext])
            tmp.replace(dst)
    log.info("baselines ready in %s", out)


def fetch_all(cfg: Config) -> None:
    fetch_baselines(cfg)
    fetch_voices(cfg)
    fetch_negatives(cfg)
    fetch_backgrounds(cfg)

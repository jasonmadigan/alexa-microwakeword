from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="HQ")
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """any audio file as float32 mono 16 khz; formats libsndfile lacks go through ffmpeg."""
    path = Path(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        if not shutil.which("ffmpeg"):
            raise
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "wav", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            check=True,
            capture_output=True,
        ).stdout
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return to_mono_16k(audio, sr)


def decode_bytes(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr)


def save_wav(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), (pcm * 32767).astype(np.int16), SAMPLE_RATE, subtype="PCM_16")


def trim_silence(audio: np.ndarray, threshold_db: float, margin_s: float = 0.02) -> np.ndarray:
    """drop leading and trailing audio more than threshold_db below the peak frame."""
    frame = int(0.01 * SAMPLE_RATE)
    if audio.size < frame:
        return audio
    n = audio.size // frame
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms)
    loud = np.flatnonzero(level >= level.max() - threshold_db)
    margin = int(margin_s * SAMPLE_RATE)
    start = max(0, loud[0] * frame - margin)
    end = min(audio.size, (loud[-1] + 1) * frame + margin)
    return audio[start:end]


def normalise_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio if top == 0 else (audio * (peak / top)).astype(np.float32)

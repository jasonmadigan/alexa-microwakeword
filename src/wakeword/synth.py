"""piper synthesis of the clip plan, one single-threaded onnx session per worker and voice."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig

from .audio import normalise_peak, save_wav, to_mono_16k, trim_silence

log = logging.getLogger(__name__)

_VOICES: dict[str, PiperVoice] = {}
_STATE: dict = {}


def load_voice(model: Path, threads: int = 1) -> PiperVoice:
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    config = PiperConfig.from_dict(json.loads(Path(f"{model}.json").read_text()))
    session = onnxruntime.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
    return PiperVoice(session=session, config=config)


def ipa_to_ids(voice: PiperVoice, ipa: str) -> list[int]:
    phonemes = list(unicodedata.normalize("NFD", ipa))
    missing = sorted({p for p in phonemes if p not in voice.config.phoneme_id_map})
    if missing:
        raise ValueError(f"phonemes {missing} not in the voice's phoneme map")
    return voice.phonemes_to_ids(phonemes)


def text_to_ids(voice: PiperVoice, text: str) -> list[int]:
    phonemes = [p for sentence in voice.phonemize(text) for p in sentence]
    return voice.phonemes_to_ids(phonemes)


def synthesise(voice: PiperVoice, row: dict) -> np.ndarray:
    ids = ipa_to_ids(voice, row["ipa"]) if row.get("ipa") else text_to_ids(voice, row["text"])
    speaker = row["speaker_index"] if voice.config.num_speakers > 1 else None
    audio = voice.phoneme_ids_to_audio(
        ids,
        SynthesisConfig(
            speaker_id=speaker,
            length_scale=row["length_scale"],
            noise_scale=row["noise_scale"],
            noise_w_scale=row["noise_w_scale"],
        ),
    )
    if isinstance(audio, tuple):
        audio = audio[0]
    return to_mono_16k(np.asarray(audio, dtype=np.float32).reshape(-1), voice.config.sample_rate)


def _init(voice_paths: dict[str, str], trim_db: float) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _STATE["voice_paths"] = voice_paths
    _STATE["trim_db"] = trim_db


def _voice(name: str) -> PiperVoice:
    voice = _VOICES.get(name)
    if voice is None:
        # keep at most a couple of voices resident per worker
        if len(_VOICES) >= 2:
            _VOICES.clear()
        voice = _VOICES[name] = load_voice(Path(_STATE["voice_paths"][name]))
    return voice


def _run(batch: list[dict]) -> list[dict]:
    out = []
    for row in batch:
        path = Path(row["path"])
        if not path.exists():
            audio = synthesise(_voice(row["voice"]), row)
            audio = normalise_peak(trim_silence(audio, _STATE["trim_db"]))
            save_wav(path, audio)
            duration = audio.size / 16000
        else:
            duration = sf.info(str(path)).duration
        out.append({**row, "duration_s": round(duration, 4)})
    return out


SYNTH_FIELDS = ("voice", "speaker_index", "text", "ipa", "length_scale", "noise_scale", "noise_w_scale")


def clip_name(row: dict) -> str:
    # content-addressed: a changed plan can never pick up audio synthesised for another row
    digest = hashlib.sha256(json.dumps([row[k] for k in SYNTH_FIELDS], ensure_ascii=False).encode()).hexdigest()
    return f"{row['id']}-{digest[:12]}.wav"


def run(plan: list[dict], clips_dir: Path, voice_paths: dict[str, Path], workers: int, trim_db: float) -> list[dict]:
    for row in plan:
        row["path"] = str(clips_dir / row["split"] / row["kind"] / clip_name(row))
    rows = sorted(plan, key=lambda r: (r["voice"], r["id"]))
    batches = []
    for _, group in groupby(rows, key=lambda r: r["voice"]):
        group = list(group)
        for i in range(0, len(group), 64):
            batches.append(group[i : i + 64])
    done: list[dict] = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=({k: str(v) for k, v in voice_paths.items()}, trim_db)) as pool:
        # consecutive batches share a voice; chunking keeps them on one worker
        for i, result in enumerate(pool.map(_run, batches, chunksize=4)):
            done.extend(result)
            if (i + 1) % 50 == 0 or i + 1 == len(batches):
                log.info("synthesised %d/%d clips", len(done), len(rows))
    return sorted(done, key=lambda r: r["id"])

"""held-out evaluation with the firmware's detection logic, per (gender, region) group."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, load_audio
from .config import Config
from .features import background_dirs, stable_seed
from .streaming import StreamingModel, quantize_cutoff, quantize_features, sweep, vad_gate

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm"}

_W: dict = {}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    cutoff: float | None = None
    window: int | None = None


def household_clips(root: Path) -> list[dict]:
    """household/<person>/*.wav are wake word recordings; household/ambient/ holds background audio."""
    rows = []
    if not root.is_dir():
        return rows
    for person in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "ambient"):
        for f in sorted(person.rglob("*")):
            if f.suffix.lower() in AUDIO_SUFFIXES:
                rows.append({"id": f"household-{person.name}-{f.stem}", "path": str(f), "group": f"household/{person.name}", "kind": "positive"})
    return rows


def _eval_augmenter(cond: dict, noise: list[str], rirs: list[str]):
    import audiomentations as A

    steps = []
    if cond.get("rir"):
        steps.append(A.ApplyImpulseResponse(ir_path=rirs, p=1.0, leave_length_unchanged=True))
    if cond.get("background_snr_db"):
        lo, hi = cond["background_snr_db"]
        steps.append(A.AddBackgroundNoise(sounds_path=noise, min_snr_db=lo, max_snr_db=hi, p=1.0))
    lo, hi = cond["gain_db"]
    steps.append(A.Gain(min_gain_db=lo, max_gain_db=hi, p=1.0))
    steps.append(A.Normalize(apply_to="only_too_loud_sounds", p=1.0))
    return A.Compose(steps)


def _init(models: dict[str, str], vad: str | None, conditions: dict, noise: list[str], rirs: list[str], ev: dict):
    _W["models"] = {k: StreamingModel(v) for k, v in models.items()}
    _W["vad"] = StreamingModel(vad) if vad else None
    _W["augmenters"] = {k: _eval_augmenter(c, noise, rirs) for k, c in conditions.items()}
    _W["ev"] = ev


def _features(audio: np.ndarray) -> np.ndarray:
    from microwakeword.audio.audio_utils import generate_features_for_clip

    return quantize_features(generate_features_for_clip(audio.astype(np.float32), step_ms=10))


def _infer(features: np.ndarray) -> dict:
    out = {name: m.probabilities(features) for name, m in _W["models"].items()}
    if _W["vad"] is not None:
        out["_vad"] = _W["vad"].probabilities(features)
    return out


def _clip_job(job: dict) -> dict:
    ev = _W["ev"]
    seed = stable_seed(job["id"], job["condition"])
    random.seed(seed)
    np.random.seed(seed)
    clip = load_audio(job["path"])
    pre = int(ev["preroll_s"] * SAMPLE_RATE)
    post = int(ev["postroll_s"] * SAMPLE_RATE)
    # low noise floor so the frontend's noise estimate behaves as it would on a quiet mic
    stream = np.random.normal(0, 10 ** (-70 / 20), pre + clip.size + post).astype(np.float32)
    stream[pre : pre + clip.size] += clip
    stream = _W["augmenters"][job["condition"]](samples=stream, sample_rate=SAMPLE_RATE)
    feats = _features(stream)
    stride = next(iter(_W["models"].values())).stride
    slices_per_inference = stride
    start = (pre // 160) // slices_per_inference
    end = ((pre + clip.size + post) // 160) // slices_per_inference
    return {"id": job["id"], "condition": job["condition"], "start": start, "end": end, "probs": _infer(feats)}


def _ambient_job(job: dict) -> dict:
    from mmap_ninja.ragged import RaggedMmap

    cache = _W.setdefault("mmaps", {})
    mm = cache.get(job["mmap"]) or cache.setdefault(job["mmap"], RaggedMmap(job["mmap"]))
    feats = quantize_features(np.asarray(mm[job["index"]]))
    return {"id": f"{job['mmap']}#{job['index']}", "slices": int(feats.shape[0]), "probs": _infer(feats)}


def _ambient_tracks(cfg: Config) -> list[dict]:
    jobs = []
    for name in cfg["evaluation"]["ambient_sets"]:
        root = cfg.layout.negatives / name / "testing_ambient"
        mmaps = sorted(p for p in root.glob("**/*_mmap") if p.is_dir())
        if not mmaps:
            raise FileNotFoundError(f"no testing_ambient mmaps under {root}")
        from mmap_ninja.ragged import RaggedMmap

        for mm in mmaps:
            jobs.extend({"mmap": str(mm), "index": i} for i in range(len(RaggedMmap(str(mm)))))
    return jobs


def _household_ambient(cfg: Config) -> list[Path]:
    root = cfg.layout.household / "ambient"
    return sorted(f for f in root.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES) if root.is_dir() else []


def _household_ambient_job(job: dict) -> dict:
    feats = _features(load_audio(job["path"]))
    return {"id": job["path"], "slices": int(feats.shape[0]), "probs": _infer(feats)}


def _grid(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[float]]:
    c = cfg["evaluation"]["cutoffs"]
    cutoffs = [round(x, 4) for x in np.arange(c["start"], c["stop"] + 1e-9, c["step"])]
    return np.array([quantize_cutoff(x) for x in cutoffs], np.int64), np.array(cfg["evaluation"]["windows"], np.int64), cutoffs


def _concat(results: list[dict], model: str, vad_ok: list[np.ndarray]):
    probs = [r["probs"][model].astype(np.int64) for r in results]
    offsets = np.zeros(len(probs) + 1, np.int64)
    offsets[1:] = np.cumsum([p.size for p in probs])
    return np.concatenate(probs) if probs else np.zeros(0, np.int64), np.concatenate(vad_ok) if vad_ok else np.zeros(0, np.bool_), offsets


def run(cfg: Config, candidate: Path | None, manifest: list[dict]) -> dict:
    ev = cfg["evaluation"]
    lay = cfg.layout
    vad_manifest = json.loads((lay.baselines / "vad.json").read_text())
    vad_path = str(lay.baselines / "vad.tflite") if ev.get("vad_gate", True) else None

    specs = [ModelSpec(b["name"], lay.baselines / f"{b['model']}.tflite", b["probability_cutoff"], b["sliding_window_size"]) for b in ev["baselines"]]
    if candidate is not None:
        specs.insert(0, ModelSpec("candidate", candidate))
    # baselines share one tflite at different operating points; infer each file once
    model_paths = {str(s.path): str(s.path) for s in specs}
    noise, rirs = background_dirs(cfg)

    clips = [r for r in manifest if r["split"] == "testing"]
    for r in clips:
        # positives report per speaker group, hard negatives per confusable phrase
        r["group"] = f"{r['gender']}/{r['region']}" if r["kind"] == "positive" else f"phrase/{r['text'] or r['ipa']}"
    house = household_clips(lay.household)
    jobs = [{"id": r["id"], "path": r["path"], "condition": c} for r in clips + house for c in ev["conditions"]]
    meta = {r["id"]: r for r in clips + house}
    amb_jobs = _ambient_tracks(cfg)
    house_amb = [{"path": str(p)} for p in _household_ambient(cfg)]

    init = (model_paths, vad_path, ev["conditions"], noise, rirs, ev)
    with ProcessPoolExecutor(cfg.workers, initializer=_init, initargs=init) as pool:
        log.info("evaluating %d clip streams, %d ambient tracks", len(jobs), len(amb_jobs))
        clip_results = list(pool.map(_clip_job, jobs, chunksize=16))
        amb_results = list(pool.map(_ambient_job, amb_jobs, chunksize=4))
        house_amb_results = list(pool.map(_household_ambient_job, house_amb))

    vad_cut, vad_win = vad_manifest["micro"]["probability_cutoff"], vad_manifest["micro"]["sliding_window_size"]

    def gates(results):
        first = next(iter(model_paths))
        return [vad_gate(r["probs"].get("_vad"), r["probs"][first].size, vad_cut, vad_win) for r in results]

    clip_gates, amb_gates, house_amb_gates = gates(clip_results), gates(amb_results), gates(house_amb_results)
    cutoffs_u8, windows, cutoffs = _grid(cfg)
    stride = StreamingModel(next(iter(model_paths.values()))).stride
    amb_hours = sum(r["slices"] for r in amb_results) * 0.01 / 3600
    house_hours = sum(r["slices"] for r in house_amb_results) * 0.01 / 3600

    report = {
        "ambient_hours": amb_hours,
        "household_ambient_hours": house_hours,
        "target_faph": ev["target_faph"],
        "max_confusable_accept": ev.get("max_confusable_accept"),
        "models": {},
    }
    # every model is scored over the full grid; baselines also at their configured point
    for spec in specs:
        report["models"][spec.name] = _score(
            spec, clip_results, clip_gates, amb_results, amb_gates, house_amb_results, house_amb_gates,
            meta, cutoffs_u8, windows, cutoffs, stride, amb_hours, house_hours, ev,
        )
    return report


def _score(spec, clip_results, clip_gates, amb_results, amb_gates, house_amb, house_gates, meta, cu, wi, labels, stride, amb_hours, house_hours, ev) -> dict:
    key = str(spec.path)
    p, v, off = _concat(amb_results, key, amb_gates)
    zeros = np.zeros(len(amb_results), np.int64)
    _, amb_out = sweep(p, v, off, zeros, zeros, cu, wi, stride)
    faph = amb_out.sum(axis=0) / max(amb_hours, 1e-9)  # (windows, cutoffs)

    house_faph = None
    if house_amb:
        p, v, off = _concat(house_amb, key, house_gates)
        z = np.zeros(len(house_amb), np.int64)
        _, h_out = sweep(p, v, off, z, z, cu, wi, stride)
        house_faph = h_out.sum(axis=0) / max(house_hours, 1e-9)

    p, v, off = _concat(clip_results, key, clip_gates)
    starts = np.array([r["start"] for r in clip_results], np.int64)
    ends = np.array([r["end"] for r in clip_results], np.int64)
    inside, _ = sweep(p, v, off, starts, ends, cu, wi, stride)
    detected = inside > 0  # (streams, windows, cutoffs)

    by_group: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(clip_results):
        m = meta[r["id"]]
        by_group[(m["kind"], m["group"], r["condition"])].append(i)
    conditions = sorted({r["condition"] for r in clip_results})
    pos_groups = sorted({g for (k, g, _) in by_group if k == "positive" and not g.startswith("household/")})
    house_groups = sorted({g for (k, g, _) in by_group if k == "positive" and g.startswith("household/")})

    def rate(kind, group, cond):
        idx = by_group.get((kind, group, cond))
        return detected[idx].mean(axis=0) if idx else None

    worst = np.ones(faph.shape)
    mean = np.zeros(faph.shape)
    for g in pos_groups:
        rs = [rate("positive", g, c) for c in conditions]
        rs = [r for r in rs if r is not None]
        g_mean = np.mean(rs, axis=0)
        worst = np.minimum(worst, g_mean)
        mean += g_mean / len(pos_groups)

    if spec.cutoff is not None:
        wi_idx = int(np.flatnonzero(wi == spec.window)[0]) if spec.window in wi else None
        ci_idx = int(np.argmin(np.abs(np.array(labels) - spec.cutoff)))
        if wi_idx is None or abs(labels[ci_idx] - spec.cutoff) > 1e-6:
            raise ValueError(f"{spec.name}: ({spec.cutoff}, {spec.window}) not on the evaluation grid")
        configured = (wi_idx, ci_idx)
    else:
        configured = None

    hard_idx = [i for (k, _, _), idx in by_group.items() if k == "negative" for i in idx]
    confusable = detected[hard_idx].mean(axis=0) if hard_idx else np.zeros(faph.shape)

    viable = (faph <= ev["target_faph"]) & (confusable <= ev.get("max_confusable_accept", 1.0))
    if viable.any():
        score = np.where(viable, worst * 1e6 + mean * 1e3 - faph, -np.inf)
        best = np.unravel_index(np.argmax(score), score.shape)
    else:
        best = np.unravel_index(np.argmin(faph), faph.shape)

    def point(ix):
        wi_i, ci_i = ix
        groups = {}
        for g in pos_groups + house_groups:
            groups[g] = {c: _f(rate("positive", g, c), ix) for c in conditions}
            groups[g]["n"] = len(by_group.get(("positive", g, conditions[0]), []))
        hard = {}
        for (k, g, c), idx in by_group.items():
            if k == "negative":
                hard.setdefault(g, {})[c] = float(detected[idx][:, wi_i, ci_i].mean())
        return {
            "probability_cutoff": labels[ci_i],
            "sliding_window_size": int(wi[wi_i]),
            "ambient_faph": float(faph[ix]),
            "household_ambient_faph": None if house_faph is None else float(house_faph[ix]),
            "worst_group_recall": float(worst[ix]),
            "mean_group_recall": float(mean[ix]),
            "groups": groups,
            "groups_below_min_recall": [
                g for g in pos_groups + house_groups
                if min(v for c, v in groups[g].items() if c != "n" and v is not None) < ev["min_group_recall"]
            ],
            "hard_negative_accept_rate": hard,
            "confusable_accept_rate": float(confusable[ix]),
            "meets_target": bool(viable[ix]),
        }

    # coarse grid of operating points so the trade-off can be chosen by hand
    frontier = [
        {
            "probability_cutoff": labels[ci_i],
            "sliding_window_size": int(w),
            "ambient_faph": float(faph[wi_i, ci_i]),
            "worst_group_recall": float(worst[wi_i, ci_i]),
            "mean_group_recall": float(mean[wi_i, ci_i]),
            "confusable_accept_rate": float(confusable[wi_i, ci_i]),
        }
        for wi_i, w in enumerate(wi)
        for ci_i in range(len(labels))
        if round(labels[ci_i] * 100) % 5 == 0
    ]
    out = {"path": str(spec.path), "best": point(best), "frontier": frontier}
    if configured is not None:
        out["configured"] = point(configured)
    return out


def _f(arr, ix):
    return None if arr is None else float(arr[ix])

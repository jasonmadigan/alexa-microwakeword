"""on-device detection, reproduced from esphome's micro_wake_word (2026.10.1).

features are quantised the way the firmware does it, the streaming tflite model is
invoked every `stride` feature slices, and detections follow streaming_model.cpp:
a sliding-window sum of uint8 probabilities compared against int(cutoff * 255) *
window, a 100-slice cool-off after start and after every detection, and optional
gating on the vad model.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

MIN_SLICES_BEFORE_DETECTION = 100
FEATURE_SCALE = 0.0390625  # pymicro_features float = raw microfrontend uint16 * this


def quantize_features(features: np.ndarray) -> np.ndarray:
    """microfrontend features (uint16, or pymicro_features floats) to the firmware's int8 input."""
    if np.issubdtype(features.dtype, np.integer):
        raw = features.astype(np.int64)
    else:
        raw = np.rint(features.astype(np.float64) / FEATURE_SCALE).astype(np.int64)
    value = (raw * 256 + 333) // 666 - 128
    return np.clip(value, -128, 127).astype(np.int8)


def quantize_cutoff(cutoff: float) -> int:
    return int(cutoff * 255)


class StreamingModel:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.content = self.path.read_bytes()
        self.interpreter = self._fresh()
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        if inp["dtype"] != np.int8 or out["dtype"] != np.uint8:
            raise ValueError(f"{path}: expected int8 in, uint8 out; got {inp['dtype']}, {out['dtype']}")
        self.input_index = inp["index"]
        self.output_index = out["index"]
        self.stride = int(inp["shape"][1])
        self.input_shape = tuple(inp["shape"])
        self.input_quantization = inp["quantization"]

    def _fresh(self):
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_content=self.content)
        interpreter.allocate_tensors()
        return interpreter

    def probabilities(self, features_int8: np.ndarray) -> np.ndarray:
        """uint8 output per invocation from freshly loaded state, as after a device boot."""
        # reset_all_variables zeroes the streaming state instead of rerunning the
        # model's call_once initialiser, so a new interpreter is the only faithful reset
        self.interpreter = self._fresh()
        n = features_int8.shape[0] // self.stride
        out = np.empty(n, dtype=np.uint8)
        set_tensor = self.interpreter.set_tensor
        invoke = self.interpreter.invoke
        get_tensor = self.interpreter.get_tensor
        for k in range(n):
            chunk = features_int8[k * self.stride : (k + 1) * self.stride]
            set_tensor(self.input_index, chunk.reshape(self.input_shape))
            invoke()
            out[k] = get_tensor(self.output_index)[0][0]
        return out


@numba.njit(cache=True)
def _vad_states(vad_probs, cutoff, window):
    ring = np.zeros(window, np.int64)
    last = 0
    state = np.zeros(vad_probs.shape[0], np.bool_)
    for k in range(vad_probs.shape[0]):
        last = (last + 1) % window
        ring[last] = vad_probs[k]
        state[k] = ring.sum() > cutoff * window
    return state


@numba.njit(cache=True)
def _detect(probs, vad_ok, cutoff, window, stride):
    """inference indices where the firmware reports a (non-vad-blocked) detection."""
    ring = np.zeros(window, np.int64)
    last = 0
    ignore = -MIN_SLICES_BEFORE_DETECTION
    hits = np.zeros(probs.shape[0], np.bool_)
    for k in range(probs.shape[0]):
        # slices without an invoke see the previously stored probability
        for _ in range(stride - 1):
            if ring[last] < cutoff:
                ignore = min(ignore + 1, 0)
        last = (last + 1) % window
        ring[last] = probs[k]
        if ring[last] < cutoff:
            ignore = min(ignore + 1, 0)
        if ignore >= 0 and ring.sum() > cutoff * window:
            if vad_ok[k]:
                hits[k] = True
                ring[:] = 0
                ignore = -MIN_SLICES_BEFORE_DETECTION
    return hits


@numba.njit(cache=True)
def sweep(probs, vad_ok, offsets, region_start, region_end, cutoffs, windows, stride):
    """for every stream and (window, cutoff): detections inside and outside the stream's region.

    probs/vad_ok are concatenated streams split by offsets (len n_streams + 1); region
    bounds are inference indices relative to each stream, region_end exclusive.
    returns two int32 arrays shaped (n_streams, len(windows), len(cutoffs)).
    """
    n = offsets.shape[0] - 1
    inside = np.zeros((n, windows.shape[0], cutoffs.shape[0]), np.int32)
    outside = np.zeros((n, windows.shape[0], cutoffs.shape[0]), np.int32)
    for s in range(n):
        p = probs[offsets[s] : offsets[s + 1]]
        v = vad_ok[offsets[s] : offsets[s + 1]]
        for wi in range(windows.shape[0]):
            for ci in range(cutoffs.shape[0]):
                hits = _detect(p, v, cutoffs[ci], windows[wi], stride)
                for k in np.flatnonzero(hits):
                    if region_start[s] <= k < region_end[s]:
                        inside[s, wi, ci] += 1
                    else:
                        outside[s, wi, ci] += 1
    return inside, outside


def vad_gate(vad_probs: np.ndarray | None, n: int, cutoff: float, window: int) -> np.ndarray:
    if vad_probs is None:
        return np.ones(n, np.bool_)
    if vad_probs.shape[0] < n:
        vad_probs = np.concatenate([vad_probs, np.zeros(n - vad_probs.shape[0], np.uint8)])
    return _vad_states(vad_probs[:n].astype(np.int64), quantize_cutoff(cutoff), window)

import numpy as np

from wakeword.streaming import MIN_SLICES_BEFORE_DETECTION, _detect, quantize_cutoff, quantize_features, sweep, vad_gate

STRIDE = 3
WARMUP = -(-MIN_SLICES_BEFORE_DETECTION // STRIDE)  # inferences before the cool-off can clear


def detect(probs, cutoff=0.9, window=5, vad=None):
    probs = np.asarray(probs, np.int64)
    vad = np.ones(probs.size, np.bool_) if vad is None else np.asarray(vad, np.bool_)
    return np.flatnonzero(_detect(probs, vad, quantize_cutoff(cutoff), window, STRIDE))


def test_feature_quantisation_matches_firmware():
    raw = np.array([0, 1, 333, 666, 665, 1000, 65535], np.uint16)
    # firmware: ((value * 256) + 333) / 666 - 128, clamped
    expected = [((int(v) * 256 + 333) // 666) - 128 for v in raw]
    expected = np.clip(expected, -128, 127)
    assert quantize_features(raw).tolist() == expected.tolist()
    # pymicro_features floats are raw values * 0.0390625
    assert quantize_features(raw.astype(np.float32) * 0.0390625).tolist() == expected.tolist()


def test_cutoff_truncates_like_codegen():
    assert quantize_cutoff(0.97) == 247
    assert quantize_cutoff(0.9) == 229


def test_no_detection_during_initial_cool_off():
    probs = [255] * 40
    hits = detect(probs)
    # first detection only once 100 below-cutoff slices have accrued, which never happens here
    assert hits.size == 0


def test_detection_after_quiet_warmup():
    probs = [0] * (WARMUP + 2) + [255] * 10
    hits = detect(probs, window=5)
    assert hits.tolist() == [WARMUP + 2 + 4]  # fifth high inference fills the window


def test_mean_must_strictly_exceed_cutoff():
    c = quantize_cutoff(0.9)
    at = [0] * (WARMUP + 2) + [c] * 20
    above = [0] * (WARMUP + 2) + [c + 1] * 20
    assert detect(at).size == 0
    assert detect(above).size == 1


def test_reset_and_cool_off_after_detection():
    quiet = [0] * (WARMUP + 2)
    probs = quiet + [255] * 5 + [255] * 60 + quiet + [255] * 5
    hits = detect(probs)
    # one detection per burst: continuous high probability never clears the cool-off
    assert len(hits) == 2


def test_vad_block_does_not_reset():
    probs = [0] * (WARMUP + 2) + [255] * 8
    vad = [True] * len(probs)
    first = WARMUP + 2 + 4
    vad[first] = False
    hits = detect(probs, vad=vad)
    assert hits.tolist() == [first + 1]


def test_vad_gate_uses_window_mean():
    gate = vad_gate(np.array([0, 0, 200, 200, 200], np.uint8), 5, 0.5, 2)
    # window sum must exceed int(0.5 * 255) * 2
    assert gate.tolist() == [False, False, False, True, True]


def test_sweep_counts_inside_and_outside():
    quiet = [0] * (WARMUP + 2)
    a = quiet + [255] * 5 + quiet
    b = quiet + quiet
    probs = np.array(a + b, np.int64)
    offsets = np.array([0, len(a), len(a) + len(b)], np.int64)
    vad = np.ones(probs.size, np.bool_)
    start = np.array([WARMUP + 2, 0], np.int64)
    end = np.array([WARMUP + 2 + 5, 0], np.int64)
    inside, outside = sweep(probs, vad, offsets, start, end, np.array([quantize_cutoff(0.9)]), np.array([5]), STRIDE)
    assert inside[:, 0, 0].tolist() == [1, 0]
    assert outside[:, 0, 0].tolist() == [0, 0]

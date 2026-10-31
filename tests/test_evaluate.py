import numpy as np

from wakeword.evaluate import ModelSpec, _grid, _score
from wakeword.report import markdown

STRIDE = 3
QUIET = [0] * 40


def _stream(burst: int, level: int = 255) -> np.ndarray:
    return np.array(QUIET + [level] * burst + QUIET, np.uint8)


def _clip(id_, probs, condition="clean"):
    return {"id": id_, "condition": condition, "start": len(QUIET), "end": len(QUIET) + 20, "probs": {"m.tflite": probs}}


class _Cfg:
    def __init__(self):
        self.raw = {"evaluation": {"cutoffs": {"start": 0.5, "stop": 0.99, "step": 0.01}, "windows": [3, 5]}}

    def __getitem__(self, k):
        return self.raw[k]


def test_score_picks_point_and_renders():
    cutoffs_u8, windows, labels = _grid(_Cfg())
    meta = {
        "a": {"kind": "positive", "group": "female/ireland"},
        "b": {"kind": "positive", "group": "male/scotland"},
        "c": {"kind": "negative", "group": "phrase/Alexis"},
        "h": {"kind": "positive", "group": "household/someone"},
    }
    clips = [_clip("a", _stream(6)), _clip("b", _stream(6, 200)), _clip("c", _stream(2)), _clip("h", _stream(6))]
    gates = [np.ones(c["probs"]["m.tflite"].size, np.bool_) for c in clips]
    # one hour of ambient with one spike at ~0.7, so only cutoffs >= 0.71 stay under target
    amb_probs = np.zeros(120000, np.uint8)
    amb_probs[60000:60006] = 180
    amb = [{"id": "x", "slices": 360000, "probs": {"m.tflite": amb_probs}}]
    amb_gates = [np.ones(amb_probs.size, np.bool_)]
    ev = {"target_faph": 0.5, "min_group_recall": 0.9}
    result = _score(ModelSpec("m", "m.tflite"), clips, gates, amb, amb_gates, [], [], meta,
                    cutoffs_u8, windows, labels, STRIDE, 1.0, 0.0, ev)
    best = result["best"]
    assert best["meets_target"]
    assert best["ambient_faph"] == 0.0
    # 200/255 ~ 0.78: the quieter group only passes below that cutoff
    assert 0.71 <= best["probability_cutoff"] < 0.79
    assert best["hard_negative_accept_rate"]["phrase/Alexis"]["clean"] == 0.0
    assert best["groups"]["female/ireland"]["clean"] == 1.0
    assert best["groups"]["male/scotland"]["clean"] == 1.0
    assert best["groups_below_min_recall"] == []
    md = markdown({"ambient_hours": 1.0, "target_faph": 0.5, "models": {"candidate": result}})
    assert "| female/ireland | 1 |" in md and "household/someone" in md and "Alexis" in md
    assert best["confusable_accept_rate"] == 0.0
    assert {p["sliding_window_size"] for p in result["frontier"]} == {3, 5}
    assert all(round(p["probability_cutoff"] * 100) % 5 == 0 for p in result["frontier"])


def test_confusable_limit_moves_the_operating_point():
    cutoffs_u8, windows, labels = _grid(_Cfg())
    meta = {"a": {"kind": "positive", "group": "female/ireland"}, "c": {"kind": "negative", "group": "phrase/relax a bit"}}
    # the confusable peaks at ~0.75, the wake word at 1.0
    clips = [_clip("a", _stream(6)), _clip("c", _stream(6, 190))]
    gates = [np.ones(c["probs"]["m.tflite"].size, np.bool_) for c in clips]
    amb = [{"id": "x", "slices": 360000, "probs": {"m.tflite": np.zeros(120000, np.uint8)}}]
    amb_gates = [np.ones(120000, np.bool_)]
    args = (clips, gates, amb, amb_gates, [], [], meta, cutoffs_u8, windows, labels, STRIDE, 1.0, 0.0)
    loose = _score(ModelSpec("m", "m.tflite"), *args, {"target_faph": 0.5, "min_group_recall": 0.9})
    strict = _score(ModelSpec("m", "m.tflite"), *args, {"target_faph": 0.5, "min_group_recall": 0.9, "max_confusable_accept": 0.0})
    assert loose["best"]["confusable_accept_rate"] == 1.0
    assert strict["best"]["confusable_accept_rate"] == 0.0
    assert strict["best"]["probability_cutoff"] >= 0.75
    assert strict["best"]["groups"]["female/ireland"]["clean"] == 1.0

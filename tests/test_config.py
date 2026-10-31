import numpy as np

from wakeword.audio import SAMPLE_RATE, trim_silence
from wakeword.config import REPO_ROOT, deep_merge, load_config
from wakeword.export import manifest
from wakeword.train import training_parameters


def test_overlay_null_removes_keys():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": None}, "b": None})
    assert merged == {"a": {"x": 1}}


def test_smoke_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKEWORD_DATA", str(tmp_path))
    cfg = load_config(REPO_ROOT / "config" / "smoke.yaml")
    assert cfg.name == "smoke"
    assert set(cfg["fetch"]["negatives"]) == {"dinner_party_eval"}
    assert cfg.layout.work == tmp_path / "work" / "smoke"
    assert cfg["model"]["architecture"] == "mixednet"


def test_training_parameters_reference_existing_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKEWORD_DATA", str(tmp_path))
    for name in ("alexa.yaml", "smoke.yaml"):
        params = training_parameters(load_config(REPO_ROOT / "config" / name))
        dirs = {f["features_dir"].rsplit("/", 1)[-1]: f for f in params["features"]}
        assert dirs["positive"]["truth"] is True
        assert dirs["hard_negative"]["truth"] is False
        assert dirs["dinner_party_eval"]["sampling_weight"] == 0.0
        assert params["window_step_ms"] == 10
        assert len(params["training_steps"]) == len(params["learning_rates"])


def test_manifest_shape():
    cfg = load_config(REPO_ROOT / "config" / "alexa.yaml")
    m = manifest(cfg, "x.tflite", 0.81, 6)
    assert m["version"] == 2 and m["type"] == "micro"
    assert m["micro"]["feature_step_size"] == 10
    assert m["micro"]["probability_cutoff"] == 0.81 and m["micro"]["sliding_window_size"] == 6


def test_trim_silence_keeps_the_word():
    word = 0.5 * np.sin(np.linspace(0, 200 * np.pi, SAMPLE_RATE // 2)).astype(np.float32)
    padded = np.concatenate([np.zeros(SAMPLE_RATE, np.float32), word, np.zeros(SAMPLE_RATE, np.float32)])
    trimmed = trim_silence(padded, 40, margin_s=0.02)
    assert word.size <= trimmed.size <= word.size + int(0.06 * SAMPLE_RATE)


def test_clip_names_follow_content():
    from wakeword.synth import clip_name

    row = {"id": "positive-training-0000001", "voice": "v", "speaker_index": 3, "text": "Alexa", "ipa": None,
           "length_scale": 1.0, "noise_scale": 0.6, "noise_w_scale": 0.8}
    assert clip_name(row) == clip_name(dict(row))
    assert clip_name(row) != clip_name({**row, "speaker_index": 4})
    assert clip_name(row).startswith("positive-training-0000001-")


def test_training_dir_moved_aside_when_inputs_change(tmp_path):
    from wakeword.train import _fresh_or_resumable

    train = tmp_path / "train"
    _fresh_or_resumable(train, "aaaa")
    (train / "restore").mkdir()
    _fresh_or_resumable(train, "aaaa")
    assert (train / "restore").exists()
    _fresh_or_resumable(train, "bbbb")
    assert not (train / "restore").exists()
    assert (train / "inputs.sha256").read_text().strip() == "bbbb"
    assert len(list(tmp_path.glob("train.aaaa.*"))) == 1

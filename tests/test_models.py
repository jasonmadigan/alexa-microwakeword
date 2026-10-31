"""checks against the real baseline models; skipped until `wakeword fetch` has run."""

import os
from pathlib import Path

import numpy as np
import pytest

from wakeword.export import check_model
from wakeword.streaming import StreamingModel

BASELINES = Path(os.environ.get("WAKEWORD_DATA", "data")) / "baselines"
ALEXA = BASELINES / "alexa.tflite"
pytestmark = pytest.mark.skipif(not ALEXA.exists(), reason="baselines not fetched")


def test_each_stream_starts_from_boot_state():
    rng = np.random.default_rng(0)
    loud = rng.integers(-128, 127, size=(600, 40)).astype(np.int8)
    quiet = np.full((180, 40), -128, np.int8)
    model = StreamingModel(ALEXA)
    fresh = StreamingModel(ALEXA).probabilities(quiet)
    model.probabilities(loud)
    assert np.array_equal(model.probabilities(quiet), fresh)


def test_baseline_passes_the_firmware_input_check():
    info = check_model(ALEXA)
    assert info["stride"] == 3 and info["input_zero_point"] == -128

"""print raw streaming probabilities for a few clips, to inspect model behaviour."""

import json
import sys

import numpy as np

from wakeword.audio import SAMPLE_RATE, load_audio
from wakeword.evaluate import _features
from wakeword.streaming import StreamingModel

model_paths, manifest, n = sys.argv[1].split(","), sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(manifest)]
rows = [r for r in rows if r["split"] == "testing" and r["kind"] == "positive"][:n]
models = {p: StreamingModel(p) for p in model_paths}
rng = np.random.default_rng(0)
for r in rows:
    clip = load_audio(r["path"])
    pre, post = int(1.5 * SAMPLE_RATE), SAMPLE_RATE
    stream = rng.normal(0, 10 ** (-70 / 20), pre + clip.size + post).astype(np.float32)
    stream[pre : pre + clip.size] += clip
    feats = _features(stream)
    start = (pre // 160) // 3
    for p, m in models.items():
        probs = m.probabilities(feats)
        print(f"{r['gender']:6} {r['region']:20} {p.split('/')[-1][:28]:28} pre max {probs[:start].max():3d} "
              f"pre mean {probs[:start].mean():6.1f} word max {probs[start:].max():3d}")

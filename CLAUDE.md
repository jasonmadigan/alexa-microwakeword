# alexa-microwakeword

Trains the Alexa microWakeWord model in `models/` for Home Assistant Voice PEs. The goal is recall across gender and accent at a false-accept rate no worse than the previous model; `README.md` has the pipeline and the install steps.

## Where things run

- On the training host only, through `scripts/unraid.sh` (`UNRAID_HOST`, `UNRAID_ROOT`). The workstation edits code and git: no datasets, no local venv. Source syncs to `$UNRAID_ROOT/src`; stages run in containers against `$UNRAID_ROOT/data`.
- The image holds only Python, uv and ffmpeg. The uv environment is bind-mounted from `$UNRAID_ROOT/env`, because on the original host Docker's image file sits on a spinning array disk, where installing TensorFlow and CUDA wheels takes hours. Dependency changes need `uv lock` and a sync, never an image rebuild.
- On Unraid, keep `$UNRAID_ROOT` outside `/mnt/cache/appdata`: Appdata Backup stops and tars containers with mappings there, which would kill a run.
- The GPU may be shared with other services. Runs refuse to start with less than 3 GB free; report a shortage rather than stopping anything.

## Upstream gotchas

- microWakeWord is a pinned submodule installed editable. Its built wheel drops `audio/` and `layers/` (no `__init__.py`), so a git or PyPI install imports fine until training.
- piper-sample-generator is not used: its wheel lacks `piper_train`, and its ONNX path cycles every speaker of a voice, which defeats balancing. `synth.py` drives `piper-tts` directly.
- TensorFlow 2.21 misses its pip-installed `libcusolver` and then registers no GPU; `scripts/entrypoint.sh` puts the `nvidia/*/lib` directories on `LD_LIBRARY_PATH`. It also needs `tensorboard` installed, which it no longer pulls in.
- `streaming.py` reproduces ESPHome 2026.10.1 `micro_wake_word` (feature quantisation, uint8 window sum against `int(cutoff * 255) * window`, 100-slice cool-off, VAD gate). Recheck it against `streaming_model.cpp` when ESPHome changes; `tests/test_streaming.py` pins the semantics.

## Data rules

- Every gender and accent label cites the corpus documentation in `config/voices.yaml`. Voices without documented labels stay out.
- Splits hold out whole speakers, stratified by (gender, region). Evaluation numbers are only meaningful on unseen speakers.
- Household recordings (`data/household/<name>/`) are evaluation-only, private and irreplaceable; audio never enters git.

## Baseline and publishing

- The baseline is Kevin Ahrendt's `alexa` v2 model as published in `esphome/micro-wake-word-models` (`models/v2/alexa.json` at `05b6592`; the Voice PE firmware ships no Alexa model, and the devices' earlier `kahrendt/microWakeWord` `v2.1_models` release no longer exists), scored at its default (0.9 / 5) and at 0.97 / 7, the override the original devices ran.
- `make publish` copies the exported model and its evaluation into `models/`. The README's results table quotes `models/alexa-evaluation.md`; change them together.

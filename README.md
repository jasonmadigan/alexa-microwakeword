# alexa-microwakeword

An "Alexa" wake word model for the [Home Assistant Voice PE](https://www.home-assistant.io/voice-pe/) (ESPHome [`micro_wake_word`](https://esphome.io/components/micro_wake_word/)), and the harness that trained it.

"Alexa" is a trademark of Amazon.com, Inc. or its affiliates. This is an unofficial hobby project, not affiliated with or endorsed by Amazon; the model just listens for the word.

## Why

The Voice PE doesn't ship an Alexa wake word, so ours ran Kevin Ahrendt's [microWakeWord](https://github.com/kahrendt/microWakeWord) `alexa` model: its `v2.1` release (since removed), then [`models/v2/alexa.json`](https://github.com/esphome/micro-wake-word-models/blob/main/models/v2/alexa.json) from ESPHome's micro-wake-word-models at `05b6592`, not the older v1 file beside it. That v2 model is what the comparison below uses.

It worked for me and rarely for my wife, and it was poor with female voices generally. This one was trained on synthetic speech split 50/50 between women and men, across Irish, Scottish, English, Welsh, North American, Australian, New Zealand, South African, South Asian and non-native English accents.

Held-out voices (speakers never seen in training), detection simulated the way the firmware does it:

| model | false accepts / h | Irish female, far field | worst group | mean group |
|-|-|-|-|-|
| this model, cutoff 0.52 / window 10 | 0.37 | 78 % | 72 % | 92 % |
| previous `alexa` v2, 0.97 / 7 (our old settings) | 0.00 | 24 % | 43 % | 71 % |
| previous `alexa` v2, 0.90 / 5 (its default) | 0.19 | 37 % | 57 % | 81 % |

This model and the previous one at 0.97 / 7 both accept about a quarter of confusable phrases ("Alexis", "a Lexus", "Lexa"); at its default the previous one accepts a third. [`models/alexa-evaluation.md`](models/alexa-evaluation.md) has the breakdown by gender and region, the confusable phrases and the other cutoff and window trade-offs; it calls the previous model `deployed` (0.97 / 7) and `shipped_default` (0.9 / 5).

Those test voices are synthetic. Real ones so far: 4 of 4 for me out to 7 m, 4 of 5 for my wife at 2-5 m.

## Use it on a Voice PE

Take control of the Voice PE in the ESPHome Device Builder add-on first (it shows up as discovered). ESPHome 2024.7 or newer.

1. In the device's YAML, keep the `packages:` block and add:

   ```yaml
   micro_wake_word:
     models:
       - model: github://jasonmadigan/alexa-microwakeword/models/alexa.json@main
         id: alexa
   ```

   Swap `main` for a commit SHA to pin it, so a later model can't change your device on its next rebuild. To keep it local, copy `models/alexa.json` and `models/alexa.tflite` into a folder under Home Assistant's `esphome/` config directory and use `model: <folder>/alexa.json`.

2. *Install*, then *Wirelessly*.

3. In Home Assistant, open the device and set *Wake word* to *Alexa* under *Configuration*.

4. Optionally, check the device logs (Device Builder, *Logs*) for

   ```
   - Wake Word: Alexa
     Probability cutoff: 0.52
     Sliding window size: 10
   ```

Leave `probability_cutoff` and `sliding_window_size` unset unless you want to override the tuned values; the evaluation shows what other values trade. The Voice PE's *Wake word sensitivity* setting only touches the built-in Okay Nabu, Hey Jarvis and Hey Mycroft models, not this one. To undo, remove the `models:` entry (or restore your old YAML) and install again.

## Training

Every stage runs in Docker on a host with an NVIDIA GPU. It was built on an Unraid server; `UNRAID_HOST` and `UNRAID_ROOT` point `scripts/unraid.sh` elsewhere.

| Stage | Output |
|-|-|
| `fetch` | Piper voices, microWakeWord negative feature sets, background noise and room impulse responses, the baseline `alexa` v2 and VAD models. All pinned by revision and checksum. |
| `plan` | `catalogue.tsv` (every speaker with gender, accent, region, split), `plan.jsonl` (every clip to synthesise), `coverage.txt` |
| `synth` | 16 kHz clips of "Alexa" (text and phoneme variants) and confusable phrases ("Alexis", "a Lexus", ...) |
| `features` | Augmented spectrograms in microWakeWord's ragged mmap layout |
| `train` | Quantised streaming TFLite model |
| `evaluate` | `report.md` / `report.json`: false accepts per hour on ambient audio, recall per (gender, region) and per household member, confusable-phrase accept rates, a cutoff-by-window trade-off grid, for the new model and the baselines |
| `export` | `dist/<name>/`: the `.tflite`, its ESPHome manifest with the chosen cutoff and window, a YAML snippet |

Speakers come from Piper voices whose corpora document gender and accent: VCTK, LibriTTS-R, CMU ARCTIC, L2-ARCTIC, the ARU corpus, SEMAINE, and single voices including an Irish (Jenny) and a Scottish (Alba) female voice. `config/voices.yaml` cites the source of every label. The plan splits clips evenly between genders first, then regions within each gender, caps clips per speaker and holds out whole speakers for testing.

The operating point is the cutoff and window with the best worst-group recall that stays within the configured limits on ambient false accepts per hour and confusable-phrase accepts (`evaluation` in `config/alexa.yaml`). Export refuses a model that misses them.

```sh
make image    # once, and after Dockerfile changes
make test     # unit tests, in the container
make smoke    # tiny end-to-end run; checks wiring only
make run      # full run, every stage (about 2.5 h on an RTX 5070 Ti)
make run STAGE=evaluate
make logs     # follow the latest run
make publish  # copy the exported model into models/
```

Source syncs to `$UNRAID_ROOT/src`, the Python environment lives in `$UNRAID_ROOT/env` (built from `uv.lock` on first start) and data in `$UNRAID_ROOT/data`. A full run needs about 40 GB, and won't start with less than 3 GB of free VRAM.

### Household recordings

Recordings of the people who'll actually use the speakers are the most honest test. Put them on the training host before `evaluate`:

```
$UNRAID_ROOT/data/household/<name>/*.wav|m4a|mp3|flac
$UNRAID_ROOT/data/household/ambient/*      # optional: TV, conversation, kitchen noise, no wake word
```

One "Alexa" per file, said naturally from across the room. Each person gets their own row in the report, and ambient recordings add a household false-accepts-per-hour figure. They're for evaluation only, never training.

## Licences

The code is under the [Apache License 2.0](LICENSE).

The trained model in `models/` is for personal, non-commercial use, because some of its training data is: the microWakeWord negative feature sets (CC BY-NC 4.0) and the L2-ARCTIC, SEMAINE, Hi-Fi-CAPTAIN and RyanSpeech voices. Speaker metadata in [`metadata/`](metadata/README.md) keeps its own licences (CC BY 4.0). microWakeWord, included as a submodule, is Apache 2.0.

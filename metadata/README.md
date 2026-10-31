# Speaker metadata

Gender and accent labels for multi-speaker Piper voices come from the corpora those voices were trained on.

| File | Source | Licence |
|-|-|-|
| `vctk-speaker-info.txt` | `speaker-info.txt` from VCTK 0.92, [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443), read straight out of the release zip | CC BY 4.0 |
| `librispeech-speakers.tsv` | `SPEAKERS.TXT` from LibriSpeech [raw-metadata.tar.gz](https://www.openslr.org/12/), reduced to id, gender and subset (reader names dropped) | CC BY 4.0 |

LibriTTS and LibriTTS-R reuse LibriSpeech reader ids, so the LibriSpeech table labels the `en_US-libritts_r-medium` speakers. Labels for the smaller corpora are inline in `config/voices.yaml`, each with its source.

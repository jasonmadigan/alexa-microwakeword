"""speaker catalogue, speaker-disjoint splits and the synthesis plan."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .config import REPO_ROOT, Config

SPLITS = ("training", "validation", "testing")
KINDS = ("positive", "negative")
GENDERS = ("female", "male")


@dataclass(frozen=True)
class Speaker:
    voice: str
    key: str
    index: int
    gender: str
    accent: str
    region: str
    source: str

    @property
    def uid(self) -> str:
        return f"{self.voice}:{self.key}"

    @property
    def cell(self) -> tuple[str, str]:
        return (self.gender, self.region)


def _gender(value: str) -> str:
    value = value.strip().lower()
    mapping = {"f": "female", "female": "female", "m": "male", "male": "male"}
    if value not in mapping:
        raise ValueError(f"unknown gender {value!r}")
    return mapping[value]


def _vctk(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "ID":
            continue
        out[parts[0]] = (parts[2], parts[3])
    return out


def _librispeech(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {row["reader_id"]: row["gender"] for row in csv.DictReader(f, delimiter="\t")}


def load_speakers(catalogue_path: Path, speaker_maps: dict[str, dict[str, int]]) -> list[Speaker]:
    """speaker_maps: voice name -> piper speaker_id_map ({} for single-speaker voices)."""
    catalogue = yaml.safe_load(catalogue_path.read_text())
    regions: dict[str, str] = catalogue["regions"]
    vctk = _vctk(REPO_ROOT / "metadata" / "vctk-speaker-info.txt")
    librispeech = _librispeech(REPO_ROOT / "metadata" / "librispeech-speakers.tsv")
    speakers: list[Speaker] = []

    def add(voice: str, key: str, index: int, gender: str, accent: str, source: str) -> None:
        if accent not in regions:
            raise ValueError(f"{voice}:{key}: accent {accent!r} has no region in {catalogue_path}")
        speakers.append(Speaker(voice, key, index, _gender(gender), accent, regions[accent], source))

    for voice, entry in catalogue["voices"].items():
        if voice not in speaker_maps:
            raise KeyError(f"voice {voice} missing from the downloaded piper voices")
        id_map = speaker_maps[voice]
        source = entry["source"]
        if "speaker" in entry:
            if id_map and len(id_map) > 1:
                raise ValueError(f"{voice} is multi-speaker; list its speakers")
            gender, accent = entry["speaker"]
            add(voice, "0", 0, gender, accent, source)
        elif entry.get("speakers_from") == "vctk":
            accents = entry["vctk_accents"]
            for key, index in id_map.items():
                gender, accent = vctk[key]
                if accent in accents:
                    add(voice, key, index, gender, accents[accent], source)
        elif entry.get("speakers_from") == "librispeech":
            for key, index in id_map.items():
                add(voice, key, index, librispeech[key.lstrip("p")], entry["accent"], source)
        else:
            for key, (gender, accent) in entry["speakers"].items():
                key = str(key)
                if key not in id_map:
                    raise KeyError(f"{voice}: speaker {key} not in its speaker_id_map")
                add(voice, key, id_map[key], gender, accent, source)
    return speakers


def _rank(seed: int, uid: str) -> str:
    return hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()


def held_out_counts(n: int, validation: float, testing: float) -> tuple[int, int]:
    """(validation, testing) speakers to hold out of a cell with n speakers."""
    if n <= 1:
        return 0, 0
    if n == 2:
        # one unseen speaker is worth more for testing than for checkpoint selection
        return 0, 1
    n_test = max(1, round(n * testing))
    n_val = max(1, round(n * validation))
    while n - n_test - n_val < 1:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_val, n_test


def assign_splits(speakers: Iterable[Speaker], seed: int, held_out: dict[str, float]) -> dict[str, str]:
    """speaker uid -> split, stratified by (gender, region), whole speakers only."""
    cells: dict[tuple[str, str], list[Speaker]] = defaultdict(list)
    for s in speakers:
        cells[s.cell].append(s)
    out = {}
    for members in cells.values():
        members = sorted(members, key=lambda s: _rank(seed, s.uid))
        n_val, n_test = held_out_counts(len(members), held_out["validation"], held_out["testing"])
        for i, s in enumerate(members):
            out[s.uid] = "testing" if i < n_test else "validation" if i < n_test + n_val else "training"
    return out


def allocate(total: int, weights: dict[tuple[str, str], float], capacity: dict[tuple[str, str], int]) -> dict:
    """split total over cells by weight; cells at capacity pass their excess to the rest."""
    alloc = {c: 0 for c in weights}
    remaining = total
    open_cells = {c for c, w in weights.items() if w > 0 and capacity.get(c, 0) > 0}
    while remaining > 0 and open_cells:
        weight_sum = sum(weights[c] for c in open_cells)
        shares = {c: remaining * weights[c] / weight_sum for c in open_cells}
        floors = {c: int(math.floor(v)) for c, v in shares.items()}
        leftover = remaining - sum(floors.values())
        # largest remainders take the rounding slack, deterministically
        for c in sorted(open_cells, key=lambda c: (floors[c] - shares[c], c))[:leftover]:
            floors[c] += 1
        progressed = 0
        for c in sorted(open_cells):
            take = min(floors[c], capacity[c] - alloc[c])
            alloc[c] += take
            progressed += take
        remaining -= progressed
        open_cells = {c for c in open_cells if alloc[c] < capacity[c]}
        if progressed == 0:
            break
    return alloc


def allocate_cells(total: int, gender_w: dict, region_w: dict, capacity: dict[tuple[str, str], int]) -> dict:
    """genders first, then regions within each gender, so capped regions cannot tilt the gender split."""
    genders = sorted({c[0] for c in capacity})
    by_gender = allocate(
        total,
        {g: gender_w.get(g, 0.0) for g in genders},
        {g: sum(n for c, n in capacity.items() if c[0] == g) for g in genders},
    )
    alloc = {}
    for g in genders:
        cells = {c: n for c, n in capacity.items() if c[0] == g}
        alloc.update(allocate(by_gender[g], {c: region_w.get(c[1], 0.0) for c in cells}, cells))
    return alloc


@dataclass
class ClipPlan:
    id: str
    kind: str
    split: str
    voice: str
    speaker: str
    speaker_index: int
    gender: str
    accent: str
    region: str
    text: str | None
    ipa: str | None
    length_scale: float
    noise_scale: float
    noise_w_scale: float


def build_plan(cfg: Config, speakers: list[Speaker], splits: dict[str, str]) -> list[ClipPlan]:
    plan_cfg = cfg["plan"]
    rng = random.Random(cfg.seed)
    gender_w = plan_cfg["gender_weights"]
    region_w = plan_cfg["region_weights"]
    cap = int(plan_cfg["max_clips_per_speaker"])
    synth = cfg["synthesis"]
    phrases = cfg["phrases"]
    rows: list[ClipPlan] = []
    for kind in KINDS:
        for split in SPLITS:
            total = int(plan_cfg["clips"][kind][split])
            pool: dict[tuple[str, str], list[Speaker]] = defaultdict(list)
            for s in speakers:
                if splits[s.uid] == split:
                    pool[s.cell].append(s)
            # held-out speakers are few, so only cap the training split
            capacity = {c: len(m) * cap if split == "training" else total for c, m in pool.items()}
            alloc = allocate_cells(total, gender_w, region_w, capacity)
            for cell in sorted(alloc):
                members = sorted(pool[cell], key=lambda s: _rank(cfg.seed, s.uid))
                for i in range(alloc[cell]):
                    s = members[i % len(members)]
                    text, ipa = _phrase(rng, phrases[kind], kind)
                    rows.append(
                        ClipPlan(
                            id=f"{kind}-{split}-{len(rows):07d}",
                            kind=kind,
                            split=split,
                            voice=s.voice,
                            speaker=s.key,
                            speaker_index=s.index,
                            gender=s.gender,
                            accent=s.accent,
                            region=s.region,
                            text=text,
                            ipa=ipa,
                            length_scale=round(rng.uniform(*synth["length_scale"]), 4),
                            noise_scale=round(rng.uniform(*synth["noise_scale"]), 4),
                            noise_w_scale=round(rng.uniform(*synth["noise_w_scale"]), 4),
                        )
                    )
    return rows


def _phrase(rng: random.Random, phrases: dict, kind: str) -> tuple[str | None, str | None]:
    ipa = phrases.get("ipa") or []
    if ipa and rng.random() < float(phrases.get("ipa_fraction", 0.0)):
        return None, rng.choice(ipa)
    return rng.choice(phrases["text"]), None


def write_catalogue(path: Path, speakers: list[Speaker], splits: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uid", "voice", "speaker", "index", "gender", "accent", "region", "split", "source"])
        for s in sorted(speakers, key=lambda s: s.uid):
            w.writerow([s.uid, s.voice, s.key, s.index, s.gender, s.accent, s.region, splits[s.uid], s.source])


def write_plan(path: Path, rows: list[ClipPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def coverage(speakers: list[Speaker], splits: dict[str, str], rows: list[ClipPlan] | None = None) -> str:
    """plain-text table of speakers (and planned positives) per cell and split."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in speakers:
        counts[s.cell][splits[s.uid]] += 1
    clips: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows or []:
        if r.kind == "positive" and r.split == "training":
            clips[(r.gender, r.region)] += 1
    lines = [f"{'gender':8} {'region':22} {'train':>6} {'val':>5} {'test':>5} {'train clips':>12}"]
    for cell in sorted(counts):
        c = counts[cell]
        lines.append(
            f"{cell[0]:8} {cell[1]:22} {c['training']:6d} {c['validation']:5d} {c['testing']:5d} {clips[cell]:12d}"
        )
    return "\n".join(lines)

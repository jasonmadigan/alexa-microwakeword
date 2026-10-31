import csv
from collections import Counter

import pytest
import yaml

from wakeword.catalogue import (
    Speaker,
    allocate,
    allocate_cells,
    assign_splits,
    build_plan,
    held_out_counts,
    load_speakers,
)
from wakeword.config import REPO_ROOT, load_config


def _speaker_maps():
    """speaker_id_maps shaped like the piper voice configs, from the committed metadata."""
    catalogue = yaml.safe_load((REPO_ROOT / "config" / "voices.yaml").read_text())
    vctk = [l.split()[0] for l in (REPO_ROOT / "metadata" / "vctk-speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    with (REPO_ROOT / "metadata" / "librispeech-speakers.tsv").open() as f:
        libri = [r["reader_id"] for r in csv.DictReader(f, delimiter="\t") if r["subset"].startswith("train")][:300]
    maps = {}
    for voice, entry in catalogue["voices"].items():
        if entry.get("speakers_from") == "vctk":
            maps[voice] = {k: i for i, k in enumerate(vctk)}
        elif entry.get("speakers_from") == "librispeech":
            maps[voice] = {k: i for i, k in enumerate(libri)}
        elif "speakers" in entry:
            maps[voice] = {str(k): i for i, k in enumerate(entry["speakers"])}
        else:
            maps[voice] = {}
    return maps


@pytest.fixture(scope="module")
def speakers():
    return load_speakers(REPO_ROOT / "config" / "voices.yaml", _speaker_maps())


def test_every_speaker_has_binary_gender_and_region(speakers):
    assert {s.gender for s in speakers} == {"female", "male"}
    regions = set(yaml.safe_load((REPO_ROOT / "config" / "voices.yaml").read_text())["regions"].values())
    assert {s.region for s in speakers} <= regions


def test_both_genders_in_every_region(speakers):
    cells = Counter(s.cell for s in speakers)
    for region in {s.region for s in speakers}:
        assert cells[("female", region)] > 0, region
        assert cells[("male", region)] > 0, region


def test_vctk_unknown_accent_excluded(speakers):
    assert not any(s.uid == "en_GB-vctk-medium:p280" for s in speakers)


def test_irish_voices_present(speakers):
    irish = {s.uid for s in speakers if s.region == "ireland"}
    assert "en_GB-jenny_dioco-medium:0" in irish
    assert sum(1 for s in speakers if s.region == "ireland" and s.gender == "male") >= 3


@pytest.mark.parametrize("n,expected", [(1, (0, 0)), (2, (0, 1)), (3, (1, 1)), (10, (1, 1)), (40, (4, 4))])
def test_held_out_counts(n, expected):
    assert held_out_counts(n, 0.1, 0.1) == expected


def test_splits_are_speaker_disjoint_deterministic_and_stratified(speakers):
    a = assign_splits(speakers, 7, {"validation": 0.1, "testing": 0.1})
    b = assign_splits(speakers, 7, {"validation": 0.1, "testing": 0.1})
    assert a == b
    assert set(a) == {s.uid for s in speakers}
    cells = {}
    for s in speakers:
        cells.setdefault(s.cell, []).append(a[s.uid])
    for cell, splits in cells.items():
        assert "training" in splits, cell
        if len(splits) >= 2:
            assert "testing" in splits, cell


def test_allocate_respects_capacity_and_total():
    weights = {("female", "a"): 1.0, ("male", "a"): 1.0, ("female", "b"): 1.0}
    capacity = {("female", "a"): 10, ("male", "a"): 1000, ("female", "b"): 1000}
    alloc = allocate(300, weights, capacity)
    assert sum(alloc.values()) == 300
    assert alloc[("female", "a")] == 10
    assert abs(alloc[("male", "a")] - alloc[("female", "b")]) <= 1


def test_allocate_stops_when_everything_is_full():
    alloc = allocate(100, {("f", "a"): 1.0}, {("f", "a"): 30})
    assert alloc == {("f", "a"): 30}


def test_plan_balances_gender_and_keeps_splits(speakers):
    cfg = load_config(REPO_ROOT / "config" / "smoke.yaml")
    splits = assign_splits(speakers, cfg.seed, cfg["plan"]["held_out"])
    rows = build_plan(cfg, speakers, splits)
    by_uid = {s.uid: splits[s.uid] for s in speakers}
    assert all(by_uid[f"{r.voice}:{r.speaker}"] == r.split for r in rows)
    train_pos = [r for r in rows if r.kind == "positive" and r.split == "training"]
    assert len(train_pos) == cfg["plan"]["clips"]["positive"]["training"]
    genders = Counter(r.gender for r in train_pos)
    assert abs(genders["female"] - genders["male"]) <= 1
    per_speaker = Counter((r.voice, r.speaker) for r in train_pos)
    assert max(per_speaker.values()) <= cfg["plan"]["max_clips_per_speaker"]
    assert len({r.id for r in rows}) == len(rows)


def test_speaker_uid():
    s = Speaker("v", "k", 0, "female", "irish", "ireland", "src")
    assert s.uid == "v:k" and s.cell == ("female", "ireland")


def test_capped_regions_do_not_tilt_gender_split():
    # female cells are small and capped; their spare quota must stay with female cells
    capacity = {("female", "a"): 10, ("female", "b"): 1000, ("male", "a"): 1000, ("male", "b"): 1000}
    alloc = allocate_cells(400, {"female": 1.0, "male": 1.0}, {"a": 1.0, "b": 1.0}, capacity)
    female = alloc[("female", "a")] + alloc[("female", "b")]
    assert female == 200 and alloc[("female", "a")] == 10


def test_gender_short_of_capacity_hands_excess_over():
    capacity = {("female", "a"): 50, ("male", "a"): 1000}
    alloc = allocate_cells(400, {"female": 1.0, "male": 1.0}, {"a": 1.0}, capacity)
    assert alloc == {("female", "a"): 50, ("male", "a"): 350}

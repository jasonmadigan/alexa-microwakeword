from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

from .config import Config, load_config

log = logging.getLogger("wakeword")


def _speaker_maps(cfg: Config) -> tuple[dict[str, dict[str, int]], dict[str, Path]]:
    from .fetch import catalogue_voices, voice_files, voices_index

    index = voices_index(cfg)
    maps, paths = {}, {}
    for voice in catalogue_voices(cfg):
        model, config = voice_files(cfg, index, voice)
        if not model.exists():
            raise FileNotFoundError(f"{model} missing; run fetch")
        maps[voice] = json.loads(config.read_text()).get("speaker_id_map") or {}
        paths[voice] = model
    return maps, paths


def cmd_fetch(cfg: Config, args) -> None:
    from . import fetch

    steps = {"baselines": fetch.fetch_baselines, "voices": fetch.fetch_voices, "negatives": fetch.fetch_negatives, "backgrounds": fetch.fetch_backgrounds}
    for name in args.only or list(steps):
        steps[name](cfg)


def cmd_plan(cfg: Config, args) -> None:
    from . import catalogue as cat

    maps, _ = _speaker_maps(cfg)
    speakers = cat.load_speakers(cfg.resolve(cfg["voices"]), maps)
    splits = cat.assign_splits(speakers, cfg.seed, cfg["plan"]["held_out"])
    rows = cat.build_plan(cfg, speakers, splits)
    work = cfg.layout.work
    cat.write_catalogue(work / "catalogue.tsv", speakers, splits)
    cat.write_plan(work / "plan.jsonl", rows)
    table = cat.coverage(speakers, splits, rows)
    (work / "coverage.txt").write_text(table + "\n")
    print(table)
    kinds = Counter((r.kind, r.split) for r in rows)
    print("\n".join(f"{k} {s}: {n}" for (k, s), n in sorted(kinds.items())))


def cmd_synth(cfg: Config, args) -> None:
    from . import synth
    from .catalogue import read_jsonl

    _, paths = _speaker_maps(cfg)
    plan = read_jsonl(cfg.layout.work / "plan.jsonl")
    _check_phonemes(cfg, paths)
    manifest = synth.run(plan, cfg.layout.clips, paths, cfg.workers, cfg["synthesis"]["trim_db"])
    with (cfg.layout.work / "manifest.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("manifest: %d clips", len(manifest))


def _check_phonemes(cfg: Config, paths: dict[str, Path]) -> None:
    from .synth import ipa_to_ids, load_voice

    variants = [v for kind in ("positive", "negative") for v in cfg["phrases"][kind].get("ipa") or []]
    for voice, path in paths.items():
        v = load_voice(path)
        for ipa in variants:
            try:
                ipa_to_ids(v, ipa)
            except ValueError as exc:
                raise SystemExit(f"{voice}: {ipa!r}: {exc}") from None


def _manifest(cfg: Config) -> list[dict]:
    from .catalogue import read_jsonl

    path = cfg.layout.work / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing; run synth")
    return read_jsonl(path)


def cmd_features(cfg: Config, args) -> None:
    from . import features

    features.build(cfg, _manifest(cfg))


def cmd_train(cfg: Config, args) -> None:
    from . import train

    model = train.run(cfg, train=not args.convert_only)
    log.info("model: %s", model)


def cmd_evaluate(cfg: Config, args) -> None:
    from . import evaluate, report
    from .train import MODEL_RELPATH

    candidate = None if args.baseline_only else Path(args.model) if args.model else cfg.layout.train / MODEL_RELPATH
    if candidate is not None and not candidate.exists():
        raise SystemExit(f"{candidate} missing; run train")
    result = evaluate.run(cfg, candidate, _manifest(cfg))
    out = cfg.layout.eval
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    md = report.markdown(result)
    (out / "report.md").write_text(md)
    print(md)


def cmd_export(cfg: Config, args) -> None:
    from . import export
    from .train import MODEL_RELPATH

    report_path = cfg.layout.eval / "report.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} missing; run evaluate")
    out = export.run(cfg, cfg.layout.train / MODEL_RELPATH, json.loads(report_path.read_text()))
    print(f"exported to {out}")


def cmd_doctor(cfg: Config, args) -> None:
    import platform

    import tensorflow as tf

    lay = cfg.layout
    lay.root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(lay.root)
    print(f"data root  {lay.root} ({usage.free / 1e9:.0f} GB free)")
    print(f"python     {platform.python_version()} on {platform.machine()}")
    print(f"tensorflow {tf.__version__}, gpus: {[d.name for d in tf.config.list_physical_devices('GPU')] or 'none'}")
    print(f"workers    {cfg.workers}")
    if usage.free < 50e9 and cfg.name != "smoke":
        print("warning: a full run needs roughly 40 GB")


COMMANDS = {
    "fetch": cmd_fetch,
    "plan": cmd_plan,
    "synth": cmd_synth,
    "features": cmd_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "doctor": cmd_doctor,
}
PIPELINE = ("fetch", "plan", "synth", "features", "train", "evaluate", "export")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wakeword", description="train a microwakeword model")
    parser.add_argument("--config", default="config/alexa.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="download voices, negative features, backgrounds, baselines")
    p.add_argument("--only", nargs="+", choices=["baselines", "voices", "negatives", "backgrounds"])
    sub.add_parser("plan", help="speaker catalogue, speaker-disjoint splits, clip plan")
    sub.add_parser("synth", help="synthesise planned clips with piper")
    sub.add_parser("features", help="augment clips and write spectrogram features")
    p = sub.add_parser("train", help="train and quantise the streaming model")
    p.add_argument("--convert-only", action="store_true", help="skip training; convert and test the best weights")
    p = sub.add_parser("evaluate", help="firmware-equivalent evaluation against the baselines")
    p.add_argument("--model", help="tflite to evaluate instead of the trained model")
    p.add_argument("--baseline-only", action="store_true", help="evaluate only the baseline models")
    sub.add_parser("export", help="write the esphome manifest and model to dist/")
    sub.add_parser("doctor", help="check environment, gpu and disk")
    sub.add_parser("all", help="run every stage in order")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    # huggingface_hub logs every http request at info
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.command == "all":
        for name in PIPELINE:
            log.info("stage %s", name)
            COMMANDS[name](cfg, argparse.Namespace(only=None, convert_only=False, model=None, baseline_only=False))
    else:
        COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    main()

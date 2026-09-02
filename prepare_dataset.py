#!/usr/bin/env python3
"""
Step 1 of the pipeline: turn the raw export into a training-ready dataset.

INPUT (the "manifest contract")
-------------------------------
  --audio_dir   folder containing the exported WAV clips (48kHz/16-bit)
  --manifest    a CSV file with EXACTLY these three columns:
                    file        clip filename, relative to --audio_dir
                    transcript  verbatim Mongolian transcript
                    speaker_id  stable ID of the speaker (any string)

WHAT IT DOES
------------
  1. Validates the manifest (missing files, empty transcripts, missing
     speaker IDs are DROPPED and counted — you see the numbers).
  2. Drops clips longer than 30s (Whisper's input window — text past 30s
     is unlearnable) and shorter than 0.5s (junk). Counted and reported.
  3. Resamples every clip to 16kHz mono WAV (Whisper's expected input).
  4. Splits BY SPEAKER: every speaker lands wholly in exactly one of
     train / validation / test. Test speakers appear in ZERO training
     clips. This is the integrity rule from the spec — it is enforced
     here in code and asserted before saving.
  5. Saves a HuggingFace DatasetDict to disk and writes
     split_report.json with exact hours / clips / speakers per split.

USAGE
-----
  python prepare_dataset.py \
      --audio_dir /workspace/data/wavs \
      --manifest  /workspace/data/manifest.csv \
      --output_dir /workspace/prepared
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from tqdm import tqdm

TARGET_SR = 16000
MAX_CLIP_SECONDS = 30.0
MIN_CLIP_SECONDS = 0.5


def load_and_validate_manifest(manifest_path: Path, audio_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(manifest_path, dtype=str)
    required = {"file", "transcript", "speaker_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        sys.exit(f"ERROR: manifest is missing required column(s): {sorted(missing_cols)}. "
                 f"Required columns: file, transcript, speaker_id")

    stats = {"rows_in_manifest": len(df)}

    df = df.dropna(subset=["file", "transcript", "speaker_id"])
    df["transcript"] = df["transcript"].str.strip()
    df["speaker_id"] = df["speaker_id"].str.strip()
    df = df[(df["transcript"] != "") & (df["speaker_id"] != "")]
    stats["dropped_empty_fields"] = stats["rows_in_manifest"] - len(df)

    df["path"] = df["file"].apply(lambda f: str(audio_dir / f))
    exists_mask = df["path"].apply(os.path.isfile)
    stats["dropped_missing_files"] = int((~exists_mask).sum())
    if stats["dropped_missing_files"]:
        for p in df.loc[~exists_mask, "file"].head(10):
            print(f"  missing file (dropped): {p}")
    df = df[exists_mask].reset_index(drop=True)
    return df, stats


def read_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return -1.0  # unreadable → will be dropped


def resample_one(args: tuple[str, str]) -> float:
    """Resample one clip to 16kHz mono. Returns duration in seconds."""
    src, dst = args
    audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


def split_by_speaker(df: pd.DataFrame, test_fraction: float, val_fraction: float,
                     seed: int) -> pd.DataFrame:
    """Assign every SPEAKER (not clip) to exactly one split, by duration."""
    per_speaker = df.groupby("speaker_id")["duration"].sum()
    total = per_speaker.sum()
    speakers = list(per_speaker.index)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    test_speakers, val_speakers = set(), set()
    acc = 0.0
    it = iter(speakers)
    for s in it:
        test_speakers.add(s)
        acc += per_speaker[s]
        if acc >= test_fraction * total:
            break
    acc = 0.0
    for s in it:
        val_speakers.add(s)
        acc += per_speaker[s]
        if acc >= val_fraction * total:
            break
    train_speakers = set(speakers) - test_speakers - val_speakers

    if len(test_speakers) < 3:
        print("WARNING: fewer than 3 speakers in the test set. The WER number "
              "will be statistically weak. Consider a larger test fraction or "
              "more speakers before quoting it to buyers.")
    if not train_speakers:
        sys.exit("ERROR: no speakers left for training after the split. "
                 "Your dataset has too few speakers for these fractions.")

    def assign(s):
        if s in test_speakers:
            return "test"
        if s in val_speakers:
            return "validation"
        return "train"

    df = df.copy()
    df["split"] = df["speaker_id"].apply(assign)

    # THE INTEGRITY ASSERTION: no speaker may appear in more than one split.
    overlap = (df.groupby("speaker_id")["split"].nunique() > 1)
    assert not overlap.any(), "Speaker leaked across splits — this is a bug, do not proceed."
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test_fraction", type=float, default=0.15)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    resampled_dir = out_dir / "wav16k"
    resampled_dir.mkdir(parents=True, exist_ok=True)

    print("== 1/5 Reading and validating manifest ==")
    df, stats = load_and_validate_manifest(Path(args.manifest), audio_dir)

    print("== 2/5 Reading clip durations ==")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        durations = list(tqdm(ex.map(read_duration, df["path"], chunksize=64),
                              total=len(df)))
    df["duration"] = durations

    unreadable = df["duration"] < 0
    too_long = df["duration"] > MAX_CLIP_SECONDS
    too_short = (df["duration"] >= 0) & (df["duration"] < MIN_CLIP_SECONDS)
    stats["dropped_unreadable"] = int(unreadable.sum())
    stats["dropped_over_30s"] = int(too_long.sum())
    stats["dropped_under_0.5s"] = int(too_short.sum())
    df = df[~(unreadable | too_long | too_short)].reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: no usable clips left after filtering.")

    print("== 3/5 Resampling to 16kHz mono (this is the slow step) ==")
    df["path16"] = [str(resampled_dir / f"{i:07d}.wav") for i in range(len(df))]
    jobs = list(zip(df["path"], df["path16"]))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        durs16 = list(tqdm(ex.map(resample_one, jobs, chunksize=16),
                           total=len(jobs)))
    df["duration"] = durs16  # authoritative durations post-resample

    print("== 4/5 Splitting by SPEAKER (integrity rule) ==")
    df = split_by_speaker(df, args.test_fraction, args.val_fraction, args.seed)

    print("== 5/5 Building and saving HuggingFace dataset ==")
    splits = {}
    for name in ["train", "validation", "test"]:
        part = df[df["split"] == name]
        ds = Dataset.from_dict({
            "audio": part["path16"].tolist(),
            "sentence": part["transcript"].tolist(),
            "speaker_id": part["speaker_id"].tolist(),
        }).cast_column("audio", Audio(sampling_rate=TARGET_SR))
        splits[name] = ds
    DatasetDict(splits).save_to_disk(str(out_dir / "dataset"))

    report = {
        "split_rule": "by_speaker (no speaker appears in more than one split)",
        "seed": args.seed,
        "filters": stats,
        "splits": {},
    }
    for name in ["train", "validation", "test"]:
        part = df[df["split"] == name]
        report["splits"][name] = {
            "clips": int(len(part)),
            "hours": round(part["duration"].sum() / 3600, 2),
            "unique_speakers": int(part["speaker_id"].nunique()),
        }
    with open(out_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== SPLIT REPORT (also saved to split_report.json) =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDataset saved to: {out_dir / 'dataset'}")
    print("Next step: the smoke test (see README, Phase 6).")


if __name__ == "__main__":
    main()

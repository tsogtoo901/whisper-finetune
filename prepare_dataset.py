#!/usr/bin/env python3
"""
Step 1 of the pipeline: turn the raw export into a training-ready dataset.

INPUT (the "manifest contract")
-------------------------------
  --audio_dir   folder containing the exported WAV clips (48kHz/16-bit)
  --manifest    a CSV file with these three columns (required):
                    file        clip filename, relative to --audio_dir
                    transcript  verbatim Mongolian transcript
                    speaker_id  stable ID of the speaker (any string)
                and, optionally, three more (the conversational export writes them):
                    start_sec   segment start inside `file`, in seconds (blank = whole file)
                    end_sec     segment end inside `file`, in seconds   (blank = whole file)
                    source      "rs" (read speech, whole clip) or "conv" (a segment of a
                                conversation track). Several rows may point at ONE file.

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
     here in code and asserted before saving. A conversation segment
     carries its own speaker (each speaker is on their own track), so
     the rule holds for conversational data unchanged.
  4b. Optionally repeats conversational segments in TRAIN (--conv_repeat N)
     so a small conversational share still moves the model. Test and
     validation are never repeated.
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
OPTIONAL_COLS = ("start_sec", "end_sec", "source")


def load_and_validate_manifest(manifest_path: Path, audio_dir: Path) -> tuple[pd.DataFrame, dict]:
    # encoding="utf-8-sig" strips the BOM the export writes for Excel.
    # Without it, the first column name arrives as "\ufefffile" and the
    # required-columns check below fails on an otherwise-correct manifest.
    df = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig")
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

    # optional segment columns (conversational rows); whole-file rows leave them blank
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df["source"] = df["source"].fillna("").astype(str).str.strip().str.lower().replace("", "rs")
    df["is_segment"] = df["start_sec"].notna() & df["end_sec"].notna()
    bad_seg = df["is_segment"] & (df["end_sec"] <= df["start_sec"])
    stats["dropped_bad_segment_times"] = int(bad_seg.sum())
    df = df[~bad_seg]
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


def resample_one(args: tuple[str, str, object, object]) -> float:
    """Resample one clip — or one SEGMENT of a long track — to 16kHz mono.
    Returns duration in seconds. Segments are read with start/stop frames so a
    25-minute conversation track is never loaded whole for each of its rows."""
    src, dst, start, end = args
    if start is None:
        audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    else:
        info = sf.info(src)
        a, b = int(float(start) * info.samplerate), int(float(end) * info.samplerate)
        audio, sr = sf.read(src, start=a, stop=min(b, info.frames), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    sf.write(dst, audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


def split_by_speaker(df: pd.DataFrame, test_fraction: float, val_fraction: float,
                     seed: int, max_test_speaker_share: float) -> pd.DataFrame:
    """Assign every SPEAKER (not clip) to exactly one split, by DURATION.

    Splitting by duration (not by speaker count) already handles the common
    case. But a highly skewed corpus — where one speaker (e.g. SPK30) holds a
    large share of all audio — creates a second failure mode: if that speaker
    lands in the test set, the "unheard speakers" WER becomes essentially
    "how well does it do on that ONE voice", which is not a number a buyer
    should trust. So we also cap how much any single speaker may contribute to
    the test set: any speaker whose audio exceeds max_test_speaker_share of the
    TARGET test duration is ineligible for test/validation and is placed in
    train. The test set is then filled from the remaining speakers, keeping it
    both duration-appropriate AND composed of many distinct voices.
    """
    per_speaker = df.groupby("speaker_id")["duration"].sum()
    total = float(per_speaker.sum())
    target_test = test_fraction * total
    target_val = val_fraction * total

    # Dominant speakers: too big to sit in test without dominating it.
    cap = max_test_speaker_share * target_test
    dominant = [s for s in per_speaker.index if per_speaker[s] > cap]
    eligible = [s for s in per_speaker.index if per_speaker[s] <= cap]
    if dominant:
        print(f"  {len(dominant)} dominant speaker(s) forced into TRAIN "
              f"(each exceeds {max_test_speaker_share:.0%} of the target test "
              f"hours): {', '.join(sorted(dominant)[:8])}"
              f"{' …' if len(dominant) > 8 else ''}")

    rng = random.Random(seed)
    rng.shuffle(eligible)

    # If the non-dominant pool can't fill BOTH targets (a heavily skewed
    # corpus — e.g. five speakers holding 95% of the audio), share the pool
    # between validation and test in the same ratio as the requested
    # fractions, instead of giving everything to test and leaving validation
    # empty. Validation is filled FIRST because it must never be empty
    # (early stopping depends on it) and it is the smaller of the two.
    eligible_total = float(sum(per_speaker[s] for s in eligible))
    scarce = eligible_total < (target_test + target_val)
    if scarce:
        val_share = val_fraction / (test_fraction + val_fraction)
        target_val_eff = val_share * eligible_total
        print(f"  NOTE: non-dominant speakers total only {eligible_total / 3600:.1f}h, "
              f"less than the {(target_test + target_val) / 3600:.1f}h needed for "
              f"test+validation. Sharing that pool ~{1 - val_share:.0%} test / "
              f"~{val_share:.0%} validation by speaker; all dominant speakers train.")
    else:
        target_val_eff = target_val

    test_speakers, val_speakers = set(), set()
    it = iter(eligible)
    acc = 0.0
    for s in it:
        val_speakers.add(s)
        acc += per_speaker[s]
        if acc >= target_val_eff:
            break
    acc = 0.0
    for s in it:
        test_speakers.add(s)
        acc += per_speaker[s]
        if not scarce and acc >= target_test:
            break
    achieved_test = acc
    train_speakers = set(per_speaker.index) - test_speakers - val_speakers

    if not val_speakers or not test_speakers:
        sys.exit("ERROR: could not form BOTH a validation and a test set from the "
                 "non-dominant speakers — there are too few of them. Options: add "
                 "more speakers to the corpus, or raise --max_test_speaker_share "
                 "(with care: a larger share means one voice weighs more in the "
                 "benchmark).")
    if len(test_speakers) < 3:
        print("WARNING: fewer than 3 speakers in the test set. The WER number "
              "will be statistically weak. Add more speakers, or raise "
              "--test_fraction, before quoting it to buyers.")
    if achieved_test < 0.6 * target_test:
        print(f"WARNING: test set is only {achieved_test / 3600:.1f}h vs a "
              f"target of {target_test / 3600:.1f}h — not enough eligible "
              f"(non-dominant) speakers to fill it. The number is still valid, "
              f"just smaller than requested.")
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
    ap.add_argument("--max_test_speaker_share", type=float, default=0.34,
                    help="Cap any one speaker's share of the target test hours; "
                         "speakers above this go to train (skew protection)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--conv_repeat", type=int, default=1,
                    help="Repeat each conversational (source=conv) TRAIN segment this many "
                         "times so a small conversational share still influences training "
                         "(1 = no repetition; validation/test are never repeated)")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    resampled_dir = out_dir / "wav16k"
    resampled_dir.mkdir(parents=True, exist_ok=True)

    print("== 1/5 Reading and validating manifest ==")
    df, stats = load_and_validate_manifest(Path(args.manifest), audio_dir)

    print("== 2/5 Reading clip durations ==")
    whole = ~df["is_segment"]
    durations = [0.0] * len(df)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        whole_paths = df.loc[whole, "path"].tolist()
        whole_durs = list(tqdm(ex.map(read_duration, whole_paths, chunksize=64), total=len(whole_paths)))
    it = iter(whole_durs)
    for i, seg in enumerate(df["is_segment"].tolist()):
        durations[i] = (float(df["end_sec"].iat[i]) - float(df["start_sec"].iat[i])) if seg else next(it)
    df["duration"] = durations
    n_seg = int(df["is_segment"].sum())
    if n_seg:
        print(f"  {n_seg} conversational segment(s) across {df.loc[df['is_segment'], 'file'].nunique()} track(s)")

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
    jobs = [(p, q, (s if seg else None), (e if seg else None))
            for p, q, seg, s, e in zip(df["path"], df["path16"], df["is_segment"], df["start_sec"], df["end_sec"])]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        durs16 = list(tqdm(ex.map(resample_one, jobs, chunksize=16),
                           total=len(jobs)))
    df["duration"] = durs16  # authoritative durations post-resample

    print("== 4/5 Splitting by SPEAKER (integrity rule) ==")
    df = split_by_speaker(df, args.test_fraction, args.val_fraction, args.seed,
                          args.max_test_speaker_share)

    print("== 5/5 Building and saving HuggingFace dataset ==")
    for name in ["train", "validation", "test"]:
        if (df["split"] == name).sum() == 0:
            sys.exit(f"ERROR: the '{name}' split is empty — refusing to save. "
                     f"See the split messages above.")
    splits = {}
    for name in ["train", "validation", "test"]:
        part = df[df["split"] == name]
        if name == "train" and args.conv_repeat > 1:
            conv = part[part["source"] == "conv"]
            if len(conv):
                part = pd.concat([part] + [conv] * (args.conv_repeat - 1), ignore_index=True)
                print(f"  train: conversational segments repeated x{args.conv_repeat} "
                      f"({len(conv)} → {len(conv) * args.conv_repeat} rows)")
        ds = Dataset.from_dict({
            "audio": part["path16"].tolist(),
            "sentence": part["transcript"].tolist(),
            "speaker_id": part["speaker_id"].tolist(),
            "source": part["source"].tolist(),
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
            "by_source": {src: {"clips": int(len(g)), "hours": round(g["duration"].sum() / 3600, 2),
                                "unique_speakers": int(g["speaker_id"].nunique())}
                          for src, g in part.groupby("source")},
        }
    report["conv_repeat_in_train"] = args.conv_repeat
    with open(out_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== SPLIT REPORT (also saved to split_report.json) =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDataset saved to: {out_dir / 'dataset'}")
    print("Next step: the smoke test (see README, Phase 6).")


if __name__ == "__main__":
    main()

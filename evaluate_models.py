#!/usr/bin/env python3
"""
Step 3 of the pipeline: measure WER + CER on the held-out test set.

Run it TWICE — once for stock Whisper (the baseline), once for the
fine-tuned model — on the SAME test set. Both runs use the SAME
normalization (imported from normalization.py). That is what makes the
comparison valid.

  Baseline:
      python evaluate_models.py --model openai/whisper-medium \
          --dataset_dir /workspace/prepared/dataset \
          --label baseline_medium --output_dir /workspace/results

  Fine-tuned:
      python evaluate_models.py --model /workspace/runs/medium/final \
          --dataset_dir /workspace/prepared/dataset \
          --label finetuned_medium --output_dir /workspace/results

OUTPUTS (per run)
-----------------
  <output_dir>/<label>.json          headline numbers (WER %, CER %, counts)
  <output_dir>/<label>_utterances.csv  every test clip: reference vs. model
                                       output — useful for eyeballing quality
                                       and for showing buyers real examples.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jiwer
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from normalization import normalize_mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace id (openai/whisper-medium) or local path (/workspace/runs/medium/final)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--label", required=True,
                    help="Short name used for the output files, e.g. baseline_medium")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()

    ds = load_from_disk(args.dataset_dir)[args.split]
    print(f"Evaluating on split '{args.split}': {len(ds)} clips")

    refs_raw, hyps_raw, speakers = [], [], []
    t0 = time.time()
    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[i:i + args.batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        inputs = processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.no_grad():
            ids = model.generate(
                inputs, language="mongolian", task="transcribe",
                max_new_tokens=225,
            )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        hyps_raw.extend(texts)
        refs_raw.extend(batch["sentence"])
        speakers.extend(batch["speaker_id"])
    elapsed = time.time() - t0

    # SAME normalization on BOTH sides — the validity rule.
    rows, refs, hyps = [], [], []
    skipped_empty = 0
    for ref_raw, hyp_raw, spk in zip(refs_raw, hyps_raw, speakers):
        r, h = normalize_mn(ref_raw), normalize_mn(hyp_raw)
        rows.append({"speaker_id": spk, "reference": ref_raw,
                     "model_output": hyp_raw,
                     "reference_normalized": r, "output_normalized": h})
        if r:
            refs.append(r)
            hyps.append(h)
        else:
            skipped_empty += 1

    wer = 100.0 * jiwer.wer(refs, hyps)
    cer = 100.0 * jiwer.cer(refs, hyps)

    result = {
        "model": args.model,
        "label": args.label,
        "split": args.split,
        "clips_scored": len(refs),
        "clips_skipped_empty_reference": skipped_empty,
        "wer_percent": round(wer, 2),
        "cer_percent": round(cer, 2),
        "normalization": "normalization.normalize_mn (shared module)",
        "eval_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = out_dir / f"{args.label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / f"{args.label}_utterances.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== RESULT =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}  (per-clip outputs — worth eyeballing)")


if __name__ == "__main__":
    main()

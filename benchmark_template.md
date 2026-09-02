# Mongolian ASR Benchmark — Whisper Fine-Tune (Run 1)

**Date:** ____  **Prepared by:** MONGOLIANDATA

## Headline result

| Model | WER (%) | CER (%) |
|---|---|---|
| Stock openai/whisper-medium (baseline) | ____ | ____ |
| Fine-tuned whisper-medium (ours) | ____ | ____ |
| **Improvement (absolute)** | ____ | ____ |

Fill from `results/baseline_medium.json` and `results/finetuned_medium.json`.
CER is reported alongside WER because Mongolian is agglutinative — a single
wrong suffix flips a whole "word" in WER terms; CER reflects quality more
fairly. Report both, always.

## Test set

| | Train | Validation | Test |
|---|---|---|---|
| Clips | ____ | ____ | ____ |
| Hours | ____ | ____ | ____ |
| Unique speakers | ____ | ____ | ____ |

Fill from `prepared/split_report.json`.

**Split rule:** by speaker. Every speaker appears in exactly one split; the
test set contains only speakers whose voices appear in **zero** training
clips. The reported numbers therefore measure performance on **unheard
speakers**, not memorized voices.

## Reproducibility note

- **Base model:** openai/whisper-medium (HuggingFace)
- **Fine-tuning recipe:** standard HuggingFace `transformers` Seq2Seq
  fine-tune for Whisper, task=transcribe, language=Mongolian
- **Key hyperparameters:** learning rate ____ ; effective batch size ____ ;
  epochs trained ____ (early stopping on validation WER, patience 2)
- **Text normalization:** identical function applied to references and
  predictions for BOTH models (Unicode NFC, lowercase, punctuation removed,
  digits kept, whitespace collapsed) — see `normalization.py`
- **Split seed:** 42
- **Exact eval command:**
  `python evaluate_models.py --model <MODEL> --dataset_dir <DATASET> --label <LABEL> --output_dir results`
- **Hardware:** 1× NVIDIA A100 (RunPod), training wall-clock ____ hours

Scripts and run-book are versioned; the run can be repeated as the dataset
grows.

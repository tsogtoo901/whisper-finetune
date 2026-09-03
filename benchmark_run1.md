# Mongolian ASR Benchmark — Whisper Fine-Tune (Run 1)

**Date:** 3 September 2026  **Prepared by:** MONGOLIANDATA LLC
**Model artifact:** `mnkh-dataset / models / whisper-medium-mn-run1 / deliverables.tar` (3.06 GB)

## Headline result

Held-out test set: **2,191 utterances, 2.70 hours, 14 speakers** — none of whom
appear in training.

| Model | WER (%) | CER (%) |
|---|---|---|
| Stock `openai/whisper-medium` (baseline, no Mongolian tuning) | 120.22 | 90.37 |
| **Fine-tuned whisper-medium on MONGOLIANDATA read speech** | **6.46** | **2.01** |
| **Improvement (absolute)** | **−113.8 pts** | **−88.4 pts** |

Stock Whisper's WER above 100% means it produced more erroneous tokens than
there were words to recognize — it is unusable for Mongolian out of the box.
The fine-tuned model recognizes roughly 19 of 20 words and 49 of 50 characters
correctly on new voices.

**Strict variant — unseen sentences only.** 2 of the 2,191 test utterances use a
prompt that also occurs in training. Excluding them: **WER 6.45 / CER 2.00 on
2,189 utterances.** The result therefore holds on both axes: speakers never
heard *and* sentences never read.

CER is reported alongside WER because Mongolian is agglutinative — a single
wrong suffix flips a whole "word" in WER terms; CER reflects transcript quality
more fairly.

## Data and split

| | Train | Validation | Test |
|---|---|---|---|
| Utterances | 98,383 | 4,207 | 2,191 |
| Hours | 138.13 | 4.95 | 2.70 |
| Unique speakers | 5 | 1 | 14 |

- **Split rule: by speaker.** Every speaker is wholly inside one split; the
  test set contains only speakers absent from training. Seed 42.
- Skew protection: any speaker holding more than 34% of the target test hours
  is assigned to training, so no single voice can dominate the benchmark. Five
  such speakers (the bulk of the corpus) train; the remaining fifteen form
  validation and test.
- Source: 104,784 approved read-speech utterances (48 kHz/16-bit WAV, verbatim
  Mongolian prompt as transcript). 3 dropped by length filters (>30 s or
  <0.5 s); 0 missing files; 0 empty transcripts.

## Reproducibility note

- **Base model:** `openai/whisper-medium` (769M parameters)
- **Recipe:** standard HuggingFace `transformers` Seq2Seq fine-tune,
  task = transcribe, language = Mongolian. Audio resampled to 16 kHz mono.
- **Software:** transformers 4.57.6, datasets 3.6.0, PyTorch 2.8 / CUDA 12.8,
  jiwer 4.0
- **Hyperparameters:** learning rate 1e-5 (500 warm-up steps, linear decay),
  batch size 16, fp16, gradient checkpointing (non-reentrant), max 10 epochs
  with early stopping (patience 2) on validation WER
- **Training outcome:** stopped after epoch 7; best checkpoint = epoch 5,
  which is the delivered model.

  | Epoch | 1 | 2 | 3 | 4 | **5** | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | Val WER (%) | 9.17 | 6.80 | 6.33 | 5.88 | **5.26** | 5.48 | 5.89 |
  | Val CER (%) | 2.90 | 2.08 | 2.07 | 1.92 | **1.71** | 1.79 | 2.02 |

- **Text normalization** (identical for references and predictions, both
  models): Unicode NFC, lowercase, punctuation removed, digits kept,
  whitespace collapsed — `normalization.py` in the repo.
- **Eval command:**
  `python evaluate_models.py --model <MODEL> --dataset_dir <DATASET> --label <LABEL> --output_dir results`
- **Hardware:** 1× NVIDIA A100 80 GB PCIe (RunPod). Training wall-clock ≈ 11 h;
  full job including preparation and evaluation ≈ $23 of compute.
- **Code:** github.com/tsogtoo901/whisper-finetune (scripts + run-book; no data).
- **Per-utterance outputs** for both models are included in the artifact
  (`results/*_utterances.csv`) for spot-checking or buyer demonstrations.

## Scope

This measures **read speech in clean recording conditions** — scripted
sentences read aloud. It does not claim performance on spontaneous
conversation, telephone audio, or noisy environments. The training corpus is
concentrated in five speakers; the benchmark's purpose is precisely to show how
well that generalizes to new voices, and 6.46% WER on 14 unheard speakers is
the answer.

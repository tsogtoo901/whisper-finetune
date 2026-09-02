# RUN-BOOK — Whisper Fine-Tune for Mongolian ASR (Run 1)

Written for the operator. No coding knowledge assumed. Follow the phases in
order. Anything in `ANGLE_BRACKETS` or `ALL_CAPS` is a placeholder you replace.

**⚠️ Two rules before anything else**

1. **This GitHub repo is PUBLIC. It holds scripts only.** Never commit,
   drag-and-drop, or upload audio files, `manifest.csv`, or B2 credentials
   to the repo. Data goes directly to the GPU instance and nowhere else.
2. **Money is running while the pod exists.** An A100 pod bills by the hour
   whether it is working or idle. Do the phases in one or two sittings and
   terminate the pod (Phase 10) as soon as deliverables are downloaded.

---

## Phase 0 — What you need

- A RunPod account with a payment method and ~$100 of budget headroom.
- The data export (Phase 1 below).
- This repo pushed to GitHub (drag-and-drop the 7 files via the GitHub web
  UI into a new repo, e.g. `whisper-finetune`).

**Cost expectations (estimates, flagged per convention):**

| Item | Rough cost |
|---|---|
| Smoke test (whisper-small, ~30–60 min) | $2–5 |
| Real run (whisper-medium, incl. two evals) | $20–80 depending on dataset hours |
| Idle pod you forgot to terminate | ~$2/hour, forever — see Phase 10 |

## Phase 1 — Export the data (the "manifest contract")

Produce two things:

1. A folder of the WAV clips (48kHz/16-bit, as stored).
2. `manifest.csv` — UTF-8 CSV with **exactly** these columns:

```
file,transcript,speaker_id
clip_000001.wav,Сайн байна уу,SPK_0042
```

- `file` — clip filename, relative to the audio folder
- `transcript` — the verbatim Mongolian transcript
- `speaker_id` — a stable ID per speaker. **This column is mandatory.**
  The train/test split is done by speaker; without it the benchmark is
  invalid and the run must not start.

Only include clips that passed Final Review. Zip both together as `data.zip`
(audio folder + manifest at the top level).

## Phase 2 — Create the pod

On RunPod → Deploy → GPU Pod:

- **GPU:** 1× A100 80GB (Secure Cloud; any region with availability)
- **Template:** the official **RunPod PyTorch** template (PyTorch 2.x, CUDA)
- **Container disk:** 20 GB
- **Volume:** 250 GB, mounted at `/workspace` (default)
- Start the pod, open the **Web Terminal** (or connect via SSH if set up).

## Phase 3 — Get the data onto the pod

**Option A — pull straight from B2 (preferred: datacenter-to-datacenter,
much faster than uploading from Mongolia).** In the pod terminal:

```bash
cd /workspace
pip install b2
b2 account authorize <B2_KEY_ID> <B2_APP_KEY>     # paste keys; never commit them
mkdir -p /workspace/data/wavs/approved
b2 sync b2://<BUCKET_NAME>/approved /workspace/data/wavs/approved
# then upload manifest.csv via RunPod's file-upload button INTO /workspace/data/
# so it lands exactly at /workspace/data/manifest.csv
```

NOTE ON THE PATH: the manifest's `file` column holds full keys like
`approved/MNKH_SPK01/MNKH_SPK01_RS_0006.wav`. Syncing the `approved` folder
INTO `/workspace/data/wavs/approved` makes the on-disk path equal the key, so
`--audio_dir /workspace/data/wavs` resolves every clip. Do not drop the
`approved` part of the destination, or every clip will be reported missing.

**Option B — upload from your computer** (if the export only exists locally):

```bash
# On your own computer (install runpodctl once from runpod.io/docs):
runpodctl send data.zip
# It prints a one-time code. Then in the POD terminal:
runpodctl receive <CODE>
cd /workspace && unzip data.zip -d /workspace/data
```

Either way, you should end with:
- `/workspace/data/wavs/` — the clips
- `/workspace/data/manifest.csv`

## Phase 4 — Install the scripts

In the pod terminal:

```bash
cd /workspace
git clone https://github.com/<YOUR_GITHUB_USER>/whisper-finetune.git
cd whisper-finetune
pip install -r requirements.txt
python normalization.py        # should print "Normalization self-check OK"
```

## Phase 5 — Prepare the dataset

```bash
cd /workspace/whisper-finetune
python prepare_dataset.py \
    --audio_dir /workspace/data/wavs \
    --manifest  /workspace/data/manifest.csv \
    --output_dir /workspace/prepared
```

This resamples everything and prints a **SPLIT REPORT** (clips / hours /
speakers per split). **Copy that report into the chat with the builder
before continuing.** If it warns about fewer than 3 test speakers, stop and
check with the builder — the benchmark would be too weak to quote.

## Phase 6 — Smoke test (cheap validation pass, whisper-small)

Proves the whole pipeline works before spending real money.

```bash
python train.py \
    --dataset_dir /workspace/prepared/dataset \
    --model_name openai/whisper-small \
    --output_dir /workspace/runs/smoke \
    --smoke_test
```

**What "green" looks like:** it runs ~30–60 minutes, prints training loss
going down, prints an eval with `wer` and `cer` numbers (they will be bad —
that's fine, it barely trained), and ends with
`Best model saved to: /workspace/runs/smoke/final`. Any crash: copy the
last ~30 lines of output into the chat with the builder. Do not proceed to
Phase 7 until the smoke test is green.

## Phase 7 — The real run (whisper-medium)

```bash
python train.py \
    --dataset_dir /workspace/prepared/dataset \
    --model_name openai/whisper-medium \
    --output_dir /workspace/runs/medium
```

- Runs for several hours (depends on dataset size). It evaluates after
  every epoch and **stops itself** when validation WER stops improving —
  do not stop it manually unless it has clearly crashed.
- The terminal must stay alive. If your connection is unstable, start it
  inside tmux so it survives disconnects:
  `tmux new -s train` → run the command → detach with `Ctrl-B` then `D` →
  later `tmux attach -t train`.
- If validation `wer` prints as `nan` or the loss explodes upward: stop,
  tell the builder — the fix is usually a lower `--learning_rate 5e-6`.

## Phase 8 — Evaluation (the deliverable numbers)

Run BOTH, in this order, same test set:

```bash
python evaluate_models.py --model openai/whisper-medium \
    --dataset_dir /workspace/prepared/dataset \
    --label baseline_medium --output_dir /workspace/results

python evaluate_models.py --model /workspace/runs/medium/final \
    --dataset_dir /workspace/prepared/dataset \
    --label finetuned_medium --output_dir /workspace/results
```

Each prints a RESULT block with WER % and CER %. Copy both blocks into the
chat with the builder.

## Phase 9 — Download the deliverables

```bash
cd /workspace
zip -r deliverables.zip \
    runs/medium/final \
    results \
    prepared/split_report.json
runpodctl send deliverables.zip     # then `runpodctl receive <CODE>` on your computer
```

**Verify before wiping:** on your own computer, unzip it and confirm the
`final/` folder contains files (several GB — `model.safetensors` is the big
one) and the two results JSONs open and show numbers.

## Phase 10 — Wipe checklist (data-handling rule — do not skip)

Only after Phase 9 is verified locally:

- [ ] `rm -rf /workspace/data /workspace/prepared /workspace/runs` in the pod
- [ ] If B2 keys were used: revoke that application key in the B2 console
- [ ] **Terminate** the pod in the RunPod console (not just Stop)
- [ ] Confirm the attached **volume is deleted** too — a stopped pod's
      volume keeps billing AND keeps a copy of the dataset
- [ ] Confirm in RunPod billing that nothing is still accruing

## Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too big | Re-run with `--batch_size 8 --grad_accum 2` |
| Loss becomes `nan` / explodes | Learning rate too high (catastrophic forgetting) | `--learning_rate 5e-6`, restart the run |
| `command not found: python` | Wrong terminal/template | Use `python3`, or confirm the PyTorch template was selected |
| Download from B2 very slow | Wrong region pairing | Acceptable; or fall back to Option B |
| Pod terminal froze mid-training | Connection dropped | If tmux was used: `tmux attach -t train`. If not, the run may be dead — check `nvidia-smi` for GPU activity |
| Anything else | — | Copy the last ~30 lines of terminal output into the chat with the builder |

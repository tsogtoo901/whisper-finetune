# MNKH conversational draft worker — run-book

Transcribes uploaded conversation sessions with the fine-tuned Mongolian Whisper
(Run 1 weights in B2) and posts drafts back to the portal. Runs on a rented GPU,
not on Railway. Same style as the fine-tune run-book: spin up, run, terminate.

## One-time setup (portal side) — admin, once
1. Railway → `web` → Variables → add `CONV_JOB_API_KEY` = a long random string
   (e.g. run `python3 -c "import secrets;print(secrets.token_urlsafe(32))"` anywhere).
   Keep the value: the worker needs the same one.
2. Railway → Variables → add `CONV_AUTO_DRAFT` = `0` so the portal stops using OpenAI
   for drafts. Sessions then wait in `uploaded` until this worker takes them.
3. Railway redeploys by itself after the variable change.
4. B2 → App Keys → create a key **restricted to bucket `mnkh-dataset`, read only**
   for the worker (it reads `models/` and `conv/`). Keep key id + app key.

## One-time setup (RunPod) — once
1. RunPod → Storage → **New Network Volume**, ~20 GB, in the same region you'll rent
   GPUs in. Name it `mnkh`. This caches the 3 GB model so each run doesn't re-download it.

## Per run (whenever there are `uploaded` sessions — e.g. once a day after recordings)
1. RunPod → Pods → Deploy. Pick a cheap GPU (an A10 / L4 / RTX 4090 is plenty — this is
   inference), template **RunPod PyTorch 2.x**, and attach the `mnkh` network volume
   (mounted at `/workspace`). Start it and open the web terminal.
2. First run only:
   ```
   cd /workspace && git clone https://github.com/tsogtoo901/whisper-finetune.git
   ```
   Later runs: `cd /workspace/whisper-finetune && git pull`
3. ```
   cd /workspace/whisper-finetune/conv_worker
   pip install -r requirements.txt
   ```
4. Export the environment (values from Railway / B2):
   ```
   export MNKH_BASE_URL=https://mgl-speech-portal.us
   export CONV_JOB_API_KEY=<same value as on Railway>
   export B2_KEY_ID=<key id>  B2_APP_KEY=<app key>
   export B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com  B2_BUCKET=mnkh-dataset
   export MODEL_TAR_KEY=models/whisper-medium-mn-run1/deliverables.tar
   export MODEL_DIR=/workspace/model
   ```
5. Run: `python worker.py --once`
   - First ever run downloads the 3 GB model archive from B2 into `/workspace/model`
     (a few minutes). Every later run reuses it.
   - It drafts every pending session and exits. To keep it polling instead (e.g. during a
     recording day), run `python worker.py` — it exits by itself after 20 idle minutes
     (`IDLE_EXIT_MIN`) so an idle pod doesn't burn money.
6. When it exits: RunPod → **Stop** the pod, then **Terminate** it. The network volume
   (and the cached model) survives; the pod's own disk does not, which is fine.

## What "done" looks like
- Portal → Admin → Conversations → the session shows `drafted`, with utterance counts and
  `model whisper-medium-mn-run1`.
- The circle's editor sees it in her queue (**Харилцан ярианы хэсэг** button on the editor
  dashboard) as `drafted` → **Засах**.
- If a session shows `drafting` for more than 2 h (worker died mid-way), the portal reverts
  it to `uploaded` with an "interrupted" note and the next run picks it up again.

## Smoke test (optional, no portal calls)
`python worker.py --dry-run some_track.wav` → prints `start end text` lines for a local WAV.

## Swapping in a better model (Run 2, Run 3 …)
Upload the new deliverables tar to B2, set `MODEL_TAR_KEY` (and `MODEL_VERSION`) to the
new path, delete `/workspace/model`, run again. Nothing else changes — every draft records
which model produced it.

## Notes
- Run 1 was trained on READ speech; expect heavier corrections on the first conversational
  sessions. Those corrected transcripts are the training data for Run 2.
- Cost: minutes of GPU time per session — a few cents per session on an hourly pod. Most of
  a run's cost is the pod's startup, so batch sessions: one run per recording day.
- While no worker is running, new sessions simply wait in `uploaded` (editors see
  "ноорог бэлтгэгдэж байна…"). Nothing is lost.

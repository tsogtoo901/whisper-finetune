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


## Serverless (automatic) — the normal mode once set up
With this set up, nobody runs a pod: the portal wakes the worker when a session is uploaded,
the worker drafts everything waiting and shuts down. Pay only for minutes used.

### One-time setup
1. **RunPod → Serverless → New Endpoint → "Import from GitHub".** Authorize GitHub once, pick
   the `whisper-finetune` repo, branch `main`, **Dockerfile path `conv_worker/Dockerfile`**
   (build context `conv_worker`). RunPod builds the image (10–20 min the first time) and
   rebuilds on every push to `main`.
2. Endpoint settings:
   - GPU: 24 GB class (RTX 4090 / L4 / A5000 tier). Min workers **0**, max workers **1**.
   - Idle timeout **60 s**; execution timeout **1800 s** (a batch of sessions).
   - **Network volume**: attach `mnkh` if the endpoint's region offers it; otherwise create a
     20 GB volume in a serverless region and attach that (the model downloads itself on the
     first job, once).
   - Environment variables (use RunPod Secrets for the three keys):
     ```
     MNKH_BASE_URL=https://mgl-speech-portal.us
     CONV_JOB_API_KEY=<same value as on Railway>
     B2_KEY_ID=<read-only key id>
     B2_APP_KEY=<read-only application key>
     B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com
     B2_BUCKET=mnkh-dataset
     MODEL_TAR_KEY=models/whisper-medium-mn-run1/deliverables.tar
     MODEL_DIR=/runpod-volume/model
     ```
3. RunPod → Settings → API Keys → create a key (restricted to this endpoint if offered).
4. Railway → `web` → Variables: add `RUNPOD_API_KEY` (that key) and `RUNPOD_ENDPOINT_ID`
   (the endpoint's id, shown on its page). `CONV_AUTO_DRAFT` must be `0`. Railway restarts.
5. Test: Admin → Conversations → **Run worker now**. The endpoint's Requests tab shows a job;
   within a few minutes any `uploaded` session is `drafted`.

### How it behaves
- A session reaching `uploaded` triggers one job (repeat triggers inside 2 minutes are
  ignored — one job drains everything waiting).
- The portal's hourly sweep re-triggers if any session has waited 10+ minutes (covers a
  missed trigger or a failed job).
- Admin → Conversations shows the worker mode, the last trigger, and a **Run worker now**
  button; a session's page has the same button while it is `uploaded`.
- A job killed mid-session (execution timeout, GPU shortage) leaves that session `drafting`;
  the portal reverts it after 2 h, or press **Re-queue for drafting** on the session page.
- Cold start is 1–2 minutes (image + model load), then a few minutes per session.

### Swapping in a new model (Run 2 …)
Upload the new tar to B2, change `MODEL_TAR_KEY` (and add `MODEL_VERSION`) on the endpoint,
delete the `model` folder on the volume (or attach a fresh one), and the next job downloads
the new model.

The manual pod routine below stays as the fallback when RunPod serverless is unavailable.

## Manual fallback — per run (whenever there are `uploaded` sessions — e.g. once a day after recordings)
1. RunPod → **Storage → `mnkh` → "Configure Pod with volume"** (this guarantees the volume is
   attached). Pick RTX 4090 (or RTX PRO 4500 / RTX 4000 Ada), template **Runpod PyTorch 2.x**,
   Deploy On-Demand. Wait for Running → Connect → Web Terminal.
2. `ls /workspace` must show `model`, `env.sh`, `whisper-finetune`. If it shows nothing, the
   volume is NOT attached: Stop, Terminate, redeploy from step 1.
3. Four lines, one at a time:
   ```
   cd /workspace/whisper-finetune && git pull
   cd conv_worker && pip install -r requirements.txt
   source /workspace/env.sh
   python worker.py --once
   ```
   Expected output: `generation ids: eos=… pad=…`, then per session
   `A: N regions (M without text)`, `B: …`, `imported N utterances`, then
   `nothing pending; exiting`.
4. RunPod → **Stop**, then **Terminate** the pod. The volume (model, keys, code) stays.

First-time only (once per volume): create the key file, then step 3 works forever:
```
cat > /workspace/env.sh <<'EOF'
export MNKH_BASE_URL="https://mgl-speech-portal.us"
export CONV_JOB_API_KEY="<same value as on Railway>"
export B2_KEY_ID="<read-only key id>"
export B2_APP_KEY="<read-only application key>"
export B2_ENDPOINT="https://s3.us-west-004.backblazeb2.com"
export B2_BUCKET="mnkh-dataset"
export MODEL_TAR_KEY="models/whisper-medium-mn-run1/deliverables.tar"
export MODEL_DIR="/workspace/model"
EOF
cd /workspace && git clone https://github.com/tsogtoo901/whisper-finetune.git
```

If `git pull` says "Your local changes … would be overwritten":
`cd /workspace/whisper-finetune && git checkout -- conv_worker/worker.py && git pull`

If a session shows `drafting` in the portal after a crashed run: Admin → Conversations →
session → **"Draft job stuck? Re-queue for drafting"**, then run again.

## What "done" looks like
- Portal → Admin → Conversations → the session shows `drafted`, with utterance counts and
  `model whisper-medium-mn-run1`.
- The circle's editor sees it in her queue (**Харилцан ярианы хэсэг** button on the editor
  dashboard) as `drafted` → **Засах**.
- Rows are cut only at pauses (never inside a word), ≤ 20 s each; the partner's voice leaking
  through earphones is ignored (a moment counts as yours only if your track is at least as
  loud as your partner's). Speech the model left without text shows as amber rows.
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

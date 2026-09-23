#!/usr/bin/env python3
"""
MNKH conversational draft worker
================================
Polls the MNKH portal's job API for uploaded conversation sessions, transcribes each
speaker's track with the FINE-TUNED Mongolian Whisper (Run 1 weights from B2), and posts
timestamped utterances back. Runs on any machine with a GPU (RunPod / Modal / a laptop
with a GPU); nothing here touches Railway's compute.

Flow per session:
  GET  {MNKH_BASE_URL}/api/conv/pending      -> sessions moved to 'drafting' for us
  download both tracks from B2 (48 kHz mono WAV) -> resample to 16 kHz
  energy-based VAD -> transcribe only voiced regions (avoids Whisper hallucinating
  text during the partner's turns, which are silence on this track)
  POST {MNKH_BASE_URL}/api/conv/import       -> session becomes 'drafted'

Environment (see RUNBOOK.md):
  MNKH_BASE_URL       https://mgl-speech-portal.us
  CONV_JOB_API_KEY    same value as the Railway variable
  B2_KEY_ID / B2_APP_KEY / B2_ENDPOINT / B2_BUCKET   (bucket: mnkh-dataset)
  MODEL_TAR_KEY       models/whisper-medium-mn-run1/deliverables.tar   (default)
  MODEL_DIR           where the extracted model is cached (default ./model)
  IDLE_EXIT_MIN       exit after this many idle minutes (0 = run forever; default 20)
"""
import os, sys, io, json, time, math, wave, array, tarfile, argparse, tempfile, traceback
import urllib.request, urllib.error

BASE     = os.environ.get("MNKH_BASE_URL", "https://mgl-speech-portal.us").rstrip("/")
API_KEY  = os.environ.get("CONV_JOB_API_KEY", "")
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_BUCKET   = os.environ.get("B2_BUCKET", "mnkh-dataset")
MODEL_TAR_KEY = os.environ.get("MODEL_TAR_KEY", "models/whisper-medium-mn-run1/deliverables.tar")
MODEL_DIR   = os.environ.get("MODEL_DIR", "./model")
IDLE_EXIT_MIN = float(os.environ.get("IDLE_EXIT_MIN", "20"))
POLL_SEC    = float(os.environ.get("POLL_SEC", "15"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "whisper-medium-mn-run1")
LANGUAGE    = "mongolian"
TARGET_SR   = 16000

# ───────────────────────────── portal API ──────────────────────────────────────
def api(path, method="GET", body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                                 data=(json.dumps(body).encode() if body is not None else None))
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode() or "{}")

# ───────────────────────────── B2 ──────────────────────────────────────────────
def b2():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=B2_ENDPOINT,
                        aws_access_key_id=os.environ["B2_KEY_ID"], aws_secret_access_key=os.environ["B2_APP_KEY"],
                        config=Config(signature_version="s3v4"))

def ensure_model():
    """Download + extract the fine-tuned model once; return the directory holding config.json."""
    for root, _, files in os.walk(MODEL_DIR):
        if "config.json" in files and any(f.startswith("model") or f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            return root
    os.makedirs(MODEL_DIR, exist_ok=True)
    tar_path = os.path.join(MODEL_DIR, "deliverables.tar")
    if not os.path.exists(tar_path):
        print(f"[worker] downloading model {MODEL_TAR_KEY} …", flush=True)
        b2().download_file(B2_BUCKET, MODEL_TAR_KEY, tar_path)
    print("[worker] extracting …", flush=True)
    with tarfile.open(tar_path) as t:
        t.extractall(MODEL_DIR)
    for root, _, files in os.walk(MODEL_DIR):
        if "config.json" in files:
            return root
    raise RuntimeError("model archive did not contain a config.json (HF format expected)")

# ───────────────────────────── audio ───────────────────────────────────────────
def load_wav_16k(path):
    """48 kHz mono 16-bit WAV -> float32 numpy at 16 kHz."""
    import numpy as np
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    assert ch == 1 and sw == 2, f"expected mono 16-bit, got {ch}ch/{sw*8}-bit"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != TARGET_SR:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(sr, TARGET_SR)
            x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
        except Exception:
            f = sr // TARGET_SR                      # 48k -> 16k: 3:1 average (crude but adequate for ASR)
            m = (len(x) // f) * f
            x = x[:m].reshape(-1, f).mean(axis=1).astype(np.float32)
    return x

def voiced_regions(x, sr=TARGET_SR, frame_ms=20, margin_db=12.0, pad_s=0.25, gap_s=0.6, min_s=0.4):
    """Energy VAD: regions where the track is clearly above its own quiet floor.
    On a dual-channel recording the partner's turns are near-silent on this track,
    and Whisper hallucinates on silence — so we only transcribe voiced regions."""
    import numpy as np
    n = int(sr * frame_ms / 1000)
    m = (len(x) // n) * n
    if m == 0:
        return []
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr * fr).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10); thr = floor + margin_db
    on = db > thr
    regions = []; start = None; last_on = None
    for i, v in enumerate(on):
        t = i * frame_ms / 1000.0
        if v:
            if start is None: start = t
            last_on = t
        elif start is not None and (t - last_on) > gap_s:
            regions.append((start, last_on + frame_ms / 1000.0)); start = None
    if start is not None:
        regions.append((start, last_on + frame_ms / 1000.0))
    out = []
    for s, e in regions:
        s, e = max(0.0, s - pad_s), min(len(x) / sr, e + pad_s)
        if e - s >= min_s:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
    return out

# ───────────────────────────── ASR ─────────────────────────────────────────────
class Transcriber:
    def __init__(self, model_dir):
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if device == 0 else torch.float32
        print(f"[worker] loading {model_dir} on {'cuda' if device == 0 else 'cpu'} …", flush=True)
        self.pipe = pipeline("automatic-speech-recognition", model=model_dir, device=device, torch_dtype=dtype,
                             chunk_length_s=30, stride_length_s=(4, 2))
        self.gen = {"language": LANGUAGE, "task": "transcribe", "num_beams": 1, "no_repeat_ngram_size": 4}

    def segments(self, x16k, sr=TARGET_SR):
        """[(start_sec, end_sec, text)] on the track's own timeline."""
        out = []
        for s, e in voiced_regions(x16k, sr):
            clip = x16k[int(s * sr):int(e * sr)]
            if len(clip) < sr // 4:
                continue
            res = self.pipe({"raw": clip, "sampling_rate": sr}, return_timestamps=True, generate_kwargs=self.gen)
            chunks = res.get("chunks") or [{"timestamp": (0.0, e - s), "text": res.get("text", "")}]
            for c in chunks:
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                t0, t1 = c.get("timestamp") or (0.0, None)
                t0 = float(t0 or 0.0)
                t1 = float(t1) if t1 is not None else min(e - s, t0 + 10.0)
                if t1 <= t0:
                    t1 = t0 + 0.2
                out.append((s + t0, s + t1, text))
        return out

# ───────────────────────────── assembly ────────────────────────────────────────
def to_utterances(seg_by_channel, offsets_ms):
    """Place both channels on the session timeline and enforce the portal's import rule
    (sorted, non-overlapping per channel) by clamping."""
    utts = []
    for ch, segs in seg_by_channel.items():
        base = offsets_ms.get(ch, 0) / 1000.0
        rows = sorted([[max(0.0, base + s), max(0.0, base + e), t] for s, e, t in segs], key=lambda r: r[0])
        for k in range(1, len(rows)):
            if rows[k][0] < rows[k - 1][1]:
                rows[k][0] = rows[k - 1][1]
                if rows[k][1] <= rows[k][0]:
                    rows[k][1] = rows[k][0] + 0.05
        utts += [{"channel": ch, "start_sec": round(s, 3), "end_sec": round(e, 3), "text": t} for s, e, t in rows]
    return utts

def offsets_for(tracks):
    """B relative to A from the client clocks + measured skew (ms), clamped to ±60 s."""
    ta = next((t for t in tracks if t["channel"] == "A"), None)
    tb = next((t for t in tracks if t["channel"] == "B"), None)
    off = {"A": 0, "B": 0}
    if ta and tb and ta.get("client_start_ms") and tb.get("client_start_ms"):
        d = int((tb["client_start_ms"] + int(tb.get("clock_skew_ms") or 0)) - ta["client_start_ms"])
        off["B"] = max(-60000, min(60000, d))
    return off

# ───────────────────────────── main loop ───────────────────────────────────────
def process(session, asr, s3):
    code = session["session_id"]
    print(f"[worker] {code}: {len(session['tracks'])} tracks", flush=True)
    seg_by_channel = {}
    with tempfile.TemporaryDirectory() as tmp:
        for t in session["tracks"]:
            local = os.path.join(tmp, f"{t['channel']}.wav")
            s3.download_file(B2_BUCKET, t["b2_key"], local)
            x = load_wav_16k(local)
            segs = asr.segments(x)
            print(f"[worker] {code} {t['channel']}: {len(segs)} segments from {len(x)/TARGET_SR/60:.1f} min", flush=True)
            seg_by_channel[t["channel"]] = segs
    off = offsets_for(session["tracks"])
    utts = to_utterances(seg_by_channel, off)
    if not utts:
        raise RuntimeError("no utterances produced (silent tracks?)")
    r = api("/api/conv/import", "POST", {"session_id": code, "model_version": MODEL_VERSION,
                                         "alignment_offset_ms": off, "utterances": utts})
    print(f"[worker] {code}: imported {r.get('utterances')} utterances", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process what is pending, then exit")
    ap.add_argument("--dry-run", metavar="WAV", help="transcribe a local WAV and print segments; no API calls")
    args = ap.parse_args()
    model_dir = ensure_model()
    asr = Transcriber(model_dir)
    if args.dry_run:
        for s, e, t in asr.segments(load_wav_16k(args.dry_run)):
            print(f"{s:8.2f} {e:8.2f}  {t}")
        return
    if not API_KEY:
        sys.exit("CONV_JOB_API_KEY is not set")
    s3 = b2(); idle_since = time.time()
    while True:
        try:
            pending = api("/api/conv/pending?limit=5").get("sessions", [])
        except urllib.error.HTTPError as e:
            print(f"[worker] pending failed: HTTP {e.code} ({'set CONV_JOB_API_KEY on Railway' if e.code == 503 else 'check API key'})", flush=True)
            pending = []
        if pending:
            idle_since = time.time()
            for s in pending:
                try:
                    process(s, asr, s3)
                except Exception:
                    print(f"[worker] {s.get('session_id')} FAILED:\n" + traceback.format_exc(), flush=True)
                    # leave it 'drafting'; the portal reverts it to 'uploaded' after 2 h and we retry
        elif args.once:
            print("[worker] nothing pending; exiting (--once)", flush=True); return
        elif IDLE_EXIT_MIN and (time.time() - idle_since) > IDLE_EXIT_MIN * 60:
            print(f"[worker] idle {IDLE_EXIT_MIN:.0f} min; exiting to save GPU time", flush=True); return
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()

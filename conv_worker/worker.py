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
                                 # Cloudflare blocks Python's default "Python-urllib" client; a named agent passes.
                                 headers={"X-API-Key": API_KEY, "Content-Type": "application/json",
                                          "User-Agent": "MNKH-conv-worker/1.0"},
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

def frame_db(x, sr=TARGET_SR, frame_ms=20):
    """Per-frame level in dB for a 16 kHz float signal."""
    import numpy as np
    n = int(sr * frame_ms / 1000)
    m = (len(x) // n) * n
    if m == 0:
        return np.zeros(0)
    fr = x[:m].reshape(-1, n)
    return 20 * np.log10(np.sqrt((fr * fr).mean(axis=1)) + 1e-9)

def voiced_regions(x, sr=TARGET_SR, frame_ms=20, margin_db=12.0, pad_s=0.25, gap_s=0.6, min_s=0.4,
                   x_other=None, other_shift_s=0.0, own_db=6.0, max_s=20.0):
    """Regions of MY speech on this track's own timeline.
    A frame counts when (1) it is clearly above my track's own quiet floor and (2) my track is at
    least as loud as my partner's track at the same moment (x_other, whose timeline is mine
    shifted by other_shift_s). The partner's voice leaking through my earphones is quieter on
    my track than on theirs, so it is skipped; genuine overlap (both loud) is kept on both.
    Regions are merged across pauses shorter than gap_s, padded, and regions longer than max_s
    are split at their quietest internal dip. Rows are cut ONLY here — never inside a region —
    so a boundary can never fall in the middle of a word."""
    import numpy as np
    db = frame_db(x, sr, frame_ms)
    if len(db) == 0:
        return []
    fs = frame_ms / 1000.0
    floor = np.percentile(db, 10); thr = floor + margin_db
    on = db > thr
    if x_other is not None:
        dbo = frame_db(x_other, sr, frame_ms)
        shift = int(round(other_shift_s / fs))      # my frame i ↔ other frame i + shift
        other = np.full(len(db), -1e9)
        lo, hi = max(0, -shift), min(len(db), len(dbo) - shift)
        if hi > lo:
            other[lo:hi] = dbo[lo + shift:hi + shift]
        on = on & (db >= other - own_db)
    regions = []; start = None; last_on = None
    for i, v in enumerate(on):
        t = i * fs
        if v:
            if start is None: start = t
            last_on = t
        elif start is not None and (t - last_on) > gap_s:
            regions.append((start, last_on + fs)); start = None
    if start is not None:
        regions.append((start, last_on + fs))
    total = len(x) / sr
    merged = []
    for a, b in regions:
        a, b = max(0.0, a - pad_s), min(total, b + pad_s)
        if b - a < min_s:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    stack = list(reversed(merged))
    while stack:
        a, b = stack.pop()
        if b - a <= max_s:
            out.append((a, b)); continue
        L = b - a; lo, hi = int((a + 0.3 * L) / fs), int((b - 0.3 * L) / fs); mid = (lo + hi) / 2.0
        if hi - lo < 2:
            out.append((a, b)); continue
        k = min(range(lo, hi), key=lambda i: (round(float(db[i]), 1), abs(i - mid)))
        cut = (k + 0.5) * fs
        stack.append([cut, b]); stack.append([a, cut])
    return out

# ───────────────────────────── ASR ─────────────────────────────────────────────
class Transcriber:
    def __init__(self, model_dir):
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if device == 0 else torch.float32
        print(f"[worker] loading {model_dir} on {'cuda' if device == 0 else 'cpu'} …", flush=True)
        try:
            self.pipe = pipeline("automatic-speech-recognition", model=model_dir, device=device, dtype=dtype,
                                 chunk_length_s=30, stride_length_s=(4, 2))
        except TypeError:   # transformers < 4.56 uses the old name
            self.pipe = pipeline("automatic-speech-recognition", model=model_dir, device=device, torch_dtype=dtype,
                                 chunk_length_s=30, stride_length_s=(4, 2))
        # Newer transformers (4.5x) can crash in Whisper's timestamp processor when the model's
        # generation config stores eos/pad ids as a LIST ("slice indices must be integers");
        # older ones (<4.45) cannot read this model's tokenizer file at all. Normalise the ids
        # to plain ints so the current 4.x line works.
        # The pipeline keeps its OWN copy of the generation config and passes that to generate(),
        # so fix both copies, and additionally force plain-int ids through generate_kwargs
        # (those override whatever config generate() ends up with).
        def _as_int(v):
            return int(v[0]) if isinstance(v, (list, tuple)) and v else v
        for gc in (getattr(self.pipe.model, "generation_config", None), getattr(self.pipe, "generation_config", None)):
            if gc is None: continue
            for k in ("eos_token_id", "pad_token_id", "decoder_start_token_id", "bos_token_id"):
                v = getattr(gc, k, None)
                if isinstance(v, (list, tuple)): setattr(gc, k, _as_int(v))
        tok = self.pipe.tokenizer
        eos = _as_int(getattr(self.pipe.model.generation_config, "eos_token_id", None)) or tok.eos_token_id
        pad = _as_int(getattr(self.pipe.model.generation_config, "pad_token_id", None)) or tok.pad_token_id or eos
        self.gen = {"language": LANGUAGE, "task": "transcribe", "num_beams": 1, "no_repeat_ngram_size": 4,
                    "eos_token_id": int(eos), "pad_token_id": int(pad)}
        print(f"[worker] generation ids: eos={eos} pad={pad}", flush=True)

    def segments(self, x16k, sr=TARGET_SR, x_other=None, other_shift_s=0.0):
        """[(start_sec, end_sec, text)] on the track's own timeline: exactly one row per
        speech region (regions are cut only at pauses, ≤ 20 s), so no boundary can land inside
        a word. Empty text is kept as an empty row so the editor still hears the audio."""
        out = []
        for s, e in voiced_regions(x16k, sr, x_other=x_other, other_shift_s=other_shift_s):
            clip = x16k[int(s * sr):int(e * sr)]
            if len(clip) < sr // 4:
                continue
            try:
                res = self.pipe({"raw": clip, "sampling_rate": sr}, return_timestamps=False, generate_kwargs=self.gen)
                text = (res.get("text") or "").strip()
            except Exception as ex:
                print(f"[worker] region {s:.1f}-{e:.1f}s failed ({ex}); leaving empty", flush=True)
                text = ""
            out.append((s, e, text))
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
    off = offsets_for(session["tracks"])
    off_s = {ch: v / 1000.0 for ch, v in off.items()}
    seg_by_channel = {}
    with tempfile.TemporaryDirectory() as tmp:
        audio = {}
        for t in session["tracks"]:
            local = os.path.join(tmp, f"{t['channel']}.wav")
            s3.download_file(B2_BUCKET, t["b2_key"], local)
            audio[t["channel"]] = load_wav_16k(local)
        for ch, x in audio.items():
            other_ch = "B" if ch == "A" else "A"
            x_other = audio.get(other_ch)
            # my file time t ↔ session time t + off[ch] ↔ other's file time t + off[ch] - off[other]
            shift = off_s.get(ch, 0.0) - off_s.get(other_ch, 0.0)
            segs = asr.segments(x, x_other=x_other, other_shift_s=shift)
            n_empty = sum(1 for _, _, t in segs if not t)
            print(f"[worker] {code} {ch}: {len(segs)} regions ({n_empty} without text) from {len(x)/TARGET_SR/60:.1f} min", flush=True)
            seg_by_channel[ch] = segs
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

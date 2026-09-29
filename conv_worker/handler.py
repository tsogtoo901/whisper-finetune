#!/usr/bin/env python3
"""
RunPod Serverless entry point for the MNKH conversational draft worker.

The portal calls the endpoint's /run whenever a session reaches 'uploaded' (and the hourly
sweep re-checks). Each job loads the fine-tuned model once (cached on the attached network
volume at /runpod-volume/model), drains every pending session through the portal's job API,
and returns. Idle time costs nothing; the worker container exits after the endpoint's idle
timeout. All configuration comes from the endpoint's environment variables — see RUNBOOK.md.
"""
import os, sys, time, traceback

os.environ.setdefault("MODEL_DIR", "/runpod-volume/model")   # network volume mount on serverless
os.environ.setdefault("IDLE_EXIT_MIN", "0")

import runpod                                 # pip install runpod
import worker                                 # the same code the manual pod runs

_asr = None
_s3 = None

def _init():
    global _asr, _s3
    if _asr is None:
        model_dir = worker.ensure_model()
        _asr = worker.Transcriber(model_dir)
        _s3 = worker.b2()

def handler(job):
    """Drain the portal's pending sessions. Returns a summary the RunPod console shows."""
    reason = (job.get("input") or {}).get("reason", "")
    t0 = time.time()
    if not worker.API_KEY:
        return {"error": "CONV_JOB_API_KEY is not set on the endpoint"}
    try:
        _init()
    except Exception:
        return {"error": "model load failed", "detail": traceback.format_exc()[-1500:]}
    done, failed = [], []
    for _round in range(50):                  # each round takes up to 5 sessions
        try:
            pending = worker.api("/api/conv/pending?limit=5").get("sessions", [])
        except Exception as e:
            return {"error": f"portal unreachable: {e}", "reason": reason, "done": done, "failed": failed}
        if not pending:
            break
        for s in pending:
            code = s.get("session_id")
            try:
                worker.process(s, _asr, _s3)
                done.append(code)
            except Exception:
                print(f"[worker] {code} FAILED:\n" + traceback.format_exc(), flush=True)
                failed.append(code)           # left 'drafting'; the portal re-queues it after 2 h
    return {"reason": reason, "done": done, "failed": failed, "seconds": round(time.time() - t0, 1)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

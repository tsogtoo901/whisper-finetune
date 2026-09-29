# MNKH conversational draft worker — RunPod Serverless image.
# Built by RunPod from the whisper-finetune repo (path: conv_worker). Same base image as the
# manual pod, so the environment cannot drift between runs.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt runpod
COPY worker.py handler.py /app/

ENV MODEL_DIR=/runpod-volume/model \
    PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/handler.py"]

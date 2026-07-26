# MiniGPT (TensorFlow) Dockerfile
# -------------------------------
# NOTE: build with the REPO ROOT as context so the shared `common/` folder is
# available, e.g. from the repo root:
#     docker build -f tensorflow/Dockerfile -t minigpt-tf .
# (docker-compose in this folder already sets context: .. for you.)

# ---- Base image (TensorFlow + Keras pre-installed, CPU) ---------------------
FROM tensorflow/tensorflow:2.16.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# ---- Copy shared code + the TensorFlow project ------------------------------
COPY common /app/common              # shared tokenizer + data
COPY tensorflow /app/tensorflow      # the TensorFlow implementation

# ---- Run from the tensorflow folder -----------------------------------------
WORKDIR /app/tensorflow
CMD ["python", "chat.py"]

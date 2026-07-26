# MiniGPT (PyTorch) Dockerfile
# ----------------------------
# NOTE: build with the REPO ROOT as context so the shared `common/` folder is
# available, e.g. from the repo root:
#     docker build -f pytorch/Dockerfile -t minigpt-pytorch .
# (docker-compose in this folder already sets context: .. for you.)

# ---- Base image -------------------------------------------------------------
FROM python:3.11-slim

# ---- Environment ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ---- Install CPU-only PyTorch (small image) ---------------------------------
COPY pytorch/requirements.txt /app/pytorch/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ---- Copy shared code + the PyTorch project ---------------------------------
COPY common /app/common          # shared tokenizer + data
COPY pytorch /app/pytorch        # the PyTorch implementation

# ---- Run from the pytorch folder --------------------------------------------
WORKDIR /app/pytorch
CMD ["python", "chat.py"]

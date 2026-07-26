# MiniGPT (TensorFlow) — Docker Guide

Run the TensorFlow version of MiniGPT in a container — no local Python/TensorFlow needed.

---

## Files

```
tensorflow/
├── Dockerfile            # TensorFlow CPU image
├── .dockerignore
└── docker-compose.yml    # train / chat shortcuts
```

---

## Prerequisites

Install **Docker Desktop** and verify:
```bash
docker --version
docker compose version
```

---

## Quick Start (docker compose)

```bash
# from inside tensorflow/
docker compose build
docker compose run --rm train     # trains, saves checkpoints/ on host
docker compose run --rm chat      # interactive chat
```

---

## Plain Docker commands

```bash
# Build
docker build -t minigpt-tf .

# Train (persist checkpoints on host)
# PowerShell:
docker run --rm -v ${PWD}/checkpoints:/app/checkpoints minigpt-tf python train.py
# Bash:
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints minigpt-tf python train.py

# Chat (interactive needs -it)
docker run --rm -it -v ${PWD}/checkpoints:/app/checkpoints minigpt-tf python chat.py

# Generate
docker run --rm -v ${PWD}/checkpoints:/app/checkpoints minigpt-tf python generate.py "twinkle"
```

---

## Notes

- The official `tensorflow/tensorflow` image is **CPU-only** and Linux-based, which is
  perfect here (MiniGPT trains in ~1 minute on CPU).
- The `checkpoints/` volume mount persists your trained model on the host.
- Interactive chat needs `-it` (or the `chat` compose service with `tty: true`).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker: command not found` | Install Docker Desktop, restart terminal |
| Chat exits immediately | Use `-it` or the `chat` compose service |
| `No trained model found` | Run the `train` step first |
| Model gone after restart | Mount `-v .../checkpoints:/app/checkpoints` |

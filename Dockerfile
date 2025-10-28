# syntax=docker/dockerfile:1.6

########## Stage 1: Private Git-Repo holen (inkl. LFS) ##########
FROM debian:bookworm-slim AS fetcher
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/*

ARG REPO_PATH="mariusbrd/buba_dashboard_prod"   # <- korrigierter Default
ARG REPO_REF="main"

# Token NUR als Build-Secret mounten
RUN --mount=type=secret,id=gh_token \
    set -eux; \
    git lfs install --system; \
    token="$(cat /run/secrets/gh_token)"; \
    git clone --depth 1 --branch "$REPO_REF" \
      "https://oauth2:${token}@github.com/${REPO_PATH}.git" /src; \
    git -C /src remote set-url origin "https://github.com/${REPO_PATH}.git"; \
    git -C /src lfs pull

########## Stage 2: Runtime ##########
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Europe/Berlin
WORKDIR /app

# Code zuerst kopieren
COPY --from=fetcher /src/ /app/

# Dependencies installieren (falls vorhanden)
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8080
CMD ["python", "app.py"]

# syntax=docker/dockerfile:1.6
FROM debian:bookworm-slim AS fetcher
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git git-lfs \
 && rm -rf /var/lib/apt/lists/*

ARG REPO_PATH="mariusbrd/buba_dashboard_prod"
ARG REPO_REF="main"

RUN set -eux; \
    git lfs install --system; \
    git clone --depth 1 --branch "$REPO_REF" "https://github.com/${REPO_PATH}.git" /src; \
    git -C /src lfs pull

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Europe/Berlin
WORKDIR /app
COPY --from=fetcher /src/ /app/
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    pip install --no-cache-dir gunicorn && \
    chmod +x /app/start.sh
EXPOSE 8080
CMD ["/app/start.sh"]

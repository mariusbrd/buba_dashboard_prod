# BuBa Dashboard – Docker Deployment

Dieses Dokument beschreibt den **reproduzierbaren** Docker-Deploy des BuBa Dashboards unter dem Projektnamen **`buba-dashboard`**.  
Der Build klont die GitHub-Repo **1:1 im Build** (inkl. großer Dateien via **Git LFS**), läuft **nicht als root**, startet mit **gunicorn** und enthält einen **Healthcheck**. Es werden **keine Volumes** verwendet, damit nichts die im Image enthaltene Repo-Struktur überdeckt.

---

## Voraussetzungen

- Linux Server mit:
  - Docker Engine & Docker Compose v2  
  - Internetzugriff auf `https://github.com/mariusbrd/buba_dashboard_prod`

---

## Versionen

- Docker version: 28.5.1, build e180ab8  
- Docker Compose version: v2.40.2

---

## Clean Deploy

Bereinigt alte Docker-Reste und legt ein frisches Arbeitsverzeichnis für einen sauberen Neu-Deploy an.

```bash
sudo docker compose down --remove-orphans --volumes || true
sudo docker system prune -f
cd ~
sudo rm -rf ~/buba-deploy
mkdir -p ~/buba-deploy
cd ~/buba-deploy
```

---

## Commit-SHA der Hauptbranch ermitteln & prüfen

Ermittelt die neueste Commit-SHA der `main`-Branch, gibt sie aus, validiert, dass sie gültig ist, und prüft beim Remote, dass genau diese SHA existiert.

```bash
SHA=$(git ls-remote https://github.com/mariusbrd/buba_dashboard_prod.git -h refs/heads/main | cut -f1)
echo "$SHA"
[ -n "$SHA" ] && [ ${#SHA} -ge 7 ] || { echo "SHA leer/ungültig"; exit 1; }
git ls-remote https://github.com/mariusbrd/buba_dashboard_prod.git "$SHA"
```

---

## Dateien erstellen

### `.dockerignore`

Schließt unnötige Dateien (Git-Ordner, Caches, Logs) vom Build-Kontext aus.

```bash
sudo tee .dockerignore >/dev/null <<'EOF'
.git
**/__pycache__/
**/*.pyc
**/.ipynb_checkpoints
*.log
EOF
```

### `Dockerfile`

Zweistufige Dockerfile:  
**Fetcher-Stage** klont die GitHub-Repo 1:1 (inkl. LFS), pinnt auf Branch/Tag/SHA und schreibt die Commit-ID nach `.build_commit`.  
**Runtime-Stage** nutzt `python:3.11-slim`, installiert Requirements (mit Pip-Cache), kopiert den Code, legt optional leere Verzeichnisse an, wechselt auf einen Nicht-Root-User, setzt einen **Healthcheck ohne curl** (exec-Form) und startet produktionsreif via **gunicorn** (`app:server`).

```bash
sudo tee Dockerfile >/dev/null <<'EOF'
# syntax=docker/dockerfile:1.6

########## Stage 1: Repo holen (Branch/Tag/SHA) inkl. Git LFS ##########
FROM debian:bookworm-slim AS fetcher
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git git-lfs curl \
 && git lfs install --system \
 && rm -rf /var/lib/apt/lists/*

ARG REPO_PATH="mariusbrd/buba_dashboard_prod"
ARG REPO_REF="main"   # darf Branch, Tag oder Commit-SHA sein

# Universal-Fetch: init + fetch --depth 1 + checkout FETCH_HEAD
RUN mkdir -p /src && cd /src \
 && git init \
 && git remote add origin "https://github.com/${REPO_PATH}.git" \
 && git fetch --depth 1 origin "${REPO_REF}" \
 && git checkout --detach FETCH_HEAD \
 && git lfs pull \
 && git rev-parse HEAD > /src/.build_commit

########## Stage 2: Runtime ##########
FROM python:3.11-slim

# Build-Args (für Labels)
ARG REPO_PATH="mariusbrd/buba_dashboard_prod"
ARG REPO_REF="unknown"

# OCI-Labels
LABEL org.opencontainers.image.title="BuBa Dashboard" \
      org.opencontainers.image.source="https://github.com/${REPO_PATH}" \
      org.opencontainers.image.revision="${REPO_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies (mit BuildKit-Pip-Cache schneller)
COPY --from=fetcher /src/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r /app/requirements.txt

# App-Code 1:1 übernehmen
COPY --from=fetcher /src/ /app/

# (Optional) leere Ordner aus Manifest erzeugen, wenn .docker.dirs existiert
RUN if [ -f "/app/.docker.dirs" ]; then xargs -I{} mkdir -p "/app/{}" < /app/.docker.dirs; fi

# Nicht als root laufen
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck ohne curl (einzeilig, prüft /_dash-health oder /)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
  ["python","-c","import sys,urllib.request,contextlib; ok=0\nfor p in ('/_dash-health','/'):\n  try:\n    import urllib.error\n    with contextlib.closing(urllib.request.urlopen('http://127.0.0.1:8080'+p,timeout=3)) as r:\n      ok=1 if 200<=r.getcode()<500 else ok\n      break\n  except Exception:\n    pass\nsys.exit(0 if ok else 1)"]

EXPOSE 8080

# Produktionsstart via gunicorn
# WICHTIG: Falls dein WSGI-Objekt nicht "app:server" ist, hier anpassen.
CMD ["gunicorn","-b","0.0.0.0:8080","app:server","--workers","2","--threads","4","--timeout","120"]
EOF
```

### `docker-compose.yml`

Definiert den Compose-Stack für das Dashboard: baut das Image aus der `Dockerfile`, erstellt den Container **buba-dashboard**, veröffentlicht **Port 8080→8080**, setzt Umgebungsvariablen, aktiviert **restart unless-stopped**, erzwingt **no-new-privileges**, vergibt den Netzwerk-Alias und hinterlegt OCI-Labels.

```bash
sudo tee docker-compose.yml >/dev/null <<'EOF'
name: buba-dashboard

services:
  app:
    image: buba-dashboard:latest
    container_name: buba-dashboard
    build:
      context: .
      dockerfile: Dockerfile
      args:
        REPO_PATH: "mariusbrd/buba_dashboard_prod"
        REPO_REF: "main"   # wird beim Build per --build-arg auf die SHA gesetzt
    ports:
      - "8080:8080"
    environment:
      TZ: Europe/Berlin
      SCENARIO_FORCE_REFRESH: "1"
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    networks:
      default:
        aliases:
          - buba-dashboard
    labels:
      org.opencontainers.image.title: "BuBa Dashboard"
      org.opencontainers.image.source: "https://github.com/mariusbrd/buba_dashboard_prod"
      org.opencontainers.image.revision: "${REPO_REF:-main}"
EOF
```

---

## Build & Start

Baut das Image frisch (ohne Cache, mit aktuellem Base-Image, auf die Commit-SHA `$SHA` gepinnt) und startet den Container im Hintergrund.

```bash
sudo docker compose build --no-cache --pull --build-arg REPO_REF="$SHA"
sudo docker compose up -d
```

---

## Verifizieren

Prüft den **gebauten Commit** (`.build_commit`), die **Struktur** im Image, bestätigt **openpyxl**, zeigt **Status** und streamt **Logs**.

```bash
# exakt gebaute Commit-ID
sudo docker compose exec app sh -lc 'cat /app/.build_commit'

# Struktur sichtbar machen
sudo docker compose exec app sh -lc 'ls -la /app | sed -n "1,150p"'

# openpyxl installiert?
sudo docker compose exec app python -c 'import openpyxl; print("openpyxl", openpyxl.__version__)' || true

# Status / Logs
sudo docker compose ps
sudo docker compose logs -f --tail=200 app
```

---

## Update auf neuen Repo-Stand (später)

Holt die neueste Commit-SHA der `main`-Branch, baut das Image neu darauf gepinnt, startet den aktualisierten Container und prüft anschließend, dass genau diese SHA im laufenden Build steckt.

```bash
cd ~/buba-deploy
NEW_SHA=$(git ls-remote https://github.com/mariusbrd/buba_dashboard_prod.git -h refs/heads/main | cut -f1)
echo "$NEW_SHA"
sudo docker compose build --no-cache --pull --build-arg REPO_REF="$NEW_SHA"
sudo docker compose up -d
sudo docker compose exec app sh -lc 'cat /app/.build_commit'
```

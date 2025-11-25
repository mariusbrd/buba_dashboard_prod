# BuBa Dashboard – Docker Development Environment

Dieses Dokument beschreibt den **reproduzierbaren** Docker-Deploy des BuBa Dashboards für die **lokale Entwicklung** unter dem Projektnamen **`buba-dashboard-dev`**.  
Der Build klont das GitHub-Repo **1:1 im Build** (inkl. großer Dateien via **Git LFS**), verwendet **Volumes für Live-Entwicklung**, startet mit **Python Development Server** und enthält einen **Healthcheck**. Entwicklungs-Daten bleiben in **Named Volumes** erhalten.

---

## Voraussetzungen

- Entwicklungs-Rechner mit:
  - Docker Engine & Docker Compose v2  
  - Internetzugriff auf `https://github.com/mariusbrd/buba_dashboard`
  - Git (optional, für lokale Entwicklung)

---

## Versionen

- Docker version: 28.5.1, build e180ab8  
- Docker Compose version: v2.40.2

---

## Clean Deploy

Bereinigt alte Docker-Reste und legt ein frisches Arbeitsverzeichnis für einen sauberen Neu-Deploy an.

```bash
docker compose down --remove-orphans --volumes || true
docker system prune -f
cd ~
rm -rf ~/buba-dev
mkdir -p ~/buba-dev
cd ~/buba-dev
```

---

## Commit-SHA der Hauptbranch ermitteln & prüfen

Ermittelt die neueste Commit-SHA der `main`-Branch, gibt sie aus, validiert, dass sie gültig ist, und prüft beim Remote, dass genau diese SHA existiert.

```bash
SHA=$(git ls-remote https://github.com/mariusbrd/buba_dashboard.git -h refs/heads/main | cut -f1)
echo "$SHA"
[ -n "$SHA" ] && [ ${#SHA} -ge 7 ] || { echo "SHA leer/ungültig"; exit 1; }
git ls-remote https://github.com/mariusbrd/buba_dashboard.git "$SHA"
```

---

## Dateien erstellen

### `.dockerignore`

Schließt unnötige Dateien (Git-Ordner, Caches, Logs) vom Build-Kontext aus.

```bash
tee .dockerignore > /dev/null <<'EOF'
.git
**/__pycache__/
**/*.pyc
**/.ipynb_checkpoints
*.log
.env
.venv
EOF
```

### `Dockerfile.dev`

Zweistufige Dockerfile für **Development**:  
**Fetcher-Stage** klont das GitHub-Repo 1:1 (inkl. LFS), pinnt auf Branch/Tag/SHA und schreibt die Commit-ID nach `.build_commit`.  
**Runtime-Stage** nutzt `python:3.11-slim`, installiert Requirements plus **Development-Tools** (ipdb, pytest, black, flake8), kopiert den Code, legt Verzeichnisse für Volumes an, und startet einen **Development-Server** (kein gunicorn).

```bash
tee Dockerfile.dev > /dev/null <<'EOF'
# syntax=docker/dockerfile:1.6

########## Stage 1: Repo holen (Branch/Tag/SHA) inkl. Git LFS ##########
FROM debian:bookworm-slim AS fetcher
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git git-lfs \
 && git lfs install --system \
 && rm -rf /var/lib/apt/lists/*

ARG REPO_PATH="mariusbrd/buba_dashboard"
ARG REPO_REF="main"

# Universal-Fetch: init + fetch --depth 1 + checkout FETCH_HEAD
RUN mkdir -p /src && cd /src \
 && git init \
 && git remote add origin "https://github.com/${REPO_PATH}.git" \
 && git fetch --depth 1 origin "${REPO_REF}" \
 && git checkout --detach FETCH_HEAD \
 && git lfs pull \
 && git rev-parse HEAD > /src/.build_commit

########## Stage 2: Runtime (Development) ##########
FROM python:3.11-slim

# Build-Args (für Labels)
ARG REPO_PATH="mariusbrd/buba_dashboard"
ARG REPO_REF="unknown"

# OCI-Labels
LABEL org.opencontainers.image.title="BuBa Dashboard (Development)" \
      org.opencontainers.image.source="https://github.com/${REPO_PATH}" \
      org.opencontainers.image.revision="${REPO_REF}" \
      org.opencontainers.image.description="Development environment for BuBa Dashboard"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=development

WORKDIR /app

# Dependencies (mit BuildKit-Pip-Cache schneller)
COPY --from=fetcher /src/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r /app/requirements.txt

# Development-Tools installieren
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ipdb pytest black flake8

# App-Code 1:1 übernehmen
COPY --from=fetcher /src/ /app/

# Verzeichnisse für Volumes vorbereiten
RUN mkdir -p /app/forecaster/user_presets \
             /app/forecaster/trained_models \
             /app/scenario/data \
             /app/loader/financial_cache

# Als root laufen (für Development, einfachere Volume-Permissions)
# In Production würde man einen non-root user nutzen

# Healthcheck (einzeilig, prüft ob App läuft)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/', timeout=3)" || exit 1

EXPOSE 8080

# Development-Server (kein gunicorn)
CMD ["python", "app.py"]
EOF
```

### `docker-compose.yml`

Definiert den Compose-Stack für das **Development-Dashboard**: baut das Image aus `Dockerfile.dev`, erstellt den Container **buba-dashboard-dev**, veröffentlicht **Port 8080→8080**, setzt Development-Umgebungsvariablen, verwendet **Named Volumes** für Persistenz, aktiviert **restart unless-stopped**, und hinterlegt OCI-Labels.

```bash
tee docker-compose.yml > /dev/null <<'EOF'
name: buba-dashboard-dev

services:
  app:
    image: buba-dashboard:dev
    container_name: buba-dashboard-dev
    build:
      context: .
      dockerfile: Dockerfile.dev
      args:
        REPO_PATH: "mariusbrd/buba_dashboard"
        REPO_REF: "main"   # wird beim Build per --build-arg auf die SHA gesetzt
    ports:
      - "8080:8080"
    environment:
      TZ: Europe/Berlin
      FLASK_ENV: development
      PYTHONUNBUFFERED: "1"
      SCENARIO_FORCE_REFRESH: "1"
    # Named Volumes für Entwicklungs-Persistenz
    volumes:
      - buba_dev_user_presets:/app/forecaster/user_presets
      - buba_dev_trained_models:/app/forecaster/trained_models
      - buba_dev_scenario_data:/app/scenario/data
      - buba_dev_cache:/app/loader/financial_cache
    restart: unless-stopped
    networks:
      default:
        aliases:
          - buba-dashboard-dev
    labels:
      org.opencontainers.image.title: "BuBa Dashboard (Development)"
      org.opencontainers.image.source: "https://github.com/mariusbrd/buba_dashboard"
      org.opencontainers.image.revision: "${REPO_REF:-main}"

# Named Volumes (überleben Container-Löschung)
volumes:
  buba_dev_user_presets:
    name: buba_dev_user_presets
  buba_dev_trained_models:
    name: buba_dev_trained_models
  buba_dev_scenario_data:
    name: buba_dev_scenario_data
  buba_dev_cache:
    name: buba_dev_cache
EOF
```

---

## Build & Start

Baut das Image frisch (ohne Cache, mit aktuellem Base-Image, auf die Commit-SHA `$SHA` gepinnt) und startet den Container im Hintergrund.

```bash
docker compose build --no-cache --pull --build-arg REPO_REF="$SHA"
docker compose up -d
```

---

## Verifizieren

Prüft den **gebauten Commit** (`.build_commit`), die **Struktur** im Image, bestätigt **openpyxl**, zeigt **Status** und streamt **Logs**.

```bash
# Exakt gebaute Commit-ID
docker compose exec app sh -lc 'cat /app/.build_commit'

# Struktur sichtbar machen
docker compose exec app sh -lc 'ls -la /app | sed -n "1,150p"'

# openpyxl installiert?
docker compose exec app python -c 'import openpyxl; print("openpyxl", openpyxl.__version__)' || true

# Development-Tools verfügbar?
docker compose exec app python -c 'import ipdb, pytest, black, flake8; print("Dev-Tools OK")' || true

# Status / Logs
docker compose ps
docker compose logs -f --tail=200 app
```

---

## Development-Workflows

### Shell im Container öffnen

```bash
docker compose exec app /bin/bash
```

### Python-Abhängigkeiten hinzufügen

```bash
# 1. Lokal requirements.txt bearbeiten (auf dem Host)
echo "neue-library==1.0.0" >> requirements.txt

# 2. Container neu bauen
docker compose build

# 3. Container neu starten
docker compose up -d
```

### Code-Änderungen (via Volumes)

Falls Sie Live-Code-Editing möchten (optional), erweitern Sie `docker-compose.yml`:

```yaml
volumes:
  # Bestehende Volumes...
  - ./app.py:/app/app.py              # Haupt-App
  - ./loader:/app/loader              # Loader-Modul
  - ./forecaster:/app/forecaster      # Forecaster-Modul
  - ./geospacial:/app/geospacial      # Geospacial-Modul
  - ./overview:/app/overview          # Overview-Modul
  - ./scenario:/app/scenario          # Scenario-Modul (bereits vorhanden)
```

**Hinweis**: Nach Änderungen an Code muss Dash eventuell manuell neu geladen werden (Browser-Refresh).

### Tests ausführen

```bash
# Im Container pytest ausführen
docker compose exec app pytest tests/ -v

# Code-Qualität prüfen
docker compose exec app flake8 app.py loader/ forecaster/ --max-line-length=120

# Code formatieren (Dry-run)
docker compose exec app black app.py loader/ forecaster/ --check
```

---

## Volumes verwalten

### Volumes anzeigen

```bash
docker volume ls | grep buba_dev
```

### Volume-Inhalt inspizieren

```bash
# User Presets
docker run --rm -v buba_dev_user_presets:/data alpine ls -lh /data

# Cache
docker run --rm -v buba_dev_cache:/data alpine du -sh /data
```

### Volumes sichern

```bash
# Backup erstellen
docker run --rm -v buba_dev_user_presets:/data -v $(pwd):/backup alpine tar czf /backup/presets_backup.tar.gz -C /data .

# Wiederherstellen
docker run --rm -v buba_dev_user_presets:/data -v $(pwd):/backup alpine tar xzf /backup/presets_backup.tar.gz -C /data
```

### Volumes zurücksetzen

```bash
# VORSICHT: Löscht alle Daten im Volume!
docker volume rm buba_dev_cache
docker compose up -d  # Volume wird neu erstellt
```

---

## Update auf neuen Repo-Stand (später)

Holt die neueste Commit-SHA der `main`-Branch, baut das Image neu darauf gepinnt, startet den aktualisierten Container und prüft anschließend, dass genau diese SHA im laufenden Build steckt.

```bash
cd ~/buba-dev
NEW_SHA=$(git ls-remote https://github.com/mariusbrd/buba_dashboard.git -h refs/heads/main | cut -f1)
echo "$NEW_SHA"
docker compose build --no-cache --pull --build-arg REPO_REF="$NEW_SHA"
docker compose up -d
docker compose exec app sh -lc 'cat /app/.build_commit'
```

**Hinweis**: Die Named Volumes bleiben erhalten – Ihre Entwicklungsdaten gehen nicht verloren!

---

## Debugging

### Logs in Echtzeit verfolgen

```bash
docker compose logs -f app
```

### Detaillierte Container-Info

```bash
docker inspect buba-dashboard-dev
```

### In laufenden Container attachen (für ipdb)

```bash
# 1. Breakpoint in Code setzen
#    import ipdb; ipdb.set_trace()

# 2. An Container attachen
docker attach buba-dashboard-dev

# 3. Mit Ctrl+P, Ctrl+Q detachen (ohne Container zu stoppen)
```

### Healthcheck-Status

```bash
docker inspect --format='{{.State.Health.Status}}' buba-dashboard-dev
```

---

## Cleanup

### Container stoppen und entfernen (Volumes behalten)

```bash
docker compose down
```

### Container + Volumes entfernen (ALLE Daten löschen!)

```bash
docker compose down -v
```

### Images aufräumen

```bash
docker rmi buba-dashboard:dev
```

### Komplettes Cleanup

```bash
docker compose down -v
docker system prune -af
docker volume prune -f
```

---

## Unterschiede zu Production-Deployment

| Feature | Development | Production |
|---------|------------|------------|
| **Repository** | `mariusbrd/buba_dashboard` | `mariusbrd/buba_dashboard_prod` |
| **Server** | `python app.py` (Debug) | `gunicorn` (Production) |
| **User** | root (Entwicklung) | appuser (Security) |
| **Volumes** | Named Volumes (Persistenz) | Keine Volumes |
| **Dev-Tools** | pytest, ipdb, black, flake8 | Keine |
| **Auto-Reload** | Optional via Volumes | Nein |
| **Zweck** | Entwicklung & Testing | Production-Deploy |

---

## Troubleshooting

### Port bereits belegt

```bash
# Port-Nutzung prüfen
netstat -an | grep 8080

# In docker-compose.yml anderen Port nutzen:
# ports:
#   - "8081:8080"
```

### Container startet nicht

```bash
# Detaillierte Logs
docker compose logs app

# Build-Logs anzeigen
docker compose build --progress=plain
```

### Volume-Permission-Probleme

```bash
# In Container als root ausführen
docker compose exec -u root app chown -R 1000:1000 /app/forecaster/user_presets
```

---

## Weitere Informationen

- **[README.md](README.md)** - Projekt-Übersicht und Schnellstart
- **[README_Deployment.md](README_Deployment.md)** - Production-Deployment
- **[README_Portainer.md](README_Portainer.md)** - Portainer-Deployment
- **[Makefile](Makefile)** - Build-Automatisierung

---

**Viel Erfolg bei der Entwicklung! 🚀**

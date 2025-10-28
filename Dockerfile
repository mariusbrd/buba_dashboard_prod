# syntax=docker/dockerfile:1.6

########## Stage 1: Private Git-Repo komplett holen ##########
FROM debian:bookworm-slim AS fetcher
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Passe REPO_REF auf Tag/Commit-SHA an für reproduzierbare Builds
ARG REPO_PATH="mariusbrd/buba_dashbord_prod"
ARG REPO_REF="main"

# Token nur zur Buildzeit als Secret mounten (landet NICHT im finalen Image)
RUN --mount=type=secret,id=gh_token \
    set -eu; \
    token="$(cat /run/secrets/gh_token)"; \
    git clone --depth 1 --branch "$REPO_REF" \
      "https://oauth2:${token}@github.com/${REPO_PATH}.git" /src; \
    # Remote-URL ohne Token hinterlegen (nur Hygiene)
    git -C /src remote set-url origin "https://github.com/${REPO_PATH}.git"

########## Stage 2: Runtime ##########
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Europe/Berlin
WORKDIR /app

# 1) Dependencies zuerst installieren (Cache-freundlich)
#    Falls die Repo keine requirements.txt hat, kannst du diesen Block anpassen/entfernen.
COPY --from=fetcher /src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# 2) Gesamte Repo als App-Code übernehmen
COPY --from=fetcher /src/ /app/

# 3) Extern erreichbarer Port (laut app.py intern 8080)
EXPOSE 8080

# 4) Start – wichtig: __main__-Pfad der app.py wird ausgeführt
CMD ["python", "app.py"]

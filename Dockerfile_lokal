# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Berlin

# (optional) Zeitzone installieren, damit keine Interaktion nötig ist
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# zuerst nur requirements kopieren, damit Layer gecached werden können
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# jetzt den kompletten Projektinhalt kopieren
# (inkl. overview/, forecaster/, scenario/, loader/ usw.)
COPY . /app

# interner Port aus app.py
EXPOSE 8080

# Standard: direkter Start von app.py (wichtig, da __main__-Block scenario/instructor triggert)
CMD ["python", "app.py"]

# BuBa Dashboard – Portainer Deployment Guide

Diese Anleitung beschreibt, wie Sie das BuBa Dashboard als **Development-Umgebung** in **Portainer** deployen.

---

## 📋 Voraussetzungen

- **Portainer** installiert und läuft
- **Internet-Zugriff** auf GitHub (`https://github.com/mariusbrd/buba_dashboard.git`)
- **Docker Engine** auf dem Portainer-Host

---

## 🚀 Deployment in Portainer

### Methode 1: Repository-basiert (Empfohlen)

Diese Methode klont das Repository automatisch und hält es aktuell.

#### Schritt-für-Schritt:

1. **Öffnen Sie Portainer Web-UI**
   - Navigieren Sie zu Ihrem Portainer-Dashboard

2. **Neuen Stack erstellen**
   - `Stacks` → `+ Add stack`
   - **Name**: `buba-dashboard-dev`

3. **Build-Methode wählen**
   - ✅ **Repository** auswählen

4. **Repository-Konfiguration**
   ```
   Repository URL: https://github.com/mariusbrd/buba_dashboard.git
   Repository reference: main
   Compose path: docker-compose.portainer.yaml
   ```

5. **Environment Variables** (optional)
   ```
   FLASK_ENV=development
   SCENARIO_FORCE_REFRESH=1
   ```

6. **Deploy the stack**
   - Klicken Sie auf `Deploy the stack`
   - Warten Sie, bis der Build abgeschlossen ist (kann 2-5 Minuten dauern)

7. **Zugriff prüfen**
   - Dashboard: `http://<portainer-host>:8080`
   - Status in Portainer: `Stacks` → `buba-dashboard-dev`

---

### Methode 2: Web Editor (Manuelle Konfiguration)

Falls Sie die Konfiguration anpassen möchten:

1. **Neuen Stack erstellen**
   - `Stacks` → `+ Add stack`
   - **Name**: `buba-dashboard-dev`

2. **Build-Methode wählen**
   - ✅ **Web editor** auswählen

3. **Compose-Datei einfügen**
   
   Kopieren Sie den Inhalt von `docker-compose.portainer.yaml`:

   ```yaml
   version: '3.8'
   
   services:
     buba-dev:
       image: buba-dashboard:dev
       container_name: buba-dashboard-dev
       
       build:
         context: https://github.com/mariusbrd/buba_dashboard.git
         dockerfile: Dockerfile.portainer
       
       ports:
         - "8080:8080"
       
       environment:
         FLASK_ENV: development
         SCENARIO_FORCE_REFRESH: "1"
         TZ: Europe/Berlin
       
       volumes:
         - buba_user_presets:/app/forecaster/user_presets
         - buba_trained_models:/app/forecaster/trained_models
         - buba_scenario_data:/app/scenario/data
         - buba_cache:/app/loader/financial_cache
       
       restart: unless-stopped
       
       healthcheck:
         test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/', timeout=3)"]
         interval: 30s
         timeout: 10s
         retries: 3
   
   volumes:
     buba_user_presets:
     buba_trained_models:
     buba_scenario_data:
     buba_cache:
   ```

4. **Deploy**
   - Klicken Sie auf `Deploy the stack`

---

## 📊 Stack-Management in Portainer

### Stack-Status überwachen

1. **Dashboard-Ansicht**
   - `Stacks` → `buba-dashboard-dev`
   - Zeigt laufende Container, Volumes und Netzwerke

2. **Container-Logs**
   - Klicken Sie auf den Container `buba-dashboard-dev`
   - Tab: `Logs`
   - ✅ Auto-refresh aktivieren

3. **Healthcheck-Status**
   - Container-Detailansicht
   - Zeigt `healthy` wenn alles läuft

### Stack aktualisieren

#### Option A: Git Pull (bei Repository-Methode)

1. `Stacks` → `buba-dashboard-dev`
2. Klicken Sie auf `Pull and redeploy`
3. Portainer holt die neuesten Änderungen und baut neu

#### Option B: Manuelles Rebuild

1. `Stacks` → `buba-dashboard-dev`
2. Klicken Sie auf `Update the stack`
3. Aktivieren Sie `Re-pull image and redeploy`
4. Klicken Sie auf `Update`

### Stack stoppen/starten

- **Stoppen**: `Stacks` → `buba-dashboard-dev` → `Stop this stack`
- **Starten**: `Stacks` → `buba-dashboard-dev` → `Start this stack`
- **Entfernen**: `Stacks` → `buba-dashboard-dev` → `Delete this stack`

---

## 💾 Volumes und Datenpersistenz

### Verwaltete Volumes

Der Stack erstellt folgende Named Volumes:

| Volume | Zweck | Pfad im Container |
|--------|-------|-------------------|
| `buba_dev_user_presets` | Benutzer-Konfigurationen | `/app/forecaster/user_presets` |
| `buba_dev_trained_models` | ML-Modelle | `/app/forecaster/trained_models` |
| `buba_dev_scenario_data` | Szenario-Analysen | `/app/scenario/data` |
| `buba_dev_cache` | API-Cache | `/app/loader/financial_cache` |

### Volume-Backup in Portainer

1. **Volume anzeigen**
   - `Volumes` → Wählen Sie Volume (z.B. `buba_dev_cache`)

2. **Backup erstellen**
   - `Browse` → Dateien einsehen
   - Container mit Volume verbinden für manuellen Export

3. **Volume löschen** (Vorsicht!)
   - Nur wenn Stack gestoppt ist
   - Alle Daten im Volume gehen verloren

---

## 🔧 Konfiguration anpassen

### Port ändern

In der Compose-Datei:

```yaml
ports:
  - "8081:8080"  # Host-Port:Container-Port
```

### Umgebungsvariablen

```yaml
environment:
  FLASK_ENV: production        # development oder production
  SCENARIO_FORCE_REFRESH: "0"  # "0" oder "1"
  TZ: Europe/Berlin
  # Weitere Variablen nach Bedarf
```

### Resource-Limits setzen

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

---

## 🐛 Troubleshooting

### Container startet nicht

1. **Logs prüfen**
   - `Containers` → `buba-dashboard-dev` → `Logs`

2. **Häufige Probleme**
   - **Port belegt**: Anderer Container nutzt Port 8080
     - Lösung: Port in Compose-Datei ändern
   - **Build fehlgeschlagen**: GitHub nicht erreichbar
     - Lösung: Internet-Verbindung prüfen
   - **Dependencies fehlen**: requirements.txt nicht gefunden
     - Lösung: Stack neu deployen

### Dashboard nicht erreichbar

1. **Healthcheck prüfen**
   - Container-Status sollte `healthy` sein

2. **Port-Mapping prüfen**
   - `Containers` → `buba-dashboard-dev` → `Published Ports`
   - Sollte `8080:8080` zeigen

3. **Firewall**
   - Stellen Sie sicher, dass Port 8080 auf dem Host erreichbar ist

### Build-Fehler

**Fehler: "Cannot clone repository"**
- Lösung: Prüfen Sie die GitHub-URL
- Stellen Sie sicher, dass das Repository öffentlich ist

**Fehler: "Requirements installation failed"**
- Lösung: Prüfen Sie die Logs auf fehlende System-Dependencies
- Eventuell Dockerfile.portainer anpassen

---

## 🔄 Updates und Maintenance

### Automatische Updates (mit Watchtower)

Optional können Sie Watchtower für automatische Updates nutzen:

```yaml
services:
  watchtower:
    image: containrrr/watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    command: --interval 3600 buba-dashboard-dev
```

### Manuelle Updates

Regelmäßig (z.B. wöchentlich):

1. Stack-Seite öffnen
2. `Pull and redeploy` klicken
3. Neue Version wird automatisch deployed
4. Logs auf Fehler prüfen

---

## 📈 Monitoring

### In Portainer

1. **Resource-Nutzung**
   - `Containers` → `buba-dashboard-dev`
   - Zeigt CPU, RAM, Network, Disk

2. **Logs**
   - Real-time Log-Streaming
   - Filter und Suche verfügbar

### Externes Monitoring

Falls Sie externe Tools verwenden:

- **Prometheus**: Metrics-Endpoint könnte hinzugefügt werden
- **Grafana**: Dashboard für Visualisierung
- **Uptime Kuma**: Healthcheck-Monitoring

---

## 🔐 Sicherheit

### Best Practices

- ✅ **Rootless Container**: Dockerfile nutzt nicht root (optional aktivierbar)
- ✅ **Read-Only Volumes**: Für bestimmte Pfade aktivierbar
- ✅ **Network Isolation**: Eigenes Bridge-Netzwerk
- ✅ **Resource Limits**: CPU/RAM begrenzen

### Secrets Management

Falls Sie API-Keys benötigen:

1. **Portainer Secrets erstellen**
   - `Secrets` → `Add secret`
   - Name: `buba_api_key`

2. **In Compose referenzieren**
   ```yaml
   secrets:
     - buba_api_key
   
   secrets:
     buba_api_key:
       external: true
   ```

---

## 📚 Weitere Ressourcen

- **[README.md](README.md)** - Projekt-Übersicht
- **[README_Development.md](README_Development.md)** - Lokale Entwicklung
- **[README_Deployment.md](README_Deployment.md)** - Produktions-Deployment
- **[Portainer Dokumentation](https://docs.portainer.io/)** - Offizielle Portainer-Docs

---

## 🆘 Support

Bei Problemen:

1. **Portainer-Logs prüfen**
2. **GitHub Issues** im Repository öffnen
3. **Portainer Community** für Portainer-spezifische Fragen

---

**Viel Erfolg mit Ihrem BuBa Dashboard in Portainer! 🚀**

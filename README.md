# BuBa Dashboard

Ein professionelles Dashboard zur Geldvermögensbildung mit Prognose-Suite, Szenario-Analysen und geografischen Visualisierungen.

## 🚀 Schnellstart

### Voraussetzungen

- **Python 3.11+** (erforderlich)
- **Docker & Docker Compose** (optional, nur für Container-Deployment)
- **Git** (für Versionskontrolle)

### Installation und Start

Das Projekt bietet zwei Build-Systeme für verschiedene Betriebssysteme:

#### **Linux/macOS** (mit Makefile):

```bash
# 1. Alle verfügbaren Befehle anzeigen
make help

# 2. Abhängigkeiten installieren
make install

# 3. Anwendung starten
make run
```

#### **Windows** (mit PowerShell):

```powershell
# 1. Alle verfügbaren Befehle anzeigen
.\make.ps1 help

# 2. Abhängigkeiten installieren
.\make.ps1 install

# 3. Anwendung starten
.\make.ps1 run
```

> **Hinweis für Windows-Nutzer**: Falls PowerShell-Skripte blockiert sind, führen Sie einmalig aus:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### **Windows** (Schnellstart ohne Konfiguration):

Für den einfachsten Start unter Windows können Sie auch einfach auf die Datei `quick-start.bat` doppelklicken. Diese:
- Prüft automatisch die Python-Installation
- Installiert alle Abhängigkeiten
- Startet das Dashboard

Das Dashboard ist dann verfügbar unter: **http://localhost:8080**

---

## 📋 Makefile-Befehle

### Setup und Installation

| Befehl | Beschreibung |
|--------|--------------|
| `make check-python` | Überprüft Python-Version (3.11+) |
| `make check-deps` | Prüft System-Abhängigkeiten |
| `make install` | Installiert alle Python-Pakete |
| `make setup` | Vollständiges Setup |
| `make upgrade-deps` | Aktualisiert alle Pakete |

### Entwicklung

| Befehl | Beschreibung |
|--------|--------------|
| `make run` | Startet die Anwendung (Produktionsmodus) |
| `make dev` | Startet mit Auto-Reload (Entwicklungsmodus) |
| `make test` | Führt Tests aus |
| `make lint` | Prüft Code-Qualität |
| `make format` | Formatiert Python-Code |

### Docker

| Befehl | Beschreibung |
|--------|--------------|
| `make docker-build` | Baut Docker-Image |
| `make docker-run` | Startet Anwendung in Docker |
| `make docker-stop` | Stoppt Docker-Container |
| `make docker-logs` | Zeigt Container-Logs |
| `make docker-restart` | Neustart der Container |
| `make docker-shell` | Öffnet Shell im Container |
| `make docker-clean` | Entfernt Docker-Ressourcen |

### Wartung

| Befehl | Beschreibung |
|--------|--------------|
| `make clean` | Entfernt temporäre Dateien |
| `make clean-all` | Vollständiger Cleanup (inkl. Docker) |
| `make info` | Zeigt Projektinformationen |

---

## 🛠️ Manuelle Installation (ohne Makefile)

Falls du das Makefile nicht verwenden möchtest:

```bash
# 1. Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # oder: venv\Scripts\activate (Windows)

# 2. Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt

# 3. Anwendung starten
python app.py
```

---

## 🐳 Docker-Deployment

### Mit Makefile:

```bash
make docker-build
make docker-run
```

### Mit Docker Compose (manuell):

```bash
docker-compose build
docker-compose up -d
```

### Logs anzeigen:

```bash
make docker-logs
# oder
docker-compose logs -f
```

---

## 📁 Projektstruktur

```
buba_dashboard/
├── app.py                    # Hauptanwendung (Dash-App)
├── Makefile                  # Build- und Deployment-Automatisierung
├── requirements.txt          # Python-Abhängigkeiten
├── Dockerfile               # Container-Definition
├── docker-compose.yaml      # Docker-Orchestrierung
│
├── forecaster/              # Prognose-Module
│   ├── forecaster_main.py   # Hauptlogik für Prognosen
│   ├── user_presets/        # Gespeicherte Benutzer-Konfigurationen
│   └── trained_models/      # Trainierte ML-Modelle
│
├── geospacial/              # Geografische Analysen
│   ├── geospacial_main.py   # Geo-Visualisierungen
│   └── geospacial_viz.py    # Karten und Regionen
│
├── loader/                  # Daten-Loader
│   ├── loader.py            # Datenimport und -verarbeitung
│   └── instructor.py        # Daten-Instruktionen
│
├── overview/                # Übersichts-Dashboard
│   └── overview_main.py     # KPIs und Hauptansicht
│
└── scenario/                # Szenario-Analysen
    ├── scenario_main.py     # Szenario-Berechnungen
    ├── scenario_analyzer.py # Analysewerkzeuge
    └── scenario_dataloader.py # Szenario-Daten
```

---

## 🔧 Entwicklung

### Code-Qualität prüfen:

```bash
make lint
```

### Code formatieren:

```bash
make format
```

### Tests ausführen:

```bash
make test
```

---

## 🌐 Features

- **📊 Übersichts-Dashboard**: KPIs und interaktive Charts zur Geldvermögensbildung
- **🔮 Prognose-Suite**: Decision Tree und ARIMAX-Modelle für Vorhersagen
- **🎯 Szenario-Analyse**: Regionale Anpassungen und What-If-Szenarien
- **🗺️ Geo-Visualisierung**: Geografische Darstellung von Finanzdaten
- **💾 Datenpersistenz**: Automatisches Caching und Modellspeicherung

---

## 📝 Umgebungsvariablen

Folgende Umgebungsvariablen können gesetzt werden:

| Variable | Beschreibung | Standard |
|----------|--------------|----------|
| `SCENARIO_FORCE_REFRESH` | Szenario-Daten neu laden | `0` |
| `FORECASTER_DATA_DIR` | Datenverzeichnis | `./data` |
| `FORECASTER_PRESETS_DIR` | Preset-Verzeichnis | `./forecaster/user_presets` |
| `FORECASTER_MODELS_DIR` | Modell-Verzeichnis | `./forecaster/trained_models` |

---

## 🔍 Troubleshooting

### Port bereits belegt

```bash
make check-ports
```

Falls Port 8080 belegt ist, kannst du in `docker-compose.yaml` einen anderen Port setzen.

### Python-Version zu alt

```bash
make check-python
```

Das Projekt benötigt Python 3.11 oder höher.

### Dependencies fehlen

```bash
make clean
make install
```

---

## 📚 Weitere Dokumentation

- [Deployment-Anleitung](README_Deployment.md)
- [GitHub-Informationen](README_Github.md)

---

## 👥 Autor

Data Science Team - BuBa Dashboard Projekt

---

## 📄 Lizenz

Internes Projekt - Alle Rechte vorbehalten

---

## 📝 Changelog

### [2025-11-25] - Build-Automatisierung hinzugefügt

#### ✨ Neue Features

**Build-System für einfacheren Projekt-Start:**
- **Makefile** - Vollständige Build-Automatisierung für Linux/macOS mit Befehlen für Setup, Entwicklung, Docker und Wartung
- **make.ps1** - PowerShell-Skript mit identischer Funktionalität für Windows-Nutzer
- **quick-start.bat** - Ein-Klick-Startskript für Windows (keine Kommandozeilen-Kenntnisse erforderlich)

#### 📋 Verfügbare Make-Befehle

Alle drei Build-Systeme bieten die gleichen Funktionen:

**Setup & Installation:**
- `check-python` - Überprüft Python-Version (3.11+)
- `check-deps` - Prüft System-Abhängigkeiten
- `install` - Installiert alle Python-Pakete
- `setup` - Vollständiges Setup
- `upgrade-deps` - Aktualisiert alle Pakete

**Entwicklung:**
- `run` - Startet die Anwendung (Produktionsmodus)
- `dev` - Startet mit Auto-Reload (Entwicklungsmodus)
- `test` - Führt Tests aus
- `lint` - Prüft Code-Qualität
- `format` - Formatiert Python-Code

**Docker:**
- `docker-build` - Baut Docker-Image
- `docker-run` - Startet Anwendung in Docker
- `docker-stop` - Stoppt Docker-Container
- `docker-logs` - Zeigt Container-Logs
- `docker-restart` - Neustart der Container
- `docker-shell` - Öffnet Shell im Container
- `docker-clean` - Entfernt Docker-Ressourcen

**Wartung:**
- `clean` - Entfernt temporäre Dateien
- `clean-all` - Vollständiger Cleanup (inkl. Docker)
- `info` - Zeigt Projektinformationen
- `check-ports` - Prüft Port-Verfügbarkeit

#### 🎯 Vorteile für Dritte

Diese Build-Tools ermöglichen es Dritten, das Projekt mit **minimalem Aufwand** zu starten:

**Linux/macOS:**
```bash
make install && make run
```

**Windows mit PowerShell:**
```powershell
.\make.ps1 install
.\make.ps1 run
```

**Windows (einfachster Weg):**
- Doppelklick auf `quick-start.bat`
- Fertig! 🚀

#### 📚 Dokumentation

Die README wurde umfassend erweitert mit:
- Plattform-spezifischen Schnellstart-Anleitungen
- Vollständiger Befehlsreferenz für alle Build-Systeme
- Detaillierter Projektstruktur-Übersicht
- Troubleshooting-Sektion
- Feature-Beschreibungen



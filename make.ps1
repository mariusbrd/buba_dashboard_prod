#!/usr/bin/env pwsh
# ============================================================================
# BuBa Dashboard - PowerShell Build Script (Windows Alternative zu Makefile)
# ============================================================================
# Verwendung: .\make.ps1 <command>
# Beispiele:
#   .\make.ps1 help
#   .\make.ps1 install
#   .\make.ps1 run
# ============================================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Konfiguration
$PYTHON = "python"
$PIP = "pip"
$DOCKER = "docker"
$DOCKER_COMPOSE = "docker-compose"
$APP_NAME = "gvb-dashboard"
$PORT = 8080

# Farben für Ausgabe
function Write-Cyan { Write-Host $args -ForegroundColor Cyan }
function Write-Green { Write-Host $args -ForegroundColor Green }
function Write-Yellow { Write-Host $args -ForegroundColor Yellow }
function Write-Red { Write-Host $args -ForegroundColor Red }

# ============================================================================
# Hilfsfunktionen
# ============================================================================

function Show-Help {
    Write-Cyan ""
    Write-Cyan "════════════════════════════════════════════════════════════"
    Write-Cyan "  BuBa Dashboard - Verfügbare Befehle"
    Write-Cyan "════════════════════════════════════════════════════════════"
    Write-Host ""
    Write-Green "Setup und Installation:"
    Write-Host "  check-python        Überprüft Python-Version (3.11+)"
    Write-Host "  check-deps          Prüft System-Abhängigkeiten"
    Write-Host "  install             Installiert alle Python-Pakete"
    Write-Host "  setup               Vollständiges Setup"
    Write-Host "  upgrade-deps        Aktualisiert alle Pakete"
    Write-Host ""
    Write-Green "Entwicklung:"
    Write-Host "  run                 Startet die Anwendung (Produktionsmodus)"
    Write-Host "  dev                 Startet mit Auto-Reload (Entwicklungsmodus)"
    Write-Host "  test                Führt Tests aus"
    Write-Host "  lint                Prüft Code-Qualität"
    Write-Host "  format              Formatiert Python-Code"
    Write-Host ""
    Write-Green "Docker:"
    Write-Host "  docker-build        Baut Docker-Image"
    Write-Host "  docker-run          Startet Anwendung in Docker"
    Write-Host "  docker-stop         Stoppt Docker-Container"
    Write-Host "  docker-logs         Zeigt Container-Logs"
    Write-Host "  docker-restart      Neustart der Container"
    Write-Host "  docker-shell        Öffnet Shell im Container"
    Write-Host "  docker-clean        Entfernt Docker-Ressourcen"
    Write-Host ""
    Write-Green "Wartung:"
    Write-Host "  clean               Entfernt temporäre Dateien"
    Write-Host "  clean-all           Vollständiger Cleanup (inkl. Docker)"
    Write-Host "  info                Zeigt Projektinformationen"
    Write-Host ""
}

# ============================================================================
# Setup und Installation
# ============================================================================

function Check-Python {
    Write-Cyan "Prüfe Python-Version..."
    try {
        $version = & $PYTHON --version 2>&1
        if ($version -match "Python 3\.([1-9][1-9]|[2-9][0-9])") {
            Write-Green "✓ Python-Version OK: $version"
            return $true
        } else {
            Write-Red "Fehler: Python 3.11 oder höher wird benötigt!"
            Write-Red "Aktuelle Version: $version"
            return $false
        }
    } catch {
        Write-Red "Fehler: Python nicht gefunden!"
        return $false
    }
}

function Check-Dependencies {
    Write-Cyan "Prüfe System-Abhängigkeiten..."
    
    if (-not (Check-Python)) {
        return $false
    }
    
    # Docker prüfen
    try {
        & $DOCKER --version | Out-Null
    } catch {
        Write-Yellow "Warnung: Docker nicht gefunden (nur für Container-Deployment benötigt)"
    }
    
    # Docker Compose prüfen
    try {
        & $DOCKER_COMPOSE --version | Out-Null
    } catch {
        Write-Yellow "Warnung: Docker Compose nicht gefunden (nur für Container-Deployment benötigt)"
    }
    
    Write-Green "✓ Abhängigkeiten geprüft"
    return $true
}

function Install-Dependencies {
    Write-Cyan "Installiere Python-Abhängigkeiten..."
    
    if (-not (Check-Python)) {
        return $false
    }
    
    & $PIP install --upgrade pip
    & $PIP install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Green "✓ Installation abgeschlossen"
        return $true
    } else {
        Write-Red "Fehler bei der Installation!"
        return $false
    }
}

function Setup-Project {
    if (Install-Dependencies) {
        Write-Green "✓ Setup abgeschlossen - Projekt ist bereit!"
        return $true
    }
    return $false
}

function Upgrade-Dependencies {
    Write-Cyan "Aktualisiere Abhängigkeiten..."
    & $PIP install --upgrade -r requirements.txt
    Write-Green "✓ Abhängigkeiten aktualisiert"
}

# ============================================================================
# Entwicklung
# ============================================================================

function Run-App {
    Write-Cyan "Starte BuBa Dashboard..."
    Write-Green "Dashboard läuft auf: http://localhost:$PORT"
    & $PYTHON app.py
}

function Run-Dev {
    Write-Cyan "Starte BuBa Dashboard (Entwicklungsmodus)..."
    Write-Green "Dashboard läuft auf: http://localhost:$PORT"
    Write-Yellow "Auto-Reload aktiviert - Änderungen werden automatisch übernommen"
    $env:FLASK_ENV = "development"
    & $PYTHON app.py
}

function Run-Tests {
    Write-Cyan "Führe Tests aus..."
    if (Test-Path "tests") {
        & $PYTHON -m pytest tests/ -v
    } else {
        Write-Yellow "Keine Tests gefunden (tests/ Verzeichnis existiert nicht)"
    }
}

function Run-Lint {
    Write-Cyan "Prüfe Code-Qualität..."
    try {
        & flake8 app.py forecaster/ geospacial/ loader/ overview/ scenario/ --max-line-length=120 --ignore=E501,W503
        Write-Green "✓ Code-Qualität OK"
    } catch {
        Write-Yellow "flake8 nicht installiert - überspringe Lint-Check"
        Write-Yellow "Installation: pip install flake8"
    }
}

function Run-Format {
    Write-Cyan "Formatiere Code..."
    try {
        & black app.py forecaster/ geospacial/ loader/ overview/ scenario/
        Write-Green "✓ Code formatiert"
    } catch {
        Write-Yellow "black nicht installiert"
        Write-Yellow "Installation: pip install black"
    }
}

# ============================================================================
# Docker
# ============================================================================

function Docker-Build {
    Write-Cyan "Baue Docker-Image..."
    & $DOCKER_COMPOSE build
    if ($LASTEXITCODE -eq 0) {
        Write-Green "✓ Docker-Image gebaut"
    }
}

function Docker-Run {
    Write-Cyan "Starte Anwendung in Docker..."
    & $DOCKER_COMPOSE up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Green "✓ Anwendung läuft in Docker auf http://localhost:$PORT"
        Write-Yellow "Logs anzeigen: .\make.ps1 docker-logs"
    }
}

function Docker-Stop {
    Write-Cyan "Stoppe Docker-Container..."
    & $DOCKER_COMPOSE down
    Write-Green "✓ Container gestoppt"
}

function Docker-Logs {
    Write-Cyan "Docker-Logs (Strg+C zum Beenden):"
    & $DOCKER_COMPOSE logs -f
}

function Docker-Restart {
    Docker-Stop
    Docker-Run
}

function Docker-Shell {
    Write-Cyan "Öffne Shell im Container..."
    & $DOCKER exec -it $APP_NAME /bin/bash
}

function Docker-Clean {
    Write-Cyan "Räume Docker-Ressourcen auf..."
    & $DOCKER_COMPOSE down -v --rmi all
    Write-Green "✓ Docker-Ressourcen entfernt"
}

# ============================================================================
# Wartung
# ============================================================================

function Clean-Temp {
    Write-Cyan "Räume temporäre Dateien auf..."
    
    # Python Cache
    Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -File -Filter "*.pyo" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Temporäre Daten
    $paths = @(
        "forecaster\trained_models\*",
        "forecaster\trained_outputs\*",
        "scenario\data\*",
        "scenario\models_scenario\*",
        "scenario\scenario_cache\*",
        "loader\runs\*",
        "loader\gvb_output.parquet",
        "loader\gvb_output.xlsx",
        "scenario\.scenario_month.stamp"
    )
    
    foreach ($path in $paths) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Green "✓ Cleanup abgeschlossen"
}

function Clean-All {
    Clean-Temp
    Docker-Clean
    Write-Green "✓ Vollständiger Cleanup abgeschlossen"
}

# ============================================================================
# Utilities
# ============================================================================

function Show-Info {
    Write-Cyan ""
    Write-Cyan "════════════════════════════════════════════════════════════"
    Write-Cyan "  BuBa Dashboard - Projektinformationen"
    Write-Cyan "════════════════════════════════════════════════════════════"
    Write-Host ""
    Write-Green "Python-Version:"
    & $PYTHON --version
    Write-Host ""
    Write-Green "Installierte Pakete:"
    & $PIP list
    Write-Host ""
    Write-Green "Projekt-Struktur:"
    Write-Host "  app.py              - Hauptanwendung"
    Write-Host "  forecaster/         - Prognose-Module"
    Write-Host "  geospacial/         - Geografische Analysen"
    Write-Host "  loader/             - Daten-Loader"
    Write-Host "  overview/           - Übersichts-Dashboard"
    Write-Host "  scenario/           - Szenario-Analysen"
    Write-Host ""
}

function Check-Ports {
    Write-Cyan "Prüfe Port-Verfügbarkeit..."
    $connections = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
    if ($connections) {
        Write-Red "Warnung: Port $PORT ist bereits belegt!"
    } else {
        Write-Green "✓ Port $PORT ist verfügbar"
    }
}

# ============================================================================
# Hauptlogik
# ============================================================================

switch ($Command.ToLower()) {
    "help" { Show-Help }
    "check-python" { Check-Python }
    "check-deps" { Check-Dependencies }
    "install" { Install-Dependencies }
    "setup" { Setup-Project }
    "upgrade-deps" { Upgrade-Dependencies }
    "run" { Run-App }
    "dev" { Run-Dev }
    "test" { Run-Tests }
    "lint" { Run-Lint }
    "format" { Run-Format }
    "docker-build" { Docker-Build }
    "docker-run" { Docker-Run }
    "docker-stop" { Docker-Stop }
    "docker-logs" { Docker-Logs }
    "docker-restart" { Docker-Restart }
    "docker-shell" { Docker-Shell }
    "docker-clean" { Docker-Clean }
    "clean" { Clean-Temp }
    "clean-all" { Clean-All }
    "info" { Show-Info }
    "check-ports" { Check-Ports }
    default {
        Write-Red "Unbekannter Befehl: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}

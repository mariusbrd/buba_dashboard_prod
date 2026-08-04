# ============================================================================
# BuBa Dashboard - Makefile
# ============================================================================
# Dieses Makefile erleichtert die häufigsten Entwicklungs- und 
# Deployment-Aufgaben für das BuBa Dashboard Projekt.
#
# Hauptbefehle:
#   make help        - Zeigt alle verfügbaren Befehle
#   make install     - Installiert alle Python-Abhängigkeiten
#   make run         - Startet die Anwendung lokal
#   make docker-build - Baut das Docker-Image
#   make docker-run  - Startet die Anwendung in Docker
#   make clean       - Räumt temporäre Dateien auf
# ============================================================================

.PHONY: help install run dev docker-build docker-run docker-stop docker-logs clean test lint format check-deps

# Standardziel: Hilfe anzeigen
.DEFAULT_GOAL := help

# Farben für die Ausgabe
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Konfiguration
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
APP_NAME := gvb-dashboard
PORT := 8080

# ============================================================================
# Hilfe und Dokumentation
# ============================================================================

help: ## Zeigt diese Hilfe an
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(CYAN)  BuBa Dashboard - Verfügbare Make-Befehle$(NC)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Setup und Installation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'install|setup|check' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Entwicklung:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'run|dev|test|lint|format' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'docker' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Wartung:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'clean|upgrade' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Setup und Installation
# ============================================================================

check-python: ## Überprüft, ob Python 3.11+ installiert ist
	@echo "$(CYAN)Prüfe Python-Version...$(NC)"
	@$(PYTHON) --version 2>&1 | grep -E "Python 3\.(1[1-9]|[2-9][0-9])" > /dev/null || \
		(echo "$(RED)Fehler: Python 3.11 oder höher wird benötigt!$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python-Version OK$(NC)"

check-deps: check-python ## Überprüft alle System-Abhängigkeiten
	@echo "$(CYAN)Prüfe System-Abhängigkeiten...$(NC)"
	@command -v $(DOCKER) >/dev/null 2>&1 || echo "$(YELLOW)Warnung: Docker nicht gefunden (nur für Container-Deployment benötigt)$(NC)"
	@command -v $(DOCKER_COMPOSE) >/dev/null 2>&1 || echo "$(YELLOW)Warnung: Docker Compose nicht gefunden (nur für Container-Deployment benötigt)$(NC)"
	@echo "$(GREEN)✓ Abhängigkeiten geprüft$(NC)"

install: check-python ## Installiert alle Python-Abhängigkeiten
	@echo "$(CYAN)Installiere Python-Abhängigkeiten...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Installation abgeschlossen$(NC)"

setup: install ## Vollständiges Setup (Alias für install)
	@echo "$(GREEN)✓ Setup abgeschlossen - Projekt ist bereit!$(NC)"

upgrade-deps: ## Aktualisiert alle Python-Pakete auf die neueste Version
	@echo "$(CYAN)Aktualisiere Abhängigkeiten...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Abhängigkeiten aktualisiert$(NC)"

# ============================================================================
# Entwicklung
# ============================================================================

run: ## Startet die Anwendung lokal (Produktionsmodus)
	@echo "$(CYAN)Starte BuBa Dashboard...$(NC)"
	@echo "$(GREEN)Dashboard läuft auf: http://localhost:$(PORT)$(NC)"
	$(PYTHON) app.py

dev: ## Startet die Anwendung im Entwicklungsmodus (mit Auto-Reload)
	@echo "$(CYAN)Starte BuBa Dashboard (Entwicklungsmodus)...$(NC)"
	@echo "$(GREEN)Dashboard läuft auf: http://localhost:$(PORT)$(NC)"
	@echo "$(YELLOW)Auto-Reload aktiviert - Änderungen werden automatisch übernommen$(NC)"
	FLASK_ENV=development $(PYTHON) app.py

test: ## Führt Tests aus (wenn vorhanden)
	@echo "$(CYAN)Führe Tests aus...$(NC)"
	@if [ -d "tests" ]; then \
		$(PYTHON) -m pytest tests/ -v; \
	else \
		echo "$(YELLOW)Keine Tests gefunden (tests/ Verzeichnis existiert nicht)$(NC)"; \
	fi

lint: ## Prüft Code-Qualität mit flake8 (falls installiert)
	@echo "$(CYAN)Prüfe Code-Qualität...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 app.py forecaster/ geospacial/ loader/ overview/ scenario/ --max-line-length=120 --ignore=E501,W503; \
		echo "$(GREEN)✓ Code-Qualität OK$(NC)"; \
	else \
		echo "$(YELLOW)flake8 nicht installiert - überspringe Lint-Check$(NC)"; \
	fi

format: ## Formatiert Python-Code mit black (falls installiert)
	@echo "$(CYAN)Formatiere Code...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black app.py forecaster/ geospacial/ loader/ overview/ scenario/; \
		echo "$(GREEN)✓ Code formatiert$(NC)"; \
	else \
		echo "$(YELLOW)black nicht installiert - Installation: pip install black$(NC)"; \
	fi

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Baut das Docker-Image
	@echo "$(CYAN)Baue Docker-Image...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker-Image gebaut$(NC)"

docker-run: ## Startet die Anwendung in Docker
	@echo "$(CYAN)Starte Anwendung in Docker...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Anwendung läuft in Docker auf http://localhost:$(PORT)$(NC)"
	@echo "$(YELLOW)Logs anzeigen: make docker-logs$(NC)"

docker-stop: ## Stoppt die Docker-Container
	@echo "$(CYAN)Stoppe Docker-Container...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Container gestoppt$(NC)"

docker-logs: ## Zeigt Docker-Logs an
	@echo "$(CYAN)Docker-Logs (Strg+C zum Beenden):$(NC)"
	$(DOCKER_COMPOSE) logs -f

docker-restart: docker-stop docker-run ## Neustart der Docker-Container

docker-shell: ## Öffnet eine Shell im laufenden Container
	@echo "$(CYAN)Öffne Shell im Container...$(NC)"
	$(DOCKER) exec -it $(APP_NAME) /bin/bash

docker-clean: ## Entfernt Docker-Images und -Container
	@echo "$(CYAN)Räume Docker-Ressourcen auf...$(NC)"
	$(DOCKER_COMPOSE) down -v --rmi all
	@echo "$(GREEN)✓ Docker-Ressourcen entfernt$(NC)"

# ============================================================================
# Wartung und Cleanup
# ============================================================================

clean: ## Räumt temporäre Dateien und Cache-Verzeichnisse auf
	@echo "$(CYAN)Räume temporäre Dateien auf...$(NC)"
	@# Python Cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@# Temporäre Daten
	rm -rf forecaster/trained_models/* 2>/dev/null || true
	rm -rf forecaster/trained_outputs/* 2>/dev/null || true
	rm -rf scenario/data/* 2>/dev/null || true
	rm -rf scenario/models_scenario/* 2>/dev/null || true
	rm -rf scenario/scenario_cache/* 2>/dev/null || true
	rm -rf loader/runs/* 2>/dev/null || true
	rm -f loader/gvb_output.parquet 2>/dev/null || true
	rm -f loader/gvb_output.xlsx 2>/dev/null || true
	rm -f scenario/.scenario_month.stamp 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup abgeschlossen$(NC)"

clean-all: clean docker-clean ## Vollständiger Cleanup (inkl. Docker)
	@echo "$(GREEN)✓ Vollständiger Cleanup abgeschlossen$(NC)"

# ============================================================================
# Nützliche Utilities
# ============================================================================

info: ## Zeigt Projektinformationen an
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(NC)"
	@echo "$(CYAN)  BuBa Dashboard - Projektinformationen$(NC)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Python-Version:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(GREEN)Installierte Pakete:$(NC)"
	@$(PIP) list | grep -E "(dash|plotly|pandas|numpy|scikit-learn|statsmodels)" || echo "Pakete noch nicht installiert"
	@echo ""
	@echo "$(GREEN)Projekt-Struktur:$(NC)"
	@echo "  app.py              - Hauptanwendung"
	@echo "  forecaster/         - Prognose-Module"
	@echo "  geospacial/         - Geografische Analysen"
	@echo "  loader/             - Daten-Loader"
	@echo "  overview/           - Übersichts-Dashboard"
	@echo "  scenario/           - Szenario-Analysen"
	@echo ""

quick-start: check-deps install run ## Schnellstart: Installation + Start

status: ## Zeigt den Status aller Dienste an
	@echo "$(CYAN)Service Status:$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Container:$(NC)"
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "  Keine Container laufen"
	@echo ""

# ============================================================================
# Entwickler-Tools
# ============================================================================

freeze: ## Erstellt/aktualisiert requirements.txt mit aktuellen Versionen
	@echo "$(CYAN)Erstelle requirements.txt...$(NC)"
	$(PIP) freeze > requirements_frozen.txt
	@echo "$(GREEN)✓ requirements_frozen.txt erstellt$(NC)"
	@echo "$(YELLOW)Hinweis: Manuelle Überprüfung empfohlen bevor requirements.txt ersetzt wird$(NC)"

shell: ## Startet eine Python-Shell mit importierten Modulen
	@echo "$(CYAN)Starte Python-Shell...$(NC)"
	$(PYTHON) -i -c "import pandas as pd; import numpy as np; import plotly.express as px; print('Module geladen: pd, np, px')"

check-ports: ## Prüft, ob Port 8080 verfügbar ist
	@echo "$(CYAN)Prüfe Port-Verfügbarkeit...$(NC)"
	@netstat -an 2>/dev/null | grep ":$(PORT)" | grep LISTEN > /dev/null && \
		echo "$(RED)Warnung: Port $(PORT) ist bereits belegt!$(NC)" || \
		echo "$(GREEN)✓ Port $(PORT) ist verfügbar$(NC)"

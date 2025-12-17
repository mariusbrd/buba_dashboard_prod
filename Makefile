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

# Konfiguration
PYTHON := python
PIP := $(PYTHON) -m pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
APP_NAME := gvb-dashboard
PORT := 8080

PROJECT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
WORKSPACE=${PROJECT_DIR}/data/workspace


# ============================================================================
# Setup und Installation
# ============================================================================

check-python:
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" || \
		(echo "$(RED)Fehler: Python 3.11 bis 3.13 ist aktuell unterstützt!$(NC)" && exit 1)


install: check-python 
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt


upgrade-deps: ## Aktualisiert alle Python-Pakete auf die neueste Version
	$(PIP) install --upgrade -r requirements.txt

# ============================================================================
# Entwicklung
# ============================================================================

run: ## Startet die Anwendung lokal (Produktionsmodus)
	$(PYTHON) -m src.app

dev: ## Startet die Anwendung im Entwicklungsmodus (mit Auto-Reload)
	FLASK_ENV=development $(PYTHON) app.py

test: ## Führt Tests aus (wenn vorhanden)
	@if [ -d "tests" ]; then \
		$(PYTHON) -m pytest tests/ -v; \
	else \
		echo "$(YELLOW)Keine Tests gefunden (tests/ Verzeichnis existiert nicht)$(NC)"; \
	fi

lint: ## Prüft Code-Qualität mit flake8 (falls installiert)
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 app.py forecaster/ geospacial/ loader/ overview/ scenario/ --max-line-length=120 --ignore=E501,W503; \
		echo "$(GREEN)✓ Code-Qualität OK$(NC)"; \
	else \
		echo "$(YELLOW)flake8 nicht installiert - überspringe Lint-Check$(NC)"; \
	fi

format: ## Formatiert Python-Code mit black (falls installiert)
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
	$(DOCKER_COMPOSE) build

docker-run: ## Startet die Anwendung in Docker
	$(DOCKER_COMPOSE) up -d

docker-stop: ## Stoppt die Docker-Container
	$(DOCKER_COMPOSE) down

docker-logs: ## Zeigt Docker-Logs an
	$(DOCKER_COMPOSE) logs -f

docker-restart: docker-stop docker-run ## Neustart der Docker-Container

docker-shell: ## Öffnet eine Shell im laufenden Container
	$(DOCKER) exec -it $(APP_NAME) /bin/bash

docker-clean: ## Entfernt Docker-Images und -Container
	$(DOCKER_COMPOSE) down -v --rmi all

# ============================================================================
# Wartung und Cleanup
# ============================================================================

clean: ## Räumt temporäre Dateien und Cache-Verzeichnisse auf
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

clean-all: clean docker-clean ## Vollständiger Cleanup inkl. Docker

# ============================================================================
# Nützliche Utilities
# ============================================================================


quick-start: check-deps install run ## Schnellstart: Installation + Start

status: ## Zeigt den Status aller Dienste an
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "  Keine Container laufen"

# ============================================================================
# Entwickler-Tools
# ============================================================================

freeze: ## Erstellt/aktualisiert requirements.txt mit aktuellen Versionen
	$(PIP) freeze > requirements_frozen.txt

shell: ## Startet eine Python-Shell mit importierten Modulen
	$(PYTHON) -i -c "import pandas as pd; import numpy as np; import plotly.express as px; print('Module geladen: pd, np, px')"

check-ports: ## Prüft, ob Port 8080 verfügbar ist
	@netstat -an 2>/dev/null | grep ":$(PORT)" | grep LISTEN > /dev/null && \
		echo "$(RED)Warnung: Port $(PORT) ist bereits belegt!$(NC)" || \
		echo "$(GREEN)✓ Port $(PORT) ist verfügbar$(NC)"

@echo off
REM ============================================================================
REM BuBa Dashboard - Windows Schnellstart (Batch-Datei)
REM ============================================================================
REM Einfaches Batch-Skript für Windows-Nutzer ohne PowerShell-Kenntnisse
REM Doppelklick auf diese Datei, um das Dashboard zu starten
REM ============================================================================

echo.
echo ============================================================
echo   BuBa Dashboard - Schnellstart
echo ============================================================
echo.

REM Prüfe Python-Installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python nicht gefunden!
    echo Bitte installieren Sie Python 3.11 oder hoeher von python.org
    pause
    exit /b 1
)

echo [OK] Python erkannt
echo.

REM Prüfe ob requirements.txt existiert
if not exist "requirements.txt" (
    echo [FEHLER] requirements.txt nicht gefunden!
    echo Stellen Sie sicher, dass Sie im Projektverzeichnis sind.
    pause
    exit /b 1
)

echo Installiere/Aktualisiere Abhaengigkeiten...
echo.
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo.
    echo [FEHLER] Installation der Abhaengigkeiten fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo [OK] Alle Abhaengigkeiten installiert
echo.
echo ============================================================
echo   Starte BuBa Dashboard...
echo   URL: http://localhost:8080
echo ============================================================
echo.
echo Druecken Sie Strg+C um das Dashboard zu beenden
echo.

REM Starte die Anwendung
python app.py

pause

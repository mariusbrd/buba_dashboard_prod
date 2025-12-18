# =========================================================
# GVB Dashboard (Refactored Single-File Layout)
# - Import cleanup
# - Modern Dash ctx trigger (back-compatible)
# - Structured sections for maintainability
# NOTE: Functional behavior preserved; heavy logic unchanged.
# =========================================================

import logging
import os
import signal
import atexit
import shutil
import warnings
import threading
import time
from pathlib import Path

"""
GVB Dashboard - Mit echten Daten aus instructor.py/loader.py
Ein professionelles Dashboard zur Geldvermögensbildung für Bankenvorstände

Features:
- Übersichtsseite mit KPIs und interaktiven Charts
- Prognose-Suite mit Decision Tree und ARIMAX
- Szenario-Analyse für regionale Anpassungen
- Nutzt echte Daten aus gvb_output.xlsx
- Produktionsreifer, modularer Code

Autor: Data Science Team
Version: 1.1 - Real Data Edition
"""

# ==========================
# 0) Logging setup
# ==========================
_logger = logging.getLogger("GVB_Dashboard")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
logger = _logger  # Convenience alias


# ==============================================================================
# FOUNDATION: Zentrale Log-API (Namespaces + Komfort)
# ==============================================================================
class Log:
    """Zentrale Logging-Fassade mit thematischen Präfixen.

    Ziel:
      - konsistente, filterbare Logs im gesamten Dashboard
      - klare Trennung nach Feature-Bereichen (Data, Forecast, Scenario, UI, Perf)
    Nutzung:
      Log.data("LoadExcel | opened | path=gvb_output.xlsx")
      Log.perf("MA | window=3 | rows=1024")
    """

    @staticmethod
    def info(msg: str):        logger.info(msg)
    @staticmethod
    def warn(msg: str):        logger.warning(msg)
    @staticmethod
    def error(msg: str):       logger.error(msg)
    @staticmethod
    def exception(msg: str):   logger.exception(msg)

    # Feature-Namespaces
    @staticmethod
    def data(msg: str):        logger.info(f"[Data] {msg}")
    @staticmethod
    def forecast(msg: str):    logger.info(f"[Forecast] {msg}")
    @staticmethod
    def scenario(msg: str):    logger.info(f"[Scenario] {msg}")
    @staticmethod
    def ui(msg: str):          logger.info(f"[UI] {msg}")
    @staticmethod
    def perf(msg: str):        logger.info(f"[Perf] {msg}")

    # Optional: feinerer Namespace speziell für die Szenario-Tabelle
    SCENARIO_TABLE_NS = "ScenarioTable"

    @staticmethod
    def scenario_table(msg: str, also_root: bool = True):
        import logging as _pylogging
        _pylogging.getLogger(Log.SCENARIO_TABLE_NS).info(f"[ScenarioTable] {msg}")
        if also_root:
            logger.info(f"[ScenarioTable] {msg}")

    @staticmethod
    def enable_namespace(ns: str, level: int = logging.INFO):
        """Namespace-Logger gezielt lauter/leiser stellen."""
        import logging as _pylogging
        _pylogging.getLogger(ns).setLevel(level)


# ==============================================================================
# 1) Imports (Dash, Plotly, Pandas, ...)
# ==============================================================================
import dash
from dash import dcc, html, Input, Output, State, dash_table

# Backward compatibility for Dash trigger context
try:
    from dash import ctx  # Dash 2.12+
except Exception:  # Older Dash versions
    from dash import callback_context as ctx

from textwrap import shorten

import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json

warnings.filterwarnings("ignore")

# System imports für Subprocess
import subprocess
import sys

# ML Imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.info("Hinweis: statsmodels nicht installiert - ARIMAX-Features sind deaktiviert")

from sklearn.utils import resample

# Python 3.9+
try:
    from collections.abc import Iterable
except ImportError:  # sehr alte Umgebungen
    from typing import Iterable


# =============================================================================
# 2) Presets & Model Persistence – SAUBER KONSOLIDIERT
# =============================================================================
from typing import Any, Dict, Optional

_APP_DIR: Path = Path(__file__).resolve().parent

# Optional via ENV variierbar
_DATA_DIR: Path = Path(os.getenv("FORECASTER_DATA_DIR", _APP_DIR / "data")).resolve()
PRESETS_DIR: Path = Path(os.getenv("FORECASTER_PRESETS_DIR", _APP_DIR / "forecaster" / "user_presets")).resolve()
MODELS_DIR: Path = Path(os.getenv("FORECASTER_MODELS_DIR", _APP_DIR / "forecaster" / "trained_models")).resolve()
RUNS_DIR: Path = Path(os.getenv("FORECASTER_RUNS_DIR", _APP_DIR / "loader" / "runs")).resolve()

PRESETS_FILE: Path = PRESETS_DIR / "user_presets.json"

# Verzeichnisse sicherstellen
for _p in (_DATA_DIR, PRESETS_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3) Self-Cleaning: zwei getrennte Routinen
#    - Hard Clean: bei Start/Stop → alles weg
#    - Periodic Clean: alle 12h → nur Szenario-Artefakte
# =============================================================================

try:
    APP_ROOT: Path = Path(__file__).resolve().parent
except Exception:
    APP_ROOT = Path.cwd()

# Hard-Clean: diese Pfade sollen bei jedem Start/Stop geleert/gelöscht werden
# WICHTIG: scenario/data ist NICHT hier, weil dort output.xlsx/analysis_data.xlsx
# gespeichert werden, die beim Start heruntergeladen werden!
ALWAYS_PURGE_PATHS = [
    APP_ROOT / "forecaster" / "trained_models",
    APP_ROOT / "forecaster" / "trained_outputs",
    APP_ROOT / "loader" / "gvb_output.parquet",
    APP_ROOT / "loader" / "gvb_output.xlsx",
    APP_ROOT / "loader" / "runs", 
    # ENTFERNT: APP_ROOT / "scenario" / "data",  ← Enthält output.xlsx/analysis_data.xlsx!
    APP_ROOT / "scenario" / "models_scenario",
    APP_ROOT / "scenario" / "scenario_cache",
    APP_ROOT / "scenario" / ".scenario_month.stamp",
]

# Periodic-Clean: nur diese Ordner/Dateien werden nach 12h aufgeräumt
SCENARIO_DATA_DIR = APP_ROOT / "scenario" / "data"
SCENARIO_MODELS_DIR = APP_ROOT / "scenario" / "models_scenario"
SCENARIO_MAX_AGE_HOURS = 12  # Dateien älter als 12h werden entfernt

# Flags gegen Mehrfachausführung (z. B. durch Flask/Dash Reload)
_SELF_CLEAN_START_RAN = False
_SELF_CLEAN_STOP_RAN = False


def _is_under_app_root(path: Path) -> bool:
    """Verhindert, dass außerhalb des Projektordners gelöscht wird."""
    try:
        path.resolve().relative_to(APP_ROOT.resolve())
        return True
    except Exception:
        return False


def _empty_dir_safe(path: Path) -> None:
    """Leert ein Verzeichnis sicher (Dateien/Symlinks löschen, Unterordner rekursiv)."""
    try:
        if not _is_under_app_root(path):
            logger.warning(f"[SelfClean] Skip (outside APP_ROOT): {path}")
            return
        path.mkdir(parents=True, exist_ok=True)
        for entry in path.iterdir():
            try:
                if entry.is_file() or entry.is_symlink():
                    entry.unlink(missing_ok=True)
                elif entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
            except Exception as e:
                logger.warning(f"[SelfClean] Could not remove {entry}: {e}")
        logger.info(f"[SelfClean] Emptied: {path}")
    except Exception as e:
        logger.error(f"[SelfClean] Error cleaning {path}: {e}")


def _delete_file_safe(path: Path) -> None:
    """Löscht eine einzelne Datei sicher, falls vorhanden."""
    try:
        if path.exists() and path.is_file():
            if not _is_under_app_root(path):
                logger.warning(f"[SelfClean] Skip file (outside APP_ROOT): {path}")
                return
            path.unlink(missing_ok=True)
            logger.info(f"[SelfClean] Removed file: {path.name}")
    except Exception as e:
        logger.warning(f"[SelfClean] Could not remove file {path}: {e}")


def _delete_older_than(path: Path, hours: int, pattern: str = "*") -> None:
    """Löscht nur Dateien im Verzeichnis, die älter als `hours` Stunden sind."""
    try:
        if not _is_under_app_root(path):
            logger.warning(f"[SelfClean] Skip (outside APP_ROOT): {path}")
            return

        cutoff = time.time() - hours * 3600
        path.mkdir(parents=True, exist_ok=True)

        for entry in path.glob(pattern):
            try:
                if entry.is_file():
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff:
                        entry.unlink(missing_ok=True)
                        logger.info(f"[SelfClean] [12h] Removed old file: {entry.name}")
            except Exception as e:
                logger.warning(f"[SelfClean] [12h] Could not remove {entry}: {e}")
    except Exception as e:
        logger.error(f"[SelfClean] [12h] Error age-cleaning {path}: {e}")


def self_clean_startup() -> None:
    """HARTE Routine: wird beim Start ausgeführt, löscht alle relevanten Artefakte."""
    global _SELF_CLEAN_START_RAN
    if _SELF_CLEAN_START_RAN:
        return
    _SELF_CLEAN_START_RAN = True

    # Reloader-Schutz (Dev)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" and os.environ.get("FLASK_ENV") == "development":
        pass

    logger.info("[SelfClean] Startup: cleaning all relevant temp artefacts …")

    for p in ALWAYS_PURGE_PATHS:
        if p.is_dir():
            _empty_dir_safe(p)
        else:
            _delete_file_safe(p)

    # direkt den 12h-Cleaner anwerfen
    _start_periodic_scenario_cleaner()


def self_clean_shutdown(*_args, **_kwargs) -> None:
    """HARTE Routine: wird beim Stop ausgeführt, löscht alle relevanten Artefakte."""
    global _SELF_CLEAN_STOP_RAN
    if _SELF_CLEAN_STOP_RAN:
        return
    _SELF_CLEAN_STOP_RAN = True

    logger.info("[SelfClean] Shutdown: cleaning all relevant temp artefacts …")

    for p in ALWAYS_PURGE_PATHS:
        if p.is_dir():
            _empty_dir_safe(p)
        else:
            _delete_file_safe(p)


def _periodic_scenario_clean() -> None:
    """FEINE Routine: wird alle 12h ausgeführt, löscht nur Szenario-Artefakte."""
    logger.info("[SelfClean] [12h] Periodic scenario clean …")
    _delete_older_than(SCENARIO_DATA_DIR, SCENARIO_MAX_AGE_HOURS, pattern="analysis_data_*.xlsx")
    _delete_older_than(SCENARIO_MODELS_DIR, SCENARIO_MAX_AGE_HOURS, pattern="*.pkl")
    # nächsten Lauf wieder planen
    _start_periodic_scenario_cleaner()


def _start_periodic_scenario_cleaner() -> None:
    """Startet einen Timer, der nach 12h den Szenario-Cleaner aufruft."""
    t = threading.Timer(12 * 3600, _periodic_scenario_clean)
    t.daemon = True
    t.start()


# Beim Beenden/Signal aufräumen (Hard Clean)
atexit.register(self_clean_shutdown)
try:
    signal.signal(signal.SIGINT,  lambda *a, **k: (self_clean_shutdown(), os._exit(0)))
    signal.signal(signal.SIGTERM, lambda *a, **k: (self_clean_shutdown(), os._exit(0)))
except Exception:
    pass






# -- Utilities ----------------------------------------------------------------
def _slugify(name: str) -> str:
    """
    Erzeugt eine stabile, Dateinamen- und URL-freundliche ID aus einem Namen.
    Beispiel: "Meine Prognose V1" -> "meine-prognose-v1"
    """
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-_.]+", "-", s)   # nur a-z, 0-9, -, _, .
    s = re.sub(r"-{2,}", "-", s).strip("-") # doppelte - entfernen
    return s or f"preset-{int(time.time())}"

def _now_iso() -> str:
    """Zeitstempel als ISO-String (YYYY-MM-DDTHH:MM:SS)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _normalize_target_slug(value: Optional[str]) -> str:
    """
    Vereinheitlicht Ziel/Target-Werte für UI/Keys.
    - Normalisiert Synonyme (einlage→einlagen, kredit→kredite, …)
    - Fällt ansonsten auf slugify zurück (z. B. 'Gesamt GVB' → 'gesamt-gvb')
    """
    if not value:
        return ""
    v = str(value).strip().lower()
    synonyms = {
        "einlage": "einlagen",
        "einlagen": "einlagen",
        "kredit": "kredite",
        "kredite": "kredite",
        "versicherung": "versicherungen",
        "versicherungen": "versicherungen",
        "wertpapier": "wertpapiere",
        "wertpapiere": "wertpapiere",
        "gesamt": "gesamt",
    }
    if v in synonyms:
        return synonyms[v]
    # Fallback: konsistenter Slug für sonstige Targets
    return _slugify(v)



def _log_geo(msg: str, level: str = "info"):
    lg = logging.getLogger("GVB_Dashboard")
    # Versuche, ein bestehendes Log-Objekt zu nutzen
    try:
        if 'Log' in globals():
            # Versuche Log.geo, sonst Log.info, sonst Log.debug
            fn = getattr(Log, 'geo', None) or getattr(Log, 'info', None) or getattr(Log, 'debug', None)
            if callable(fn):
                fn(msg)
                return
    except Exception:
        pass
    # Fallback auf Standard-Logger
    getattr(lg, level, lg.info)(msg)


# -- JSON I/O (atomare Writes) ------------------------------------------------
def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Schreibt Text atomar (erst in .tmp, dann os.replace). Verhindert kaputte Dateien
    bei Abbrüchen während des Schreibens.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _load_user_presets_from_disk() -> Dict[str, Any]:
    """
    Lädt die User-Presets aus der JSON-Datei (falls vorhanden) und liefert ein {name: meta}-Dict zurück.

    Erwartete Struktur (Beispiel):
      {
        "Mein Preset": {
          "id": "mein-preset",
          "created_at": "2025-10-01T09:30:00",
          "target": "Wertpapiere",
          "exog": ["ICP.M.DE.N...", "..."],
          "model_path": "data/models/wertpapiere__20251001_093000.pkl"   # optional
        },
        ...
      }
    """
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_user_presets_to_disk(presets: Dict[str, Any]) -> None:
    """Persistiert die übergebene Preset-Diktstruktur atomar nach PRESETS_FILE."""
    try:
        _atomic_write_text(PRESETS_FILE, json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
        Log.data(f"Presets | saved | file={PRESETS_FILE.name} items={len(presets)}")
    except Exception as e:
        logger.error(f"[Presets] Write error: {e}")


def _model_path_for(target_slug: str, *, stamp: Optional[str] = None, ext: str = ".pkl") -> Path:
    """
    Generiert einen standardisierten Pfad für ein trainiertes Modell.
    Beispiel: forecaster/trained_models/wertpapiere__20251001_093000.pkl
    """
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{target_slug}__{stamp}{ext}"
    return (MODELS_DIR / fname).resolve()


def _save_model_artifact(obj: Any, path: Optional[Path] = None, *, target_slug: Optional[str] = None) -> Path:
    """
    Speichert ein Modellobjekt (Pickle) unter 'path' oder (falls nicht gesetzt)
    unter einem Standardpfad basierend auf target_slug. Gibt den endgültigen Pfad zurück.
    """
    import pickle
    if path is None:
        if not target_slug:
            raise ValueError("Either 'path' or 'target_slug' must be provided for _save_model_artifact().")
        path = _model_path_for(target_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path

def _load_model_artifact(path: Path) -> Any:
    """Lädt ein modelliertes Objekt (Pickle) von 'path'."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------- USER PRESET HELPERS (vollständig) -------------------
# === app.py – Preset/Model Pfade & Helper =================================
import os, json, time
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR: Path = Path(__file__).resolve().parent
PRESETS_DIR: Path = (APP_DIR / "forecaster" / "user_presets").resolve()
PRESETS_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_FILE: Path = PRESETS_DIR / "user_presets.json"

MODELS_DIR: Path = (APP_DIR / "forecaster" / "trained_models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)

def _load_user_presets_from_disk() -> Dict[str, Any]:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_user_presets_to_disk(presets: Dict[str, Any]) -> None:
    _atomic_write_text(PRESETS_FILE, json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- Name/Hash Utils ----
def _slugify_name(name: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "preset"

def _stable_cfg_hash(*, target: str, exog: List[str], horizon: int, flows: bool, extra: Optional[Dict[str, Any]] = None) -> str:
    import hashlib
    payload = {
        "target": str(target or ""),
        "exog": sorted({str(x) for x in (exog or [])}),
        "horizon": int(horizon or 0),
        "use_flows": bool(flows),
        "extra": extra or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# ---- Public Helper API ----
def create_user_preset_from_ui_state(
    *, name: str, target: str, exog: List[str], horizon: int,
    is_fluss_mode: bool, model_payload: Optional[Dict[str, Any]] = None,
    extra_ui_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    from datetime import datetime
    if not (name or "").strip():
        raise ValueError("Bitte einen Preset-Namen angeben.")
    preset_id = _slugify_name(name)
    model_path = (model_payload or {}).get("path")
    exog_snapshot_path = (model_payload or {}).get("exog_snapshot_path")
    cfg_hash = _stable_cfg_hash(
        target=target, exog=list(exog or []), horizon=int(horizon or 0),
        flows=bool(is_fluss_mode), extra=(extra_ui_options or {})
    )
    preset: Dict[str, Any] = {
        "id": preset_id,
        "name": name.strip(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target": target,
        "exog": list(exog or []),
        "horizon": int(horizon or 0),
        "is_fluss_mode": bool(is_fluss_mode),
        "cfg_hash": cfg_hash,
        "model_path": model_path,
        "exog_snapshot_path": exog_snapshot_path
    }
    if extra_ui_options:
        preset["ui_opts"] = extra_ui_options
    return preset

def upsert_user_preset(name: str, preset_obj: Dict[str, Any]) -> Dict[str, Any]:
    data = _load_user_presets_from_disk()
    data[name] = preset_obj
    _save_user_presets_to_disk(data)
    return data

def merge_hc_and_user_presets_for_dropdown(
    hc_presets: Dict[str, Any], user_presets: Dict[str, Any],
    *, hc_label: str = "H&C Presets", user_label: str = "Eigene Presets"
) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    if isinstance(hc_presets, dict) and hc_presets:
        options.append({"label": hc_label, "value": "__GROUP__HC", "disabled": True})
        for disp_name, meta in hc_presets.items():
            pid = (meta or {}).get("id") or _slugify_name(f"preset-{disp_name}")
            options.append({"label": disp_name, "value": f"preset_{pid}"})
    if isinstance(user_presets, dict) and user_presets:
        options.append({"label": user_label, "value": "__GROUP__USER", "disabled": True})
        for disp_name, meta in user_presets.items():
            pid = (meta or {}).get("id") or _slugify_name(disp_name)
            options.append({"label": disp_name, "value": f"user_{pid}"})
    return options

def delete_user_preset(*, preset_identifier: str, delete_files: bool = False):
    """
    Löscht ein User-Preset aus der JSON-Datei. 'preset_identifier' kann entweder
    der Anzeigename (Key im JSON) oder die Preset-ID sein. Wenn delete_files=True,
    werden referenzierte Dateien (model_path, exog_snapshot_path, final_dataset_path) entfernt.
    """
    data = _load_user_presets_from_disk()
    if not isinstance(data, dict) or not data:
        return {}

    # Key finden (Display-Name) – entweder via Name oder via ID
    to_delete_key = None
    for disp_name, meta in list(data.items()):
        pid = (meta or {}).get("id")
        if disp_name == preset_identifier or str(pid) == str(preset_identifier):
            to_delete_key = disp_name
            break

    if not to_delete_key:
        return data  # nichts zu tun

    if delete_files:
        try:
            meta = data.get(to_delete_key, {}) or {}
            for path_key in ("model_path", "exog_snapshot_path", "final_dataset_path"):
                p = meta.get(path_key)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"[Presets] Datei nicht gelöscht ({p}): {e}")
        except Exception as e:
            logger.warning(f"[Presets] Fehler beim Datei-Cleanup: {e}")

    data.pop(to_delete_key, None)
    _save_user_presets_to_disk(data)
    return data



# =============================================================================
# ==========================
# 3) Constants & Enums (Canonical)
# ========================== (KANONISCH)
# =============================================================================
from typing import Final, Dict, Tuple

# Kanonische Hierarchieebenen (nur diese drei sind erlaubt)
LEVELS: Final[Tuple[str, str, str]] = ("ebene1", "ebene2", "ebene3")

# Kanonische Value-Spaltennamen (Datenarten)
VALUE_COLS: Final[Tuple[str, str]] = ("bestand", "fluss")

# Kanonische KPI-Kategorien auf Ebene 1 (E1)
KPI_E1: Final[Tuple[str, str, str, str]] = (
    "Einlagen",
    "Wertpapiere",
    "Versicherungen",
    "Kredite",
)

# Reihenfolge für E1 in Tabellen/Charts (falls nötig)
E1_DISPLAY_ORDER: Final[Tuple[str, str, str, str]] = KPI_E1

# Mapping für Sektoren aus dem UI auf kanonische Codes
SEKTOR_ALIASES: Final[Dict[str, str]] = {
    # Privathaushalte
    "Privathaushalte": "PH",
    "PH": "PH",
    "ph": "PH",
    # Nichtfinanzunternehmen
    "Nichtfinanzunternehmen": "NFK",
    "NFK": "NFK",
    "nfk": "NFK",
    # Fallbacks / sonstige Labels werden unverändert gelassen
}

# (Optional) Kanonische Spalten-Aliase für spätere Normalisierung
# -> wird in Punkt 2/3 von der Roadmap nützlich (keine aktive Nutzung hier)
CANON_COLUMN_ALIASES: Final[Dict[str, str]] = {
    # Datum
    "datum": "date",
    "date": "date",
    "Date": "date",
    "Datum": "date",

    # Ebenen
    "Ebene1": "ebene1",
    "Ebene2": "ebene2",
    "Ebene3": "ebene3",
    "E1": "ebene1",
    "E2": "ebene2",
    "E3": "ebene3",

    # Werte
    "Bestand": "bestand",
    "Fluss": "fluss",
    "flow": "fluss",
    "stock": "bestand",

    # Sektor
    "sektor": "sektor",
    "sector": "sektor",
    "Sektor": "sektor",
}




# =============================================================================
# ==========================
# 4) Date & Quarter Helpers
# ========================== (KANONISCH)
# =============================================================================
from typing import Tuple, List, Optional, Union, Dict, Any

import pandas as pd
import numpy as np

QUARTER_START_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}
QUARTER_END_MONTH   = {1: 3, 2: 6, 3: 9, 4: 12}
QUARTER_END_DAY     = {3: 31, 6: 30, 9: 30, 12: 31}

def parse_year_quarter_float(value: float) -> Tuple[int, int]:
    """
    Konvertiert Slider-Wert wie 2024.75 in (jahr, quartal).
    Robust gegen Rundungsartefakte (0.24/0.25 etc.).
    """
    y = int(value)
    frac = round(value - y, 2)
    if frac <= 0.01:
        q = 1
    elif abs(frac - 0.25) <= 0.01:
        q = 2
    elif abs(frac - 0.50) <= 0.01:
        q = 3
    elif abs(frac - 0.75) <= 0.01:
        q = 4
    else:
        # Fallback, falls Slider ungewöhnliche Werte liefert
        q = int(round(frac * 4)) + 1
        q = min(max(q, 1), 4)
    return y, q

def get_quarter_start_date(year: int, quarter: int) -> pd.Timestamp:
    """Erster Kalendertag des Quartals."""
    return pd.Timestamp(year, QUARTER_START_MONTH[quarter], 1)

def get_quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    """Letzter Kalendertag des Quartals."""
    m_end = QUARTER_END_MONTH[quarter]
    d_end = QUARTER_END_DAY[m_end]
    return pd.Timestamp(year, m_end, d_end)

def window_from_slider(slider_range: List[float]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Nimmt den 2-Werte-Slider (z.B. [2020.00, 2024.75]) und liefert
    (start_date, end_date) als Kalendergrenzen.
    """
    sy, sq = parse_year_quarter_float(slider_range[0])
    ey, eq = parse_year_quarter_float(slider_range[1])
    return get_quarter_start_date(sy, sq), get_quarter_end_date(ey, eq)



def quarters_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """
    Anzahl vollständiger Quartalsschritte zwischen d1 und d2 (inklusive Endquartal-Indexing).
    Beispiel: 2023Q1 → 2024Q1 = 4 Schritte.
    """
    p1 = d1.to_period("Q")
    p2 = d2.to_period("Q")
    return int((p2.year - p1.year) * 4 + (p2.quarter - p1.quarter))

def year_tickvals_biennial(datetime_index: pd.DatetimeIndex) -> Tuple[List[pd.Timestamp], List[str]]:
    """
    Liefert 2-jährliche X-Ticks als (tickvals, ticktext).
    Nutze das im Performance-Chart, damit die Achse ruhig bleibt.
    """
    if datetime_index.empty:
        return [], []
    idx = pd.to_datetime(datetime_index)
    years = idx.year.values
    start_year = int(years.min())
    biennial_years = pd.Index(years[((years - start_year) % 2) == 0]).unique()
    tickvals = [idx[idx.year == y][0] for y in biennial_years if (idx.year == y).any()]
    ticktext = [f"{int(y)}" for y in biennial_years]
    return tickvals, ticktext

import re
_QUARTER_LABEL_RE = re.compile(r"^\s*(\d{4})-Q([1-4])\s*$")

def label_to_quarter_end(label: str) -> Optional[pd.Timestamp]:
    """
    'YYYY-Qn' → letzter Kalendertag des Quartals als Timestamp.
    Gibt None zurück, wenn das Label nicht passt.
    """
    m = _QUARTER_LABEL_RE.match(str(label))
    if not m:
        return None
    year, quarter = int(m.group(1)), int(m.group(2))
    return get_quarter_end_date(year, quarter)

from typing import Any, Optional
import re
import math

_NUM_RE = re.compile(r'^[\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?$')

def parse_german_number(value: Any) -> Optional[float]:
    """
    Parst Zahlen in DE/EN-Format robust:
      - "1.234,56"  → 1234.56
      - "1,234.56"  → 1234.56
      - "1234,56"   → 1234.56
      - "2,5 %"     → 0.025  (Prozent werden automatisch /100 gerechnet)
      - Leere/ungültige Werte → None
    """
    if value is None:
        return None
    # Direkte Zahlen
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    # Prozent erkennen (z.B. "2,5 %")
    has_percent = "%" in s

    # Nur relevante Zeichen behalten
    s = re.sub(r"[^0-9,\.\+\-eE%]", "", s)

    # Beide Separatoren vorhanden?
    if "," in s and "." in s:
        # Deutsches Format, wenn das letzte Zeichen ein Komma als Dezimaltrenner nahe dem Ende ist
        # Heuristik: Wenn das letzte Komma nach dem letzten Punkt kommt → deutsch
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")     # Punkte als Tausendertrennzeichen entfernen
            s = s.replace(",", ".")    # Komma als Dezimalpunkt
        else:
            s = s.replace(",", "")     # US: Kommas als Tausendertrennzeichen entfernen
            # Punkt bleibt Dezimalpunkt
    else:
        # Nur Komma → deutsch
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        # Nur Punkt → already fine

    # Prozentzeichen entfernen
    s = s.replace("%", "")

    # Validieren und parsen
    s = s.strip()
    if not _NUM_RE.match(s):
        # Letzter Fallback: Minus/Plus übriglassen, alles andere raus
        s2 = re.sub(r"[^0-9\.\+\-eE]", "", s)
        if not _NUM_RE.match(s2):
            return None
        s = s2

    try:
        num = float(s)
    except Exception:
        return None

    if has_percent:
        num = num / 100.0
    return num


# =============================================================================
# ==========================
# 5) Data Normalization Helpers
# ========================== (KANONISCH)
# =============================================================================
import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def normalize_dataframe_dates(
    df: pd.DataFrame,
    *,
    target_col: str = "date",
    candidates: Tuple[str, ...] = ("date", "Date", "DATE", "Datum", "datum")
) -> pd.DataFrame:
    """
    Erzwingt eine Spalte 'date' als pandas.Timestamp:
    - nimmt die erste existierende Kandidaten-Spalte
    - konvertiert hart zu datetime (errors='coerce')
    - droppt NaN-Datumszeilen
    - sortiert nach 'date'
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[target_col])

    work = df.copy()
    src = None
    for c in candidates:
        if c in work.columns:
            src = c
            break

    if src is None:
        # defensiv: 'date' anlegen (leer)
        work[target_col] = pd.NaT
        return work.dropna(subset=[target_col])

    if src != target_col:
        work = work.rename(columns={src: target_col})

    work[target_col] = pd.to_datetime(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).sort_values(target_col).reset_index(drop=True)
    return work


def normalize_value_columns(
    df: pd.DataFrame,
    *,
    ensure: Tuple[str, ...] = ("bestand", "fluss"),
    candidates_map: dict = None
) -> pd.DataFrame:
    """
    Mapped/erzwingt die kanonischen Wertespalten 'bestand' und 'fluss' (numerisch).
    candidates_map erlaubt alternative Quellnamen.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss"])

    work = df.copy()

    # Default-Mapping: ergänze hier ggf. deine realen Quellspalten
    if candidates_map is None:
        candidates_map = {
            "bestand": ["bestand", "Bestand", "value", "Value", "Bestände"],
            "fluss":   ["fluss", "Fluss", "flow", "Flow", "Zufluss", "Abfluss"]
        }

    for canonical in ensure:
        if canonical not in work.columns:
            # suche passende Quellspalte
            src = None
            for cand in candidates_map.get(canonical, []):
                if cand in work.columns:
                    src = cand
                    break
            if src is not None:
                work[canonical] = pd.to_numeric(work[src], errors="coerce")
            else:
                work[canonical] = np.nan
        else:
            work[canonical] = pd.to_numeric(work[canonical], errors="coerce")

    return work


# =============================================================================
# ==========================
# 6) Store Helpers
# ========================== (KANONISCH)
# =============================================================================
from typing import Any, Optional
import json
import pandas as pd

def load_dataframe_from_store(
    payload: Any,
    fallback: Optional[pd.DataFrame] = None,
    *,
    date_col: Optional[str] = None,
    coerce_dates: bool = True,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    """
    Robustes Laden eines DataFrames aus Dash-Store-Payloads.
    Unterstützt:
      - JSON-String (orient='split' bevorzugt, sonst auto)
      - Dict mit {'columns': [...], 'data': [...]}
      - Dict (wird zu einzeiligem DF)
      - Liste von Dicts
    Args:
        payload: Inhalt des Stores
        fallback: DF bei Fehler/leer (sonst leeres DF)
        date_col: Name einer Datumsspalte (optional)
        coerce_dates: Datumsspalte zu datetime konvertieren
        sort_by_date: nach Datum sortieren
    """
    def _empty() -> pd.DataFrame:
        return fallback.copy() if isinstance(fallback, pd.DataFrame) else pd.DataFrame()

    if payload is None:
        return _empty()

    # 1) String → JSON / DataFrame
    if isinstance(payload, str):
        # Versuch 1: read_json(split)
        try:
            df = pd.read_json(payload, orient="split")
            if isinstance(df, pd.DataFrame):
                pass
            else:
                df = _empty()
        except Exception:
            # Versuch 2: json.loads → weiterverarbeiten
            try:
                payload = json.loads(payload)
            except Exception:
                return _empty()
        else:
            # optional: Datum parsen
            if date_col and coerce_dates and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                if sort_by_date:
                    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
            return df

    # 2) Dict
    if isinstance(payload, dict):
        # typischer DataTable-Export
        if {"columns", "data"} <= set(payload.keys()):
            try:
                cols = payload.get("columns", [])
                cols = [c["id"] if isinstance(c, dict) and "id" in c else c for c in cols]
                data = payload.get("data", [])
                df = pd.DataFrame(data, columns=cols if cols else None)
            except Exception:
                return _empty()
        else:
            # generisch: einzeiliges DF
            try:
                df = pd.DataFrame([payload]) if payload else _empty()
            except Exception:
                return _empty()

    # 3) Liste (von Dicts)
    elif isinstance(payload, list):
        if len(payload) == 0:
            return _empty()
        if isinstance(payload[0], dict):
            try:
                df = pd.DataFrame(payload)
            except Exception:
                return _empty()
        else:
            # Unsupported: Liste primitiver Typen → leeres DF
            return _empty()
    else:
        # Unbekannter Typ
        return _empty()

    # 4) Optionale Datumskonvertierung
    if date_col and coerce_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if sort_by_date:
            df = df.sort_values(date_col).reset_index(drop=True)

    return df



# =============================================================================
# ==========================
# 7) Data Layer
# ========================== (Unified)
# =============================================================================
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple
import pathlib
import pandas as pd
import numpy as np

# Erwartet in FOUNDATION vorhanden:
# - load_dataframe_from_store(payload, fallback)
# - normalize_dataframe_dates(df, target_col="date")
# - normalize_value_columns(df)  -> erzwingt 'bestand' & 'fluss' numerisch


# =============================================================================
# StoreSource · lädt DataFrames aus Dash-Stores (beliebige JSON-Formate)
# =============================================================================
@dataclass
class StoreSource:
    """Quelle: Dash-Stores (JSON). Liefert (gvb_df, exog_df) im Kanon-Schema.

Garantien:
  gvb_df: ['date','ebene1','ebene2','ebene3','bestand','fluss', ...]
  exog_df: mindestens ['date', ...]

Der Loader:
  - akzepiert mehrere Payload-Formate
  - erzwingt Datums- und Wertespalten
  - sortiert stabil
"""
    gvb_payload: object
    exog_payload: object

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 1) Roh-Laden (robust gegen unterschiedliche Store-Formate)
        gvb_df = load_dataframe_from_store(self.gvb_payload, fallback=pd.DataFrame())
        exog_df = load_dataframe_from_store(self.exog_payload, fallback=pd.DataFrame())

        # 2) Datum vereinheitlichen → 'date' + sortiert
        gvb_df = normalize_dataframe_dates(gvb_df, target_col="date") if not gvb_df.empty else pd.DataFrame(columns=["date"])
        exog_df = normalize_dataframe_dates(exog_df, target_col="date") if not exog_df.empty else pd.DataFrame(columns=["date"])

        # 3) Level-Spalten sicherstellen
        if not gvb_df.empty:
            for col in ("ebene1", "ebene2", "ebene3"):
                if col not in gvb_df.columns:
                    gvb_df[col] = np.nan
            # Whitespace säubern (falls Strings)
            for col in ("ebene1", "ebene2", "ebene3"):
                if col in gvb_df.columns and gvb_df[col].dtype == "object":
                    gvb_df[col] = gvb_df[col].astype(str).str.strip().replace({"nan": np.nan})

        # 4) Werte-Spalten kanonisieren → erzwingt 'bestand' & 'fluss' (numerisch)
        gvb_df = normalize_value_columns(gvb_df) if not gvb_df.empty else pd.DataFrame(columns=["date","ebene1","ebene2","ebene3","bestand","fluss"])

        # 5) Sortierung stabil
        if not gvb_df.empty:
            sort_cols = [c for c in ["date", "ebene1", "ebene2", "ebene3"] if c in gvb_df.columns]
            gvb_df = gvb_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        if not exog_df.empty:
            exog_df = exog_df.sort_values(["date"], kind="stable").reset_index(drop=True)

        # 6) Minimales Schema garantieren (falls oben alles leer)
        if gvb_df.empty:
            gvb_df = pd.DataFrame(columns=["date","ebene1","ebene2","ebene3","bestand","fluss"])
        if exog_df.empty:
            exog_df = pd.DataFrame(columns=["date"])

        # Logging (failsafe, optional)
        try:
            Log.data(f"[StoreSource] GVB: {len(gvb_df)} rows | EXOG: {len(exog_df)} rows")  # type: ignore[name-defined]
        except Exception:
            pass

        return gvb_df, exog_df


# =============================================================================
# DiskSource · lädt DataFrames von Disk (Excel/CSV/Parquet/Feather)
# =============================================================================
@dataclass
class DiskSource:
    """Quelle: Dateien auf Disk (Excel/CSV/Parquet/Feather).
MIT automatischer Spalten-Korrektur für vertauschte bestand/fluss.
"""
    gvb_path: pathlib.Path
    exog_path: Optional[pathlib.Path] = None
    gvb_sheet: Optional[str] = None
    exog_sheet: Optional[str] = None

    def _read_any(self, path: Optional[pathlib.Path], sheet: Optional[str] = None) -> pd.DataFrame:
        """Liest beliebiges Format"""
        if not path or not isinstance(path, pathlib.Path) or not path.exists():
            return pd.DataFrame()
        
        suf = path.suffix.lower()
        try:
            if suf == ".parquet":
                return pd.read_parquet(path)
            if suf == ".feather":
                import pyarrow.feather as _fe
                return _fe.read_feather(path)
            if suf in (".csv", ".txt"):
                return pd.read_csv(path)
            if suf in (".xlsx", ".xls"):
                # 🔧 Spezialbehandlung für Excel → nutze _load_excel()
                return self._load_excel(path)
        except Exception as e:
            try:
                Log.data(f"[DiskSource] Lesen fehlgeschlagen für {path.name}: {e}")
            except Exception:
                pass
            return pd.DataFrame()
        return pd.DataFrame()

    def _load_excel(self, path: pathlib.Path) -> pd.DataFrame:
        """
        Lädt eine GVB-Excel-Datei (mit den vier erwarteten Sheets) und
        macht dabei:
        - LFS-/Mini-Dateien erkennen
        - falls nötig instructor.py starten (run_instructor_loader)
        - Spalten sichern
        - bestand/fluss vertauschen falls nötig
        - sektor + datatype setzen
        """
        def _is_lfs_pointer(p: pathlib.Path) -> bool:
            try:
                if p.stat().st_size < 1024:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    if txt.startswith("version https://git-lfs.github.com/spec/v1"):
                        return True
            except Exception:
                pass
            return False

        # 1) Datei prüfen
        if not path.exists():
            # Fallback: Wenn gvb_output.xlsx fehlt (z.B. weil in .gitignore und frisch ausgecheckt),
            # versuchen wir sie zu generieren.
            if path.name == "gvb_output.xlsx":
                logger.warning(f"LoadExcel | {path} fehlt – versuche instructor.py …")
                try:
                    # Versuche, die globale Funktion aufzurufen (falls definiert)
                    # run_instructor_loader() liegt weiter unten im Skript
                    rebuilt = run_instructor_loader()  # type: ignore
                    if rebuilt and rebuilt.exists():
                        logger.info(f"LoadExcel | benutze neu erzeugte Datei: {rebuilt}")
                        # Wir arbeiten mit dem neuen Pfad weiter (könnte Parquet sein, aber hier erwarten wir Excel-Logik)
                        # Da _load_excel spezifisch für Excel ist, hoffen wir, dass run_instructor_loader eine XLSX liefert oder wir sie lesen können.
                        # run_instructor_loader returns Path.
                        # Wenn es Parquet zurückgibt, wird pd.ExcelFile unten crashen.
                        # ABER: Die Methode _read_any ruft _load_excel nur für .xlsx auf.
                        # Wenn wir hier neu bauen, und run_instructor_loader gibt Parquet zurück, haben wir ein Problem,
                        # weil wir hier in _load_excel sind.
                        
                        # Fix: Wenn rebuilt KEIN Excel ist, müssen wir aufgeben oder rekursiv _read_any aufrufen (geht nicht wegen self).
                        # Wir prüfen die Extension.
                        if rebuilt.suffix.lower() == ".xlsx":
                            path = rebuilt
                        else:
                            # Falls Parquet zurückkommt, können wir es hier nicht verarbeiten (wir sind in _load_excel).
                            # Aber wir können es in die Logik von oben durchreichen? Nein.
                            # Wir loggen Warning und returnen empty, ABER beim nächsten Start wird _read_any Parquet nehmen.
                            # Oder wir geben hier auf.
                             logger.warning(f"LoadExcel | Neu erzeugte Datei ist {rebuilt.suffix} ({rebuilt}), aber Excel erwartet.")
                             # Wir versuchen es trotzdem, vllt hat instructor AUCH xlsx erzeugt (macht es meistens).
                             # Wir checken, ob die ursprünglich angefragte Datei jetzt existiert?
                             if path.exists():
                                 pass # Alles gut, path existiert jetzt
                             else:
                                 # Wir haben rebuilt (zB parquet) aber path (xlsx) fehlt immer noch.
                                 return pd.DataFrame()
                    else:
                        return pd.DataFrame()
                except NameError:
                    logger.error("LoadExcel | run_instructor_loader() nicht verfügbar.")
                    return pd.DataFrame()
                except Exception as e:
                    logger.error(f"LoadExcel | Generierung fehlgeschlagen: {e}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

        use_path = path

        # 2) Zu klein oder LFS? → versuchen neu zu bauen
        if use_path.stat().st_size < 2048 or _is_lfs_pointer(use_path):
            logger.warning(f"LoadExcel | {use_path} ist zu klein oder ein Git-LFS-Pointer – versuche instructor.py …")
            rebuilt = None
            try:
                # run_instructor_loader ist weiter unten im Modul definiert,
                # kann hier aber zur Laufzeit aufgerufen werden
                rebuilt = run_instructor_loader()  # type: ignore
            except NameError:
                logger.error("LoadExcel | run_instructor_loader() nicht definiert – kann Datei nicht neu erstellen.")
            except Exception as e:
                logger.error(f"LoadExcel | instructor.py konnte nicht ausgeführt werden: {e}")

            if rebuilt and rebuilt.exists():
                use_path = rebuilt
                logger.info(f"LoadExcel | benutze neu erzeugte Datei: {use_path}")
            else:
                # nichts brauchbares erzeugt → leer zurück
                return pd.DataFrame()

        # 3) Jetzt wirklich ein Excel öffnen
        Log.data(f"LoadExcel | open | file={use_path.name}")
        try:
            xl = pd.ExcelFile(use_path, engine="openpyxl")
            Log.data(f"LoadExcel | sheets | names={xl.sheet_names}")
        except Exception as e:
            logger.error(f"❌ Excel-Fehler: {e}")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        # wir erwarten genau diese vier Sheets, wie in deiner Originalversion
        for sheet in ["bestand_ph", "bestand_nfk", "fluss_ph", "fluss_nfk"]:
            if sheet not in xl.sheet_names:
                try:
                    Log.warn(f"LoadExcel | missing sheet | name={sheet}")
                except Exception:
                    logger.warning(f"LoadExcel | missing sheet | name={sheet}")
                continue

            try:
                df_raw = xl.parse(sheet)
                if df_raw.empty:
                    continue

                # 1) Datumsspalte finden
                date_col = None
                for cand in ["date", "Date", "Datum", "datum", "zeit", "Zeit"]:
                    if cand in df_raw.columns:
                        date_col = cand
                        break
                if date_col and date_col != "date":
                    df_raw = df_raw.rename(columns={date_col: "date"})

                df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
                df_raw = df_raw.dropna(subset=["date"])

                # 2) Pflichtspalten sicherstellen
                for col in ["ebene1", "ebene2", "ebene3", "bestand", "fluss"]:
                    if col not in df_raw.columns:
                        df_raw[col] = np.nan

                # 3) Numerisch konvertieren
                for col in ["bestand", "fluss"]:
                    if col in df_raw.columns:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

                df = df_raw.copy()

                # 4) Swap-Logik wie bei dir: manchmal steht alles in fluss, obwohl es ein bestand-Sheet ist
                bestand_nn = df["bestand"].notna().sum()
                fluss_nn = df["fluss"].notna().sum()
                Log.data(f"LoadExcel | values | sheet={sheet} bestand_nn={bestand_nn} fluss_nn={fluss_nn}")

                if sheet.startswith("bestand") and bestand_nn == 0 and fluss_nn > 0:
                    Log.warn(f"LoadExcel | swap applied | sheet={sheet} src=fluss->bestand")
                    df["bestand"] = df["fluss"].copy()
                    df["fluss"] = np.nan

                if sheet.startswith("fluss") and fluss_nn == 0 and bestand_nn > 0:
                    Log.warn(f"LoadExcel | swap applied | sheet={sheet} src=bestand->fluss")
                    df["fluss"] = df["bestand"].copy()
                    df["bestand"] = np.nan

                # 5) sektor & datatype setzen
                if sheet.endswith("_ph"):
                    df["sektor"] = "PH"
                elif sheet.endswith("_nfk"):
                    df["sektor"] = "NFK"
                else:
                    df["sektor"] = "UNK"

                if sheet.startswith("fluss"):
                    df["datatype"] = "fluss"
                else:
                    df["datatype"] = "bestand"

                frames.append(df)

            except Exception as e:
                logger.error(f"❌ Fehler in Sheet '{sheet}': {e}")
                import traceback
                traceback.print_exc()

        # 6) alles zusammenführen
        if not frames:
            logger.error("❌ Keine Frames aus Excel erstellt.")
            return pd.DataFrame(columns=["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss", "sektor", "datatype"])

        Log.data(f"LoadExcel | merge | frames={len(frames)}")
        out = pd.concat(frames, axis=0, ignore_index=True)
        out = out.sort_values(["date", "sektor", "datatype"], kind="stable").reset_index(drop=True)

        Log.data(f"LoadExcel | final | shape={out.shape}")
        Log.data(f"LoadExcel | final nn | bestand={out['bestand'].notna().sum()} fluss={out['fluss'].notna().sum()}")

        return out


    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Lädt GVB + Exog mit Schema-Garantien"""
        # 1) Roh-Lesen (nutzt _read_any → _load_excel für Excel)
        gvb_df = self._read_any(self.gvb_path, self.gvb_sheet)
        exog_df = self._read_any(self.exog_path, self.exog_sheet) if self.exog_path else pd.DataFrame()

        # 2) Datum vereinheitlichen
        gvb_df = normalize_dataframe_dates(gvb_df, target_col="date") if not gvb_df.empty else pd.DataFrame(columns=["date"])
        exog_df = normalize_dataframe_dates(exog_df, target_col="date") if not exog_df.empty else pd.DataFrame(columns=["date"])

        # 3) Level-Spalten sicherstellen
        if not gvb_df.empty:
            for col in ("ebene1", "ebene2", "ebene3"):
                if col not in gvb_df.columns:
                    gvb_df[col] = np.nan

        # 4) Werte-Spalten kanonisieren (ACHTUNG: Macht nichts mehr, da schon in _load_excel() erledigt)
        # gvb_df = normalize_value_columns(gvb_df) if not gvb_df.empty else pd.DataFrame(...)
        # → Übersprungen, da bereits korrekt

        # 5) Sortierung
        if not gvb_df.empty:
            sort_cols = [c for c in ["date", "ebene1", "ebene2", "ebene3"] if c in gvb_df.columns]
            gvb_df = gvb_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        if not exog_df.empty:
            exog_df = exog_df.sort_values(["date"], kind="stable").reset_index(drop=True)

        # 6) Minimales Schema garantieren
        if gvb_df.empty:
            gvb_df = pd.DataFrame(columns=["date","ebene1","ebene2","ebene3","bestand","fluss"])
        if exog_df.empty:
            exog_df = pd.DataFrame(columns=["date"])

        # Logging
        try:
            Log.data(f"[DiskSource] GVB: {len(gvb_df)} rows | EXOG: {len(exog_df)} rows")
        except Exception:
            pass

        return gvb_df, exog_df


# =========================
# ==========================
# 8) DataManager (with momentum KPIs)
# ==========================
# =========================

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union, Iterable
import numpy as np
import pandas as pd
import logging
from scipy.stats import linregress

# ========== KONSTANTEN ==========
CANON_LEVELS = ("ebene1", "ebene2", "ebene3")
CANON_VALUES = ("bestand", "fluss")

# ========== LOGGER ==========
logger = logging.getLogger("GVB_Dashboard")

# ========== HELPER ==========
def _normalize_value_columns_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisiert bestand/fluss Spaltennamen"""
    if df is None or df.empty:
        return df
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"bestand", "stock", "stocks", "level"}:
            rename_map[c] = "bestand"
        elif cl in {"fluss", "flow", "flows", "nettofluss", "net_flow"}:
            rename_map[c] = "fluss"
    if rename_map:
        df = df.rename(columns=rename_map)
    for v in CANON_VALUES:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    return df


def _kpi_zero_dict() -> dict:
    """Leere KPI-Struktur für Fehler-Fallback"""
    return {
        "Einlagen": 0.0,
        "Wertpapiere": 0.0,
        "Versicherungen": 0.0,
        "Kredite": 0.0,
        "Gesamt GVB": 0.0,
        "Netto GVB": 0.0,
    }


# ======================================================================
#                           DATA MANAGER
# ======================================================================
class DataManager:
    """Kanonischer Zugriff auf GVB Daten (inkl. Momentum-KPIs)."""

    def __init__(self, source: object):
        gvb_raw, exog_raw = source.load()
        self._gvb = self._ensure_canonical_schema_gvb(gvb_raw)
        self._exog = self._ensure_canonical_schema_exog(exog_raw)
        self._cache: Dict[Tuple, pd.DataFrame] = {}

        logger.info(f"[DataManager] Initialisiert: GVB={len(self._gvb)} rows, EXOG={len(self._exog)} rows")

    # ------------------------------------------------------------------
    # Hilfsfunktionen (Canon, Cache, Filter, Aggregation)
    # ------------------------------------------------------------------
    def _canon_date_str(self, x: Optional[Union[str, pd.Timestamp]]) -> str:
        if x is None:
            return ""
        try:
            t = pd.to_datetime(x, errors="coerce")
            if pd.isna(t):
                return ""
            return str(t.normalize().date())
        except Exception:
            return ""

    def _canon_sektor_key(self, sektor: Optional[Union[str, Iterable[str]]]) -> Optional[Tuple[str, ...]]:
        if sektor is None:
            return None
        if isinstance(sektor, (str,)):
            val = str(sektor).strip()
            return (val.lower(),) if val else None
        try:
            vals = [str(s).strip().lower() for s in sektor if str(s).strip()]
            return tuple(sorted(set(vals))) if vals else None
        except Exception:
            s = str(sektor).strip().lower()
            return (s,) if s else None

    def _make_cache_key(
        self,
        *,
        level: str,
        data_type: str,
        smoothing: int,
        start_date: Optional[Union[str, pd.Timestamp]],
        end_date: Optional[Union[str, pd.Timestamp]],
        sektor: Optional[Union[str, Iterable[str]]],
    ) -> Tuple:
        return (
            str(level).lower(),
            str(data_type).lower(),
            int(smoothing),
            self._canon_date_str(start_date),
            self._canon_date_str(end_date),
            self._canon_sektor_key(sektor),
        )

    @property
    def gvb_data(self) -> pd.DataFrame:
        return self._gvb.copy()

    @property
    def exog_data(self) -> pd.DataFrame:
        return self._exog.copy()

    # ------------------------------------------------------------------
    # Öffentliche Funktionen
    # ------------------------------------------------------------------
    def get_aggregated_data(
        self,
        level: str,
        data_type: str,
        smoothing: int = 1,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        sektor: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        level = (level or "ebene1").lower()
        if level not in CANON_LEVELS:
            raise ValueError(f"Unbekanntes level '{level}'. Erlaubt: {', '.join(CANON_LEVELS)}")
        data_type = (data_type or "bestand").lower()
        if data_type not in CANON_VALUES:
            raise ValueError(f"Unbekannter data_type '{data_type}'. Erlaubt: {', '.join(CANON_VALUES)}")
        k = max(int(smoothing or 1), 1)
        key = self._make_cache_key(
            level=level, data_type=data_type, smoothing=k,
            start_date=start_date, end_date=end_date, sektor=sektor
        )
        if key in self._cache:
            return self._cache[key].copy()
        df = self._select(level=level, preferred_value_col=data_type, sektor=sektor)
        df = self._apply_window(df, start_date, end_date)
        if k > 1:
            df = self._apply_ma(df, level=level, k=k)
        expected_cols = {"date", level, data_type}
        for c in expected_cols - set(df.columns):
            df[c] = np.nan
        df = df[["date", level, data_type]]
        self._cache[key] = df
        return df.copy()

    # ------------------------------------------------------------------
    # KPI-Berechnungen
    # ------------------------------------------------------------------
    def calculate_kpis(self, sektor=None, data_type="bestand", smoothing=1,
                       start_date=None, end_date=None) -> dict:
        df = self.get_aggregated_data("ebene1", data_type, smoothing, start_date, end_date, sektor)
        if df.empty:
            return _kpi_zero_dict()
        pv = df.pivot_table(index='date', columns='ebene1', values=data_type,
                            aggfunc='sum').sort_index().fillna(0.0)
        if pv.empty:
            return _kpi_zero_dict()
        last = pv.iloc[-1]
        e = float(last.get('Einlagen', 0.0))
        w = float(last.get('Wertpapiere', 0.0))
        v = float(last.get('Versicherungen', 0.0))
        k = float(last.get('Kredite', 0.0))
        gesamt = e + w + v
        netto = gesamt - k
        return {
            "Einlagen": e, "Wertpapiere": w, "Versicherungen": v,
            "Kredite": k, "Gesamt GVB": gesamt, "Netto GVB": netto
        }

    def calculate_kpis_with_changes(self, sektor=None) -> dict:
        df_e1 = self.get_aggregated_data("ebene1", "bestand", 1, sektor=sektor)
        if df_e1.empty:
            return {k: {"current": 0.0, "qoq": None, "yoy": None}
                    for k in ["Gesamt GVB", "Netto GVB",
                              "Einlagen", "Wertpapiere",
                              "Versicherungen", "Kredite"]}
        pv = df_e1.pivot(index="date", columns="ebene1", values="bestand").sort_index().fillna(0.0)
        def pct_change(series, periods):
            try:
                s = series.dropna()
                if len(s) <= abs(periods):
                    return None
                base, curr = s.iloc[-(periods+1)], s.iloc[-1]
                if base == 0 or pd.isna(base) or pd.isna(curr):
                    return None
                return (curr / base - 1.0) * 100.0
            except Exception:
                return None
        e = pv.get("Einlagen", pd.Series(dtype=float))
        w = pv.get("Wertpapiere", pd.Series(dtype=float))
        v = pv.get("Versicherungen", pd.Series(dtype=float))
        k = pv.get("Kredite", pd.Series(dtype=float))
        gesamt = e.add(w, fill_value=0.0).add(v, fill_value=0.0)
        netto = gesamt.sub(k, fill_value=0.0)
        result = {
            "Gesamt GVB": {"current": float(gesamt.iloc[-1]),
                           "qoq": pct_change(gesamt, 1),
                           "yoy": pct_change(gesamt, 4)},
            "Netto GVB": {"current": float(netto.iloc[-1]),
                          "qoq": pct_change(netto, 1),
                          "yoy": pct_change(netto, 4)}
        }
        for cat in ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite"]:
            s = pv.get(cat, pd.Series(dtype=float))
            result[cat] = {"current": float(s.iloc[-1]) if not s.empty else 0.0,
                           "qoq": pct_change(s, 1),
                           "yoy": pct_change(s, 4)}
        return result
    # ------------------------------------------------------------------
    # Momentum-KPIs
    # ------------------------------------------------------------------
    @staticmethod
    def _short_term_trend(series, periods=3):
        """Berechnet die Steigung (Trend) über die letzten 'periods' Werte."""
        if len(series.dropna()) < periods:
            return None
        y = series.dropna().iloc[-periods:]
        x = range(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope

    @staticmethod
    def _momentum_score(series, periods=4):
        """Durchschnittliche prozentuale Veränderung der letzten n Quartale."""
        returns = series.pct_change().dropna()
        if len(returns) < periods:
            return None
        return returns.iloc[-periods:].mean() * 100

    @staticmethod
    def _rsi(series, periods=8):
        """Relative Strength Index (RSI) auf Basis der Quartalsveränderungen."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(periods).mean().iloc[-1]
        avg_loss = loss.rolling(periods).mean().iloc[-1]
        if pd.isna(avg_gain) or pd.isna(avg_loss) or avg_loss == 0:
            return None
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_momentum_kpis(
        self,
        *,
        level: str = "ebene1",
        parent: Optional[Union[str, Tuple[str, str]]] = None,
        sektor: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Berechnet Kurzfrist-Trend, Momentum-Score und RSI je Kategorie – auf der gewünschten Ebene.
        - level: "ebene1" | "ebene2" | "ebene3"
        - parent:
            * ebene2: Name der übergeordneten ebene1 (z. B. "Einlagen")
            * ebene3: Tuple (ebene1_name, ebene2_name) oder nur ebene1_name (dann alle Unterkategorien)
        """
        level = (level or "ebene1").lower()
        if level not in CANON_LEVELS:
            return pd.DataFrame(columns=["Kurzfristiger Trend", "Momentum Score (%)", "RSI"])

        # Aggregierte Daten der gewünschten Ebene ziehen
        df = self.get_aggregated_data(level, "bestand", smoothing=1, sektor=sektor)
        if df.empty or "date" not in df.columns or level not in df.columns:
            return pd.DataFrame(columns=["Kurzfristiger Trend", "Momentum Score (%)", "RSI"])

        # Parent-Filter anwenden (analog zu Performance/Veränderungen)
        if parent:
            mapping = self.gvb_data[["ebene1", "ebene2", "ebene3"]].drop_duplicates()
            allowed: List[str] = []
            if level == "ebene2":
                # parent ist ebene1
                p1 = parent if not isinstance(parent, (list, tuple)) else parent[0]
                allowed = (
                    mapping.loc[mapping["ebene1"] == p1, "ebene2"]
                    .dropna().astype(str).unique().tolist()
                )
            elif level == "ebene3":
                if isinstance(parent, (list, tuple)) and len(parent) >= 2:
                    p1, p2 = parent[0], parent[1]
                    allowed = (
                        mapping.loc[(mapping["ebene1"] == p1) & (mapping["ebene2"] == p2), "ebene3"]
                        .dropna().astype(str).unique().tolist()
                    )
                else:
                    # Nur ebene1 gegeben → alle ebene3 unterhalb dieser ebene1
                    p1 = parent if not isinstance(parent, (list, tuple)) else parent[0]
                    allowed = (
                        mapping.loc[mapping["ebene1"] == p1, "ebene3"]
                        .dropna().astype(str).unique().tolist()
                    )
            if allowed:
                df = df[df[level].isin(allowed)]

        if df.empty:
            return pd.DataFrame(columns=["Kurzfristiger Trend", "Momentum Score (%)", "RSI"])

        pv = (
            df.pivot(index="date", columns=level, values="bestand")
            .sort_index()
        )

        results = {}
        for col in pv.columns:
            s = pd.to_numeric(pv[col], errors="coerce").dropna()
            if len(s) < 4:
                continue
            results[col] = {
                "Kurzfristiger Trend": self._short_term_trend(s),
                "Momentum Score (%)": self._momentum_score(s),
                "RSI": self._rsi(s),
            }
        return pd.DataFrame(results).T


    # ------------------------------------------------------------------
    # Interne Methoden (Schemas, Filter, MA etc.)
    # ------------------------------------------------------------------
    def _ensure_canonical_schema_gvb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erzwingt kanonisches Schema: ['date', ebene1..3, bestand, fluss]
        """
        logger.info(f"[_ensure_canonical_schema_gvb] START | Input shape: {df.shape if df is not None else 'None'}")

        if df is None or df.empty:
            return pd.DataFrame(columns=["date", *CANON_LEVELS, *CANON_VALUES])

        out = df.copy()
        # 1️⃣ Date normalisieren
        date_col = None
        for cand in ("date", "Date", "Datum", "datum"):
            if cand in out.columns:
                date_col = cand
                break
        if date_col and date_col != "date":
            out = out.rename(columns={date_col: "date"})
        if "date" not in out.columns:
            out["date"] = pd.NaT
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])

        # 2️⃣ Ebenen sicherstellen
        for lvl in CANON_LEVELS:
            if lvl not in out.columns:
                out[lvl] = np.nan

        # 3️⃣ Werte-Spalten normalisieren
        out = _normalize_value_columns_fallback(out)
        if not any(v in out.columns for v in CANON_VALUES):
            out["bestand"] = np.nan

        # 4️⃣ datatype-Filter (falls vorhanden)
        if "datatype" in out.columns:
            valid_rows = (
                ((out["datatype"] == "bestand") & out["bestand"].notna()) |
                ((out["datatype"] == "fluss") & out["fluss"].notna())
            )
            out = out[valid_rows].copy()

        # 5️⃣ Sortieren
        out = out.sort_values(["date", *CANON_LEVELS], kind="stable").reset_index(drop=True)
        return out
    def _ensure_canonical_schema_exog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erzwingt kanonisches Schema: ['date', ...]"""
        if df is None or df.empty:
            return pd.DataFrame(columns=["date"])

        out = df.copy()
        date_col = None
        for cand in ("date", "Date", "Datum", "datum"):
            if cand in out.columns:
                date_col = cand
                break
        if date_col and date_col != "date":
            out = out.rename(columns={date_col: "date"})
        if "date" not in out.columns:
            out["date"] = pd.NaT

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
        return out

    def _pick_value_col(self, df: pd.DataFrame, preferred: str) -> str:
        """Wählt beste verfügbare Werte-Spalte"""
        if preferred in df.columns:
            return preferred
        alt = "fluss" if preferred == "bestand" else "bestand"
        if alt in df.columns:
            return alt
        for c in df.columns:
            if c in ("date", *CANON_LEVELS, "sektor", "sector"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return preferred

    def _apply_sektor_filter(self, df: pd.DataFrame,
                             sektor: Optional[Union[str, List[str]]]) -> pd.DataFrame:
        """Robuster Sektor-Filter (case-insensitive, tolerant gegenüber fehlenden Spalten)."""
        if df is None or df.empty or sektor in (None, [], (), set()):
            return df
        cand_cols = [c for c in df.columns if str(c).lower() in ("sektor", "sector")]
        if not cand_cols:
            return df
        col = cand_cols[0]

        if isinstance(sektor, str):
            targets = {sektor.strip().lower()} if sektor.strip() else set()
        else:
            targets = {str(s).strip().lower() for s in sektor if str(s).strip()}
        if not targets:
            return df

        tmp = df.copy()
        tmp["_sector_lower"] = tmp[col].astype(str).str.strip().str.lower()
        out = tmp[tmp["_sector_lower"].isin(targets)].drop(columns=["_sector_lower"], errors="ignore")
        return out

    def _select(self, level: str, preferred_value_col: str,
                sektor: Optional[Union[str, List[str]]]) -> pd.DataFrame:
        """Aggregiert auf gewünschter Ebene."""
        if self._gvb.empty:
            return pd.DataFrame(columns=["date", level, preferred_value_col])

        df = self._apply_sektor_filter(self._gvb, sektor)
        if "datatype" in df.columns:
            df = df[df["datatype"] == preferred_value_col].copy()

        actual_col = self._pick_value_col(df, preferred_value_col)
        base = df[["date", level, actual_col]].copy()

        out = (
            base.groupby(["date", level], dropna=False)[actual_col]
                .sum(min_count=1)
                .reset_index()
                .sort_values(["date", level], kind="stable")
        )
        if actual_col != preferred_value_col:
            out = out.rename(columns={actual_col: preferred_value_col})
        return out

    def _apply_window(self, df: pd.DataFrame,
                      start_date: Optional[Union[str, pd.Timestamp]],
                      end_date: Optional[Union[str, pd.Timestamp]]) -> pd.DataFrame:
        """Filtert Zeitfenster."""
        if df.empty:
            return df
        s = pd.to_datetime(start_date, errors="coerce") if start_date is not None else None
        e = pd.to_datetime(end_date, errors="coerce") if end_date is not None else None
        if s is not None:
            df = df[df["date"] >= s]
        if e is not None:
            df = df[df["date"] <= e]
        return df

    def _apply_ma(self, df: pd.DataFrame, level: str, k: int) -> pd.DataFrame:
        """Wendet Moving Average (Glättung) an."""
        if df.empty or int(k) <= 1:
            return df
        val_cols = [c for c in df.columns if c not in ("date", level)]
        if not val_cols:
            return df
        col = val_cols[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values(["date", level], kind="stable").copy()
        df[col] = df.groupby(level, dropna=False)[col].transform(
            lambda s: s.rolling(int(k), min_periods=1).mean()
        )
        return df





# ------------------------------------------------------------------------------
# ==========================
# 9) Pipeline integration
# ========================== (robust)
# ------------------------------------------------------------------------------
import sys
from pathlib import Path

forecaster_root = Path(__file__).resolve().parent / "forecaster"
if forecaster_root.exists() and str(forecaster_root) not in sys.path:
    sys.path.insert(0, str(forecaster_root))

HAS_PIPELINE = False
DashboardForecastAdapter = None

try:
    # Falls 'forecaster' ein richtiges Package ist (mit __init__.py)
    from forecaster.forecast_integration import DashboardForecastAdapter as _Adapter
    DashboardForecastAdapter = _Adapter
    HAS_PIPELINE = True
    logger.info("✓ Pipeline-Integration geladen (forecaster.forecast_integration)")
except Exception as e_pkg:
    try:
        # Fallback: Modul direkt unter forecaster/ via sys.path
        from forecast_integration import DashboardForecastAdapter as _Adapter
        DashboardForecastAdapter = _Adapter
        HAS_PIPELINE = True
        logger.info("✓ Pipeline-Integration geladen (forecast_integration)")
    except Exception as e_mod:
        logger.warning(f"⚠️ Pipeline nicht verfügbar: {e_mod}")
# Loader integration
loader_path = Path(__file__).parent / "loader"
if str(loader_path) not in sys.path:
    sys.path.insert(0, str(loader_path))

try:
    from exog_instructor import download_ecb_indicators
    HAS_INSTRUCTOR = True
    logger.info("✓ exog_instructor erfolgreich importiert")
except ImportError as e:
    HAS_INSTRUCTOR = False
    logger.warning(f"⚠️ exog_instructor konnte nicht importiert werden: {e}")
# --- Scenario-Downloader Integration (beim App-Start) ---
scenario_path = Path(__file__).parent / "scenario"
if str(scenario_path) not in sys.path:
    sys.path.insert(0, str(scenario_path))

try:
    from scenario_dataloader import DashDownloadConfig, DashDataDownloader  # nutzt deinen Adapter
    HAS_SCENARIO_DOWNLOADER = True
    logger.info("✓ Scenario-Downloader geladen")
except ImportError as e:
    HAS_SCENARIO_DOWNLOADER = False
    logger.warning(f"⚠️ Scenario-Downloader nicht verfügbar: {e}")


# ==============================================================================
# ==========================
# 10) Configuration
# ==========================
# ==============================================================================
THEME = dbc.themes.FLATLY
BRAND_COLOR = "#14324E"
SUCCESS_COLOR = "#28a745"
DANGER_COLOR = "#dc3545"
WARNING_COLOR = "#ffc107"
INFO_COLOR = "#17a2b8"

# Globale Farbkonfiguration
GVB_COLORS = {
    'Gesamt GVB': '#14324E',      # Dunkelblau (Hauptfarbe)
    'Einlagen': '#17a2b8',        # Türkis/Cyan
    'Wertpapiere': '#28a745',     # Grün  
    'Versicherungen': '#ffc107',  # Gelb/Orange
    'Kredite': '#dc3545'          # Rot
}

# Plotly-kompatible Farbliste in derselben Reihenfolge
GVB_COLOR_SEQUENCE = ['#17a2b8', '#28a745', '#ffc107', '#dc3545', '#14324E']

# ECB Presets
SIMPLIFIED_PRESETS = {
    "aktien": {
        "target": "Wertpapiere",
        "exog": ["zinssatz_10y", "inflation_rate", "verfuegbares_einkommen"],
        "horizon": 6,
        "description": "Wertpapier-/Aktienprognosen"
    },
    "versicherungen": {
        "target": "Versicherungen",
        "exog": ["inflation_rate", "verfuegbares_einkommen", "arbeitslosenquote"],
        "horizon": 8,
        "description": "Versicherungs-/Altersvorsorge"
    },
    "immobilien": {
        "target": "gesamt",
        "exog": ["zinssatz_10y", "immobilienpreise", "verfuegbares_einkommen"],
        "horizon": 8,
        "description": "Gesamtbetrachtung mit Immobilienfokus"
    }
}


# ==============================================================================
# ==========================
# 11) Shared UI Helpers (Colors)
# ==========================
# ==============================================================================

def get_category_color(name: str, fallback: str = "#6c757d") -> str:
    """Kanonischer Farb-Lookup auf Basis von GVB_COLORS."""
    try:
        return GVB_COLORS.get(name, fallback)
    except Exception:
        return fallback

def get_hierarchical_color(
    category: str,
    level: str,
    data_manager,
    fallback: str = "#6c757d"
) -> str:
    """
    Farb-Lookup mit Hierarchie: 
    - ebene1 → direkte Farbe
    - ebene2/3 → Farbe des ebene1-Parents
    """
    if level == 'ebene1':
        return get_category_color(category, fallback)
    try:
        mapping = data_manager.gvb_data[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
        if level == 'ebene2':
            parent = mapping.loc[mapping['ebene2'] == category, 'ebene1']
        else:
            parent = mapping.loc[mapping['ebene3'] == category, 'ebene1']
        if not parent.empty:
            return get_category_color(parent.iloc[0], fallback)
    except Exception:
        pass
    return fallback




# ==============================================================================
# ==========================
# 12) Data Loading & Management
# ==========================
# ==============================================================================
def run_instructor_loader() -> Path:
    """Führt loader/instructor.py aus und liefert den Pfad zur erzeugten GVB-Datei zurück.
    Bevorzugt die neue loader/gvb_output.parquet.
    """
    logger.info("📊 Lade echte GVB-Daten...")
    current_dir = APP_ROOT
    loader_dir = current_dir / "loader"
    instructor_py = loader_dir / "instructor.py"

    if not instructor_py.exists():
        raise FileNotFoundError(f"instructor.py nicht gefunden in {loader_dir}")

    # UTF-8 für stdout/stderr im Child-Prozess erzwingen (Windows-sicher)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        logger.info(f"Führe aus: {instructor_py}")
        result = subprocess.run(
            [sys.executable, str(instructor_py)],
            cwd=str(loader_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode != 0:
            if result.stderr:
                logger.info(f"STDERR: {result.stderr}")
            raise RuntimeError(f"instructor.py fehlgeschlagen (Exit Code: {result.returncode})")

        logger.info("instructor.py erfolgreich ausgeführt")
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")

        # NEU: zuerst nach Parquet im loader/ schauen
        output_loader_parquet = loader_dir / "gvb_output.parquet"
        if output_loader_parquet.exists():
            logger.info(f"✅ gvb_output.parquet bereit: {output_loader_parquet}")
            return output_loader_parquet

        # ab hier: Legacy-Pfade wie bisher
        output_root_xlsx = current_dir / "gvb_output.xlsx"
        output_loader_xlsx = loader_dir / "gvb_output.xlsx"

        # Wenn die Datei im loader liegt, aber im Root fehlt → ins Root kopieren (Kompatibilität)
        if output_loader_xlsx.exists() and not output_root_xlsx.exists():
            import shutil
            try:
                shutil.copy2(str(output_loader_xlsx), str(output_root_xlsx))
            except Exception:
                # notfalls ignorieren – wir haben ja loader/gvb_output.xlsx
                pass

        if output_root_xlsx.exists():
            logger.info(f"✅ gvb_output.xlsx bereit: {output_root_xlsx}")
            return output_root_xlsx
        if output_loader_xlsx.exists():
            logger.info(f"✅ gvb_output.xlsx gefunden (loader): {output_loader_xlsx}")
            return output_loader_xlsx

        raise FileNotFoundError("gvb_output.* wurde nicht erstellt (weder Parquet noch Excel).")

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("instructor.py Timeout nach 5 Minuten") from e
    except Exception as e:
        raise RuntimeError(f"Fehler beim Ausführen von instructor.py: {e}")



def load_gvb_excel_or_build() -> pathlib.Path | None:
    """
    Versucht zuerst, eine vorhandene gvb_output.* an den bekannten Stellen zu finden.
    Bevorzugt die neue Parquet-Datei im loader/.
    Falls nichts Brauchbares liegt, wird der loader/instructor.py gestartet und
    die neu erzeugte Datei zurückgegeben.
    """
    current_dir = APP_ROOT
    loader_dir = current_dir / "loader"

    # 1) bekannte Orte durchgehen – jetzt mit Parquet ganz oben
    candidates = [
        loader_dir / "gvb_output.parquet",           # NEU: unser Ziel
        current_dir / "gvb_output.parquet",          # falls du sie mal ins Root kopierst
        current_dir / "gvb_output.xlsx",
        loader_dir / "gvb_output.xlsx",
        current_dir / "overview" / "gvb_output.xlsx",
        current_dir / "overview" / "loader" / "gvb_output.xlsx",
    ]

    def _is_lfs_or_too_small(p: Path) -> bool:
        try:
            if not p.exists():
                return True
            # Parquet und Excel unter 2KB sind sehr wahrscheinlich Stubs
            if p.stat().st_size < 2048:
                return True
            # bei .xlsx nicht weiter prüfen, bei .parquet auch nicht zwingend
            if p.suffix.lower() == ".xlsx":
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if txt.startswith("version https://git-lfs.github.com/spec/v1"):
                    return True
        except Exception:
            return True
        return False

    # 2) gibt es schon eine brauchbare Datei?
    for cand in candidates:
        if cand.exists() and not _is_lfs_or_too_small(cand):
            logger.info(f"✅ gvb_output.* gefunden und verwendbar: {cand}")
            return cand

    # 3) wenn wir hier sind: nichts Brauchbares → instructor anwerfen
    try:
        rebuilt_path = run_instructor_loader()
        if rebuilt_path and rebuilt_path.exists() and not _is_lfs_or_too_small(rebuilt_path):
            logger.info(f"✅ gvb_output.* neu erstellt: {rebuilt_path}")
            return rebuilt_path
    except Exception as e:
        logger.error(f"❌ Konnte gvb_output.* nicht neu erstellen: {e}")

    logger.error("❌ gvb_output.* weder gefunden noch erstellen können.")
    return None





def create_synthetic_exog_data(gvb_data):
    """Erstellt synthetische exogene Daten basierend auf dem Zeitraum der echten GVB-Daten"""
    logger.info("📄 Generiere passende Makrodaten...")
    # Zeitraum aus GVB-Daten ableiten
    start_date = gvb_data['date'].min()
    end_date = gvb_data['date'].max()
    dates = pd.date_range(start_date, end_date, freq='Q')
    n_periods = len(dates)
    
    logger.info(f"Makrodaten-Zeitraum: {start_date} bis {end_date} ({n_periods} Quartale)")
    # Zeittrend für realistische Entwicklung
    time_trend = np.arange(n_periods) / n_periods
    
    # Makroökonomische Variablen generieren (realistisch für deutschen Markt)
    np.random.seed(42)  # Reproduzierbarkeit
    
    # Zinssatz 10Y (fallender Trend seit 2000er)
    base_interest = 4.0 - 3.0 * time_trend
    cycle_component = 1.2 * np.sin(2 * np.pi * time_trend * 1.5)
    noise = np.random.normal(0, 0.3, n_periods)
    zinssatz_10y = np.clip(base_interest + cycle_component + noise, 0.01, 6.0)
    
    # Inflation mit COVID/Ukraine-Effekten
    base_inflation = 2.0 + 0.3 * np.sin(2 * np.pi * time_trend * 2.5)
    if end_date.year >= 2020:
        recent_periods = int(round(n_periods * 0.10))
        recent_periods = max(0, min(recent_periods, n_periods))
        inflation_shock = np.zeros(n_periods)
        if recent_periods > 0:
            pattern = np.array([2.0, 4.5, 6.0, 3.5, 2.0], dtype=float)
            shock_seq = np.tile(pattern, int(np.ceil(recent_periods / len(pattern))))[:recent_periods]
            inflation_shock[-recent_periods:] = shock_seq
        base_inflation = base_inflation + inflation_shock

    inflation_rate = base_inflation + np.random.normal(0, 0.4, n_periods)
    
    # Arbeitslosenquote (Deutschland-typisch)
    base_unemployment = 9.0 - 4.0 * time_trend  # Langfristig sinkend
    unemployment_cycle = 2.0 * np.sin(2 * np.pi * time_trend * 1.2 + np.pi)
    arbeitslosenquote = np.clip(base_unemployment + unemployment_cycle + 
                               np.random.normal(0, 0.3, n_periods), 3.0, 12.0)
    
    # Verfügbares Einkommen (stetiges Wachstum)
    income_growth = np.full(n_periods, 1.006)  # ~0.6% pro Quartal
    income_shocks = 1 + np.random.normal(0, 0.02, n_periods)
    verfuegbares_einkommen = 100 * np.cumprod(income_growth * income_shocks)
    
    # Immobilienpreise (mit Boom-Phasen)
    house_base_growth = np.full(n_periods, 1.012)  # ~1.2% pro Quartal
    house_boom = np.ones(n_periods)
    mid_point = n_periods // 2
    house_boom[mid_point:] *= 1.008  # Verstärktes Wachstum
    house_variations = 1 + np.random.normal(0, 0.025, n_periods)
    immobilienpreise = 100 * np.cumprod(house_base_growth * house_boom * house_variations)
    
    # Weitere Variablen
    hauptrefinanzierungssatz = np.clip(zinssatz_10y - 0.8 + 
                                      np.random.normal(0, 0.2, n_periods), 0, 4.5)
    
    # BIP (Deutschland-typisches Wachstum)
    gdp_growth = np.full(n_periods, 1.004)  # ~0.4% pro Quartal
    gdp_variations = 1 + np.random.normal(0, 0.012, n_periods)
    bruttoinlandsprodukt = 100 * np.cumprod(gdp_growth * gdp_variations)
    
    # Sparquote (Deutschland-typisch höher)
    base_sparquote = 12.0 + 2.0 * np.sin(np.pi * time_trend)
    sparquote = np.clip(base_sparquote + np.random.normal(0, 0.8, n_periods), 8.0, 18.0)
    
    exog_df = pd.DataFrame({
        'date': dates,
        'zinssatz_10y': zinssatz_10y,
        'inflation_rate': inflation_rate,
        'arbeitslosenquote': arbeitslosenquote,
        'verfuegbares_einkommen': verfuegbares_einkommen,
        'immobilienpreise': immobilienpreise,
        'hauptrefinanzierungssatz': hauptrefinanzierungssatz,
        'bruttoinlandsprodukt': bruttoinlandsprodukt,
        'sparquote': sparquote
    })
    
    logger.info(f"✅ Makrodaten generiert: {exog_df.shape}")
    return exog_df



from datetime import datetime
from pathlib import Path

def ensure_monthly_gvb_refresh() -> Path | None:
    """
    Stellt sicher, dass die GVB-Daten mindestens 1x pro Monat neu geladen werden.
    - merkt sich den letzten erfolgreichen Monat in loader/.gvb_last_refresh
    - wenn aktueller Monat != gespeicherter Monat → run_instructor_loader()
    - gibt den Pfad zur (neu) erzeugten gvb_output.* zurück, wenn vorhanden
    """
    base_dir = APP_ROOT
    loader_dir = base_dir / "loader"
    marker_file = loader_dir / ".gvb_last_refresh"

    current_month = datetime.today().strftime("%Y-%m")

    # Marker lesen
    last_month = None
    if marker_file.exists():
        try:
            last_month = marker_file.read_text(encoding="utf-8").strip()
        except Exception:
            last_month = None

    # gleicher Monat → nix machen, aber vorhandene Datei zurückgeben
    if last_month == current_month:
        for p in [
            loader_dir / "gvb_output.parquet",   # NEU: das ist unser Favorit
            base_dir / "gvb_output.parquet",
            base_dir / "gvb_output.xlsx",
            loader_dir / "gvb_output.xlsx",
        ]:
            if p.exists():
                return p
        return None

    # anderer Monat oder kein Marker → neu laden
    logger = logging.getLogger("GVB_Dashboard")
    logger.info(f"[MonthlyRefresh] Neuer Monat erkannt (alt={last_month}, neu={current_month}) – lade GVB neu …")

    new_path = None
    try:
        new_path = run_instructor_loader()
    except Exception as e:
        logger.error(f"[MonthlyRefresh] Fehler beim monatlichen Reload: {e}")
        return None

    # Marker aktualisieren
    try:
        loader_dir.mkdir(parents=True, exist_ok=True)
        marker_file.write_text(current_month, encoding="utf-8")
    except Exception as e:
        logger.warning(f"[MonthlyRefresh] Konnte Marker-Datei nicht schreiben: {e}")

    return new_path





















# ==============================================================================
# 13) Chart Helpers (Axes)
# ==============================================================================
def format_axis_quarters(fig, date_iterable):
    """
    Formatiert die X-Achse eines Plotly-Figures auf 'Qx YYYY'.
    - date_iterable: Liste/Series von Datums-Objekten (oder Strings).
    - Verwendet tickmode='array' für volle Kontrolle.
    - Setzt Ticks auf Quartalsanfänge.
    """
    try:
        if date_iterable is None or len(date_iterable) == 0:
            return

        # Normalisieren zu Datetime
        dt_index = pd.to_datetime(list(date_iterable))
        if dt_index.empty:
            return

        min_date = dt_index.min()
        max_date = dt_index.max()
        
        # Check against NaT
        if pd.isna(min_date) or pd.isna(max_date):
            return

        # Erzeuge Quartals-Range (Quartals-Start)
        # 'QS' = Quarter Start (01.01., 01.04., 01.07., 01.10.)
        start_q = pd.Timestamp(min_date).to_period('Q').start_time
        end_q = pd.Timestamp(max_date).to_period('Q').end_time
        
        # Generiere Ticks für jedes Quartal in diesem Bereich
        qs = pd.date_range(start=start_q, end=end_q, freq='QS')

        if len(qs) == 0:
            return

        # Labels bauen: "Q1 2024", "Q2 2024" etc.
        tick_vals = []
        tick_text = []

        for d in qs:
            tick_vals.append(d)
            q_label = f"Q{d.quarter} {d.year}"
            tick_text.append(q_label)

        # Ausdünnung bei sehr vielen Daten, um Overlap zu vermeiden
        # (Beispiel: Mehr als 40 Quartale -> 10 Jahre -> jedes 2. Quartal)
        if len(tick_vals) > 40:
             tick_vals = tick_vals[::2]
             tick_text = tick_text[::2]
        elif len(tick_vals) > 80:
             tick_vals = tick_vals[::4]
             tick_text = tick_text[::4]

        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-45,
        )
    except Exception as e:
        logger.warning(f"Failed to format axis quarters: {e}")

    
# ==============================================================================
# UI-KOMPONENTEN UND LAYOUT
# ==============================================================================

def create_navbar():
    """Navigationsleiste erstellen"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink(
                    [html.I(className="bi bi-house-door me-1"), "Übersicht"],
                    href="/",
                    id="nav-overview",
                    className="d-flex align-items-center px-3"
                ),
                className="mx-2"
            ),
            dbc.NavItem(
                dbc.NavLink(
                    [html.I(className="bi bi-graph-up-arrow me-1"), "Prognose-Suite"],
                    href="/forecast",
                    id="nav-forecast",
                    className="d-flex align-items-center px-3"
                ),
                className="mx-2"
            ),
            dbc.NavItem(
                dbc.NavLink(
                    [html.I(className="bi bi-sliders me-1"), "Szenario-Analyse"],
                    href="/scenario",
                    id="nav-scenario",
                    className="d-flex align-items-center px-3"
                ),
                className="mx-2"
            ),
            dbc.NavItem(
                dbc.NavLink(
                    [html.I(className="bi bi-geo-alt me-1"), "Geo-Analyse"],
                    href="/geo",
                    id="nav-geo",
                    className="d-flex align-items-center px-3"
                ),
                className="mx-2"
            ),

        ],
        brand=[html.I(className="bi bi-bank me-2"), "Horn & Company - GVB Dashboard"],
        brand_href="/",
        color="primary",
        dark=True,
        className="navbar-custom mb-4",
        fluid=True
    )


def create_global_settings():
    """Globale Einstellungen"""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Sektor:", className="fw-bold"),
                    dcc.Dropdown(
                        id='sektor-dropdown',
                        options=[
                            {'label': 'Private Haushalte', 'value': 'haushalte'},
                            {'label': 'Nichtfinanzielle Unternehmen', 'value': 'unternehmen'}
                        ],
                        value='haushalte',
                        clearable=False
                    )
                ], width=2),

                dbc.Col([
                    dbc.Label("Zeitraum:", className="fw-bold"),
                    dcc.RangeSlider(
                        id='zeitraum-slider',
                        min=2000,
                        max=2026,
                        step=0.25,
                        value=[2020, 2026],
                        marks={year: str(year) for year in range(2000, 2030, 5)},
                        tooltip={
                            "placement": "bottom", 
                            "always_visible": True,
                            "transform": "decimalYearToQuarter"
                        }
                    )
                ], width=3),

                dbc.Col([
                    dbc.Label("Detail-Ebene:", className="fw-bold"),
                    dcc.Dropdown(
                        id='detail-ebene-dropdown',
                        options=[
                            {'label': 'Hauptkategorien', 'value': 'ebene1'},
                            {'label': 'Unterkategorien', 'value': 'ebene2'},
                            {'label': 'Detailkategorien', 'value': 'ebene3'}
                        ],
                        value='ebene1',
                        clearable=False
                    )
                ], width=2),

                dbc.Col([
                    dbc.Label("Glättung:", className="fw-bold"),
                    dcc.Dropdown(
                        id='glaettung-dropdown',
                        options=[
                            {'label': 'Keine', 'value': 1},
                            {'label': '2 Quartale', 'value': 2},
                            {'label': '4 Quartale', 'value': 4}
                        ],
                        value=1,
                        clearable=False
                    )
                ], width=1),

                dbc.Col([
                    dbc.Label("Datenmodus:", className="fw-bold"),
                    dbc.Switch(
                        id='datenmodus-switch',
                        label="Bestands- vs. Flussdaten",
                        value=False
                    ),
                ], width=2),

                dbc.Col([
                    dbc.Label("Skalierung:", className="fw-bold"),
                    dbc.Switch(
                        id='log-transform-switch',
                        label="Log-Transformation",
                        value=False
                    ),
                ], width=2),
            ], align="end")
        ])
    ], className="settings-panel")

def create_kpi_cards(color_titles: bool = False):
    """KPI-Karten — klickbar, ohne data_manager.
    
    Neu:
    - In der 'Gesamt GVB'-Karte wird rechts dezent der Netto-Wert angezeigt (id: kpi-card-gesamt-netto).
    """

    # Hex-Farben je Kategorie
    color_map_hex = {
        "Einlagen":       get_category_color("Einlagen"), 
        "Wertpapiere":    get_category_color("Wertpapiere"), 
        "Versicherungen":  get_category_color("Versicherungen"), 
        "Kredite":         get_category_color("Kredite"), 
    }

    # Bootstrap-Icons
    icon_map = {
        "Gesamt GVB":     "bi-building",
        "Einlagen":       "bi-piggy-bank",
        "Wertpapiere":    "bi-graph-up-arrow",
        "Versicherungen": "bi-shield-check",
        "Kredite":        "bi-credit-card",
    }

    # Kontextfarben für Icon-Farbe
    icon_context = {
        "Gesamt GVB":     "primary",
        "Einlagen":       "info",
        "Wertpapiere":    "success",
        "Versicherungen": "warning",
        "Kredite":        "secondary",
    }

    # Karten-IDs
    id_map = {
        "Gesamt GVB":     "kpi-card-gesamt",
        "Einlagen":       "kpi-card-einlagen",
        "Wertpapiere":    "kpi-card-wertpapiere",
        "Versicherungen": "kpi-card-versicherungen",
        "Kredite":        "kpi-card-kredite",
    }

    # Spaltenbreiten & Reihenfolge
    widths     = [4, 2, 2, 2, 2]
    categories = ["Gesamt GVB", "Einlagen", "Wertpapiere", "Versicherungen", "Kredite"]

    cards = []
    for i, category in enumerate(categories):
        icon_cls   = icon_map[category]
        icon_color = icon_context[category]
        card_id    = id_map[category]

        # Titel- und Zahlen-Styles
        if category == "Gesamt GVB":
            title_classes = "mb-0 fw-light text-muted" if not color_titles else "mb-0 fw-light"
            title_style   = {} if not color_titles else {"color": "#212529"}
            value_style   = {}
        else:
            hex_color     = color_map_hex[category]
            title_classes = "mb-0 fw-light text-muted" if not color_titles else "mb-0 fw-light"
            title_style   = {} if not color_titles else {"color": hex_color}
            value_style   = {"color": hex_color}

        # Header mit Icon + Titel/Einheit
        header = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        html.I(className=f"bi {icon_cls} text-{icon_color}"),
                        className="d-flex align-items-center justify-content-center bg-light rounded-circle",
                        style={"width": "42px", "height": "42px"}
                    ),
                    width="auto",
                    className="pe-2"
                ),
                dbc.Col(
                    [
                        html.H6(category, className=title_classes, style=title_style),
                        html.Small("Mrd. EUR", className="text-muted")
                    ],
                    className="d-flex flex-column justify-content-center",
                    width=True
                ),
            ],
            align="center",
            className="mb-2 gx-2"
        )

        # Werte-Reihe (für Gesamt GVB inkl. Netto rechts)
        if category == "Gesamt GVB":
            value_row = html.Div(
                [
                    html.H3("...", className="mb-2 fw-bold", id=f"{card_id}-value", style=value_style),
                    html.Small("Netto: ...", id=f"{card_id}-netto", className="ms-2 text-muted"),
                ],
                className="d-flex align-items-baseline justify-content-between"
            )
        else:
            value_row = html.Div(
                html.H3("...", className="mb-2 fw-bold", id=f"{card_id}-value", style=value_style),
                className="d-flex align-items-baseline"
            )

        # QoQ/YoY-Platzhalter
        badges = dbc.Row(
            [
                dbc.Col(html.Div("...", id=f"{card_id}-qoq"), width=6),
                dbc.Col(html.Div("...", id=f"{card_id}-yoy"), width=6),
            ],
            className="g-2"
        )

        # Card + Klickfläche
        card = dbc.Card(
            dbc.CardBody([header, value_row, html.Hr(className="my-2"), badges],
                         style={"padding": "14px"}),
            className="kpi-card h-100"
        )
        clickable = html.Div(
            card, id=card_id, n_clicks=0,
            style={"height": "100%", "cursor": "pointer", "userSelect": "none"}
        )
        cards.append(dbc.Col(clickable, width=widths[i], className="mb-2"))

    return dbc.Row(cards, className="mb-4 gx-2")


def create_overview_layout():
    """Übersichtsseite Layout"""
    return html.Div([
        # Persistenter Drill-Down-Zustand
        dcc.Store(id='drill-store', storage_type='memory',
                  data={"eff_level": "ebene1", "parent": None}),

        create_global_settings(),
        create_kpi_cards(),

        # Hauptcharts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Gesamtentwicklung Geldvermögensbildung", className="mb-0")
                            ], width=8),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("Flächen", id="chart-type-area", size="sm", outline=True, active=True),
                                    dbc.Button("Balken", id="chart-type-bar", size="sm", outline=True)
                                ], size="sm")
                            ], width=4, className="d-flex justify-content-end align-items-center")
                        ], align="center")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='main-trend-chart',
                            style={"height": "550px"},
                            config={"displayModeBar": True, "responsive": False}
                        )
                    ])
                ], className="chart-container")
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Aktuelle Verteilung"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='distribution-chart',
                            style={"height": "490px"},
                            config={"responsive": False}
                        ),
                        html.Hr(),
                        html.Div(id="distribution-metrics")
                    ])
                ], className="chart-container")
            ], width=4)
        ], className="g-3 mb-3"),

        # Veränderungen
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.Span("Veränderungen", className="fw-semibold")
                            ], width=True),
                            dbc.Col([
                                dbc.RadioItems(
                                    id="change-metric",
                                    options=[
                                        {"label": "Absolute Veränderung", "value": "abs"},
                                        {"label": "Ø % pro Quartal", "value": "avg_pct_qoq"},
                                        {"label": "Ø % p.a.", "value": "avg_pct_pa"},
                                    ],
                                    value="abs",
                                    inline=True,
                                    className="me-2"
                                )
                            ], width="auto", className="d-flex justify-content-end align-items-center")
                        ], align="center", className="g-2")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='quarterly-changes-chart',
                            style={"height": "420px"},
                            config={"responsive": False}
                        )
                    ])
                ], className="chart-container")
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Saisonalitäten (Radar)", className="fw-semibold")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='seasonality-radar-chart',
                            style={"height": "420px"},
                            config={"displayModeBar": False, "responsive": False}
                        )
                    ])
                ], className="chart-container")
            ], width=4),
        ], className="g-3 mb-3"),

        # Performance-Analyse
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        className="d-flex justify-content-between align-items-center",
                        children=[
                            html.Span("Performance-Analyse", className="fw-semibold mb-0"),
                            html.Div(
                                dcc.Dropdown(
                                    id='performance-focus-dropdown',
                                    options=[
                                        {'label': 'Alle Kategorien', 'value': 'all'},
                                        {'label': 'Einlagen', 'value': 'Einlagen'},
                                        {'label': 'Wertpapiere', 'value': 'Wertpapiere'},
                                        {'label': 'Versicherungen', 'value': 'Versicherungen'},
                                        {'label': 'Kredite', 'value': 'Kredite'}
                                    ],
                                    value='all',
                                    style={"width": "240px"}
                                ),
                                className="ms-auto"
                            )
                        ]
                    ),
                    dbc.CardBody([
                        dcc.Graph(
                            id='performance-chart',
                            style={"height": "545px"},
                            config={"responsive": False}
                        )
                    ])
                ], className="chart-container")
            ], width=8),

            dbc.Col([
                # Haupt-Kennzahlen-Card
                dbc.Card([
                    dbc.CardHeader("Kennzahlen (Betrachtungszeitraum)"),
                    dbc.CardBody([
                        html.Div(
                            id='performance-metrics-table',
                            style={'height': '200px', 'overflowY': 'auto'}
                        )
                    ])
                ], className="chart-container", style={"marginBottom": "16px"}),

                # Momentum-Kennzahlen-Card
                dbc.Card([
                    dbc.CardHeader("Momentum-Kennzahlen (YTD)"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='momentum-kpi-table',
                            columns=[
                                {"name": "Kategorie", "id": "Kategorie"},
                                {"name": "Kurzfristiger Trend", "id": "Kurzfristiger Trend",
                                 "type": "numeric",
                                 "format": dash_table.Format.Format(
                                     precision=1,
                                     scheme=dash_table.Format.Scheme.fixed
                                 ).group(True)},
                                {"name": "Momentum Score (%)", "id": "Momentum Score (%)",
                                 "type": "numeric",
                                 "format": dash_table.FormatTemplate.percentage(1)},
                                {"name": "RSI", "id": "RSI",
                                 "type": "numeric",
                                 "format": dash_table.Format.Format(
                                     precision=1,
                                     scheme=dash_table.Format.Scheme.fixed
                                 ).group(True)},
                            ],
                            data=[],
                            style_table={'height': '200px', 'overflowY': 'auto'},
                            style_cell={
                                'textAlign': 'center',
                                'fontSize': '11px',
                                'padding': '5px',
                                'fontFamily': 'Arial, sans-serif',
                                'height': '30px'
                            },
                            style_header={
                                'backgroundColor': BRAND_COLOR,
                                'color': 'white',
                                'fontWeight': 'bold',
                                'height': '35px'
                            },
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                                {'if': {'column_id': 'RSI', 'filter_query': '{RSI} > 70'},
                                 'backgroundColor': '#f8d7da', 'color': '#721c24', 'fontWeight': '600'},
                                {'if': {'column_id': 'RSI', 'filter_query': '{RSI} < 30'},
                                 'backgroundColor': '#d4edda', 'color': '#155724', 'fontWeight': '600'},
                                {'if': {'column_id': 'Momentum Score (%)', 'filter_query': '{Momentum Score (%)} > 0'},
                                 'color': '#28a745', 'fontWeight': '600'},
                                {'if': {'column_id': 'Momentum Score (%)', 'filter_query': '{Momentum Score (%)} < 0'},
                                 'color': '#dc3545', 'fontWeight': '600'},
                            ],
                            sort_action="native",
                            page_action="none"
                        ),
                        html.Div(
                            "RSI > 70: überkauft | RSI < 30: überverkauft",
                            style={
                                "fontSize": "10px",
                                "textAlign": "center",
                                "color": "#6c757d",
                                "marginTop": "4px"
                            }
                        )
                    ])
                ], className="chart-container")
            ], width=4)
        ], className="g-3")
    ])



def create_forecast_layout():
    """Pipeline-integrierte Prognose-Suite:
    - Konfiguration (Sektor, Ziel, Horizon, Exogene, Presets, Cache)
    - Forecast-Chart
    - Metriken & Feature Importance
    - Modals: Preset speichern, Runs-Liste, Forecast-Bestätigung
    - Toasts & Stores
    """
    return html.Div(
        [
            # =========================
            # (1) PROGNOSE-KONFIGURATION
            # =========================
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H5("Prognose-Konfiguration", className="mb-0"),
                            dbc.Badge("Pipeline", color="success", className="ms-2"),
                        ]
                    ),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                # (1) Sektor & Toggles
                                dbc.Col(
                                    [
                                        dbc.Label("Sektor", className="fw-bold mb-2"),
                                        dcc.Dropdown(
                                            id="forecast-sektor-dropdown",
                                            options=[
                                                {"label": "Private Haushalte (PH)", "value": "PH"},
                                                {"label": "Nichtfinanzielle Unternehmen (NFK)", "value": "NFK"},
                                            ],
                                            value="PH",
                                            clearable=False,
                                            className="mb-3",
                                        ),
                                        dbc.Switch(
                                            id="forecast-datenmodus-switch",
                                            label="Flussdaten verwenden",
                                            value=True,
                                            className="mb-2",
                                        ),
                                        dbc.Switch(
                                            id="show-backtest-switch",
                                            label="Historische Vorhersagen",
                                            value=False,
                                            className="mt-2",
                                        ),
                                    ],
                                    width={"xs": 12, "lg": 2},
                                    className="mb-3",
                                ),

                                # (2) Zielvariable & Prognosehorizont
                                dbc.Col(
                                    [
                                        dbc.Label("Zielvariable", className="fw-bold mb-2"),
                                        dcc.Dropdown(
                                            id="forecast-target-dropdown",
                                            options=[
                                                {"label": "Gesamt GVB", "value": "gesamt"},
                                                {"label": "Einlagen", "value": "Einlagen"},
                                                {"label": "Wertpapiere", "value": "Wertpapiere"},
                                                {"label": "Versicherungen", "value": "Versicherungen"},
                                                {"label": "Kredite", "value": "Kredite"},
                                            ],
                                            value="Wertpapiere",
                                            clearable=False,
                                            className="mb-3",
                                        ),
                                        dbc.Label("Prognosehorizont", className="fw-bold mb-2"),
                                        html.Div(
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "2Q",
                                                        id={"type": "horizon-btn", "value": 2},
                                                        outline=True,
                                                        color="primary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "4Q",
                                                        id={"type": "horizon-btn", "value": 4},
                                                        outline=True,
                                                        color="primary",
                                                        size="sm",
                                                        active=True,
                                                    ),
                                                    dbc.Button(
                                                        "6Q",
                                                        id={"type": "horizon-btn", "value": 6},
                                                        outline=True,
                                                        color="primary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="w-100 mb-2",
                                            ),
                                            className="px-2",
                                        ),
                                    ],
                                    width={"xs": 12, "lg": 2},
                                    className="mb-3",
                                ),

                                # (3) Einflussfaktoren
                                dbc.Col(
                                    [
                                        dbc.Label("Einflussfaktoren", className="fw-bold mb-2"),
                                        dcc.Dropdown(
                                            id="external-exog-dropdown",
                                            options=[],
                                            multi=True,
                                            placeholder="Aus Liste wählen...",
                                            className="mb-2",
                                        ),
                                        dbc.Input(
                                            id="manual-series-input",
                                            placeholder="Serien-ID manuell eingeben (z. B. ICP.M.DE.N...)",
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Hinzufügen",
                                            id="add-manual-series-btn",
                                            size="sm",
                                            color="outline-primary",
                                            className="mb-1",
                                        ),
                                        html.Small(
                                            "Ausgewählt werden sowohl Dropdown als auch manuell eingegebene Serien",
                                            className="text-muted d-block",
                                        ),
                                    ],
                                    width={"xs": 12, "lg": 4},
                                    className="mb-3",
                                ),

                                # (4) Presets & Cache & Runs
                                dbc.Col(
                                    [
                                        dbc.Label("Presets & Caching", className="fw-bold mb-2"),
                                        dcc.Dropdown(
                                            id="forecast-preset-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="– kein Preset –",
                                            clearable=False,
                                            persistence=False,
                                            className="mb-2",
                                        ),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "Speichern",
                                                    id="save-preset-btn",
                                                    color="outline-secondary",
                                                    size="sm",
                                                ),
                                                dbc.Button(
                                                    "Laden",
                                                    id="load-preset-btn",
                                                    color="outline-info",
                                                    size="sm",
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    "Löschen",
                                                    id="delete-preset-btn",
                                                    color="outline-danger",
                                                    size="sm",
                                                    disabled=True,
                                                ),
                                            ],
                                            className="w-100 mb-3",
                                        ),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    [html.I(className="bi bi-arrow-clockwise me-1"), "Neu"],
                                                    id="retrain-model-btn",
                                                    color="outline-warning",
                                                    size="sm",
                                                ),
                                                dbc.Button(
                                                    [html.I(className="bi bi-folder me-1"), "Liste"],
                                                    id="show-runs-btn",
                                                    color="outline-info",
                                                    size="sm",
                                                ),
                                            ],
                                            className="w-100",
                                        ),
                                        dbc.Switch(
                                            id="model-cache-switch",
                                            label="Cache verwenden",
                                            value=True,
                                            className="mb-2 mt-3",
                                        ),
                                    ],
                                    width={"xs": 12, "lg": 2},
                                    className="mb-3",
                                ),

                                # (5) Aktionen
                                dbc.Col(
                                    [
                                        dbc.Label("Aktionen", className="fw-bold mb-2"),
                                        dbc.Button(
                                            [html.I(className="bi bi-graph-up-arrow me-1"), "Prognose erstellen"],
                                            id="create-forecast-btn",
                                            color="success",
                                            size="lg",
                                            className="w-100 mb-2",
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-download me-1"), "Export"],
                                            id="export-rawdata-btn",
                                            color="outline-primary",
                                            size="sm",
                                            className="w-100",
                                        ),
                                        # Upload ersetzt den früheren "H&C Presets vorbereiten" Button
                                        dcc.Upload(
                                            id="upload-custom-dataset",
                                            children=dbc.Button(
                                                [html.I(className="bi bi-upload me-1"), "Upload",],
                                                id="upload-custom-dataset-btn",
                                                color="outline-secondary",
                                                size="sm",
                                                className="w-100 mt-2",
                                            ),
                                            multiple=False,
                                            style={"width": "100%"},
                                        ),
                                        html.Div(
                                            id="upload-custom-dataset-feedback",
                                            className="small text-muted mt-1",
                                        ),
                                    ],
                                    width={"xs": 12, "lg": 2},
                                    className="mb-3",
                                ),
                            ],
                            align="start",
                            className="g-3",
                        )
                    ),
                ],
                className="settings-panel mb-4",
            ),

            # =========================
            # (2) PROGNOSE-CHART
            # =========================
            dbc.Card(
                [
                    dbc.CardHeader("Prognose-Ergebnisse"),
                    dbc.CardBody(
                        dcc.Loading(
                            html.Div(
                                dcc.Graph(id="forecast-chart", style={"height": "480px", "width": "100%"}),
                                className="graph-100",
                            ),
                            type="circle",
                            parent_className="chart-body-fixed chart-body-500",
                        ),
                        style={"height": "500px", "padding": "0"},
                    ),
                ],
                className="chart-container mb-4",
            ),

            # =========================================
            # (3) METRIKEN & FEATURE IMPORTANCE
            # =========================================
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Performance-Metriken", className="fw-bold"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Div(
                                            id="forecast-metrics",
                                            className="graph-100",
                                            style={"height": "100%", "overflowY": "auto"},
                                        ),
                                        type="circle",
                                        parent_className="chart-body-fixed",
                                    ),
                                    style={"height": "420px", "padding": "12px"},
                                ),
                            ],
                            className="chart-container forecast-metrics-card",
                            style={"height": "480px"},
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Feature Importance Hierarchie", className="fw-bold"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Div(
                                            dcc.Graph(
                                                id="feature-importance-icicle",
                                                config={"displayModeBar": False},
                                                style={"height": "400px", "width": "100%"},
                                            ),
                                            className="graph-100",
                                        ),
                                        type="circle",
                                        parent_className="chart-body-fixed",
                                    ),
                                    style={"height": "420px", "padding": "12px"},
                                ),
                            ],
                            className="chart-container feature-card",
                            style={"height": "480px"},
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Feature Importance", className="fw-bold"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Div(
                                            id="feature-importance-table",
                                            className="graph-100",
                                            style={"height": "100%", "overflowY": "auto"},
                                        ),
                                        type="circle",
                                        parent_className="chart-body-fixed",
                                    ),
                                    style={"height": "420px", "padding": "12px"},
                                ),
                            ],
                            className="chart-container feature-card",
                            style={"height": "480px"},
                        ),
                        width=4,
                    ),
                ],
                id="feature-importance-section",
                className="mb-4",
            ),

            # Downloads
            dcc.Download(id="download-rawdata"),

            # =========================
            # (4) PRESET SPEICHERN MODAL
            # =========================
            dbc.Modal(
                [
                    dbc.ModalHeader("Preset speichern"),
                    dbc.ModalBody(
                        [
                            dbc.Label("Name:", className="fw-bold"),
                            dbc.Input(
                                id="preset-name-input",
                                placeholder="z. B. 'Meine Aktienprognose'",
                                className="mb-3",
                            ),
                            html.H6("Aktuelle Einstellungen:", className="fw-bold"),
                            html.Div(id="preset-preview-content", className="bg-light p-2 rounded"),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Speichern", id="confirm-save-preset", color="primary"),
                            dbc.Button("Abbrechen", id="cancel-save-preset", color="secondary"),
                        ]
                    ),
                ],
                id="save-preset-modal",
                is_open=False,
            ),

            # =========================
            # (5) RUNS-LISTE MODAL
            # =========================
            dbc.Modal(
                [
                    dbc.ModalHeader("Runs aus loader/runs"),
                    dbc.ModalBody(html.Div(id="runs-list-body")),
                    dbc.ModalFooter(dbc.Button("Schließen", id="close-runs-list", className="ms-auto")),
                ],
                id="runs-list-modal",
                is_open=False,
                size="lg",
                scrollable=True,
            ),

            # =========================
            # (6) FORECAST-BESTÄTIGUNG MODAL (neu)
            # =========================
            dbc.Modal(
                [
                    dbc.ModalHeader("Prognose starten?"),
                    dbc.ModalBody(
                        html.Div(
                            [
                                html.Div(
                                    "Bitte bestätige, dass du mit den aktuellen Einstellungen die Prognose berechnen möchtest."
                                ),
                                html.Hr(),
                                html.Div(id="forecast-confirm-summary", className="small text-muted"),
                            ]
                        )
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Abbrechen", id="cancel-recalc-forecast", color="secondary", className="me-2"),
                            dbc.Button(
                                [html.I(className="bi bi-play-fill me-1"), "Jetzt berechnen"],
                                id="confirm-recalc-forecast",
                                color="success",
                            ),
                        ]
                    ),
                ],
                id="forecast-confirm-modal",
                is_open=False,
                size="md",
                backdrop="static",
                scrollable=False,
            ),

            # =========================
            # (7) TOASTS
            # =========================
            dbc.Toast(
                id="preset-save-toast",
                header="Benachrichtigung",
                is_open=False,
                dismissable=True,
                duration=3000,
                icon="success",
                style={"position": "fixed", "top": 70, "right": 20, "zIndex": 2000},
            ),
            dbc.Toast(
                id="hc-prewarm-toast",
                header="H&C Presets",
                is_open=False,
                dismissable=True,
                duration=4000,
                icon="success",
                style={"position": "fixed", "top": 120, "right": 20, "zIndex": 2000},
            ),
            dbc.Toast(
                id="exog-add-toast",
                header="Status",
                is_open=False,
                dismissable=True,
                duration=3500,
                icon="primary",
                style={"position": "fixed", "top": 170, "right": 20, "zIndex": 2000, "maxWidth": "420px"},
            ),
            dbc.Toast(
                id="cache-clear-toast",
                header="Cache",
                is_open=False,
                dismissable=True,
                duration=3500,
                icon="warning",
                style={"position": "fixed", "top": 220, "right": 20, "zIndex": 2000},
            ),

            # =========================
            # (8) STORES
            # =========================
            dcc.Store(id="user-presets-store", data={}),
            dcc.Store(id="forecast-horizon-store", data=6),
            dcc.Store(id="model-artifact-store", data=None),
            dcc.Store(id="hc-presets-cache-store", data={}),
            dcc.Store(id="forecast-state-store", data={"has_forecast": False}),
            # Store für das benutzerdefinierte PIPELINE_PREP-DataFrame
            dcc.Store(id="custom-final-dataset-store", data=None),
        ]
    )


def create_scenario_layout():
    """
    Szenario-Analyse Layout (optimiert):
    - Card 1: Szenario-Konfiguration
    - Card 2: Eingabetabelle (volle Breite)
    - Hauptbereich: links Verlauf & Forecast, rechts Szenario-Impact (zentrierter Spinner)
    - Treiberanalyse: zentrierter Spinner
    - Hidden Stores
    """
    return html.Div([
        # --- Card 1: Szenario-Steuerung ---------------------------------------
        dbc.Card([
            dbc.CardHeader("Szenario-Konfiguration"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Zielreihe:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="scenario-target-dropdown",
                            options=[
                                {"label": "Gesamt GVB", "value": "gesamt"},
                                {"label": "Einlagen", "value": "Einlagen"},
                                {"label": "Wertpapiere", "value": "Wertpapiere"},
                                {"label": "Versicherungen", "value": "Versicherungen"},
                                {"label": "Kredite", "value": "Kredite"},
                            ],
                            value="gesamt",
                            clearable=False,
                        ),
                    ], width=3),

                    dbc.Col([
                        dbc.Label("Vordefinierte Szenarien:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="scenario-preset-dropdown",
                            options=[
                                {"label": "Baseline (Bundesdurchschnitt)", "value": "baseline"},
                                {"label": "Zinsschock (+2% Punkte)", "value": "rate_shock"},
                                {"label": "Hohe Inflation (+3% Punkte)", "value": "high_inflation"},
                                {"label": "Immobilienpreisrückgang (-20%)", "value": "property_down"},
                                {"label": "Rezessionsszenario", "value": "recession"},
                                {"label": "Wirtschaftsboom", "value": "boom"},
                            ],
                            value="baseline",
                            clearable=False,
                        ),
                    ], width=4),

                    dbc.Col([
                        dbc.Label("\u00A0", className="fw-bold mb-1", style={"visibility": "hidden"}),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Übernehmen", id="apply-exog-overrides-btn", color="primary", size="sm"),
                                dbc.Button("Zurücksetzen", id="reset-exog-overrides-btn", color="secondary", size="sm"),
                                dbc.Button("Analyse starten", id="run-scenario-analysis-btn", color="success", size="sm"),
                            ],
                            className="flex-wrap gap-2",
                        ),
                        html.Small(id="apply-exog-overrides-note", className="text-muted d-block mt-1"),
                    ], width=5, className="d-flex flex-column justify-content-start"),
                ], className="g-2", align="start"),
            ])
        ], className="settings-panel mb-3"),

        # --- Card 2: Eingabe-Tabelle ------------------------------------------
        dbc.Card([
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.Span("Exogene Vorhersagen (nächste 4 Quartale)"), width="auto"),
                        dbc.Col(
                            html.Small(
                                [
                                    html.Span("Δ vs. Baseline wird farblich hervorgehoben. "),
                                    html.Span("Tooltip zeigt Baseline, Wert, Δ absolut und in %."),
                                ],
                                className="text-muted",
                            ),
                            className="text-end",
                        ),
                    ],
                    align="center",
                    className="g-0",
                )
            ),
            dbc.CardBody([
                dash_table.DataTable(
                    id="exog-override-table",
                    columns=[],
                    data=[],
                    editable=True,
                    cell_selectable=True,
                    tooltip_data=[],
                    tooltip_duration=None,
                    tooltip_delay=0,
                    style_as_list_view=True,
                    fixed_rows={"headers": True},
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": "12px", "padding": "6px", "textAlign": "right", "minWidth": "100px"},
                    style_cell_conditional=[{"if": {"column_id": "Variable"}, "textAlign": "left", "minWidth": "180px"}],
                    style_header={"fontWeight": "bold"},
                ),
                html.Small(
                    "Tipp: Felder leer lassen, wenn keine Überschreibung gewünscht ist.",
                    className="text-muted d-block mt-2",
                ),
            ])
        ], className="chart-container mb-3"),

        # --- Hauptbereich: links Chart / rechts KPI-Impacts -------------------
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Verlauf & Forecast"),
                    dbc.CardBody([
                        dcc.Graph(id="scenario-comparison-chart", style={"height": "520px"})
                    ])
                ], className="chart-container flex-fill h-100", style={"minHeight": "640px"}),
                width=8, className="d-flex",
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Szenario-Impact"),
                    dbc.CardBody(
                        dcc.Loading(
                            id="loading-scenario-impact",
                            type="circle",
                            delay_show=0,  # sofort zeigen
                            # Loading-Container zentriert den Spinner:
                            style={"minHeight": "600px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                            children=html.Div(
                                id="scenario-kpi-cards",
                                className="vstack gap-2",
                                style={"overflowY": "auto", "height": "600px", "width": "100%"},
                            ),
                        ),
                        style={"position": "relative"},
                        className="d-flex flex-column",
                    ),
                ], className="chart-container flex-fill h-100", style={"minHeight": "640px"}),
                width=4, className="d-flex",
            ),
        ], className="g-2", align="stretch"),

        # --- Treiberanalyse ----------------------------------------------------
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Beitrag einzelner Variablen"),
                    dbc.CardBody(
                        dcc.Loading(
                            id="loading-driver-analysis",
                            type="circle",
                            delay_show=0,
                            style={"minHeight": "360px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                            children=dcc.Graph(id="driver-analysis-chart", style={"height": "340px", "width": "100%"}),
                        ),
                        style={"position": "relative"},
                    ),
                ], className="chart-container"),
                width=12,
            ),
        ], className="mt-2"),

        # --- Hidden Stores -----------------------------------------------------
        dcc.Store(id="scenario-adjustments-store", data={}),
        dcc.Store(id="scenario-results-store", data={}),
        dcc.Store(id="scenario-data-store"),
        dcc.Store(id="exog-baseline-store", data={}),
    ])


# ==============================================================================
# HELPER-FUNKTIONEN UND UTILITIES
# ==============================================================================

from typing import Optional, Union, List

_SEKTOR_ALIASES = {
    # UI → Datenwerte
    "all": None,
    "alle": None,
    "gesamt": None,
    "": None,
    None: None,

    # Haushalte (Privathaushalte)
    "haushalte": ["PH"],
    "privathaushalte": ["PH"],
    "ph": ["PH"],

    # Unternehmen (Nichtfinanzielle Unternehmen)
    "unternehmen": ["NFK"],
    "nichtfinanzielle unternehmen": ["NFK"],
    "nfk": ["NFK"],
}

def map_sektor(value: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """
    Mappt UI-Values robust auf die **echten Werte in den Daten**.
    Gibt None (kein Filter), 'PH', 'NFK' oder Liste davon zurück.
    """
    if value is None:
        return None

    # Liste? → rekursiv mappen & flatten
    if isinstance(value, (list, tuple, set)):
        mapped: List[str] = []
        for v in value:
            mv = map_sektor(v)
            if mv is None:
                continue
            if isinstance(mv, (list, tuple, set)):
                mapped.extend(list(mv))
            else:
                mapped.append(str(mv))
        # Duplikate raus
        mapped = sorted(set(mapped))
        return mapped if mapped else None

    key = str(value).strip().lower()
    mapped = _SEKTOR_ALIASES.get(key)
    # Fallback: Wenn der Nutzer bereits 'PH'/'NFK' liefert, lass es durch
    if mapped is None and key.upper() in {"PH", "NFK"}:
        return key.upper()
    return mapped




def _extract_sektor_from_global(global_settings) -> str | None:
    """
    Versucht, einen Sektorwert aus dem global-settings-store zu ziehen.
    Erwartet dict-ähnliche Daten; erkennt mehrere mögliche Keys.
    """
    try:
        if not isinstance(global_settings, dict):
            return None
        for k in ["sektor", "sector", "selected_sektor", "selectedSector"]:
            if k in global_settings and global_settings[k]:
                return global_settings[k]
    except Exception:
        pass
    return None


def get_gvb_color(name: str) -> str:
    """
    DEPRECATED – bitte get_category_color() verwenden.
    Kompatibilitäts-Wrapper.
    """
    return get_category_color(name, fallback="#6c757d")



def _apply_last_n_quarters(df: pd.DataFrame, n_last: int) -> pd.DataFrame:
    """Filtert auf die letzten n Quartale"""
    if df.empty or not isinstance(n_last, (int, np.integer)) or n_last <= 0:
        return df
    dates = df["date"].sort_values().unique()
    if len(dates) <= n_last:
        return df
    cutoff = dates[-n_last]
    return df[df["date"] >= cutoff].copy()











# ==============================================================================
# DASH APP SETUP UND CORE CALLBACKS
# ==============================================================================

import os
import logging
from pathlib import Path

import dash

app = dash.Dash(
    __name__,
    external_stylesheets=[
        THEME,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css"
    ],
    suppress_callback_exceptions=True,
    title="Horn & Company GVB Dashboard"
)
server = app.server

# --------------------------------------------------------------------------
# Modul-Callbacks registrieren
# --------------------------------------------------------------------------
from overview import overview_main
overview_main.register_overview_callbacks(
    app,
    Log=Log,
    DataManager=DataManager,
    StoreSource=StoreSource,
    DiskSource=DiskSource,
)

from forecaster import forecaster_main
forecaster_main.register_forecaster_callbacks(app, Log)

from scenario import scenario_main
scenario_main.register_scenario_callbacks(app, Log)

from geospacial.geospacial_main import (
    create_geo_layout,
    register_geo_callbacks,
    app_preload_vgrdl,          # kannst du später auch entfernen, wenn ungenutzt
    rebuild_deutschlandatlas_files,
)

register_geo_callbacks(app)

# --------------------------------------------------------------------------
# Custom CSS / index_string
# --------------------------------------------------------------------------
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .navbar-custom {
                background-color: #14324E !important;
                border-bottom: 3px solid #1e3d5a;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .navbar-custom .navbar-brand, .navbar-custom .nav-link {
                color: white !important;
                font-weight: 500;
            }
            .navbar-custom .nav-link:hover {
                color: #e0e0e0 !important;
                background-color: rgba(255,255,255,0.1);
                border-radius: 4px;
                transition: all 0.3s ease;
            }
            .navbar-custom .nav-link.active {
                color: white !important;
                background-color: rgba(255,255,255,0.2);
                border-radius: 4px;
            }
            .kpi-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border: none;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .kpi-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            }
            .settings-panel {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 25px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }
            .chart-container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                padding: 20px;
                margin-bottom: 25px;
            }
            .page-title {
                color: #14324E;
                font-weight: 600;
                border-bottom: 3px solid #14324E;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            .metric-badge {
                font-size: 0.85rem;
                padding: 0.4rem 0.8rem;
                border-radius: 15px;
                font-weight: 600;
            }

            /* === Stabiler Overlay-Spinner + feste Innenhöhe ================= */
            .chart-body-fixed {
              position: relative;
              height: 420px;
              overflow: hidden;
              padding: 0;
              box-sizing: border-box;
            }
            .chart-body-fixed > .dash-loading-overlay {
              position: absolute;
              inset: 0;
              display: flex;
              align-items: center;
              justify-content: center;
              background: transparent;
              z-index: 2;
            }
            .chart-body-fixed .dash-spinner {
              width: 3rem;
              height: 3rem;
            }
            .graph-100 {
              height: 100%;
              width: 100%;
              box-sizing: border-box;
            }
            .spinner-center {
              display: flex;
              align-items: center;
              justify-content: center;
              width: 100%;
              height: 100%;
            }
            .chart-body-500 {
                height: 500px !important;
                padding: 0 !important;
            }
            .chart-body-500 .graph-100 {
                height: 100% !important;
            }
            .chart-body-500 .dash-graph {
                height: 100% !important;
            }
            .forecast-metrics-card,
            .feature-card {
                height: 480px !important;
            }
            .forecast-metrics-card .card-body,
            .feature-card .card-body {
                height: 420px !important;
                padding: 12px !important;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            .forecast-metrics-card .dash-graph,
            .feature-card .dash-graph {
                height: 100% !important;
                flex: 1;
            }
            #feature-importance-table {
                height: 100% !important;
                overflow-y: auto !important;
            }
            #forecast-metrics {
                height: 100% !important;
                overflow-y: auto !important;
            }
            .forecast-metrics-card .dash-spinner-container,
            .feature-card .dash-spinner-container {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 2;
            }
            #feature-importance-icicle {
                height: 100% !important.
            }
            #feature-importance-icicle .js-plotly-plot,
            #feature-importance-icicle .plotly,
            #feature-importance-icicle .main-svg {
                height: 100% !important;
            }
            #forecast-chart {
                height: 100% !important;
            }
            #forecast-chart .js-plotly-plot,
            #forecast-chart .plotly {
                height: 100% !important;
            }
            .forecast-metrics-card .card-body > div,
            .feature-card .card-body > div {
                max-height: 100%;
                overflow: hidden;
            }
            #forecast-metrics .container-fluid {
                height: 100%;
                overflow-y: auto;
            }
            #forecast-metrics .row {
                min-height: 60px;
            }
            .forecast-metrics-card .container-fluid,
            .feature-card .container-fluid {
                max-height: 100%;
                overflow: hidden;
            }
            .modal {
                z-index: 2050 !important;
            }
            .modal-backdrop {
                z-index: 2040 !important;
            }
            .Toastify__toast-container {
                z-index: 3000 !important;
            }

            /* =======================
               GEO Dropdowns - Allgemein
               ======================= */
            .geo-dropdown { 
                width: 100%; 
            }
            .geo-dropdown .Select,
            .geo-dropdown .Select-control,
            .geo-dropdown .Select__control { 
                max-width: 100% !important; 
            }
            .geo-dropdown .Select-value,
            .geo-dropdown .Select__value-container { 
                max-width: 100% !important; 
            }
            .geo-dropdown .Select-value-label,
            .geo-dropdown .Select__single-value {
                display: inline-block !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                max-width: 100% !important;
            }
            .geo-dropdown .Select-menu-outer .Select-option,
            .geo-dropdown .Select-menu-outer .VirtualizedSelectOption,
            .geo-dropdown .Select__menu-list .Select__option {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            .geo-dropdown .Select-menu-outer .Select-option[title],
            .geo-dropdown .Select__menu-list .Select__option[title] { 
                cursor: help; 
            }

            /* =======================
               GEO Region-Dropdown - Höhenbegrenzung
               ======================= */
            /* NUR das aufgeklappte Dropdown-Menü begrenzen (nicht den Input-Container) */
            #geo-region-dropdown .Select-menu-outer,
            #geo-region-dropdown .Select__menu {
                max-height: 200px !important;
                overflow-y: auto !important;
            }
            
            /* Scrollbar-Styling NUR für das Menü */
            #geo-region-dropdown .Select-menu-outer::-webkit-scrollbar,
            #geo-region-dropdown .Select__menu::-webkit-scrollbar {
                width: 8px;
            }
            
            #geo-region-dropdown .Select-menu-outer::-webkit-scrollbar-track,
            #geo-region-dropdown .Select__menu::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            
            #geo-region-dropdown .Select-menu-outer::-webkit-scrollbar-thumb,
            #geo-region-dropdown .Select__menu::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            
            #geo-region-dropdown .Select-menu-outer::-webkit-scrollbar-thumb:hover,
            #geo-region-dropdown .Select__menu::-webkit-scrollbar-thumb:hover {
                background: #555;
            }

            /* Responsive Anpassungen für Geo-Dropdowns */
            @media (min-width: 1200px) {
                .geo-dropdown .Select-value-label,
                .geo-dropdown .Select__single-value { 
                    max-width: 100% !important; 
                }
            }
            @media (max-width: 1199px) and (min-width: 992px) {
                .geo-dropdown .Select-value-label,
                .geo-dropdown .Select__single-value { 
                    max-width: 85% !important; 
                }
            }
            @media (max-width: 991px) and (min-width: 768px) {
                .geo-dropdown .Select-value-label,
                .geo-dropdown .Select__single-value { 
                    max-width: 75% !important; 
                }
            }
            @media (max-width: 767px) {
                .geo-dropdown .Select-value-label,
                .geo-dropdown .Select__single-value { 
                    max-width: 65vw !important; 
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==============================================================================
# HAUPT-LAYOUT
# ==============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Data Stores (Thread-sicher)
    dcc.Store(id='gvb-data-store', storage_type='memory'),
    dcc.Store(id='exog-data-store', storage_type='memory'),
    dcc.Store(id='data-metadata-store', storage_type='memory'),

    # Unsichtbarer Fallback
    html.Div(
        dbc.Checklist(
            id="forecast-real-switch",
            options=[{"label": "", "value": "real"}],
            value=[],
            switch=True
        ),
        style={"display": "none"},
        id="forecast-real-switch-fallback"
    ),

    create_navbar(),
    dbc.Container(
        id='page-content',
        fluid=True,
        style={'minHeight': '85vh'}
    ),

    # Global Stores
    dcc.Store(id='global-settings-store', data={}),
    dcc.Store(id='selected-categories-store', data=[]),
    dcc.Store(id='forecast-horizon-slider', data=6),
])

# ==============================================================================
# NAVIGATION CALLBACKS
# ==============================================================================

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/forecast':
        return create_forecast_layout()
    elif pathname == '/scenario':
        return create_scenario_layout()
    elif pathname == '/geo':
        # kommt aus geospacial_main
        return create_geo_layout()
    else:
        return create_overview_layout()


@app.callback(
    [
        Output('nav-overview', 'active'),
        Output('nav-forecast', 'active'),
        Output('nav-scenario', 'active'),
        Output('nav-geo', 'active')
    ],
    Input('url', 'pathname')
)
def _set_active_nav(pathname):
    return (
        pathname == '/',
        pathname == '/forecast',
        pathname == '/scenario',
        pathname == '/geo',
    )

# ==============================================================================
# PRELOAD-FUNKTION (für __main__ und für Gunicorn-Import)
# ==============================================================================

def run_startup_preloads():
    """
    Führt alle Preload-/Housekeeping-Jobs aus, ohne den Dash-Server zu starten.
    - monatlicher GVB-Refresh
    - GVB-Datei sicherstellen (Excel/Parquet)
    - Geo / Deutschlandatlas-Rebuild
    - Szenario-Downloader (monatlich)
    - Szenario Analyse-Daten
    Kann sowohl im __main__-Start als auch beim Import (z.B. im Docker)
    aufgerufen werden. Sollte idempotent sein.
    """
    # nur im echten Start / Preload, nicht bei jedem Request
    self_clean_startup()

    # Logger für diese Funktion
    lg = logging.getLogger("GVB_Dashboard")
    lg.info("=" * 60)
    lg.info("🏦 Horn & Company GVB Dashboard – Preload")
    lg.info("=" * 60)

    # Szenario Komponenten nur laden, wenn vorhanden
    # HINWEIS: scenario/ ist bereits in sys.path (siehe Zeile ~1928)
    try:
        from scenario_dataloader import (
            DashDownloadConfig,
            DashDataDownloader,
            should_run_this_month,
            mark_ran_this_month,
        )
        has_scenario_downloader_local = True
        lg.info("✅ scenario_dataloader erfolgreich importiert")
    except Exception as e:
        lg.warning(f"⚠️ scenario_dataloader konnte nicht geladen werden: {e}")
        import traceback
        lg.warning(f"Traceback: {traceback.format_exc()}")
        has_scenario_downloader_local = False

    # 1) GVB Datei sicherstellen
    try:
        refreshed_path = ensure_monthly_gvb_refresh()
        if refreshed_path:
            lg.info(f"✅ Monatlicher GVB Refresh ok: {refreshed_path}")
        else:
            lg.info("ℹ️ Monatlicher GVB Refresh nicht nötig oder fehlgeschlagen, nutze bestehende Datei.")
    except Exception as e:
        lg.warning(f"⚠️ Monatlicher GVB Refresh fehlgeschlagen: {e}")

    try:
        gvb_path = load_gvb_excel_or_build()
        if gvb_path is not None and gvb_path.exists():
            lg.info(f"✅ GVB Datei vorhanden oder erzeugt: {gvb_path}")
        else:
            lg.warning("⚠️ Keine gültige gvb_output.xlsx gefunden, Dashboard nutzt Fallback.")
    except Exception as e:
        lg.warning(f"⚠️ GVB Preload fehlgeschlagen: {e}")

    # 2) Geo / Deutschlandatlas: Deutschlandatlas_*_merged neu schreiben (alle Ebenen)
    try:
        geo_results = rebuild_deutschlandatlas_files(
            levels=("krs", "gem", "vbgem"),  # explizit alle Ebenen
            export_excel=False,              # kein Excel-Export
            logger=lambda m: lg.info(str(m))
        )

        if geo_results:
            for lvl, info in sorted(geo_results.items()):
                indicator_name = info.get("indicator_name")
                matched = info.get("matched")
                total = info.get("total")
                merged_parquet = info.get("merged_parquet")
                merged_excel = info.get("merged_excel")
                rows = info.get("rows")

                # Fall A: VGRdL-Merge hat geklappt → wir haben indicator/matched/total
                if indicator_name is not None and matched is not None and total is not None:
                    lg.info(
                        f"✅ Geo Rebuild {lvl}: {indicator_name} "
                        f"(matched {matched}/{total}) → {merged_parquet} / {merged_excel}"
                    )
                # Fall B: nur Basis-Datei aus Deutschlandatlas.xlsx aufgebaut
                else:
                    lg.info(
                        f"✅ Geo Basis-Rebuild {lvl}: {merged_parquet} / {merged_excel} "
                        f"(Zeilen: {rows})"
                    )
        else:
            lg.info("ℹ️ Geo Rebuild: keine Ebene erfolgreich aktualisiert (siehe Log oben).")
    except Exception as e:
        lg.warning(f"⚠️ Geo Rebuild fehlgeschlagen: {e}")

    # 3) Szenario Daten wie bisher (max 1 mal pro Monat)
    if has_scenario_downloader_local:
        try:
            cfg_file = scenario_path / "config.yaml"
            if not cfg_file.exists():
                raise FileNotFoundError(f"scenario/config.yaml nicht gefunden unter {cfg_file}")

            force_refresh = os.getenv("SCENARIO_FORCE_REFRESH", "0") == "1"
            
            # Prüfe ob output.xlsx existiert - wenn nicht, erzwinge Download
            # HINWEIS: config.yaml hat output_path: "output.xlsx",
            # scenario_dataloader._resolve_under_scenario_data() macht daraus /app/scenario/data/output.xlsx
            output_check_path = scenario_path / "data" / "output.xlsx"
            if not output_check_path.exists():
                lg.warning(f"⚠️ {output_check_path} fehlt – erzwinge Szenario-Download")
                force_refresh = True

            lg.info("📥 Szenario Preload (monatlich gesteuert)…")
            if force_refresh or should_run_this_month(scenario_path):
                cfg = DashDownloadConfig.from_yaml(str(cfg_file))
                # HINWEIS: cfg.output_path ist bereits korrekt aufgelöst durch
                # DashDownloadConfig._resolve_under_scenario_data() → /app/scenario/data/output.xlsx
                
                out_path = Path(cfg.output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Cache relativ zu scenario
                cache_dir = Path(cfg.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                lg.info(f"⏬ Lade Szenario Daten ({cfg.start_date} → {cfg.end_date}) …")
                runner = DashDataDownloader(
                    cfg,
                    logger=lambda m: (Log.scenario(m) if 'Log' in globals() else lg.info(str(m)))
                )
                final_df, written_file = runner.run(save=True)
                mark_ran_this_month(scenario_path)

                if written_file:
                    lg.info(f"✅ Szenario Datei geschrieben: {written_file.resolve()}")
                else:
                    lg.info(f"✅ Szenario Daten im Speicher erzeugt (Output: {out_path.resolve()})")
            else:
                lg.info("⏭️ Szenario Download wird übersprungen (diesen Monat schon erledigt).")
        except Exception as e:
            lg.warning(f"⚠️ Szenario Download fehlgeschlagen: {e}")

        # Analyse Daten beim Start sicherstellen
        try:
            # WICHTIG: Wir nutzen die Logik aus scenario_main, da diese
            # output.xlsx -> analysis_data.xlsx konvertieren kann.
            from scenario.scenario_main import ensure_analysis_data_on_startup as _ensure_analysis
            _ensure_analysis()
        except Exception as e:
            lg.warning(f"⚠️ Konnte scenario Analyse Daten nicht initialisieren: {e}")

# ==============================================================================
# OPTIONALER IMPORT-PRELOAD (für Gunicorn & Co.) mit File-Lock
# ==============================================================================
# Wenn z.B. im Docker-Container mit gunicorn gestartet wird, führt der erste
# Worker die Preloads aus, während andere warten. Dies verhindert Race Conditions
# bei der Datei-Generierung.

import time

try:
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except Exception:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False


def _run_preload_with_lock():
    """Führt Preload mit File-Lock aus - nur ein Worker generiert Dateien.

    Fallback: Wenn kein `fcntl` verfügbar ist (z.B. Windows), läuft Preload ohne Lock.
    """
    lg = logging.getLogger("GVB_Dashboard")

    # Cross-Platform Lockfile-Pfad (statt hardcoded '/app/...')
    lock_file = (APP_ROOT / ".gvb_preload.lock")
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Windows / kein fcntl: einfach ohne Lock laufen lassen
    if not _HAS_FCNTL:
        lg.info("🔓 Preload-Lock nicht verfügbar (kein fcntl) – führe Initialisierung ohne Lock aus")
        run_startup_preloads()
        return

    try:
        # Lock-Datei öffnen/erstellen
        with open(lock_file, "w") as f:
            try:
                # Versuche exklusiven Lock zu bekommen (non-blocking)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Wir haben den Lock - führe Preload aus
                lg.info("🔒 Preload-Lock erhalten - führe Initialisierung aus")
                run_startup_preloads()
                lg.info("✅ Preload abgeschlossen")

            except BlockingIOError:
                # Ein anderer Worker hat den Lock - warte bis er fertig ist
                lg.info("⏳ Warte auf Preload durch anderen Worker...")
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Blocking wait
                lg.info("✅ Preload durch anderen Worker abgeschlossen")

    except Exception as e:
        lg.warning(f"⚠️ Preload-Lock Fehler: {e}")
        # Letzter Fallback: lieber ohne Lock starten, als gar nicht zu starten
        try:
            run_startup_preloads()
        except Exception as e2:
            lg.warning(f"⚠️ Preload-Fallback ohne Lock fehlgeschlagen: {e2}")



# ==============================================================================
# PRELOADS (if requested)
# ==============================================================================

if __name__ == "__main__":
    lg = logging.getLogger("GVB_Dashboard")
    lg.info("🚀 Starte Dash-Server im __main__-Modus …")

    # Preloads lokal ausführen (falls nicht schon beim Import gelaufen)
    # run_startup_preloads()  # Nicht nötig, läuft schon beim Import

    # Dash starten
    app.run(
        host="0.0.0.0",
        port=8080,
        # debug=True
        debug=False
    )

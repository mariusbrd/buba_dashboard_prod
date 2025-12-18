# forecaster_main.py
# # Aufgabe: Dash Ebene. Callbacks, Anzeige, Laden und Speichern von Presets.

#  -*- coding: utf-8 -*-
"""
Forecasting-Ansicht / -Callbacks für das Dashboard.
Arbeitet Store-basiert (GVB/EXOG JSON) und integriert die Pipeline bei Bedarf.
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import base64
import sys
import os
import io
import json
import re
import unicodedata
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path
from textwrap import shorten
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

import dash
from dash import html, dcc, no_update, callback_context
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.colors import sequential as pseq

# dash-bootstrap-components optional
try:
    import dash_bootstrap_components as dbc
except Exception:
    dbc = None


import shutil


FORECASTER_DIR = Path(__file__).resolve().parent

try:
    APP_ROOT: Path = FORECASTER_DIR.parent
except Exception:
    APP_ROOT = Path.cwd()


# ==============================================================================
# KONFIGURATION & KONSTANTEN
# ==============================================================================

# Farben/Theming
try:
    from app import GVB_COLORS, GVB_COLOR_SEQUENCE, BRAND_COLOR
except Exception:
    GVB_COLORS = {
        'Gesamt GVB': '#14324E',
        'Einlagen': '#17a2b8',
        'Wertpapiere': '#28a745',
        'Versicherungen': '#ffc107',
        'Kredite': '#dc3545',
    }
    GVB_COLOR_SEQUENCE = ['#17a2b8', '#28a745', '#ffc107', '#dc3545', '#14324E']
    BRAND_COLOR = "#14324E"





def format_axis_quarters(fig, date_iterable):
    """
    Formatiert die X-Achse als KATEGORISCH (String-basiert), um "Qx YYYY"
    als Header im Hover zu erzwingen, während die visuelle Sortierung erhalten bleibt.
    Ticks weiterhin nur alle 5 Jahre (z.B. Q1 2020).
    """
    try:
        if date_iterable is None or len(date_iterable) == 0:
            return

        dt_index = pd.to_datetime(list(date_iterable))
        if dt_index.empty:
            return

        min_date = dt_index.min()
        max_date = dt_index.max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return

        # 1. Master-Timeline erstellen (Alle Quartale im Bereich)
        try:
            start_q = pd.Timestamp(min_date).to_period('Q').start_time
            end_q = pd.Timestamp(max_date).to_period('Q').end_time
        except Exception:
            return
            
        full_qs = pd.date_range(start=start_q, end=end_q, freq='QS')
        if len(full_qs) == 0:
            return

        # Mapping: Timestamp -> "Qx YYYY"
        # UND: Erstellen der geordneten Kategorie-Liste
        def to_q_str(d):
            return f"Q{d.quarter} {d.year}"
            
        category_order = [to_q_str(d) for d in full_qs]
        
        # 2. Alle Traces auf String-Werte mappen
        for trace in fig.data:
            if getattr(trace, 'x', None) is None:
                continue
            try:
                # Altdaten (Datetimes) zu Strings konvertieren
                ts_series = pd.to_datetime(pd.Series(trace.x), errors='coerce')
                # NaT ignorieren/leeren
                new_x = ts_series.apply(lambda x: to_q_str(x) if pd.notna(x) else None).tolist()
                trace.x = new_x
                
                # Hovertemplate bereinigen (Header macht jetzt den Job)
                trace.hovertemplate = "%{fullData.name}: %{y}<extra></extra>"
            except Exception:
                pass

        # 3. Ticks berechnen (5 Jahres Abstand)
        min_year = min_date.year
        max_year = max_date.year
        tick_vals = []
        
        years = range(min_year, max_year + 1)
        span = max_year - min_year
        
        target_years = []
        if span >= 5:
            target_years = [y for y in years if y % 5 == 0]
        else:
            target_years = list(years)
            
        # Wir setzen den Tick genau auf das String-Label "Q1 YYYY"
        for y in target_years:
            val = f"Q1 {y}"
            if val in category_order:
                tick_vals.append(val)
        
        # 4. Layout Update
        fig.update_xaxes(
            type='category',
            categoryorder='array',
            categoryarray=category_order,
            
            tickmode='array',
            tickvals=tick_vals,
            tickangle=0
        )
            
    except Exception:
        pass





INFO_COLOR = GVB_COLORS.get('Einlagen', '#0d6efd')

# Pfade für Presets/Snapshots
PRESETS_DIR = FORECASTER_DIR / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOTS_DIR = PRESETS_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

HCPRESET_CACHE_FILE = PRESETS_DIR / "hc_presets_cache.json"


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

try:
    from app import _logger as APP_LOGGER
except Exception:
    APP_LOGGER = None

_logger = APP_LOGGER or logging.getLogger("GVB_Dashboard")
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | GVB_Dashboard | %(message)s")
    )
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Log-Adapter
try:
    from app import Log as _AppLog
    Log = _AppLog
except Exception:
    class Log:
        @staticmethod
        def scenario_table(msg: str): _logger.info(msg)
        @staticmethod
        def info(msg: str): _logger.info(msg)
        @staticmethod
        def warning(msg: str): _logger.warning(msg)
        @staticmethod
        def error(msg: str): _logger.error(msg)
        @staticmethod
        def exception(msg: str): _logger.exception(msg)

# ==============================================================================
# CALLBACK PROXY
# ==============================================================================

class _CallbackProxy:
    """Sammelt Callback-Registrierungen bis zur echten App-Registrierung."""
    
    def __init__(self):
        self._registrations = []

    def callback(self, *args, **kwargs):
        def _decorator(func):
            self._registrations.append((args, kwargs, func))
            return func
        return _decorator

app = _CallbackProxy()

# ==============================================================================
# PIPELINE-ADAPTER
# ==============================================================================

try:
    from forecast_integration import DashboardForecastAdapter, HAS_PIPELINE
except Exception:
    DashboardForecastAdapter = None
    HAS_PIPELINE = False

# ==============================================================================
# BASIS UTILITIES
# ==============================================================================

def _filter_gvb_json_by_sektor(gvb_json: str, sektor_value: str) -> str:
    """Filtert den GVB-Store nach 'sektor' (PH/NFK) und gibt JSON (orient='split') zurück.
       Fällt robust zurück (ungefiltert), wenn Spalte fehlt oder sektor_value None/leer ist."""
    try:
        df = _parse_store_df(gvb_json)
        if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor_value:
            sekt = str(sektor_value).strip().upper()
            df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            _logger.debug(f"GVB-Daten nach Sektor '{sekt}' gefiltert: {len(df)} Zeilen")
        return df.to_json(orient="split", date_format="iso")
    except Exception as e:
        _logger.warning(f"Sektorfilterung fehlgeschlagen: {e}, verwende ungefilterte Daten")
        return gvb_json


def _parse_store_df(payload) -> pd.DataFrame:
    """
    Robust gegen verschiedene Store-Formate:
    - pandas JSON orient='split'
    - dict mit 'data' + 'columns'
    - dict mit 'records'
    - list von records
    - None -> leerer DataFrame
    """
    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, str):
        try:
            return pd.read_json(payload, orient="split")
        except Exception:
            pass
        try:
            obj = json.loads(payload)
        except Exception:
            _logger.debug("Store-Payload konnte nicht als JSON geparst werden")
            return pd.DataFrame()
    else:
        obj = payload

    if isinstance(obj, dict):
        if set(obj.keys()) >= {"index", "columns", "data"}:
            try:
                return pd.DataFrame(data=obj["data"], columns=obj["columns"])
            except Exception:
                pass
        if "records" in obj and isinstance(obj["records"], list):
            try:
                return pd.DataFrame.from_records(obj["records"])
            except Exception:
                pass
        if "data" in obj and "columns" in obj:
            try:
                return pd.DataFrame(obj["data"], columns=obj["columns"])
            except Exception:
                pass

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        try:
            return pd.DataFrame.from_records(obj)
        except Exception:
            pass

    _logger.debug("Store-Payload konnte nicht in DataFrame konvertiert werden")
    return pd.DataFrame()


def _make_metadata_jsonable(obj):
    """Wandelt Metadata aus der Pipeline in JSON-kompatible Strukturen um."""
    import numpy as np
    import pandas as pd
    from datetime import date, datetime

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_metadata_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_metadata_jsonable(v) for v in obj]
    # Fallback: als String
    return str(obj)


def _flatten_metadata_to_df(meta: dict):
    """Macht aus einem verschachtelten Metadata-Dict eine 2-Spalten-Tabelle."""
    import pandas as pd
    rows = []

    def _walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(f"{prefix}[{i}]", v)
        else:
            rows.append({"key": prefix, "value": value})

    if isinstance(meta, dict):
        _walk("", meta)
    return pd.DataFrame(rows)


def _safe_load_store(payload) -> Optional[pd.DataFrame]:
    """Lädt DataFrame aus Store (robust) → None, wenn nicht darstellbar."""
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            return pd.read_json(payload, orient="split")
        except Exception:
            try:
                return pd.DataFrame(json.loads(payload))
            except Exception:
                return None
    if isinstance(payload, (list, dict)):
        try:
            return pd.DataFrame(payload)
        except Exception:
            return None
    return None


def _normalize_dates(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalisiert Datums-Spalte → 'date' als datetime64[ns]."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    date_candidates = ["date", "Date", "Datum", "datum", "period", "Period"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    
    return df.sort_values("date").reset_index(drop=True)


def _safe_series(df: pd.DataFrame, col: str, index_like=None) -> pd.Series:
    """Gibt eine numerische Series mit gegebenem Index zurück (fehlende -> 0.0)."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    else:
        s = pd.Series(0.0, index=df.index if index_like is None else index_like)
    
    if index_like is not None:
        s = s.reindex(index_like, fill_value=0.0)
    
    return s


def _slugify(text: str) -> str:
    """ASCII-Slug: Kleinbuchstaben, Ziffern, Bindestriche/Unterstriche."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]+", "", text)
    text = re.sub(r"[\s]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-_")


def _current_quarter_end() -> str:
    """Aktuelles Quartalsende als YYYY-MM."""
    today = date.today()
    q_end_month = ((today.month - 1) // 3 + 1) * 3
    return f"{today.year}-{q_end_month:02d}"


def _looks_like_ecb_code(s: str) -> bool:
    """Grobe Heuristik für ECB-Codes."""
    if not s:
        return False
    s = str(s)
    return (s.count(".") >= 2) or bool(re.search(r"[A-Z]{2,}\.[A-Z0-9]{1,}\.", s))


def _build_main_e1_table_from_store(
    gvb_json,
    *,
    data_type: str = "bestand",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    smoothing: int = 1,
    use_log: bool = False,
    sektor: Optional[str] = None
) -> pd.DataFrame:
    """
    Erzeugt die Ebene-1 Haupttabelle direkt aus dem gvb-data-store:
    Spalten: Einlagen, Wertpapiere, Versicherungen, Kredite, Gesamt_GVB, Netto_GVB
    """
    df = _parse_store_df(gvb_json)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    # Sicherstellen dass alle Spalten existieren
    required_cols = ("ebene1", "ebene2", "ebene3", "sektor", "bestand", "fluss")
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Sektor-Filter
    if sektor:
        df = df[df["sektor"] == sektor].copy()

    # Datum normalisieren
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return pd.DataFrame()

    # Zeitraum festlegen
    if start_date is None or end_date is None:
        start_date = df["date"].min()
        end_date = df["date"].max()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Wert-Spalte auswählen
    valcol = "fluss" if data_type == "fluss" else "bestand"
    if valcol not in df.columns:
        return pd.DataFrame()

    # Filtern und aggregieren
    d = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if d.empty:
        return pd.DataFrame()

    agg = (
        d.groupby(["date", "ebene1"], dropna=False, as_index=False)[valcol]
        .sum()
        .rename(columns={valcol: "wert"})
    )

    # Glättung
    if isinstance(smoothing, (int, float)) and int(smoothing) > 1:
        s = int(smoothing)
        agg = agg.sort_values(["date", "ebene1"]).set_index("date")
        agg["wert"] = agg.groupby("ebene1")["wert"].transform(
            lambda x: x.rolling(s, min_periods=1).mean()
        )
        agg = agg.reset_index()

    # Pivot
    pv = (
        agg.pivot(index="date", columns="ebene1", values="wert")
        .sort_index()
        .fillna(0.0)
    )

    if pv.empty:
        return pd.DataFrame()

    # Log-Transformation
    if use_log:
        pv = np.sign(pv) * np.log1p(np.abs(pv))

    # Spalten extrahieren
    einlagen = _safe_series(pv, "Einlagen")
    wertpapiere = _safe_series(pv, "Wertpapiere", index_like=einlagen.index)
    versicherungen = _safe_series(pv, "Versicherungen", index_like=einlagen.index)
    kredite = _safe_series(pv, "Kredite", index_like=einlagen.index)

    gesamt = einlagen + wertpapiere + versicherungen
    netto = gesamt - kredite

    return pd.DataFrame({
        "date": einlagen.index,
        "Einlagen": einlagen.values,
        "Wertpapiere": wertpapiere.values,
        "Versicherungen": versicherungen.values,
        "Kredite": kredite.values,
        "Gesamt_GVB": gesamt.values,
        "Netto_GVB": netto.values,
    }).set_index("date")

# ==============================================================================
# ECB-DATABASE & PRESETS
# ==============================================================================

def _find_ecb_db() -> Optional[Path]:
    """Findet ecb_database.xlsx im Projektumfeld auf Basis von APP_ROOT."""
    base_dir = APP_ROOT
    forecaster_dir = FORECASTER_DIR

    # 1) Bevorzugte, feste Pfade relativ zum Code
    candidates = [
        forecaster_dir / "ecbdata" / "ecb_database.xlsx",  # forecaster/ecbdata/ecb_database.xlsx
        base_dir / "ecbdata" / "ecb_database.xlsx",        # <APP_ROOT>/ecbdata/ecb_database.xlsx
        base_dir / "ecb_database.xlsx",                    # <APP_ROOT>/ecb_database.xlsx
    ]

    for p in candidates:
        if p.exists():
            return p

    # 2) Suche nach generischen ecb_* Dateien im Projekt
    search_roots = [forecaster_dir, base_dir]
    for root in search_roots:
        for pattern in ["ecbdata/ecb_*.xlsx", "ecb_*.xlsx"]:
            matches = list(root.glob(pattern))
            if matches:
                return matches[0]

    # 3) Letzter Fallback: aktuelles Arbeitsverzeichnis (nur zur Sicherheit)
    for pattern in ["ecbdata/ecb_*.xlsx", "ecb_*.xlsx"]:
        matches = list(Path.cwd().glob(pattern))
        if matches:
            return matches[0]

    return None



def _load_ecb_options() -> List[Dict]:
    """Lädt ECB-Indikatoren-Optionen mit Serien-ID."""
    db_path = _find_ecb_db()
    if not db_path:
        _logger.info("[ECB] Keine ecb_database.xlsx gefunden")
        return []

    try:
        df = pd.read_excel(db_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Spalten identifizieren
        series_id_col = next(
            (c for c in ["series_id", "serien_id", "id"] if c in df.columns), None
        )
        code_col = next(
            (c for c in ["series_code", "code", "serieskey"] if c in df.columns), None
        )
        name_col = next(
            (c for c in ["classification", "name", "bezeichnung", "title"] 
             if c in df.columns), None
        )
        qc_col = next(
            (c for c in ["quality_check", "quality"] if c in df.columns), None
        )

        if not code_col:
            _logger.info(f"[ECB] Code-Spalte nicht gefunden. Verfügbar: {df.columns.tolist()}")
            return []

        # Qualitäts-Filter
        before_q = len(df)
        if qc_col:
            qc = df[qc_col].astype(str).str.lower()
            if qc.str.contains("sehr|good|ok|passed").any():
                df = df[qc.str.contains("sehr|good|ok|passed", na=False)]
        _logger.info(f"[ECB] Qualität-Filter: {before_q} -> {len(df)} Zeilen")

        # Optionen erstellen
        options = []
        for _, row in df.iterrows():
            code = str(row[code_col]).strip() if pd.notna(row.get(code_col)) else ""
            name = str(row[name_col]).strip() if (name_col and pd.notna(row.get(name_col))) else ""
            series_id = str(row[series_id_col]).strip() if (series_id_col and pd.notna(row.get(series_id_col))) else ""

            if not (isinstance(code, str) and "." in code and len(code) >= 6):
                continue

            base = name if name and name.lower() != "nan" else code
            sid = series_id if series_id and series_id.lower() != "nan" else ""

            full_label = f"[ECB] {base}" + (f" · {sid}" if sid else "")
            base_display = shorten(str(base), width=30, placeholder="…")
            id_display = shorten(str(sid), width=24, placeholder="…") if sid else ""
            short_label = f"[ECB] {base_display}" + (f" · {id_display}" if id_display else "")

            label_component = html.Span(short_label, title=full_label)
            options.append({"label": label_component, "value": code})

        _logger.info(f"[ECB] {len(options)} Indikatoren geladen")
        return options

    except Exception as e:
        _logger.exception(f"[ECB] Fehler beim Laden: {e}")
        return []


def get_ecb_presets() -> Dict:
    """Basispresets mit sprechenden Labels und ECB-Codes."""
    return {
        "einlagen": {
            "title": "Preset: Einlagen (ECB)",
            "exog": {
                "Euribor 3M (Monat)": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                "Zins-Swaps 10J (DE, Monat)": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
                "Bankzins Einlagen (Monat)": "MIR.M.DE.B.L21.A.R.A.2250.EUR.N",
                "Inflationsrate (DE, Monat)": "ICP.M.DE.N.000000.4.ANR",
                "Arbeitslosenquote (DE, Monat)": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
            },
        },
        "kredite": {
            "title": "Preset: Kredite (ECB)",
            "exog": {
                "Volkseinkommen (DE, Monat)": "BP6.M.N.DE.W1.S1.S1.T.B.IN1._Z._Z._Z.EUR._T._X.N",
                "Wohneigentumspreise (DE, Quartal)": "RESR.Q.DE._T.N._TR.TVAL.10.TB.N.IX",
                "Inflationsrate (DE, Monat)": "ICP.M.DE.N.000000.4.ANR",
                "Arbeitslosenquote (DE, Monat)": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
                "Hauptrefi-Satz (EZB, Monat)": "ILM.M.U2.C.A050100.U2.EUR",
            },
        },
        "versicherungen": {
            "title": "Preset: Versicherungen (ECB)",
            "exog": {
                "Verfügbares Einkommen (Quartal)": "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
                "Arbeitslosenquote (DE, Monat)": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
                "Zins-Swaps 10J (DE, Monat)": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
                "Inflationsrate (DE, Monat)": "ICP.M.DE.N.000000.4.ANR",
            },
        },
        "wertpapiere": {
            "title": "Preset: Wertpapiere (ECB)",
            "exog": {
                "Finanzvermögen HH gesamt (Quartal)": "QSA.Q.N.DE.W0.S1M.S1.N.A.LE.F._Z._Z.XDC._T.S.V.N._T",
                "Verfügbares Einkommen (Quartal)": "QSA.Q.N.DE.W0.S1M.S1._Z.B.B6G._Z._Z._Z.XDC._T.S.V.N._T",
                "Zins-Swaps 10J (DE, Monat)": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
                "Inflationsrate (DE, Monat)": "ICP.M.DE.N.000000.4.ANR",
                "Bruttoersparnis (Quartal)": "GFS.Q.Y.DE.W0.S13.S1._Z.B.B8G._Z._Z._Z.XDC_R_B1GQ._Z.S.V.N._T",
            },
        },
    }


def _load_ecb_series_names() -> Dict[str, str]:
    """Lädt Mapping {code -> Klarname} aus ecb_database.xlsx."""
    try:
        db_path = _find_ecb_db()
        if not db_path or not db_path.exists():
            _logger.info("[Namen-Mapping] ecb_database.xlsx nicht gefunden")
            return {}
        
        df = pd.read_excel(db_path)
        df.columns = [str(c).strip() for c in df.columns]
        
        code_col = None
        name_col = None
        for col in df.columns:
            l = col.lower()
            if code_col is None and (('series' in l and 'key' in l) or 
                                     any(x in l for x in ['code', 'key'])):
                code_col = col
            if name_col is None and any(x in l for x in 
                                       ['classification', 'bezeichnung', 'name', 
                                        'description', 'title']):
                name_col = col
        
        if not code_col or not name_col:
            _logger.info(f"[Namen-Mapping] Spalten nicht gefunden. Verfügbar: {list(df.columns)}")
            return {}
        
        name_map = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if code and name and code != 'nan' and name != 'nan' and '.' in code:
                name_map[code] = name
        
        _logger.info(f"[Namen-Mapping] {len(name_map)} Serien-Namen geladen")
        return name_map
        
    except Exception as e:
        _logger.error(f"[Namen-Mapping] Fehler beim Laden: {e}")
        return {}
    

# ==============================================================================
# PRESET-CACHE VERWALTUNG
# ==============================================================================

def _load_hc_preset_cache() -> dict:
    """Lädt den H&C Preset-Cache."""
    if HCPRESET_CACHE_FILE.exists():
        try:
            return json.loads(HCPRESET_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_hc_preset_cache(cache: dict) -> None:
    """Speichert den H&C Preset-Cache."""
    try:
        HCPRESET_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
    except Exception as e:
        _logger.error(f"[HC-Presets] Write error: {e}")


def _normalize_target_slug(target_value: Optional[str]) -> str:
    """Normalisiert Target-Werte zu Slug-Namen."""
    if not target_value:
        return "gesamt"
    
    raw = str(target_value).strip().lower()
    
    direct_map = {
        "gesamt gvb": "gesamt", "gesamt_gvb": "gesamt", "gesamtgvb": "gesamt", 
        "gesamt": "gesamt",
        "netto gvb": "netto", "netto_gvb": "netto", "nettogvb": "netto", 
        "netto": "netto",
        "einlage": "einlagen", "einlagen": "einlagen",
        "wertpapier": "wertpapiere", "wertpapiere": "wertpapiere", 
        "aktien": "wertpapiere",
        "versicherung": "versicherungen", "versicherungen": "versicherungen",
        "kredit": "kredite", "kredite": "kredite",
    }
    
    if raw in direct_map:
        return direct_map[raw]
    
    s = _slugify(raw)
    if s in ("gesamt", "gesamt-gvb", "gesamtgvb"):
        return "gesamt"
    if s in ("netto", "netto-gvb", "nettogvb", "gesamt-netto", "gesamt_gvb_netto"):
        return "netto"
    if "einlag" in s:
        return "einlagen"
    if "wertpapier" in s or "aktien" in s:
        return "wertpapiere"
    if "versicherung" in s:
        return "versicherungen"
    if "kredit" in s:
        return "kredite"
    
    return s or "gesamt"


def _infer_target_from_slug_or_title(slug: str, meta: dict) -> Optional[str]:
    """Inferiert Target aus Slug oder Titel."""
    slug = (slug or "").strip().lower()
    
    map_by_slug = {
        "einlagen": "Einlagen",
        "kredite": "Kredite",
        "versicherungen": "Versicherungen",
        "wertpapiere": "Wertpapiere",
        "gesamt": "gesamt",
    }
    
    if slug in map_by_slug:
        return map_by_slug[slug]

    title = (meta or {}).get("title", "")
    t = title.strip().lower()
    
    if "einlag" in t: 
        return "Einlagen"
    if "kredit" in t: 
        return "Kredite"
    if "versicher" in t: 
        return "Versicherungen"
    if "wertpapier" in t or "securities" in t: 
        return "Wertpapiere"
    if "gesamt" in t or "total" in t: 
        return "gesamt"
    
    return None


def _hydrate_hc_presets_with_cache(base: dict) -> dict:
    """Hydratisiert Presets mit Cache-Daten."""
    cache = _load_hc_preset_cache()
    out = {}
    
    for slug, meta in (base or {}).items():
        m = dict(meta)
        
        # Target inferieren falls nicht vorhanden
        if not m.get("target"):
            inferred = _infer_target_from_slug_or_title(slug, m)
            m["target"] = inferred
        
        # Cache-Daten einfügen
        cached = cache.get(slug, {})
        if cached.get("exog_snapshot_path"):
            m["exog_snapshot_path"] = cached["exog_snapshot_path"]
        if cached.get("final_dataset_path"):
            m["final_dataset_path"] = cached["final_dataset_path"]
        if cached.get("model_path"):
            m["model_path"] = cached["model_path"]
        
        _logger.debug(
            f"[HC-Presets] hydrate slug={slug} → target={m.get('target')} | "
            f"model={m.get('model_path')} | final={m.get('final_dataset_path')}"
        )
        out[slug] = m
    
    return out


def get_ecb_presets_hydrated() -> dict:
    """Gibt hydratisierte ECB-Presets zurück."""
    try:
        base = get_ecb_presets()
    except Exception:
        base = {}
    return _hydrate_hc_presets_with_cache(base)


def _write_final_dataset(df: pd.DataFrame, slug: str) -> str:
    """Schreibt finales Dataset als Parquet oder CSV."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = SNAPSHOTS_DIR / f"hc_{slug}_final.parquet"
    
    try:
        df.sort_values("date").to_parquet(p)
        Log.scenario_table(f"Final-Dataset gespeichert (parquet): {p}  exists={p.exists()}")
        return str(p)
    except Exception as e:
        Log.scenario_table(f"Parquet nicht möglich ({e}) → Fallback CSV")
        p = SNAPSHOTS_DIR / f"hc_{slug}_final.csv"
        df.sort_values("date").to_csv(p, index=False)
        Log.scenario_table(f"Final-Dataset gespeichert (csv): {p}  exists={p.exists()}")
        return str(p)


def _snapshot_to_store_json(path: str, selected_exogs: list):
    """Lädt Snapshot und konvertiert zu Store-JSON."""
    if not path or not os.path.exists(path):
        return dash.no_update
    
    try:
        plower = path.lower()
        if plower.endswith((".parquet", ".pq")):
            df = pd.read_parquet(path)
        elif plower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return dash.no_update
    
    # Datum normalisieren
    if "date" not in df.columns and "Datum" in df.columns:
        df = df.rename(columns={"Datum": "date"})
    if "date" not in df.columns:
        return dash.no_update
    
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass
    
    # Nur ausgewählte Spalten behalten
    selected_exogs = selected_exogs or []
    keep = ["date"] + [c for c in selected_exogs if c in df.columns]
    keep = [c for c in keep if c in df.columns]
    
    if len(keep) <= 1:
        return dash.no_update
    
    df_out = df[keep].copy().sort_values("date")
    return df_out.to_json(orient="split", date_format="iso")

# ==============================================================================
# ECB-DATEN DOWNLOAD & MERGE
# ==============================================================================
import re 
def _download_exog_codes(ecb_codes: list) -> pd.DataFrame:
    """Lädt die angegebenen ECB-Codes über loader.run_from_config.
    NEU: Wenn der Adapter gerade einen run-spezifischen Pfad übergeben hat (loader/active_run.json),
    verwenden wir genau diesen Pfad, damit Adapter und Loader dasselbe output.xlsx meinen.
    """
    Log.scenario_table(
        f"[Exog-Download] Angefordert für {len(ecb_codes)} Codes: "
        f"{ecb_codes[:6]}{' …' if len(ecb_codes) > 6 else ''}"
    )

    if not ecb_codes:
        Log.scenario_table("[Exog-Download] Keine Codes übergeben – breche ab.")
        return pd.DataFrame(columns=["date"])

    try:
        import re
        import yaml
        import json
        import loader
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from datetime import datetime, timezone

        # App-Root: .../forecaster/ → ein Ordner hoch
        loader_dir = APP_ROOT / "loader"


        # globaler Cache (dürfen wir weiter nutzen)
        cache_dir = loader_dir / "financial_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1) versuchen, Übergabe-Datei des Adapters zu lesen
        handover_path = loader_dir / "active_run.json"
        output_path = None
        if handover_path.exists():
            try:
                raw = json.loads(handover_path.read_text(encoding="utf-8"))
                # grobe Frische: maximal 2 Minuten alt
                created_at = raw.get("created_at")
                is_fresh = True
                if created_at:
                    try:
                        dt_created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        age_s = (datetime.now(timezone.utc) - dt_created).total_seconds()
                        is_fresh = age_s < 120.0
                    except Exception:
                        is_fresh = True

                requested_by_adapter = set(raw.get("requested_exog") or [])
                requested_now = set(ecb_codes or [])

                # wir akzeptieren: Dashboard will genau die gleichen oder eine Teilmenge
                codes_compatible = not requested_by_adapter or requested_now.issubset(requested_by_adapter)

                if is_fresh and codes_compatible and raw.get("expected_output"):
                    output_path = Path(raw["expected_output"])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    Log.scenario_table(
                        f"[Exog-Download] Nutze vom Adapter übergebenen Output-Pfad: {output_path}"
                    )
                else:
                    Log.scenario_table(
                        "[Exog-Download] Übergabe-Datei vorhanden, aber nicht frisch/inkompatibel – "
                        "verwende eigenen Run-Pfad."
                    )
            except Exception as _e_hand:
                Log.scenario_table(f"[Exog-Download] Übergabe-Datei konnte nicht gelesen werden: {_e_hand}")

        # 2) falls kein Übergabe-Pfad: eigenen run-spezifischen Ordner anlegen
        if output_path is None:
            runs_dir = loader_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now().strftime("exog_%Y%m%dT%H%M%S%f")
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "output.xlsx"
            Log.scenario_table(f"[Exog-Download] Eigener Run-Verz.:   {run_dir}")
            Log.scenario_table(f"[Exog-Download] Output-Ziel (neu):   {output_path}")

        # technische Namen für die YAML (ohne Sonderzeichen)
        series_defs = {re.sub(r"[^A-Za-z0-9_]+", "_", c): c for c in ecb_codes}
        Log.scenario_table(f"[Exog-Download] YAML-Series-Defs: {list(series_defs.keys())}")

        # alle Codes hart auf ECB zwingen
        source_overrides = {c.upper(): "ECB" for c in ecb_codes}

        config = {
            "start_date": "2000-01",
            "end_date": _current_quarter_end(),  # aktuelles Quartalsende
            "prefer_cache": True,
            "cache": {
                "cache_dir": str(cache_dir),
                "cache_max_age_days": 60,
            },
            "calendar_index": {
                "freq": "MS",
                "fill": "none",
            },
            "download_timeout_seconds": 30,
            "series_definitions": series_defs,
            "source_overrides": source_overrides,
            "output_path": str(output_path),  # <- jetzt entweder Adapter-Pfad oder eigener
        }

        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(
                yaml.safe_dump(config, allow_unicode=True),
                encoding="utf-8"
            )
            Log.scenario_table(f"[Exog-Download] Rufe loader.run_from_config mit Config: {cfg_path}")
            df_raw = loader.run_from_config(str(cfg_path))
            Log.scenario_table("[Exog-Download] loader.run_from_config beendet.")

    except Exception as e:
        import traceback
        _logger.error("⚠️ Pipeline nicht verfügbar")
        _logger.error("Exception: %s", e)
        _logger.error("Traceback:\n%s", traceback.format_exc())
        Log.scenario_table(f"[Exog-Download] Loader-Fehler: {e}")
        return pd.DataFrame(columns=["date"])

    if df_raw is None or df_raw.empty:
        Log.scenario_table("[Exog-Download] Loader lieferte leeres Ergebnis.")
        return pd.DataFrame(columns=["date"])

    Log.scenario_table(
        f"[Exog-Download] Roh-Daten empfangen: shape={df_raw.shape}, cols={list(df_raw.columns)[:10]}"
    )

    # wie bisher: Datum vereinheitlichen
    if "date" not in df_raw.columns and "Datum" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Datum": "date"})
    if "date" in df_raw.columns:
        try:
            df_raw["date"] = pd.to_datetime(df_raw["date"])
        except Exception:
            pass

    return df_raw


def _merge_exogs_from_sources(
    exog_names: list, 
    exog_store_json
) -> Tuple[pd.DataFrame, list]:
    """Merged Exog-Variablen aus verschiedenen Quellen."""
    Log.scenario_table(f"Merge starte – angeforderte Exogs: {exog_names}")
    
    exog_names = list(exog_names or [])
    if not exog_names:
        return pd.DataFrame(columns=["date"]), []

    # Store laden
    store_df = _safe_load_store(exog_store_json)
    if isinstance(store_df, pd.DataFrame) and "date" in store_df.columns:
        try:
            store_df["date"] = pd.to_datetime(store_df["date"])
        except Exception:
            pass
    else:
        store_df = pd.DataFrame(columns=["date"])

    # ECB vs. lokale Namen trennen
    ecb_codes = [c for c in exog_names if _looks_like_ecb_code(str(c))]
    local_names = [c for c in exog_names if c not in ecb_codes]

    # ECB-Daten laden
    df_exog = pd.DataFrame(columns=["date"])
    if ecb_codes:
        df_ecb = _download_exog_codes(ecb_codes)
        if not df_ecb.empty:
            df_exog = df_ecb.copy()

    # Lokale Spalten hinzufügen
    if local_names and not store_df.empty:
        kept = [c for c in local_names if c in store_df.columns]
        if kept:
            add = store_df[["date"] + kept].copy()
            df_exog = add if df_exog.empty else pd.merge(
                df_exog, add, on="date", how="outer"
            )

    # Sortieren
    if "date" in df_exog.columns and not df_exog.empty:
        df_exog = df_exog.sort_values("date").reset_index(drop=True)

    # Verfügbare Spalten zurückgeben
    have = [c for c in exog_names if c in df_exog.columns]
    return df_exog, have


def _extract_exog_list(preset_obj: dict) -> list:
    """Extrahiert Exog-Liste aus Preset."""
    exogs = preset_obj.get("exog")
    if isinstance(exogs, dict):
        return list(exogs.values())
    if isinstance(exogs, (list, tuple)):
        return list(exogs)
    if isinstance(exogs, str):
        return [exogs]
    return []

# ==============================================================================
# EXPORT-FUNKTIONALITÄT (mit Debug-Fallback, Schritt 10)
# ==============================================================================

def _safe_read_debug_csv(path: Any, *, add_cols: Optional[dict] = None) -> pd.DataFrame:
    """
    Liest eine Debug-CSV robust ein.
    - existiert Datei nicht → leerer DF
    - ist Datei leer/beschädigt → leerer DF
    - add_cols: optionale Zusatzspalten, die dran gehängt werden (z.B. um im Export anzuzeigen,
      welche Datei fehlte)
    """
    if not path:
        return pd.DataFrame()
    try:
        p = Path(path)
    except Exception:
        return pd.DataFrame()

    if not p.exists() or not p.is_file():
        _logger.info(f"[Export/Debug] Debug-Datei nicht gefunden: {p}")
        df = pd.DataFrame()
    else:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            _logger.warning(f"[Export/Debug] Konnte Debug-Datei nicht lesen ({p}): {e}")
            df = pd.DataFrame()

    if add_cols:
        for k, v in add_cols.items():
            df[k] = v
    return df


def _make_export_bytes(
    config_df: Optional[pd.DataFrame],
    main_df: Optional[pd.DataFrame],
    current_view_df: Optional[pd.DataFrame],
    exog_df: Optional[pd.DataFrame],
    *,
    gvb_raw_df: Optional[pd.DataFrame] = None,
    pipeline_prepared_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    future_design_df: Optional[pd.DataFrame] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    model_config_df: Optional[pd.DataFrame] = None,
    debug_train_quarterly_df: Optional[pd.DataFrame] = None,
    debug_train_design_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Erstellt Excel-Export mit allen relevanten Daten.
    Schritt 10: debug_* DataFrames sind optional – wenn leer, wird nur ein kleines Sheet mit Hinweis angelegt.
    """
    cfg_df = config_df if isinstance(config_df, pd.DataFrame) else pd.DataFrame()
    main_out = main_df if isinstance(main_df, pd.DataFrame) else pd.DataFrame()
    view_out = current_view_df if isinstance(current_view_df, pd.DataFrame) else pd.DataFrame()
    exog_out = exog_df if isinstance(exog_df, pd.DataFrame) else pd.DataFrame()

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        # README
        readme = pd.DataFrame({
            "Info": [
                "Export aus Prognose-Suite",
                "Enthält Rohdaten (GVB/EZB), aufbereitete Pipeline-Daten, Prognose und Metadata.",
                "Gesamt_GVB = Einlagen + Wertpapiere + Versicherungen (ohne Kredite)",
                "Netto_GVB  = Gesamt_GVB − Kredite",
                "Hinweis: Einige Debug-Sheets können leer sein, wenn beim Dashboard-Lauf keine Debug-Dateien vorhanden waren.",
            ]
        })
        readme.to_excel(writer, index=False, sheet_name="README")

        # bisherige Sheets
        cfg_df.to_excel(writer, index=False, sheet_name="CONFIG")
        main_out.to_excel(writer, sheet_name="GVB_MAIN")
        view_out.to_excel(writer, sheet_name="CURRENT_VIEW")
        exog_out.to_excel(writer, index=False, sheet_name="EXOG")

        # GVB-Rohdaten
        if gvb_raw_df is not None and not gvb_raw_df.empty:
            gvb_raw_df.to_excel(writer, index=False, sheet_name="GVB_RAW")

        # vorbereitete Pipeline-Daten
        if pipeline_prepared_df is not None and not pipeline_prepared_df.empty:
            pipeline_prepared_df.to_excel(writer, index=False, sheet_name="PIPELINE_PREP")

        # Forecast-Tabelle
        if forecast_df is not None and not forecast_df.empty:
            forecast_df.to_excel(writer, index=False, sheet_name="FORECAST_TS")

        # Future-Design
        if future_design_df is not None and not future_design_df.empty:
            future_design_df.to_excel(writer, index=False, sheet_name="FORECAST_DESIGN")

        # Metadata flach
        if metadata_df is not None and not metadata_df.empty:
            metadata_df.to_excel(writer, index=False, sheet_name="METADATA")

        # Modell-Config
        if model_config_df is not None and not model_config_df.empty:
            model_config_df.to_excel(writer, index=False, sheet_name="MODEL_CONFIG")

        # Schritt 10: Debug-Sheets (immer anlegen, auch wenn leer)
        if debug_train_quarterly_df is None:
            debug_train_quarterly_df = pd.DataFrame(columns=["info"])
            debug_train_quarterly_df.loc[0, "info"] = "Keine train_quarterly_debug.csv gefunden"
        debug_train_quarterly_df.to_excel(writer, index=False, sheet_name="DEBUG_QUARTERLY")

        if debug_train_design_df is None:
            debug_train_design_df = pd.DataFrame(columns=["info"])
            debug_train_design_df.loc[0, "info"] = "Keine train_design_debug.csv gefunden"
        debug_train_design_df.to_excel(writer, index=False, sheet_name="DEBUG_DESIGN")

        # leichte Formatierung
        for sheet_name in writer.sheets.keys():
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 0, 18)
            ws.set_column(1, 50, 14)

    return bio.getvalue()

# ==============================================================================
# NEU: BACKTEST-VISUALISIERUNG (Option 1: Overlay)
# ==============================================================================

def _add_backtest_to_chart(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_backtest: bool = True
) -> go.Figure:
    if not show_backtest or backtest_results is None or backtest_results.empty:
        return fig
    
    try:
        required_cols = ['date', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        
        if missing_cols:
            _logger.warning(f"[Backtest-Overlay] Fehlende Spalten: {missing_cols}")
            return fig
        
        bt = backtest_results.copy()
        if not pd.api.types.is_datetime64_any_dtype(bt['date']):
            bt['date'] = pd.to_datetime(bt['date'], errors='coerce')
        
        bt = bt.dropna(subset=['date', 'predicted'])
        
        if bt.empty:
            _logger.warning("[Backtest-Overlay] Keine gültigen Daten nach Bereinigung")
            return fig
        
        fig.add_trace(go.Scatter(
            x=bt['date'],
            y=bt['predicted'],
            mode='lines',
            name='Historische Vorhersagen',
            line=dict(
                color='rgba(255, 99, 71, 0.6)',
                width=2,
                dash='dash'
            ),
            hovertemplate='<b>Vorhersage</b>: %{y:.2f}<br><b>Datum</b>: %{x}<extra></extra>',
            visible=True
        ))
        
        _logger.info(f"[Backtest-Overlay] ✓ {len(bt)} historische Vorhersagen hinzugefügt")
    except Exception as e:
        _logger.warning(f"[Backtest-Overlay] Konnte Overlay nicht hinzufügen: {e}")
    
    return fig


def _add_backtest_error_band(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_errors: bool = True
) -> go.Figure:
    if not show_errors or backtest_results is None or backtest_results.empty:
            return fig
    try:
        required_cols = ['date', 'actual', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        
        if missing_cols:
            _logger.warning(f"[Backtest-ErrorBand] Fehlende Spalten: {missing_cols}")
            return fig
        
        df = backtest_results.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date', 'actual', 'predicted'])
        if df.empty:
            _logger.warning("[Backtest-ErrorBand] Keine gültigen Daten")
            return fig
        
        df['error'] = df['actual'] - df['predicted']
        df['upper_error'] = df['actual']
        df['lower_error'] = df['predicted']
        
        pos_mask = df['error'] > 0
        if pos_mask.any():
            pos_df = df[pos_mask].copy().sort_values('date')
            x_band = list(pos_df['date']) + list(pos_df['date'][::-1])
            y_band = list(pos_df['upper_error']) + list(pos_df['lower_error'][::-1])
            fig.add_trace(go.Scatter(
                x=x_band,
                y=y_band,
                fill='toself',
                fillcolor='rgba(144, 238, 144, 0.3)',
                line=dict(width=0),
                name='Unterschätzung',
                hoverinfo='skip',
                showlegend=True
            ))
        
        neg_mask = df['error'] < 0
        if neg_mask.any():
            neg_df = df[neg_mask].copy().sort_values('date')
            x_band = list(neg_df['date']) + list(neg_df['date'][::-1])
            y_band = list(neg_df['actual']) + list(neg_df['predicted'][::-1])
            fig.add_trace(go.Scatter(
                x=x_band,
                y=y_band,
                fill='toself',
                fillcolor='rgba(255, 99, 71, 0.3)',
                line=dict(width=0),
                name='Überschätzung',
                hoverinfo='skip',
                showlegend=True
            ))
        
        _logger.info("[Backtest-ErrorBand] ✓ Fehler-Bänder hinzugefügt")
    except Exception as e:
        _logger.warning(f"[Backtest-ErrorBand] Konnte Fehler-Bänder nicht hinzufügen: {e}")
    
    return fig


def _add_backtest_markers(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_markers: bool = True
) -> go.Figure:
    if not show_markers or backtest_results is None or backtest_results.empty:
        return fig
    
    try:
        required_cols = ['date', 'actual', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        if missing_cols:
            _logger.warning(f"[Backtest-Markers] Fehlende Spalten: {missing_cols}")
            return fig
        
        df = backtest_results.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date', 'actual', 'predicted'])
        if df.empty:
            _logger.warning("[Backtest-Markers] Keine gültigen Daten")
            return fig
        
        df['error'] = abs(df['actual'] - df['predicted'])
        if len(df) > 4:
            threshold = df['error'].quantile(0.75)
        else:
            threshold = df['error'].mean()
        
        large_errors = df[df['error'] > threshold]
        if not large_errors.empty:
            fig.add_trace(go.Scatter(
                x=large_errors['date'],
                y=large_errors['actual'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                name='Große Abweichungen',
                hovertemplate='<b>Datum</b>: %{x}<br><b>Tatsächlich</b>: %{y:.2f}<br><b>Fehler</b>: %{customdata:.2f}<extra></extra>',
                customdata=large_errors['error']
            ))
            _logger.debug(f"[Backtest-Markers] {len(large_errors)} große Fehler markiert")
    except Exception as e:
        _logger.warning(f"[Backtest-Markers] Konnte Marker nicht hinzufügen: {e}")
    
    return fig


def create_backtest_controls():
    if dbc is not None:
        return html.Div([
            dbc.Switch(
                id="show-backtest-switch",
                label="Historische Vorhersagen (gestrichelt)",
                value=False,
                className="mb-2"
            ),
            html.Small(
                "Darstellung ist fest auf Overlay (gestrichelte Linie) eingestellt.",
                className="text-muted"
            )
        ])
    else:
        return html.Div([
            html.Label([
                dcc.Checklist(
                    id="show-backtest-switch",
                    options=[{"label": " Historische Vorhersagen (gestrichelt)", "value": True}],
                    value=[],
                    inputStyle={"marginRight": "6px"}
                )
            ], style={"display": "block", "marginBottom": "8px"}),
            html.Small(
                "Darstellung ist fest auf Overlay (gestrichelte Linie) eingestellt.",
                style={"color": "#6c757d"}
            )
        ])


# ==============================================================================
# FEATURE IMPORTANCE VISUALISIERUNG
# ==============================================================================

def _split_code_attrs(code: str) -> Tuple[str, Optional[int], str, list]:
    s = str(code).strip()
    parts = s.split('__')
    base_code = parts[0]
    remainder = '__'.join(parts[1:]) if len(parts) > 1 else ''

    lag_n, lag_unit = None, ""
    patterns = [
        r'__lag[-_\s]*(-?\d+)\s*([A-Za-z]?)$',
        r'[_:\.\-]lag\s*(-?\d+)\s*([A-Za-z]?)$',
        r'\[t-?\s*(\d+)\s*([A-Za-z]?)\]$',
        r'_L(\d+)$'
    ]
    
    for pattern in patterns:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            if pattern == r'_L(\d+)$':
                lag_n = int(m.group(1))
                lag_unit = ""
            else:
                raw_n = int(m.group(1))
                lag_n = abs(raw_n)
                lag_unit = m.group(2).upper() if len(m.groups()) > 1 and m.group(2) else ""
            break

    transforms = []
    if remainder:
        for token in filter(None, remainder.split('__')):
            if not re.match(r'^lag[-_\s]*\d+[A-Za-z]?$', token, flags=re.IGNORECASE):
                transforms.append(token)

    return base_code, lag_n, lag_unit, transforms


def _fmt_indicator_label(code: str, name_map: dict, width: int = 38) -> Tuple[str, str]:
    base_code, lag_n, lag_unit, transforms = _split_code_attrs(code)
    base_name = name_map.get(base_code, base_code) or base_code
    
    lag_txt = f" (t-{lag_n}{lag_unit})" if lag_n is not None else ""
    trans_txt = f" • {' • '.join(transforms)}" if transforms else ""
    
    display_base = f"{base_name}{lag_txt}"
    full = (f"{base_name}{lag_txt} · {base_code}{trans_txt}" 
            if base_name != base_code 
            else f"{base_code}{lag_txt}{trans_txt}")
    short = shorten(str(display_base), width=width, placeholder="…")
    
    return short, full


def _create_feature_importance(metadata: dict):
    features = (metadata or {}).get('model_complexity', {}).get('top_features', {})
    if not features:
        return html.Div(
            "Keine Features verfügbar", 
            className="text-muted text-center p-3"
        )
    
    name_map = _load_ecb_series_names()
    rows = []
    
    for k, v in features.items():
        code = str(k)
        short, full = _fmt_indicator_label(code, name_map, width=38)
        rows.append({
            'FeatureCode': code, 
            'Feature': short, 
            'Full': full, 
            'Importance': float(v)
        })
    
    df = pd.DataFrame(rows).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker_color=INFO_COLOR,
        customdata=df[['Full']].values,
        hovertemplate='<b>%{customdata[0]}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        template='plotly_white',
        height=max(350, min(len(df) * 25, 400)),
        margin=dict(l=190, r=10, t=10, b=40),
        showlegend=False
    )
    
    return dcc.Graph(
        figure=fig, 
        config={'displayModeBar': False}, 
        style={"height": "100%"}
    )


def create_feature_importance_icicle(
    features: Optional[dict], 
    top_n: int = 15
) -> go.Figure:
    if not isinstance(features, dict) or len(features) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            annotations=[dict(
                text="Keine Feature Importances verfügbar",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )],
            margin=dict(l=5, r=5, t=5, b=5),
            height=400,
        )
        return fig

    items = sorted(features.items(), key=lambda kv: float(kv[1]), reverse=True)[:max(1, int(top_n))]
    codes_children = [str(k) for k, _ in items]
    values_children = [max(0.0, float(v)) for _, v in items]
    total_value = float(sum(values_children)) if values_children else 0.0

    name_map = _load_ecb_series_names()
    labels_children = []
    hovers_children = []
    
    for code in codes_children:
        short, full = _fmt_indicator_label(code, name_map, width=28)
        labels_children.append(short)
        hovers_children.append(full)

    root_label = f"Features (Top {len(labels_children)})"
    labels = [root_label] + labels_children
    parents = [""] + [root_label] * len(labels_children)
    values = [total_value] + values_children

    palette = list(pseq.Blues)
    n_col = len(palette)
    
    if len(values_children) > 0:
        v_min, v_max = min(values_children), max(values_children)
        span = (v_max - v_min) or 1.0
        
        def color_for(v):
            ratio = (v - v_min) / span
            idx = int(round(ratio * (n_col - 1)))
            return palette[idx]
        
        colors_children = [color_for(v) for v in values_children]
    else:
        colors_children = [palette[0]] * len(values_children)

    root_color = "#0B1F3B"
    colors = [root_color] + colors_children

    fig = go.Figure(go.Icicle(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="white", width=1)),
        tiling=dict(orientation="v"),
        root_color=root_color,
        customdata=["Root"] + hovers_children,
        hovertemplate="%{customdata}<br>Importance: %{value:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    
    return fig


# ==============================================================================
# CHART-ERSTELLUNG (Pipeline) – Version mit Backtest, Legende oben
# ==============================================================================

def _empty_forecast_fig(message: str = 'Klicken Sie auf "Prognose erstellen"') -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Keine Prognose erstellt",
        template='plotly_white',
        height=500,
        xaxis_title="Datum",
        yaxis_title="Bestand (Mrd EUR)",
        annotations=[{
            'text': message,
            'xref': 'paper', 
            'yref': 'paper',
            'x': 0.5, 
            'y': 0.5,
            'showarrow': False
        }]
    )
    return fig


def _error_forecast_response(
    err: Exception, 
    header: str = "Fehler bei der Prognose"
) -> Tuple[go.Figure, html.Div, dict]:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=header,
        annotations=[dict(
            text="Es ist ein Fehler aufgetreten. Details siehe Meldung daneben.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )],
        height=420,
        margin=dict(l=10, r=10, t=60, b=40),
    )

    msg_lines = [
        html.Strong(type(err).__name__),
        html.Br(),
        html.Code(str(err)),
    ]
    
    if dbc is not None:
        error_html = dbc.Alert(
            msg_lines,
            color="danger",
            dismissable=True,
            fade=True,
            className="mt-2"
        )
    else:
        error_html = html.Div(
            msg_lines,
            style={
                "background": "#f8d7da",
                "border": "1px solid #f5c2c7",
                "color": "#842029",
                "borderRadius": "6px",
                "padding": "12px",
                "marginTop": "8px",
            }
        )
    
    meta = {"error": str(err), "type": type(err).__name__}
    return fig, error_html, meta


def _create_pipeline_chart(
    forecast_df: pd.DataFrame, 
    metadata: dict, 
    gvb_data: pd.DataFrame, 
    target: str, 
    is_fluss_mode: bool,
    horizon_quarters: int = 6,
    show_backtest: bool = False,
    backtest_mode: str = 'overlay'
) -> go.Figure:
    import numpy as np

    data_type = 'fluss' if is_fluss_mode else 'bestand'
    value_col = 'fluss' if is_fluss_mode else 'bestand'
    y_title = 'Fluss (Mrd EUR)' if is_fluss_mode else 'Bestand (Mrd EUR)'

    hist = gvb_data.copy() if gvb_data is not None else pd.DataFrame()
    if not hist.empty:
        if 'datatype' in hist.columns:
            hist = hist[hist['datatype'] == data_type]
        if target != 'gesamt' and 'ebene1' in hist.columns:
            hist = hist[hist['ebene1'] == target]

        if 'date' in hist.columns:
            hist = hist[hist['date'].notna()]
            hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
        else:
            hist = pd.DataFrame(columns=['date', value_col])

        if value_col in hist.columns and not hist.empty:
            hist = (
                hist.groupby('date', as_index=False)[value_col]
                .sum()
                .sort_values('date')
                .reset_index(drop=True)
            )
        else:
            hist = pd.DataFrame({
                'date': pd.Series(dtype='datetime64[ns]'),
                value_col: pd.Series(dtype='float')
            })

    fc = forecast_df.copy() if isinstance(forecast_df, pd.DataFrame) else pd.DataFrame()
    fc_date = None
    if not fc.empty:
        for col in ['date', 'Date', 'ds']:
            if col in fc.columns:
                fc_date = col
                fc[col] = pd.to_datetime(fc[col], errors='coerce')
                break
        
        if fc_date is None and 'Quarter' in fc.columns:
            try:
                q = pd.PeriodIndex(fc['Quarter'].astype(str), freq='Q')
                fc['date'] = q.to_timestamp(how='end')
                fc_date = 'date'
            except Exception:
                pass

    y_candidates = ['yhat', 'Forecast', 'forecast', 'y_pred', 'yhat_mean', 'pred', 'value']
    fc_val = next((c for c in y_candidates if c in fc.columns), None)

    confidence_levels = [
        {
            'level': 95,
            'lower_candidates': ['yhat_lower_95', 'lower_95', 'lo95', 'y_lower_95'],
            'upper_candidates': ['yhat_upper_95', 'upper_95', 'hi95', 'y_upper_95'],
            'color': 'rgba(0, 123, 255, 0.15)',
            'label': '95% Konfidenzintervall'
        },
        {
            'level': 80,
            'lower_candidates': ['yhat_lower_80', 'lower_80', 'lo80', 'y_lower_80'],
            'upper_candidates': ['yhat_upper_80', 'upper_80', 'hi80', 'y_upper_80'],
            'color': 'rgba(0, 123, 255, 0.25)',
            'label': '80% Konfidenzintervall'
        }
    ]
    std_candidates = ['yhat_std', 'se', 'stderr', 'std']

    if fc_date and fc_val:
        fc = fc.dropna(subset=[fc_date, fc_val]).sort_values(fc_date).copy()
        fc[fc_val] = pd.to_numeric(fc[fc_val], errors='coerce')
        
        for level_config in confidence_levels:
            for c in level_config['lower_candidates'] + level_config['upper_candidates']:
                if c in fc.columns:
                    fc[c] = pd.to_numeric(fc[c], errors='coerce')
        
        for c in std_candidates:
            if c in fc.columns:
                fc[c] = pd.to_numeric(fc[c], errors='coerce')
    else:
        fc = pd.DataFrame({
            'date': pd.Series(dtype='datetime64[ns]'),
            'Forecast': pd.Series(dtype='float')
        })
        fc_date, fc_val = 'date', 'Forecast'

    x_range = None
    if not hist.empty:
        hist_start = hist['date'].min()
        hist_end = hist['date'].max()
        forecast_months = horizon_quarters * 3
        try:
            expected_end = hist_end + pd.DateOffset(months=forecast_months)
            total_span = (expected_end - hist_start).days
            buffer_days = int(total_span * 0.05)
            x_range = [
                hist_start - pd.Timedelta(days=buffer_days),
                expected_end + pd.Timedelta(days=buffer_days)
            ]
        except Exception as e:
            _logger.warning(f"[Chart] Konnte X-Achsen-Bereich nicht berechnen: {e}")
            x_range = None

    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist['date'],
            y=hist[value_col],
            mode='lines',
            name=f'Historie ({"Fluss" if is_fluss_mode else "Bestand"})',
            line=dict(color='#1f77b4', width=2),
            connectgaps=True,
            line_shape='linear'
        ))

    if not fc.empty and fc_val in fc.columns:
        def _first_present(cols):
            return next((c for c in cols if c in fc.columns), None)

        for level_config in confidence_levels:
            lower_col = _first_present(level_config['lower_candidates'])
            upper_col = _first_present(level_config['upper_candidates'])
            lower = upper = None
            
            if lower_col and upper_col:
                lower = fc[lower_col]
                upper = fc[upper_col]
                ci_label = level_config['label']
            elif level_config['level'] in [80, 95]:
                std_col = next((c for c in std_candidates if c in fc.columns), None)
                if std_col is not None and fc_val in fc.columns:
                    z_scores = {80: 1.28, 95: 1.96}
                    z = z_scores.get(level_config['level'], 1.96)
                    lower = fc[fc_val] - z * fc[std_col]
                    upper = fc[fc_val] + z * fc[std_col]
                    ci_label = f"{level_config['level']}% Konfidenzintervall (berechnet)"

            if lower is not None and upper is not None:
                mask = (~lower.isna()) & (~upper.isna()) & (~fc[fc_date].isna())
                if mask.any():
                    x_band = pd.concat([fc.loc[mask, fc_date], fc.loc[mask, fc_date][::-1]])
                    y_band = pd.concat([upper[mask], lower[mask][::-1]])
                    fig.add_trace(go.Scatter(
                        x=x_band,
                        y=y_band,
                        fill='toself',
                        fillcolor=level_config['color'],
                        line=dict(width=0),
                        hoverinfo='skip',
                        name=ci_label,
                        opacity=1.0,
                        showlegend=True
                    ))

    if not fc.empty and fc_val in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc[fc_date],
            y=fc[fc_val],
            mode='lines',
            name='Forecast',
            line=dict(dash='dot', color='#ff7f0e', width=2),
            connectgaps=True,
            line_shape='linear'
        ))

    if show_backtest and isinstance(metadata, dict):
        backtest_data = metadata.get('backtest_results')
        if backtest_data is not None and not isinstance(backtest_data, pd.DataFrame):
            try:
                backtest_data = pd.DataFrame(backtest_data)
            except Exception as e:
                _logger.warning(f"[Chart] Konnte Backtest-Daten nicht in DataFrame konvertieren: {e}")
                backtest_data = None
        
        if isinstance(backtest_data, pd.DataFrame) and not backtest_data.empty:
            bt = backtest_data.copy()
            if 'date' in bt.columns:
                bt['date'] = pd.to_datetime(bt['date'], errors='coerce')
            else:
                _logger.info("[Chart] Backtest ohne 'date' Spalte – überspringe Overlay")
                bt = pd.DataFrame()

            if 'predicted' in bt.columns:
                bt['predicted'] = pd.to_numeric(bt['predicted'], errors='coerce')

            bt = bt[['date', 'predicted']].dropna(subset=['date', 'predicted']).sort_values('date')

            if not bt.empty and bt.duplicated('date').any():
                bt = bt.groupby('date', as_index=False)['predicted'].sum()

            if not bt.empty:
                fig.add_trace(go.Scatter(
                    x=bt['date'],
                    y=bt['predicted'],
                    mode='lines',
                    name='Backtest (Vorhersage)',
                    line=dict(width=2, dash='dash', color='rgba(255, 99, 71, 0.7)'),
                    connectgaps=True,
                    line_shape='linear'
                ))

    title_target = target if target != 'gesamt' else 'Gesamt'
    legend_cfg = dict(
        title='',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        traceorder='normal',
        bgcolor='rgba(255,255,255,0.6)'
    )

    layout_config = {
        'template': 'plotly_white',
        'xaxis_title': 'Datum',
        'yaxis_title': y_title,
        'legend': legend_cfg,
        'hovermode': 'x unified',
        'height': 500,
        'margin': dict(l=50, r=20, t=80, b=40)
    }
    if x_range is not None:
        layout_config['xaxis'] = {'range': x_range, 'fixedrange': False}
    fig.update_layout(**layout_config)

    try:
        y_series = []
        if not hist.empty:
            y_series.append(hist[value_col])
        if not fc.empty and fc_val in fc.columns:
            y_series.append(fc[fc_val])

        if y_series:
            y_concat = pd.concat(y_series)
            if not y_concat.empty and np.isfinite(y_concat).all():
                y_min = float(y_concat.min())
                y_max = float(y_concat.max())
                padding = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
                fig.update_yaxes(range=[y_min - padding, y_max + padding])

        if not hist.empty:
            cutoff = hist['date'].max()
            yaxis_range = fig.layout.yaxis.range
            if yaxis_range:
                y0, y1 = yaxis_range
            else:
                y0, y1 = (0, 1)
            fig.add_trace(go.Scatter(
                x=[cutoff, cutoff],
                y=[y0, y1],
                mode='lines',
                line=dict(width=1, dash='dash', color='gray'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip',
                name=''
            ))
    except Exception as e:
        _logger.warning(f"[Chart] Konnte Y-Achse/Cutoff nicht finalisieren: {e}")

    try:
        # X-Achsen-Formatierung (Quartale)
        if x_range:
             format_axis_quarters(fig, x_range)
        elif not hist.empty:
             format_axis_quarters(fig, hist['date'])
    except Exception:
        pass

    fig.update_traces(line_shape='linear', selector=dict(type='scatter'))
    return fig


# ==============================================================================
# CALLBACKS - DROPDOWN & UI
# ==============================================================================

@app.callback(
    Output("external-exog-dropdown", "options"),
    Input("url", "pathname"),
    Input("exog-data-store", "data"),
    State("external-exog-dropdown", "value"),
    prevent_initial_call=False
)
def build_exog_options(_, exog_json, current_selection):
    ecb_opts = _load_ecb_options() or []
    ecb_by_val = {opt.get("value"): opt for opt in ecb_opts if opt.get("value")}

    options_map = {}
    
    exog_df = _safe_load_store(exog_json)
    if isinstance(exog_df, pd.DataFrame) and not exog_df.empty:
        local_cols = [c for c in exog_df.columns if str(c).lower() != "date"]
        for col in sorted(local_cols):
            v = col
            if v not in ecb_by_val:
                options_map[v] = {"label": f"[Lokal] {str(col)[:50]}", "value": v}

    for v, opt in ecb_by_val.items():
        options_map[v] = opt

    if current_selection:
        existing_values = set(options_map.keys())
        wanted = current_selection if isinstance(current_selection, list) else [current_selection]
        for val in wanted:
            if val not in existing_values:
                options_map[val] = {"label": f"[Ausgewählt] {str(val)[:50]}", "value": val}

    def sort_key(opt):
        label = opt.get("label", "")
        if str(label).startswith("[ECB]"):
            pri = 0
        elif str(label).startswith("[Lokal]"):
            pri = 1
        elif str(label).startswith("[Ausgewählt]"):
            pri = 2
        else:
            pri = 3
        return (pri, str(label).lower())

    final_opts = sorted(options_map.values(), key=sort_key)
    return final_opts


@app.callback(
    [
        Output("exog-add-toast", "is_open"),
        Output("exog-add-toast", "children"),
        Output("exog-add-toast", "header"),
        Output("exog-add-toast", "icon"),
        Output("manual-series-input", "value"),
        Output("exog-data-store", "data", allow_duplicate=True),
        Output("external-exog-dropdown", "value", allow_duplicate=True),  # ← NEU HINZUGEFÜGT!
    ],
    [Input("add-manual-series-btn", "n_clicks")],
    [
        State("external-exog-dropdown", "value"),
        State("manual-series-input", "value"),
        State("exog-data-store", "data"),
    ],
    prevent_initial_call=True,
)
def notify_exog_add(n_clicks, selected_values, manual_value, exog_store_json):
    if not n_clicks:
        return (
            True,
            "Bitte wählen Sie mindestens eine Datenreihe oder geben Sie eine Serien ID ein.",
            "Keine Auswahl",
            "danger",
            no_update,
            no_update,
            no_update,  # ← dropdown.value
        )

    # 1) Auswahl einsammeln
    selections = []
    if selected_values:
        if isinstance(selected_values, list):
            selections.extend([str(v).strip() for v in selected_values if str(v).strip()])
        else:
            selections.append(str(selected_values).strip())

    if manual_value:
        parts = re.split(r"[;,]", str(manual_value))
        parts = [p.strip() for p in parts if p and p.strip()]
        selections.extend(parts)

    selections = [s for s in selections if s]
    unique_selections = []
    seen = set()
    for s in selections:
        if s not in seen:
            unique_selections.append(s)
            seen.add(s)

    if not unique_selections:
        return (
            True,
            "Bitte wählen Sie mindestens eine Datenreihe oder geben Sie eine Serien ID ein.",
            "Keine Auswahl",
            "danger",
            no_update,
            no_update,
            no_update,  # ← dropdown.value
        )

    # 2) Bisherige Spalten im Store ermitteln
    def _extract_existing_columns(payload):
        if payload is None:
            return set()
        try:
            if isinstance(payload, str):
                try:
                    df = pd.read_json(payload, orient="split")
                    return set(df.columns.astype(str).tolist())
                except Exception:
                    pass
                obj = json.loads(payload)
            else:
                obj = payload

            if isinstance(obj, dict) and "columns" in obj:
                return set(str(c) for c in (obj.get("columns") or []))

            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                keys = set()
                for rec in obj:
                    keys.update(list(rec.keys()))
                return set(str(k) for k in keys)
        except Exception:
            pass
        return set()

    existing_cols = _extract_existing_columns(exog_store_json)
    existing_exogs = [c for c in existing_cols if c != "date"]

    to_add = [v for v in unique_selections if v not in existing_cols]
    duplicates = [v for v in unique_selections if v in existing_cols]

    if not to_add and duplicates:
        msg = html.Div([
            html.Div("Alle ausgewählten Reihen sind bereits vorhanden."),
            html.Ul([html.Li(v) for v in duplicates], className="mb-0"),
        ])
        return True, msg, "Bereits vorhanden", "warning", no_update, no_update, no_update  # ← dropdown.value

    # 3) Neu + bisherige Exogs zusammenführen und wirklich herunterladen
    new_store_json = exog_store_json
    downloaded_cols = []  # ← NEU: Track welche Spalten tatsächlich im DataFrame sind
    if to_add:
        all_exog_names = sorted(set(existing_exogs + to_add))
        merged_df, have = _merge_exogs_from_sources(all_exog_names, exog_store_json)

        if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
            new_store_json = merged_df.to_json(orient="split", date_format="iso")
            # Track welche Spalten wirklich im DataFrame sind (mit Unterstrichen!)
            downloaded_cols = [c for c in merged_df.columns if c != "date"]

    # 4) Dropdown-Selection aktualisieren mit den TATSÄCHLICHEN Spaltennamen
    current_dropdown_value = selected_values or []
    if not isinstance(current_dropdown_value, list):
        current_dropdown_value = [current_dropdown_value] if current_dropdown_value else []
    
    # Füge die heruntergeladenen Spalten zur Auswahl hinzu
    updated_dropdown_value = list(set(current_dropdown_value + downloaded_cols))

    # 5) Toast-Messages
    if to_add and not duplicates:
        msg = html.Div([
            html.Div("Die folgenden Datenreihen werden hinzugefügt:"),
            html.Ul([html.Li(v) for v in to_add], className="mb-0"),
        ])
        return True, msg, "Exogene heruntergeladen", "success", "", new_store_json, updated_dropdown_value  # ← NEU!

    if to_add and duplicates:
        msg = html.Div([
            html.Div("Einige Reihen sind bereits vorhanden und werden übersprungen:"),
            html.Ul([html.Li(v) for v in duplicates]),
            html.Hr(className="my-2"),
            html.Div("Neu hinzugefügt:"),
            html.Ul([html.Li(v) for v in to_add], className="mb-0"),
        ])
        return True, msg, "Teilweise hinzugefügt", "warning", "", new_store_json, updated_dropdown_value  # ← NEU!

    msg = html.Div([
        html.Div("Alle ausgewählten Reihen sind bereits vorhanden."),
        html.Ul([html.Li(v) for v in duplicates], className="mb-0"),
    ])
    return True, msg, "Bereits vorhanden", "warning", no_update, no_update, no_update  # ← dropdown.value




@app.callback(
    Output("forecast-horizon-store", "data"),
    Output({"type": "horizon-btn", "value": ALL}, "active"),
    Input({"type": "horizon-btn", "value": ALL}, "n_clicks"),
    State({"type": "horizon-btn", "value": ALL}, "id"),
    prevent_initial_call=True
)
def update_horizon_selection(n_clicks_list, btn_ids):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update, [dash.no_update] * len(btn_ids)

    ctx = dash.callback_context
    if not ctx.triggered:
        default_val = 6
        active_states = [b['value'] == default_val for b in btn_ids]
        return default_val, active_states

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_value = json.loads(triggered_id)['value']
    active_states = [b['value'] == selected_value for b in btn_ids]
    return selected_value, active_states


@app.callback(
    Output("horizon-display", "children"),
    Input("forecast-horizon-store", "data")
)
def update_horizon_display(quarters):
    if not quarters:
        return "6 Quartale (1.5 Jahre)"
    
    years = quarters / 4
    months = quarters * 3
    
    if quarters == 1:
        return f"{quarters} Quartal ({months} Monate)"
    elif quarters % 4 == 0:
        return f"{quarters} Quartale ({int(years)} {'Jahr' if years == 1 else 'Jahre'})"
    else:
        return f"{quarters} Quartale ({years:.1f} Jahre / {months} Monate)"


# ==============================================================================
# "Neu"-Button -> Cache (runs) leeren
# ==============================================================================

_BASE_DIR = APP_ROOT  # ein Ordner hoch von /forecaster
RUNS_DIR = Path(os.getenv("FORECASTER_RUNS_DIR", _BASE_DIR / "loader" / "runs")).resolve()


def clear_runs_directory() -> int:
    """
    Löscht alle Dateien und Unterordner im RUNS_DIR.
    Gibt die Anzahl der gelöschten Einträge zurück.
    """
    if not RUNS_DIR.exists():
        _logger.info(f"[Cache-Clear] RUNS_DIR existiert nicht: {RUNS_DIR}")
        return 0

    deleted = 0
    for item in RUNS_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted += 1
        except Exception as exc:
            _logger.warning(f"[Cache-Clear] Konnte {item} nicht löschen: {exc}")

    _logger.info(f"[Cache-Clear] {deleted} Eintrag(e) aus {RUNS_DIR} entfernt.")
    return deleted


@app.callback(
    Output("cache-clear-toast", "is_open"),
    Output("cache-clear-toast", "children"),
    Input("retrain-model-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_cache_clear(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    deleted = clear_runs_directory()

    if deleted == 0:
        msg = html.Div(
            [
                html.B("Cache leeren"),
                html.Div("Es wurden keine Einträge gefunden."),
            ]
        )
    else:
        msg = html.Div(
            [
                html.B("Cache geleert"),
                html.Div(f"{deleted} Eintrag(e) entfernt."),
            ]
        )

    return True, msg

# ==============================================================================
# "Liste"-Button: Runs aus loader/runs anzeigen
# ==============================================================================


def _load_run_meta(run_ts_dir: Path) -> dict:
    """*_run_meta.json in einem Run-Zeitstempel-Ordner laden (falls vorhanden)."""
    try:
        meta_files = list(run_ts_dir.glob("*_run_meta.json"))
        if not meta_files:
            return {}
        with meta_files[0].open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _logger.warning(f"[Runs-List] Konnte Meta aus {run_ts_dir} nicht laden: {exc}")
        return {}


def _discover_runs() -> list[dict]:
    """
    Läuft durch loader/runs und sammelt alle Runs ein.
    Erwartete Struktur:
      loader/runs/<cache_tag>/<timestamp>/*_run_meta.json
    """
    runs: list[dict] = []

    if not RUNS_DIR.exists():
        _logger.info(f"[Runs-List] RUNS_DIR existiert nicht: {RUNS_DIR}")
        return runs

    for tag_dir in sorted(RUNS_DIR.iterdir()):
        if not tag_dir.is_dir():
            continue

        # timestamps im Ordner
        for ts_dir in sorted(tag_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir():
                continue

            meta = _load_run_meta(ts_dir)
            ts_raw = ts_dir.name

            try:
                ts_parsed = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S")
                ts_display = ts_parsed.strftime("%d.%m.%Y %H:%M:%S")
            except Exception:
                ts_display = ts_raw

            runs.append(
                {
                    "cache_tag": tag_dir.name,
                    "timestamp": ts_display,
                    "timestamp_raw": ts_raw,
                    "path": str(ts_dir),
                    "target": meta.get("target_col")
                    or meta.get("config", {}).get("target_col"),
                    "selected_exog": meta.get("selected_exog")
                    or meta.get("config", {}).get("selected_exog"),
                    "horizon": meta.get("forecast_horizon")
                    or meta.get("config", {}).get("forecast_horizon"),
                    "meta_raw": meta,
                }
            )

    return runs


def _render_runs_list(runs: list[dict]) -> html.Div:
    """Modal-Body bauen."""
    if not runs:
        return html.Div("Keine Runs gefunden.", className="text-muted")

    items = []
    for r in runs:
        exog_display = ", ".join(r["selected_exog"]) if r.get("selected_exog") else "–"
        items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(r["cache_tag"]),
                            html.Span(f" | {r['timestamp']}", className="text-muted ms-2"),
                        ]
                    ),
                    html.Div(f"Target: {r.get('target') or 'unbekannt'}", className="small"),
                    html.Div(f"Horizont: {r.get('horizon') or '–'}", className="small"),
                    html.Div(f"Exogene: {exog_display}", className="small mb-1"),
                    # wichtige Stelle: dynamische ID
                    html.Button(
                        "Diesen Run verwenden",
                        id={
                            "type": "load-run-btn",
                            "cache_tag": r["cache_tag"],
                            "timestamp": r["timestamp_raw"],
                        },
                        n_clicks=0,
                        className="btn btn-sm btn-outline-primary mt-1",
                    ),
                    html.Hr(),
                ],
                className="mb-2",
            )
        )

    return html.Div(items, style={"maxHeight": "60vh", "overflowY": "auto"})




# ==============================================================================
# FORECAST: Modal öffnen und schließen
# ==============================================================================
@app.callback(
    Output("forecast-confirm-modal", "is_open"),
    Output("forecast-confirm-summary", "children"),
    Input("create-forecast-btn", "n_clicks"),          # KEIN Modal mehr, Berechnung startet direkt
    Input("confirm-recalc-forecast", "n_clicks"),      # schließt bei Confirm
    Input("cancel-recalc-forecast", "n_clicks"),       # schließt bei Cancel
    Input("model-artifact-store", "data"),             # öffnet Modal bei Run-Auswahl (Liste)
    State("forecast-confirm-modal", "is_open"),
    # Vorschau-Infos aus der UI
    State("forecast-target-dropdown", "value"),
    State("external-exog-dropdown", "value"),
    State("forecast-horizon-store", "data"),
    State("forecast-sektor-dropdown", "value"),
    State("forecast-datenmodus-switch", "value"),
    prevent_initial_call=True,
)
def toggle_recalc_modal(
    open_clicks,
    confirm_clicks,
    cancel_clicks,
    model_store,
    is_open,
    target_value,
    exog_values,
    horizon_store,
    sektor_value,
    is_fluss_mode
):
    # --- Trigger robust ermitteln (Dash >=2.9 und älter) ---
    trigger = None
    try:
        from dash import ctx as _ctx  # Dash >= 2.9
        trigger = getattr(_ctx, "triggered_id", None)
    except Exception:
        pass
    if not trigger:
        try:
            from dash import callback_context as _legacy_ctx  # ältere Dash-Versionen
            trigger = _legacy_ctx.triggered[0]["prop_id"].split(".")[0] if _legacy_ctx.triggered else None
        except Exception:
            trigger = None
    if not trigger:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

    # --- 1) „Prognose erstellen” -> KEIN Modal (Direktstart in anderem Callback) ---
    if trigger == "create-forecast-btn":
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

    # --- 2) Run aus der Liste gewählt -> Modal öffnen & Run-Vorschau ---
    if trigger == "model-artifact-store":
        from dash import html
        from dash.exceptions import PreventUpdate
        try:
            if isinstance(model_store, dict) and model_store.get("source") == "runs":
                meta = model_store.get("meta") or {}
                cache_tag = model_store.get("cache_tag") or meta.get("cache_tag")
                timestamp = model_store.get("timestamp") or meta.get("timestamp")
                target = meta.get("target_col") or meta.get("target") or "–"
                horizon = meta.get("forecast_horizon") or meta.get("horizon") or "–"
                exog = meta.get("selected_exog") or []
                exog_count = len(exog)

                preview = html.Div([
                    html.P([html.Strong("Run: "), f"{cache_tag or '–'} @ {timestamp or '–'}"], className="mb-1"),
                    html.P([html.Strong("Zielvariable: "), str(target)], className="mb-1"),
                    html.P([html.Strong("Einflussfaktoren: "), f"{exog_count} ausgewählt"], className="mb-1"),
                    html.P([html.Strong("Prognosehorizont: "), str(horizon)], className="mb-0"),
                ])
                return True, preview
            else:
                raise PreventUpdate
        except Exception:
            raise PreventUpdate

    # --- 3) Confirm / Cancel -> Modal schließen, Inhalt beibehalten ---
    if trigger in ("confirm-recalc-forecast", "cancel-recalc-forecast"):
        from dash import no_update
        return False, no_update

    # --- sonst nichts tun ---
    from dash.exceptions import PreventUpdate
    raise PreventUpdate





# ------------------------------------------------------------
# Callback: Modal öffnen/schließen + Run auswählen
# ------------------------------------------------------------
@app.callback(
    Output("runs-list-modal", "is_open"),
    Output("runs-list-body", "children"),
    Output("model-artifact-store", "data"),
    Input("show-runs-btn", "n_clicks"),
    Input("close-runs-list", "n_clicks"),
    Input({"type": "load-run-btn", "cache_tag": ALL, "timestamp": ALL}, "n_clicks"),
    State("runs-list-modal", "is_open"),
    prevent_initial_call=True,
)
def handle_runs_modal(open_click, close_click, load_clicks, is_open):
    """
    - show-runs-btn: Modal öffnen + Runs laden
    - close-runs-list: Modal schließen
    - irgendein load-run-btn: ausgewählten Run in Store legen + Modal schließen
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # 1) Modal öffnen
    if triggered_id == "show-runs-btn":
        runs = _discover_runs()
        body = _render_runs_list(runs)
        return True, body, no_update

    # 2) Modal schließen
    if triggered_id == "close-runs-list":
        return False, no_update, no_update

    # 3) dynamischer Button
    try:
        btn_id = json.loads(triggered_id)
    except json.JSONDecodeError:
        raise PreventUpdate

    if btn_id.get("type") == "load-run-btn":
        cache_tag = btn_id.get("cache_tag")
        ts = btn_id.get("timestamp")
        run_dir = RUNS_DIR / cache_tag / ts

        meta = _load_run_meta(run_dir)
        picked = {
            "source": "runs",
            "cache_tag": cache_tag,
            "timestamp": ts,
            "path": str(run_dir),
            "meta": meta,
        }
        _logger.info(f"[Runs-List] Run gewählt: {picked.get('cache_tag')}")

        # Modal schließen, Body nicht anrühren, Store füllen
        return False, no_update, picked

    raise PreventUpdate



# ==============================================================================
# PRESET-Callbacks (Dropdown befüllen, speichern, löschen, laden)
# ==============================================================================

@app.callback(
    dash.Output("forecast-preset-dropdown", "options"),
    dash.Output("user-presets-store", "data", allow_duplicate=True),
    [
        dash.Input("user-presets-store", "data"),
        dash.Input("forecast-target-dropdown", "value"),
    ],
    prevent_initial_call='initial_duplicate',
)
def populate_preset_dropdown_options(
    user_presets_from_store: Optional[Dict[str, Any]],
    target_value: Optional[str]
):
    try:
        from src.app import (
            _load_user_presets_from_disk,
            merge_hc_and_user_presets_for_dropdown,
            _normalize_target_slug as _app_norm_slug,
        )

        try:
            from src.frontend.forecaster.forecaster_main import get_ecb_presets_hydrated
        except ImportError:
            _logger.warning("[PresetDropdown] get_ecb_presets_hydrated nicht verfügbar")
            def get_ecb_presets_hydrated():
                return {}
        
        hc_dict: Dict[str, Dict[str, Any]] = {}
        try:
            all_hc_presets = get_ecb_presets_hydrated()
            target_slug = _app_norm_slug(target_value)
            if target_slug and target_slug in all_hc_presets:
                meta = all_hc_presets[target_slug]
                title = meta.get("title") or f"H&C {target_slug}"
                hc_dict[title] = {"id": target_slug}
        except Exception as e:
            _logger.warning(f"[PresetDropdown] H&C-Presets nicht verfügbar: {e}")
        
        user_dict_disk = _load_user_presets_from_disk()
        _logger.debug(f"[PresetDropdown] {len(user_dict_disk)} User-Presets von Disk geladen")
        
        options = merge_hc_and_user_presets_for_dropdown(
            hc_presets=hc_dict,
            user_presets=user_dict_disk,
            hc_label="H&C Presets",
            user_label="Eigene Presets",
        )
        
        options.insert(0, {"label": "– kein Preset –", "value": "__none__"})
        
        _logger.debug(f"[PresetDropdown] {len(options)} Optionen generiert")
        return options, user_dict_disk
    
    except Exception as e:
        _logger.exception(f"[PresetDropdown] Kritischer Fehler: {e}")
        return [{"label": "– kein Preset –", "value": "__none__"}], no_update


@app.callback(
    dash.Output("load-preset-btn", "disabled"),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=False
)
def toggle_load_preset_button(preset_value):
    return (not preset_value) or (preset_value == "__none__")


@app.callback(
    dash.Output("model-artifact-store", "data", allow_duplicate=True),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_preset_to_model_store(selected_value: Optional[str]):
    if not selected_value or selected_value == "__none__":
        raise PreventUpdate
    
    try:
        from src.app import _load_user_presets_from_disk
        if not str(selected_value).startswith("user_"):
            _logger.debug("[ApplyPreset] Kein User-Preset")
            raise PreventUpdate
        
        user_id = selected_value[5:]
        user_dict = _load_user_presets_from_disk()
        
        for display_name, meta in user_dict.items():
            if not isinstance(meta, dict):
                continue
            
            if meta.get("id") == user_id:
                model_path = meta.get("model_path")
                snapshot_path = meta.get("exog_snapshot_path")
                
                if model_path or snapshot_path:
                    model_exists = model_path and os.path.exists(model_path)
                    snapshot_exists = snapshot_path and os.path.exists(snapshot_path)
                    _logger.info(f"[ApplyPreset] Preset '{display_name}' geladen")
                    return {
                        "path": model_path if model_exists else None,
                        "exog_snapshot_path": snapshot_path if snapshot_exists else None
                    }
        
        _logger.warning(f"[ApplyPreset] Preset-ID '{user_id}' nicht in Disk-Presets gefunden")
        raise PreventUpdate
    
    except PreventUpdate:
        raise
    except Exception as e:
        _logger.warning(f"[ApplyPreset] Fehler beim Anwenden: {e}")
        raise PreventUpdate


# ==============================================================================
# PRESET: Modal öffnen und tatsächlich speichern
# ==============================================================================

@app.callback(
    Output("save-preset-modal", "is_open"),
    Output("preset-preview-content", "children"),
    Input("save-preset-btn", "n_clicks"),
    Input("confirm-save-preset", "n_clicks"),
    Input("cancel-save-preset", "n_clicks"),
    State("save-preset-modal", "is_open"),
    State("forecast-target-dropdown", "value"),
    State("external-exog-dropdown", "value"),
    State("forecast-horizon-store", "data"),
    State("forecast-sektor-dropdown", "value"),
    State("forecast-datenmodus-switch", "value"),
    prevent_initial_call=True,
)
def toggle_preset_modal(open_clicks, confirm_clicks, cancel_clicks,
                        is_open, target_value, exog_values, horizon_quarters,
                        sektor_value, is_fluss_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "save-preset-btn":
        sektor_label = {
            "PH": "Private Haushalte (PH)",
            "NFK": "Nichtfinanzielle Unternehmen (NFK)"
        }.get((sektor_value or "").upper(), str(sektor_value or "—"))
        modus_label = "Flussdaten" if bool(is_fluss_mode) else "Bestandsdaten"
        exog_count = len(exog_values or [])

        preview = html.Div([
            html.P([html.Strong("Sektor: "), sektor_label], className="mb-1"),
            html.P([html.Strong("Zielvariable: "), target_value or "Keine"], className="mb-1"),
            html.P([html.Strong("Einflussfaktoren: "), f"{exog_count} ausgewählt"], className="mb-1"),
            html.P([html.Strong("Prognosehorizont: "), f"{horizon_quarters or 6} Quartale"], className="mb-1"),
            html.P([html.Strong("Datenmodus: "), modus_label], className="mb-0"),
        ])
        return True, preview

    if trigger in ("confirm-save-preset", "cancel-save-preset"):
        return False, no_update

    raise PreventUpdate


@app.callback(
    Output("user-presets-store", "data", allow_duplicate=True),
    Output("forecast-preset-dropdown", "value", allow_duplicate=True),
    Output("preset-save-toast", "is_open", allow_duplicate=True),
    Output("preset-save-toast", "children", allow_duplicate=True),
    Output("preset-name-input", "value"),
    Input("confirm-save-preset", "n_clicks"),
    State("preset-name-input", "value"),
    State("forecast-target-dropdown", "value"),
    State("external-exog-dropdown", "value"),
    State("forecast-horizon-store", "data"),
    State("forecast-datenmodus-switch", "value"),
    State("model-artifact-store", "data"),
    State("user-presets-store", "data"),
    prevent_initial_call=True,
)
def save_preset_with_name(n_clicks, preset_name, target_value, exog_values,
                        horizon_quarters, is_fluss_mode, model_payload,
                        current_user_presets):
    if not n_clicks:
        raise PreventUpdate
    
    try:
        from app import create_user_preset_from_ui_state, upsert_user_preset
        
        _logger.debug("[SavePresetModal] Speichervorgang gestartet")
        
        if not preset_name or not preset_name.strip():
            target = str(target_value or "Preset").strip()
            exogs = list(exog_values or [])
            horizon = int(horizon_quarters or 0)
            ts = time.strftime("%Y-%m-%d %H:%M")
            preset_name = f"{target} | {len(exogs)} Exog | {horizon}Q @ {ts}"
            _logger.debug(f"[SavePresetModal] Auto-Name generiert: {preset_name}")
        else:
            preset_name = preset_name.strip()
            _logger.debug(f"[SavePresetModal] User-Name: {preset_name}")
        
        target = str(target_value or "Preset").strip()
        exogs = list(exog_values or [])
        try:
            horizon = int(horizon_quarters or 0)
        except (TypeError, ValueError):
            horizon = 0
        
        preset_obj = create_user_preset_from_ui_state(
            name=preset_name,
            target=target,
            exog=exogs,
            horizon=horizon,
            is_fluss_mode=bool(is_fluss_mode),
            model_payload=(model_payload or {}),
            extra_ui_options={},
        )
        
        all_presets = upsert_user_preset(preset_name, preset_obj)
        
        new_dropdown_value = f"user_{preset_obj['id']}"
        
        toast_children = html.Div([
            html.Div("✅ Preset erfolgreich gespeichert!", className="fw-bold"),
            html.Small(preset_name, className="text-muted d-block mt-1")
        ])
        
        return all_presets, new_dropdown_value, True, toast_children, ""
    
    except Exception as e:
        _logger.exception(f"[SavePresetModal] ✗ Fehler beim Speichern: {e}")
        
        error_toast = html.Div([
            html.Div("❌ Speichern fehlgeschlagen", className="fw-bold text-danger"),
            html.Small(str(e), className="text-danger d-block mt-1 font-monospace")
        ])
        
        return no_update, no_update, True, error_toast, no_update


# ==============================================================================
# PRESET: Löschen
# ==============================================================================

@app.callback(
    dash.Output("delete-preset-btn", "disabled"),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=False
)
def toggle_delete_button(value):
    if not value or value == "__none__":
        return True
    return not str(value).startswith("user_")


@app.callback(
    dash.Output("delete-preset-modal", "is_open"),
    dash.Output("delete-preset-body", "children"),
    dash.Input("delete-preset-btn", "n_clicks"),
    dash.Input("confirm-delete-preset", "n_clicks"),
    dash.Input("cancel-delete-preset", "n_clicks"),
    dash.State("delete-preset-modal", "is_open"),
    dash.State("forecast-preset-dropdown", "value"),
    dash.State("user-presets-store", "data"),
    prevent_initial_call=True,
)
def toggle_delete_modal(open_clicks, confirm_clicks, cancel_clicks,
                        is_open, dropdown_value, user_presets):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "delete-preset-btn":
        if not dropdown_value or not str(dropdown_value).startswith("user_"):
            raise PreventUpdate
        pid = str(dropdown_value).replace("user_", "", 1)
        disp_name = None
        for name, meta in (user_presets or {}).items():
            if str((meta or {}).get("id")) == pid:
                disp_name = name
                break
        text = f"Soll das Preset „{disp_name or pid}“ wirklich gelöscht werden?"
        return True, text

    if trigger in ("confirm-delete-preset", "cancel-delete-preset"):
        return False, no_update

    raise PreventUpdate


@app.callback(
    dash.Output("user-presets-store", "data", allow_duplicate=True),
    dash.Output("forecast-preset-dropdown", "value", allow_duplicate=True),
    dash.Output("preset-save-toast", "is_open", allow_duplicate=True),
    dash.Output("preset-save-toast", "children", allow_duplicate=True),
    dash.Input("confirm-delete-preset", "n_clicks"),
    dash.State("forecast-preset-dropdown", "value"),
    dash.State("user-presets-store", "data"),
    prevent_initial_call=True,
)
def delete_selected_preset(n_clicks, dropdown_value, user_presets):
    if not n_clicks:
        raise PreventUpdate
    if not dropdown_value or not str(dropdown_value).startswith("user_"):
        raise PreventUpdate

    pid = str(dropdown_value).replace("user_", "", 1)
    updated = dict(user_presets or {})
    key_to_remove = None
    for disp_name, meta in (updated or {}).items():
        if str((meta or {}).get("id")) == pid or disp_name == pid:
            key_to_remove = disp_name
            break
    if key_to_remove:
        updated.pop(key_to_remove, None)

    try:
        from app import delete_user_preset
        updated = delete_user_preset(preset_identifier=pid, delete_files=False) or updated
    except Exception:
        pass

    toast_children = html.Div([html.Div("🗑️ Preset gelöscht", className="fw-bold")])

    return updated, "__none__", True, toast_children


# ==============================================================================
# PRESET-LADEN (vollständig, inkl. Sektor)
# ==============================================================================

@app.callback(
    Output("external-exog-dropdown", "value", allow_duplicate=True),
    Input("forecast-preset-dropdown", "value"),
    State("forecast-target-dropdown", "value"),
    prevent_initial_call=True
)
def apply_preset_to_external_exog(preset_value, target_value):
    presets = get_ecb_presets_hydrated()
    
    if not preset_value or preset_value == "__none__" or not str(preset_value).startswith("preset_"):
        return dash.no_update
    
    slug = str(preset_value).replace("preset_", "", 1)
    if slug not in presets:
        return []
    
    exog_dict = presets[slug].get("exog")
    if isinstance(exog_dict, dict):
        return list(exog_dict.values())
    
    return []


@app.callback(
    Output("forecast-target-dropdown", "value", allow_duplicate=True),
    Output("external-exog-dropdown", "value", allow_duplicate=True),
    Output("model-artifact-store", "data", allow_duplicate=True),
    Output("exog-data-store", "data", allow_duplicate=True),
    Output("forecast-sektor-dropdown", "value", allow_duplicate=True),
    Input("load-preset-btn", "n_clicks"),
    State("forecast-preset-dropdown", "value"),
    State("user-presets-store", "data"),
    prevent_initial_call=True
)
def load_selected_preset(n_clicks, preset_value, user_presets):
    if not n_clicks or not preset_value or preset_value == "__none__":
        raise dash.exceptions.PreventUpdate

    import os

    presets = get_ecb_presets_hydrated()
    target = dash.no_update
    exogs = dash.no_update
    model_payload = dash.no_update
    exog_store_json = dash.no_update
    sektor_val = dash.no_update

    if str(preset_value).startswith("preset_"):
        slug = str(preset_value).replace("preset_", "", 1)
        p = presets.get(slug)
        if not p:
            raise dash.exceptions.PreventUpdate

        target = p.get("target") or dash.no_update
        exog_dict = p.get("exog")
        exogs = list(exog_dict.values()) if isinstance(exog_dict, dict) else (exog_dict or [])
        ui_opts = p.get("ui_opts") or {}
        sektor_val = (ui_opts.get("sektor") if isinstance(ui_opts, dict) else None) or p.get("sektor") or dash.no_update

        data_path = p.get("final_dataset_path") or p.get("exog_snapshot_path")
        mdl_path = p.get("model_path")

        if mdl_path or data_path:
            model_payload = {"path": mdl_path, "exog_snapshot_path": data_path}

        if data_path and os.path.exists(data_path):
            exog_store_json = _snapshot_to_store_json(data_path, exogs if isinstance(exogs, list) else [])

        return target, exogs, model_payload, exog_store_json, sektor_val

    if str(preset_value).startswith("user_"):
        pid = str(preset_value).replace("user_", "", 1)
        chosen = None
        for name, meta in (user_presets or {}).items():
            if str(meta.get("id")) == str(pid):
                chosen = meta
                break

        if not chosen:
            raise dash.exceptions.PreventUpdate

        target = chosen.get("target")
        exogs = chosen.get("exog") or []

        ui_opts = chosen.get("ui_opts") or {}
        sektor_val = ui_opts.get("sektor") if isinstance(ui_opts, dict) else None
        if not sektor_val:
            sektor_val = chosen.get("sektor")
        if not sektor_val:
            sektor_val = dash.no_update

        data_path = chosen.get("final_dataset_path") or chosen.get("exog_snapshot_path")
        mdl_path = chosen.get("model_path")

        if mdl_path or data_path:
            model_payload = {"path": mdl_path, "exog_snapshot_path": data_path}

        if data_path and os.path.exists(data_path):
            exog_store_json = _snapshot_to_store_json(data_path, exogs if isinstance(exogs, list) else [])

        return target, exogs, model_payload, exog_store_json, sektor_val

    raise dash.exceptions.PreventUpdate

def _compute_simple_metrics(metadata: dict) -> dict:
    """
    Extrahiert möglichst viele simple Metriken aus dem Forecaster-Metadata.
    Reihenfolge:
    1) bekannte Felder (cv_performance, diagnostics, ...)
    2) Rekonstruktion aus cv_residuals + ci_std_error + y_train_summary
    3) Fallback auf backtest_results (actual/predicted)
    """
    import numpy as np
    import pandas as pd

    # Grundgerüst
    result = {
        "mae": None,
        "rmse": None,
        "r2": None,
        "wape": None,
        "bias_pct": None,
        "smape": None,
        "directional": None,
        "coverage": {},   # z.B. {80: 92.0, 95: 98.0}
    }

    if not isinstance(metadata, dict):
        return result

    # ------------------------------------------------------------------
    # 1) aus typischen Metadata-Blöcken lesen
    # ------------------------------------------------------------------
    possible_blocks = [
        metadata.get("cv_performance"),
        metadata.get("cv_metrics"),
        metadata.get("cv"),
        (metadata.get("diagnostics") or {}).get("cv_performance"),
        (metadata.get("diagnostics") or {}).get("cv_metrics"),
        (metadata.get("diagnostics") or {}).get("cv"),
        (metadata.get("evaluation") or {}),
        (metadata.get("metrics") or {}),
        # manche Pipelines legen es unter dashboard_result ab
        (metadata.get("dashboard_result") or {}).get("cv_performance")
        if isinstance(metadata.get("dashboard_result"), dict) else None,
    ]

    def _pick(d, *names):
        if not isinstance(d, dict):
            return None
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return None

    mae = rmse = r2 = None
    bias_pct = smape = directional = None
    coverage_dict = {}

    for block in possible_blocks:
        if not isinstance(block, dict):
            continue

        mae  = mae  or _pick(block, "cv_mae", "mae", "MAE", "mean_absolute_error")
        rmse = rmse or _pick(block, "cv_rmse", "rmse", "RMSE", "root_mean_squared_error")
        r2   = r2   or _pick(block, "cv_r2", "r2", "R2", "r2_score")

        bias_pct   = bias_pct or _pick(block, "bias_pct", "mean_bias_pct", "bias_percent")
        smape      = smape or _pick(block, "smape", "sMAPE", "SMAPE")
        directional = directional or _pick(block, "directional_accuracy", "directional", "directional_pct")

        if not coverage_dict:
            cov_block = (
                block.get("interval_coverage")
                or block.get("coverage")
                or {}
            )
            if isinstance(cov_block, dict):
                for k, v in cov_block.items():
                    try:
                        lvl = int(str(k).replace("%", ""))
                    except Exception:
                        continue
                    if v is None:
                        continue
                    v = float(v)
                    if 0 < v <= 1.0:
                        v = v * 100.0
                    coverage_dict[lvl] = v

    # ------------------------------------------------------------------
    # 2) rekonstruiere Bias & CI-Deckung aus Residuen (die hast du im Log)
    # ------------------------------------------------------------------
    cv_res = metadata.get("cv_residuals")
    y_sum = metadata.get("y_train_summary") or {}
    y_mean = y_sum.get("mean") or y_sum.get("avg")  # je nach Writer
    std_error = (
        metadata.get("ci_std_error")
        or metadata.get("cv_residual_std_unscaled")
    )

    # Bias % nur berechnen, wenn noch nicht vorhanden
    if bias_pct is None and isinstance(cv_res, list) and len(cv_res) > 0 and y_mean not in (None, 0):
        cv_res_arr = np.array(cv_res, dtype=float)
        mean_resid = float(np.mean(cv_res_arr))
        bias_pct = (mean_resid / float(y_mean)) * 100.0

    # WAPE berechnen (Weighted MAPE = MAE / Mean) -> robuster als MAPE/sMAPE bei Nullen
    wape = None
    if mae is not None and y_mean is not None and abs(y_mean) > 1e-9:
        wape = (float(mae) / abs(float(y_mean))) * 100.0

    # CI-Deckung nur berechnen, wenn wir noch keine haben
    if not coverage_dict and isinstance(cv_res, list) and len(cv_res) > 0 and std_error not in (None, 0):
        cv_abs = np.abs(np.array(cv_res, dtype=float))
        levels = {
            80: 1.2815515655446004,
            95: 1.959963984540054,
        }
        cov_tmp = {}
        for lvl, z in levels.items():
            thresh = z * float(std_error)
            cov_tmp[lvl] = float(np.mean(cv_abs <= thresh) * 100.0)
        coverage_dict = cov_tmp

    # Directional ggf. in Prozent bringen (einige Writer geben 0..1 aus)
    if isinstance(directional, (int, float)) and 0 < directional <= 1.0:
        directional = directional * 100.0

    # ------------------------------------------------------------------
    # 3) Fallback: aus backtest_results fehlende Metriken nachziehen
    # ------------------------------------------------------------------
    need_bt = any(v is None for v in (mae, rmse, r2, smape, directional)) or not coverage_dict
    bt = metadata.get("backtest_results") if need_bt else None

    bt_df = pd.DataFrame()
    if need_bt and bt is not None:
        if isinstance(bt, pd.DataFrame):
            bt_df = bt.copy()
        elif isinstance(bt, (list, tuple, dict)):
            bt_df = pd.DataFrame(bt)

    if need_bt and not bt_df.empty and "actual" in bt_df.columns and "predicted" in bt_df.columns:
        actual = pd.to_numeric(bt_df["actual"], errors="coerce")
        pred   = pd.to_numeric(bt_df["predicted"], errors="coerce")
        valid  = actual.notna() & pred.notna()
        actual = actual[valid]
        pred   = pred[valid]
        err    = pred - actual

        if mae is None:
            mae = float(np.mean(np.abs(err)))
        if rmse is None:
            rmse = float(np.sqrt(np.mean(err ** 2)))

        if r2 is None:
            y_mean_bt = float(actual.mean())
            ss_tot = float(np.sum((actual - y_mean_bt) ** 2))
            ss_res = float(np.sum((actual - pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        if bias_pct is None:
            denom = float(np.mean(np.abs(actual)))
            if denom:
                bias_pct = float(np.mean(err)) / denom * 100.0

        if smape is None:
            denom2 = (np.abs(actual) + np.abs(pred))
            mask_smape = denom2 > 0
            if mask_smape.any():
                smape = 100.0 * float(
                    np.mean(2.0 * np.abs(pred[mask_smape] - actual[mask_smape]) / denom2[mask_smape])
                )

        if directional is None and len(actual) >= 2 and len(pred) >= 2:
            da = np.sign(np.diff(actual))
            dp = np.sign(np.diff(pred))
            directional = 100.0 * float(np.mean(da == dp))

        # Coverage aus Backtest-DF ziehen, falls wir immer noch keine haben
        if not coverage_dict:
            cov_out = {}
            for lvl in (80, 90, 95):
                lower_col = None
                upper_col = None
                for c in (f"yhat_lower_{lvl}", f"lower_{lvl}", f"lo{lvl}"):
                    if c in bt_df.columns:
                        lower_col = c
                        break
                for c in (f"yhat_upper_{lvl}", f"upper_{lvl}", f"hi{lvl}"):
                    if c in bt_df.columns:
                        upper_col = c
                        break
                if lower_col and upper_col:
                    lo = pd.to_numeric(bt_df[lower_col], errors="coerce")
                    hi = pd.to_numeric(bt_df[upper_col], errors="coerce")
                    m = valid & lo.notna() & hi.notna()
                    if m.any():
                        inside = (actual[m] >= lo[m]) & (actual[m] <= hi[m])
                        cov_out[lvl] = 100.0 * float(inside.mean())
            coverage_dict = cov_out

    # ------------------------------------------------------------------
    # 4) alles zurückgeben
    # ------------------------------------------------------------------
    result.update({
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "r2": r2,
        "bias_pct": bias_pct,
        "smape": smape,
        "directional": directional,
        "coverage": coverage_dict,
    })
    return result




# ==============================================================================
# EXPORT-CALLBACK (Schritt 10: Debug-Dateien optional & sicher)
# ==============================================================================

@app.callback(
    Output("download-rawdata", "data"),
    Input("export-rawdata-btn", "n_clicks"),
    State("forecast-target-dropdown", "value"),
    State("forecast-horizon-store", "data"),
    State("external-exog-dropdown", "value"),
    State("manual-series-input", "value"),
    State("forecast-datenmodus-switch", "value"),
    State("gvb-data-store", "data"),
    State("exog-data-store", "data"),
    State("model-artifact-store", "data"),
    State("forecast-sektor-dropdown", "value"),
    prevent_initial_call=True
)
def export_forecast_rawdata(
    n_clicks, target_value, horizon_value,
    exog_selection, manual_input,
    forecast_real_switch_value,
    gvb_json, exog_json, model_artifact_json,
    sektor_value
):
    if not n_clicks:
        return no_update

    import os
    from pathlib import Path





    def _flatten_metadata_to_df(meta: dict) -> pd.DataFrame:
        rows = []
        def _walk(prefix, val):
            if isinstance(val, dict):
                for k, v in val.items():
                    _walk(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    _walk(f"{prefix}[{i}]", v)
            else:
                rows.append({"key": prefix, "value": val})
        if isinstance(meta, dict):
            _walk("", meta)
        return pd.DataFrame(rows)

    def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
        try:
            df = _parse_store_df(gjson)
            if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                sekt = str(sektor).strip().upper()
                df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            return df.to_json(orient="split", date_format="iso")
        except Exception:
            return gjson

    modus = "fluss" if bool(forecast_real_switch_value) else "bestand"

    try:
        gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)
    except Exception:
        gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

    config_df = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target": target_value,
        "horizon_q": horizon_value,
        "modus": modus,
        "smoothing": 1,
        "start_date": "(auto)",
        "end_date": "(auto)",
        "sektor": str(sektor_value or "(default)"),
        "log_transform": False,
        "exog_selection_dropdown": ", ".join(exog_selection) if isinstance(exog_selection, list) else str(exog_selection or ""),
        "exog_manual_input_raw": str(manual_input or ""),
        "forecast_real_switch": forecast_real_switch_value if forecast_real_switch_value is not None else "(n/a)",
    }])

    gvb_df = _parse_store_df(gvb_json_filtered)
    if gvb_df.empty or "date" not in gvb_df.columns:
        config_df.loc[0, "status"] = "Fehler"
        config_df.loc[0, "msg"] = "GVB-Daten fehlen (nach Sektor-Filter) oder 'date' ist nicht vorhanden."
        xbytes = _make_export_bytes(
            config_df, pd.DataFrame(), pd.DataFrame(), _parse_store_df(exog_json)
        )
        return dcc.send_bytes(lambda b: b.write(xbytes), "forecast_export_error.xlsx")

    for c in ("ebene1", "ebene2", "ebene3", "bestand", "fluss", "sektor"):
        if c not in gvb_df.columns:
            gvb_df[c] = np.nan

    gvb_df["date"] = pd.to_datetime(gvb_df["date"], errors="coerce")
    gvb_df = gvb_df.dropna(subset=["date"]).sort_values("date")

    if gvb_df.empty:
        config_df.loc[0, "status"] = "Fehler"
        config_df.loc[0, "msg"] = "Alle Datumswerte sind ungültig (nach Sektor-Filter)."
        xbytes = _make_export_bytes(
            config_df, pd.DataFrame(), pd.DataFrame(), _parse_store_df(exog_json)
        )
        return dcc.send_bytes(lambda b: b.write(xbytes), "forecast_export_error.xlsx")

    start_date = gvb_df["date"].min().date()
    end_date = gvb_df["date"].max().date()
    config_df.loc[0, "start_date"] = str(start_date)
    config_df.loc[0, "end_date"] = str(end_date)

    main_df = _build_main_e1_table_from_store(
        gvb_df,
        data_type=modus,
        start_date=str(start_date),
        end_date=str(end_date),
        smoothing=1,
        use_log=False,
        sektor=sektor_value
    )

    value_col = "fluss" if modus == "fluss" else "bestand"
    view = (
        gvb_df[(gvb_df["date"] >= pd.Timestamp(start_date)) & (gvb_df["date"] <= pd.Timestamp(end_date))]
        .groupby(["date", "ebene1"], dropna=False)[value_col]
        .sum()
        .reset_index()
        .pivot(index="date", columns="ebene1", values=value_col)
        .sort_index()
        .fillna(0.0)
    )
    current_view_df = view if not view.empty else pd.DataFrame()

    exog_df = _parse_store_df(exog_json)
    if not exog_df.empty:
        for c in ["date", "Date", "DATE", "time", "Time"]:
            if c in exog_df.columns:
                if c != "date":
                    exog_df = exog_df.rename(columns={c: "date"})
                cols = ["date"] + [x for x in exog_df.columns if x != "date"]
                exog_df = exog_df[cols]
                break

    pipeline_prepared_df = pd.DataFrame()
    forecast_df = pd.DataFrame()
    metadata_df = pd.DataFrame()
    model_config_df = pd.DataFrame()
    future_design_df = pd.DataFrame()
    debug_train_quarterly_df = pd.DataFrame()
    debug_train_design_df = pd.DataFrame()

    if isinstance(model_artifact_json, dict):
        fc_json = model_artifact_json.get("forecast_df")
        if fc_json:
            try:
                forecast_df = pd.read_json(fc_json, orient="split")
            except Exception:
                forecast_df = pd.DataFrame()

        meta_obj = model_artifact_json.get("metadata") or {}
        if isinstance(meta_obj, dict):
            metadata_df = _flatten_metadata_to_df(meta_obj)
        else:
            meta_obj = {}

        snap_path = model_artifact_json.get("exog_snapshot_path")
        if snap_path and os.path.exists(snap_path):
            try:
                pipeline_prepared_df = pd.read_parquet(snap_path)
            except Exception:
                pipeline_prepared_df = pd.DataFrame()

        model_path = model_artifact_json.get("path")
        dash_export = meta_obj.get("dash_export") if isinstance(meta_obj, dict) else {}

        out_dir = None
        cfg_dict = {}

        if model_path:
            try:
                from src.forecaster.forecaster_pipeline import ModelArtifact  # type: ignore
                art = ModelArtifact.load(model_path)
                cfg_dict = art.config_dict or {}
                if cfg_dict:
                    model_config_df = pd.DataFrame(
                        [{"key": k, "value": v} for k, v in cfg_dict.items()]
                    )

                out_dir = dash_export.get("output_dir") or cfg_dict.get("output_dir")
                if out_dir:
                    out_dir = Path(out_dir)

            except Exception:
                pass

        if not out_dir:
            fallback_out = dash_export.get("output_dir")
            if fallback_out:
                out_dir = Path(fallback_out)

        # Schritt 10: Debug-Dateien robust einlesen
        if out_dir:
            tq_path = dash_export.get("train_quarterly_csv") or (out_dir / "train_quarterly_debug.csv")
            td_path = dash_export.get("train_design_csv") or (out_dir / "train_design_debug.csv")
        else:
            tq_path = dash_export.get("train_quarterly_csv") if dash_export else None
            td_path = dash_export.get("train_design_csv") if dash_export else None

        debug_train_quarterly_df = _safe_read_debug_csv(
            tq_path,
            add_cols={"_source_debug_file": str(tq_path) if tq_path else "(keine Datei übergeben)"}
        )
        debug_train_design_df = _safe_read_debug_csv(
            td_path,
            add_cols={"_source_debug_file": str(td_path) if td_path else "(keine Datei übergeben)"}
        )

        if out_dir:
            f_path = dash_export.get("future_design_csv") or (out_dir / "future_design_debug.csv")
            if f_path and Path(f_path).exists():
                try:
                    future_design_df = pd.read_csv(f_path)
                except Exception:
                    future_design_df = pd.DataFrame()

            if metadata_df.empty:
                meta_file = dash_export.get("production_metadata_json") or (out_dir / "production_forecast_metadata.json")
                if meta_file and Path(meta_file).exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            pipeline_meta_obj = json.load(f)
                        metadata_df = _flatten_metadata_to_df(pipeline_meta_obj)
                    except Exception:
                        pass

            if forecast_df.empty:
                fc_file = dash_export.get("production_forecast_csv") or (out_dir / "production_forecast.csv")
                if fc_file and Path(fc_file).exists():
                    try:
                        forecast_df = pd.read_csv(fc_file)
                    except Exception:
                        pass

        final_dataset_path = dash_export.get("final_dataset_path") if dash_export else None
        if final_dataset_path and os.path.exists(final_dataset_path):
            try:
                try:
                    pipeline_prepared_df = pd.read_excel(final_dataset_path, sheet_name="final_dataset")
                except Exception:
                    pipeline_prepared_df = pd.read_excel(final_dataset_path)
            except Exception:
                pass

    xbytes = _make_export_bytes(
        config_df=config_df,
        main_df=main_df,
        current_view_df=current_view_df,
        exog_df=exog_df,
        gvb_raw_df=gvb_df,
        pipeline_prepared_df=pipeline_prepared_df,
        forecast_df=forecast_df,
        future_design_df=future_design_df,
        metadata_df=metadata_df,
        model_config_df=model_config_df,
        debug_train_quarterly_df=debug_train_quarterly_df,
        debug_train_design_df=debug_train_design_df,
    )

    filename = f"forecast_export_{sektor_value or 'NA'}_{modus}_{pd.Timestamp.today().strftime('%Y-%m-%d')}.xlsx"
    return dcc.send_bytes(lambda b: b.write(xbytes), filename)


@app.callback(
    Output("custom-final-dataset-store", "data"),
    Output("upload-custom-dataset-feedback", "children"),
    Input("upload-custom-dataset", "contents"),
    State("upload-custom-dataset", "filename"),
    State("upload-custom-dataset", "last_modified"),
    prevent_initial_call=True,
)
def handle_custom_dataset_upload(contents, filename, last_modified):
    if contents is None:
        raise PreventUpdate

    try:
        # 1) Base64 → Bytes
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        bio = io.BytesIO(decoded)

        # 2) Excel lesen – bevorzugt PIPELINE_PREP, sonst final_dataset, sonst erstes Sheet
        xls = pd.ExcelFile(bio)
        sheet_name = None
        for cand in ("PIPELINE_PREP", "final_dataset"):
            if cand in xls.sheet_names:
                sheet_name = cand
                break
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]

        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 3) Datumsspalte auf "date" normalisieren
        if "date" not in df.columns:
            for cand in ("Datum", "DATE", "Date", "ds", "time", "Time"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "date"})
                    break

        if "date" not in df.columns:
            msg = (
                "Das hochgeladene Excel enthält weder eine Spalte 'date' noch 'Datum'. "
                "Bitte das von der Prognose-Suite exportierte File als Grundlage verwenden."
            )
            return dash.no_update, dbc.Alert(msg, color="danger", className="mt-2")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 4) Payload für den Store bauen
        payload = {
            "filename": filename,
            "uploaded_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sheet_name": sheet_name,
            "json": df.to_json(orient="split", date_format="iso"),
        }

        info = (
            f"Custom-Dataset geladen: '{filename}' "
            f"(Sheet: {sheet_name}, Zeilen: {len(df)}, Spalten: {len(df.columns)})"
        )
        return payload, dbc.Alert(info, color="success", className="mt-2")

    except Exception as e:
        _logger.exception("[Upload] Fehler beim Einlesen des Custom-Datasets: %s", e)
        msg = "Fehler beim Einlesen des Excel-Files. Bitte Struktur prüfen."
        return dash.no_update, dbc.Alert(msg, color="danger", className="mt-2")







# ==============================================================================
# ECB-Daten on demand nachladen, wenn Forecast gedrückt wird
# ==============================================================================

@app.callback(
    Output("exog-data-store", "data", allow_duplicate=True),
    Input("create-forecast-btn", "n_clicks"),
    State("external-exog-dropdown", "value"),
    State("exog-data-store", "data"),
    prevent_initial_call=True
)
def download_and_merge_exog(n_clicks, selected_codes, current_store):
    if not n_clicks or not selected_codes:
        return dash.no_update

    if not isinstance(selected_codes, list):
        selected_codes = [selected_codes]

    df_existing = _safe_load_store(current_store)
    if df_existing is not None:
        df_existing = _normalize_dates(df_existing)

    existing_cols = set()
    if df_existing is not None and not df_existing.empty:
        existing_cols = {c for c in df_existing.columns if c.lower() != "date"}

    to_download = []
    for code in selected_codes:
        if not _looks_like_ecb_code(code):
            continue
        norm = re.sub(r'[^A-Za-z0-9_]+', '_', str(code))
        already = (df_existing is not None) and (code in existing_cols or norm in existing_cols)
        if not already:
            to_download.append(code)

    if not to_download:
        return dash.no_update

    try:
        df_downloaded = _download_exog_codes(to_download)
        if df_downloaded is None or df_downloaded.empty:
            return dash.no_update
    except Exception:
        return dash.no_update

    if df_existing is None or df_existing.empty:
        df_result = df_downloaded.copy()
    else:
        df_result = pd.merge(
            df_existing, df_downloaded, on="date", how="outer", suffixes=("", "__new")
        )
        
        for col in df_downloaded.columns:
            if col == "date":
                continue
            new_col = f"{col}__new"
            if new_col in df_result.columns:
                df_result[col] = df_result[new_col].combine_first(df_result.get(col))
                df_result.drop(columns=[new_col], inplace=True)
        
        df_result = df_result.sort_values("date").reset_index(drop=True)

    keep = ["date"]
    for code in selected_codes:
        if code in df_result.columns:
            keep.append(code)
        else:
            norm = re.sub(r'[^A-Za-z0-9_]+', '_', str(code))
            if norm in df_result.columns:
                df_result = df_result.rename(columns={norm: code})
                keep.append(code)

    df_final = df_result[[c for c in keep if c in df_result.columns]]
    return df_final.to_json(orient="split", date_format="iso")


# ==============================================================================
# PREWARM (H&C Presets, unverändert, aber mit Logging)
# ==============================================================================

@app.callback(
    Output("hc-presets-cache-store", "data", allow_duplicate=True),
    Output("hc-prewarm-toast", "is_open"),
    Output("hc-prewarm-toast", "header"),
    Input("prewarm-hc-presets-btn", "n_clicks"),
    State("gvb-data-store", "data"),
    State("exog-data-store", "data"),
    prevent_initial_call=True
)
def prewarm_hc_presets(n_clicks, gvb_json, exog_store_json):
    if not n_clicks:
        raise PreventUpdate

    def _log(msg):
        try:
            Log.scenario_table(msg)
        except Exception:
            try:
                _logger.info(msg)
            except Exception:
                print(str(msg))

    _log("=" * 80)
    _log("PREWARM gestartet")
    
    try:
        _log(f"Cache-Datei: {HCPRESET_CACHE_FILE.resolve()}")
        _log(f"Snapshots-Verzeichnis: {SNAPSHOTS_DIR.resolve()}  exists={SNAPSHOTS_DIR.exists()}")
    except Exception:
        pass
    
    _log(f"GVB-JSON vorhanden: {bool(gvb_json)}, EXOG-Store-JSON vorhanden: {bool(exog_store_json)}")

    try:
        presets = get_ecb_presets_hydrated()
    except Exception as e:
        _log(f"get_ecb_presets_hydrated() Fehler: {e}")
        presets = {}

    _log(f"Anzahl Presets geladen: {len(presets)} → Slugs: {list(presets.keys()) if isinstance(presets, dict) else '(n/a)'}")
    
    if not presets:
        return no_update, True, "⚠️ Keine H&C Presets gefunden"

    try:
        cache = _load_hc_preset_cache()
    except Exception as e:
        _log(f"Cache laden fehlgeschlagen, starte leer. Grund: {e}")
        cache = {}
    
    _log(f"Aktueller Cache-Keys: {list(cache.keys())}")

    updated, skipped, failed = {}, 0, 0

    for slug, p in (presets or {}).items():
        _log("-" * 60)
        _log(f"Bearbeite Preset: {slug}")

        target = p.get("target")
        exogs = _extract_exog_list(p)
        horizon = p.get("horizon", 6)

        _log(f"Target: {target}, #Exogs: {len(exogs)}")
        
        if not target or not exogs:
            skipped += 1
            continue

        try:
            df_exog, have = _merge_exogs_from_sources(exogs, exog_store_json)
        except Exception as e:
            _log(f"Merge-Fehler: {e}")
            failed += 1
            continue

        if df_exog.empty or not have:
            skipped += 1
            continue

        try:
            final_path = _write_final_dataset(df_exog, slug)
        except Exception as e:
            _log(f"final_dataset write error: {e}")
            failed += 1
            continue

        if not os.path.exists(final_path):
            failed += 1
            continue

        model_path = None
        has_adapter = DashboardForecastAdapter is not None
        
        if HAS_PIPELINE and has_adapter and gvb_json:
            try:
                exog_json_local = df_exog.to_json(orient="split", date_format="iso")
                adapter = DashboardForecastAdapter(gvb_json, exog_json_local)
                _forecast, metadata = adapter.run_forecast(
                    target=target,
                    selected_exog=have,
                    horizon=horizon,
                    use_cached=True,
                    force_retrain=False
                )
                model_path = (metadata or {}).get("model_path")
            except Exception as e:
                _log(f"Pipeline-Fehler (nicht kritisch): {e}")

        updated_entry = {
            "exog_snapshot_path": final_path,
            "final_dataset_path": final_path
        }
        if model_path:
            updated_entry["model_path"] = model_path

        updated[slug] = updated_entry

    if not updated:
        return no_update, True, f"H&C Presets: nichts zu tun (übersprungen: {skipped}, fehlgeschlagen: {failed})"

    try:
        cache.update(updated)
        _save_hc_preset_cache(cache)
    except Exception as e:
        _log(f"Cache speichern fehlgeschlagen: {e}")
        return no_update, True, "H&C Presets vorbereitet – Fehler beim Persistieren"

    return cache, True, f"H&C Presets vorbereitet – ok: {len(updated)}, übersprungen: {skipped}, fehlgeschlagen: {failed}"


# ==============================================================================
# FORECAST: initiale Historie
# ==============================================================================

@app.callback(
    Output("forecast-chart", "figure", allow_duplicate=True),
    [
        Input("url", "pathname"),
        Input("forecast-target-dropdown", "value"),
        Input("forecast-datenmodus-switch", "value"),
    ],
    [
        State("gvb-data-store", "data"),
        State("forecast-state-store", "data"),
        State("forecast-horizon-store", "data"),
        State("forecast-sektor-dropdown", "value"),
    ],
    prevent_initial_call='initial_duplicate',
)
def show_initial_forecast_history(
    pathname, target, is_fluss_mode,
    gvb_json, fc_state, horizon, sektor_value
    ):
    if pathname != "/forecast":
        raise PreventUpdate

    ctx = dash.callback_context
    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if isinstance(fc_state, dict) and fc_state.get("has_forecast") and triggered_id == "url":
            raise PreventUpdate

    if not gvb_json:
        return _empty_forecast_fig("Lade GVB-Daten...")

    def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
        try:
            df = _parse_store_df(gjson)
            if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                sekt = str(sektor).strip().upper()
                df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            return df.to_json(orient="split", date_format="iso")
        except Exception:
            return gjson

    try:
        try:
            gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)
        except Exception:
            gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

        try:
            gvb_df = _parse_store_df(gvb_json_filtered)
        except Exception:
            gvb_df = pd.read_json(gvb_json_filtered, orient="split")

        empty_fc = pd.DataFrame()

        fig = _create_pipeline_chart(
            forecast_df=empty_fc,
            metadata={},
            gvb_data=gvb_df,
            target=target or "gesamt",
            is_fluss_mode=bool(is_fluss_mode),
            horizon_quarters=horizon or 6,
            show_backtest=False,
            backtest_mode="overlay"
        )
        return fig

    except Exception as e:
        _logger.error(f"[InitialHistory] Fehler beim Laden der Historie: {e}")
        return _empty_forecast_fig("Fehler beim Laden der historischen Daten")

# ==============================================================================
# FORECAST: Hauptcallback – mit Backtest und Export-Payload
# ==============================================================================
@app.callback(
    Output('forecast-chart', 'figure', allow_duplicate=True),
    Output('forecast-metrics', 'children'),
    Output('feature-importance-table', 'children'),
    Output('feature-importance-icicle', 'figure'),
    Output('model-artifact-store', 'data', allow_duplicate=True),
    Output('forecast-state-store', 'data', allow_duplicate=True),
    # Beide Inputs dürfen die Berechnung starten:
    Input('confirm-recalc-forecast', 'n_clicks'),
    Input('create-forecast-btn', 'n_clicks'),
    [
        State('exog-data-store', 'data'),
        State('forecast-target-dropdown', 'value'),
        State('external-exog-dropdown', 'value'),
        State('forecast-horizon-store', 'data'),
        State('model-cache-switch', 'value'),
        State('gvb-data-store', 'data'),
        State('forecast-datenmodus-switch', 'value'),
        State('show-backtest-switch', 'value'),
        State('model-artifact-store', 'data'),
        State('forecast-sektor-dropdown', 'value'),
        State('custom-final-dataset-store', 'data'),  # << NEU
    ],
    prevent_initial_call=True
)
def create_pipeline_forecast(
    confirm_clicks,
    create_clicks,
    exog_json,
    target,
    exog_vars,
    horizon,
    use_cache,
    gvb_json,
    is_fluss_mode,
    show_backtest,
    model_payload,
    sektor_value,
    custom_final_dataset,
):
    # Callback wurde ausgelöst (entweder Confirm ODER "Prognose erstellen")
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # 1) Pipeline verfügbar?
    if not HAS_PIPELINE or DashboardForecastAdapter is None:
        msg = "Pipeline-Adapter nicht gefunden – bitte Integration prüfen."
        empty_icicle = go.Figure(); empty_icicle.update_layout(template='plotly_white')
        return (
            _empty_forecast_fig(msg),
            html.Div(msg, className="text-danger"),
            html.Div(),
            empty_icicle,
            dash.no_update,
            dash.no_update
        )

    # kleine Helfer aus deiner Version
    def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
        try:
            df = _parse_store_df(gjson)
            if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                sekt = str(sektor).strip().upper()
                df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            return df.to_json(orient="split", date_format="iso")
        except Exception:
            return gjson

    def _make_metadata_jsonable(obj):
        from datetime import datetime, date
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _make_metadata_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_metadata_jsonable(v) for v in obj]
        return str(obj)

    try:
        # 2) GVB nach Sektor filtern (robust)
        try:
            # funktioniert auch, wenn _filter_gvb_json_by_sektor nicht existiert (NameError -> except)
            gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)  # noqa: F821
        except Exception:
            gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

        # 3) Adapter aufsetzen & Forecast rechnen
        adapter = DashboardForecastAdapter(
            gvb_json_filtered,
            exog_json,
            custom_final_dataset=custom_final_dataset,
        )
        adapter.pipeline_info.update({
            "ui_target": target,
            "use_flows": bool(is_fluss_mode),
            # horizon kann int oder dict sein; hier defensiv casten
            "horizon": int(horizon.get("quarters") if isinstance(horizon, dict) else (horizon or 6)),
            "sektor": sektor_value,
        })

        horizon_quarters = int(horizon.get("quarters") if isinstance(horizon, dict) else (horizon or 6))

        forecast_df, metadata = adapter.run_forecast(
            target=target,
            selected_exog=exog_vars or [],
            horizon=horizon_quarters,
            use_cached=bool(use_cache),
            force_retrain=False,
            use_flows=bool(is_fluss_mode),
            confidence_levels=[80, 95],
            preload_model_path=(model_payload or {}).get('path')
        )

        _logger.info(f"[Forecast] DataFrame-Spalten: {forecast_df.columns.tolist()}")
        if isinstance(metadata, dict):
            _logger.info(f"[Forecast] Metadata keys: {list(metadata.keys())}")
        else:
            _logger.info("[Forecast] Metadata ist kein dict – wird konvertiert")

        # 4) Chart bauen
        fig = _create_pipeline_chart(
            forecast_df=forecast_df,
            metadata=metadata if isinstance(metadata, dict) else {},
            gvb_data=adapter.gvb_data,
            target=target,
            is_fluss_mode=bool(is_fluss_mode),
            horizon_quarters=horizon_quarters,
            show_backtest=bool(show_backtest),
            backtest_mode="overlay",
        )

        # 5) Metriken
        simple = _compute_simple_metrics(metadata if isinstance(metadata, dict) else {})
        _logger.debug(f"Metriken berechnet: MAE={simple.get('mae', 0):.2f}, R²={simple.get('r2', 0):.3f}")

        def _fmt_num(x, nd=2, dash='—'):
            try:
                return dash if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
            except Exception:
                return dash

        def _fmt_pct(x, nd=1, dash='—'):
            try:
                return dash if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}%"
            except Exception:
                return dash

        mae_txt   = _fmt_num(simple.get("mae"), nd=2)
        rmse_txt  = _fmt_num(simple.get("rmse"), nd=2)
        r2_txt    = _fmt_num(simple.get("r2"), nd=3)
        wape_txt  = _fmt_pct(simple.get("wape"), nd=1)
        bias_txt  = _fmt_pct(simple.get("bias_pct"), nd=1)
        smape_txt = _fmt_pct(simple.get("smape"), nd=1)
        dir_txt   = _fmt_pct(simple.get("directional"), nd=0)

        coverage = simple.get("coverage") or {}
        cov80 = _fmt_pct(coverage.get(80), nd=1)
        cov95 = _fmt_pct(coverage.get(95), nd=1)

        # Neue UI Struktur (Anti-Confusion)
        if dbc is not None:
            # Hilfsfunktion für Label mit Tooltip-Icon
            def _lbl(text, tooltip_id, tooltip_text):
                return html.Div([
                    html.Span(text, id=tooltip_id, style={"cursor": "help", "borderBottom": "1px dotted #999"}),
                    dbc.Tooltip(tooltip_text, target=tooltip_id, placement="right"),
                ], className="text-muted small")

            metrics = dbc.Container([
                dbc.Row(
                    dbc.Col(html.Div([
                        _lbl("Relative Abweichung (WAPE)", "tt-wape", 
                             "Durchschnittliche prozentuale Abweichung relativ zum Gesamtvolumen. "
                             "Robust gegen Ausreißer und Nullen."),
                        html.H4(wape_txt, className="mb-0 mt-1 fw-bold text-primary"),
                        html.Div(f"Mittlerer Fehler (absolut): {mae_txt}", className="small text-secondary mt-1"),
                    ])),
                    className="mb-3 pb-3",
                    style={"borderBottom": "1px solid #e9ecef"},
                ),
                dbc.Row(
                    dbc.Col(html.Div([
                        _lbl("Zuverlässigkeit (95% Intervall)", "tt-cov", 
                             "Anteil der echten Datenpunkte, die im berechneten 95%-Sicherheitsbereich lagen. "
                             "Ideal sind 95%."),
                        html.H4(cov95, className="mb-0 mt-1 fw-bold text-dark"),
                        html.Div([
                            html.Span("Tendenz (Bias): ", id="tt-bias", style={"cursor": "help"}),
                            html.Span(bias_txt, className=("text-success" if (simple.get("bias_pct") or 0) < 5 else "text-danger")),
                            dbc.Tooltip("Gibt an, ob das Modell systematisch zu hoch (+) oder zu niedrig (-) schätzt.", target="tt-bias"),
                        ], className="small text-secondary mt-1"),
                    ])),
                    className="mb-3 pb-1",
                ),
                # Details (eingeklappt oder klein)
                dbc.Row(
                    dbc.Col(html.Div([
                        html.Div(f"Modellgüte (R²): {r2_txt}", className="text-muted small", title="Erklärte Varianz"),
                        html.Div(f"Typischer Fehler (RMSE): {rmse_txt}", className="text-muted small", title="Wurzeltausch mittlerer quadratischer Fehler"),
                    ])),
                    className="mt-2 pt-2 border-top",
                )
            ])
        else:
            metrics = html.Div([
                html.Div([html.B("WAPE (Rel. Fehler): "), html.Span(wape_txt), html.Small(f"  | Absolut (MAE): {mae_txt}")]),
                html.Div([html.B("Zuverlässigkeit (95% CI): "), html.Span(cov95)]),
                html.Div([html.B("Bias: "), html.Span(bias_txt)]),
            ], className="p-2")

        # 6) Feature-Importance
        feature_bar = _create_feature_importance(metadata if isinstance(metadata, dict) else {})
        features = (metadata or {}).get('model_complexity', {}).get('top_features', {}) if isinstance(metadata, dict) else {}
        icicle_fig = create_feature_importance_icicle(features, top_n=15)

        # 7) Geo-Forecast für Geo-Analyse pro Run persistieren
        try:
            pipeline_info = getattr(adapter, "pipeline_info", {}) or {}
            run_loader_dir = pipeline_info.get("run_loader_dir")
            run_cache_tag = pipeline_info.get("run_cache_tag")

            if run_loader_dir and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                run_dir = Path(str(run_loader_dir))
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    # Ordner existiert typischerweise bereits aus Step 1 – Fehler hier ignorieren
                    pass

                geo_excel_path = run_dir / "geo_forecast.xlsx"
                try:
                    # identische Struktur wie im Export: FORECAST_TS enthält forecast_df
                    forecast_df.to_excel(geo_excel_path, index=False, sheet_name="FORECAST_TS")
                    _logger.info(f"[Forecast|Geo] Geo-Forecast in {geo_excel_path} geschrieben.")
                except Exception as e_geo_write:
                    _logger.warning(f"[Forecast|Geo] Konnte Geo-Forecast-Excel nicht schreiben: {e_geo_write}")
                else:
                    # run_meta.json im gleichen Run-Ordner anreichern
                    try:
                        meta_files = sorted(run_dir.glob("*_run_meta.json"))
                        if meta_files:
                            meta_path = meta_files[0]
                            try:
                                with meta_path.open("r", encoding="utf-8") as fh:
                                    run_meta = json.load(fh) or {}
                            except Exception:
                                run_meta = {}
                        else:
                            # Fallback: neue Meta-Datei anlegen, falls aus irgendeinem Grund keine existiert
                            meta_path = run_dir / "run_meta.json"
                            run_meta = {}

                        run_meta["geo_forecast_path"] = str(geo_excel_path)
                        # kleine Zusatzinfos für UI / Debug
                        run_meta.setdefault("geo_forecast_info", {})
                        run_meta["geo_forecast_info"].update(
                            {
                                "cache_tag": run_cache_tag,
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        )

                        with meta_path.open("w", encoding="utf-8") as fh:
                            json.dump(run_meta, fh, ensure_ascii=False, indent=2)

                        _logger.info(f"[Forecast|Geo] run_meta mit Geo-Forecast aktualisiert: {meta_path}")
                    except Exception as e_meta:
                        _logger.warning(f"[Forecast|Geo] Konnte run_meta.json für Geo-Forecast nicht aktualisieren: {e_meta}")

                    # Pfad zusätzlich im Metadata ablegen (optional für andere Callbacks)
                    if isinstance(metadata, dict):
                        geo_meta = metadata.setdefault("geo_export", {})
                        geo_meta["run_loader_dir"] = str(run_dir)
                        geo_meta["geo_forecast_path"] = str(geo_excel_path)
        except Exception as e_geo:
            _logger.warning(f"[Forecast|Geo] Fehler beim Persistieren des Geo-Forecasts: {e_geo}")

        # 8) Payload für Export / Stores
        model_path = (metadata or {}).get("model_path") if isinstance(metadata, dict) else None
        snapshot_path = (metadata or {}).get("exog_snapshot_path") if isinstance(metadata, dict) else None
        dash_export_bundle = (metadata or {}).get("dash_export", {}) if isinstance(metadata, dict) else {}

        forecast_json = forecast_df.to_json(orient="split", date_format="iso")
        metadata_jsonable = _make_metadata_jsonable(metadata if isinstance(metadata, dict) else {})

        model_payload_out = {
            "path": model_path,
            "exog_snapshot_path": snapshot_path,
            "forecast_df": forecast_json,
            "metadata": metadata_jsonable,
            "dash_export": dash_export_bundle,
        }

        return (
            fig,
            metrics,
            feature_bar,
            icicle_fig,
            model_payload_out,
            {"has_forecast": True},
        )

    except Exception as e:
        _logger.exception(f"[Forecast] Fehler: {e}")
        empty_icicle = go.Figure(); empty_icicle.update_layout(template='plotly_white')
        fig, error_html, _ = _error_forecast_response(e)
        return (
            fig,
            error_html,
            html.Div(),
            empty_icicle,
            dash.no_update,
            dash.no_update,
        )


# ==============================================================================
# BACKTEST-TOGGLE (Chart neu zeichnen, aber robust)
# ==============================================================================

@app.callback(
    Output('forecast-chart', 'figure', allow_duplicate=True),
    Input('show-backtest-switch', 'value'),
    [
        State('forecast-state-store', 'data'),
        State('gvb-data-store', 'data'),
        State('forecast-target-dropdown', 'value'),
        State('forecast-datenmodus-switch', 'value'),
        State('forecast-horizon-store', 'data'),
        State('exog-data-store', 'data'),
        State('external-exog-dropdown', 'value'),
        State('forecast-sektor-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def toggle_backtest_visualization(
    show_backtest,
    fc_state, gvb_json, target, is_fluss_mode, horizon,
    exog_json, exog_vars, sektor_value
):
    if not fc_state or not fc_state.get('has_forecast'):
        _logger.debug("[Backtest-Toggle] Kein Forecast vorhanden")
        return dash.no_update

    if not gvb_json:
        _logger.warning("[Backtest-Toggle] Keine GVB-Daten")
        return dash.no_update

    def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
        try:
            df = _parse_store_df(gjson)
            if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                sekt = str(sektor).strip().upper()
                df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            return df.to_json(orient="split", date_format="iso")
        except Exception:
            return gjson

    try:
        backtest_mode = "overlay"
        _logger.debug(f"[Backtest-Toggle] show={show_backtest}, sektor={sektor_value}")

        if not HAS_PIPELINE or DashboardForecastAdapter is None:
            _logger.error("[Backtest-Toggle] Pipeline nicht verfügbar")
            return dash.no_update

        try:
            gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)
        except Exception:
            gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

        adapter = DashboardForecastAdapter(
            gvb_store_json=gvb_json_filtered,
            exog_store_json=exog_json,
            custom_final_dataset=custom_final_dataset,  # << NEU
        )
        adapter.pipeline_info.update({
            "ui_target": target,
            "use_flows": bool(is_fluss_mode),
            "horizon": int(horizon or 6),
            "sektor": sektor_value
        })

        forecast_df, metadata = adapter.run_forecast(
            target=target,
            selected_exog=exog_vars or [],
            horizon=horizon or 6,
            use_cached=True,
            force_retrain=False,
            use_flows=bool(is_fluss_mode),
            confidence_levels=[80, 95]
        )

        fig = _create_pipeline_chart(
            forecast_df=forecast_df,
            metadata=metadata,
            gvb_data=adapter.gvb_data,
            target=target or "gesamt",
            is_fluss_mode=bool(is_fluss_mode),
            horizon_quarters=horizon or 6,
            show_backtest=bool(show_backtest),
            backtest_mode=backtest_mode
        )

        _logger.debug("[Backtest-Toggle] Chart aktualisiert")
        return fig

    except Exception as e:
        _logger.exception(f"[Backtest-Toggle] Fehler: {e}")
        return dash.no_update

















# ==============================================================================
# CALLBACK-REGISTRIERUNG
# ==============================================================================

def register_forecaster_callbacks(real_app: "dash.Dash", Log):
    """
    Registriert alle gesammelten Callbacks an der echten Dash-App.
    """
    regs = getattr(app, "_registrations", [])
    _logger.info(f"Registriere {len(regs)} Forecaster-Callbacks")
    
    for args, kwargs, fn in regs:
        real_app.callback(*args, **kwargs)(fn)
    
    _logger.info("Forecaster-Callbacks erfolgreich registriert")
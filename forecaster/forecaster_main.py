# -*- coding: utf-8 -*-
"""
Forecasting-Ansicht / -Callbacks für das Dashboard.
Arbeitet Store-basiert (GVB/EXOG JSON) und integriert die Pipeline bei Bedarf.
"""

from __future__ import annotations

# ============================================================================== 
# IMPORTS
# ==============================================================================

import sys
import os
import io
import json
import re
import unicodedata
import logging
import tempfile
from datetime import date
from pathlib import Path
from textwrap import shorten
from typing import Dict, List, Optional, Tuple

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

INFO_COLOR = GVB_COLORS.get('Einlagen', '#0d6efd')

# Pfade für Presets/Snapshots
PRESETS_DIR = Path("./forecaster/presets")
PRESETS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR = PRESETS_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
HCPRESET_CACHE_FILE = PRESETS_DIR / "hc_presets_cache.json"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

try:
    from app import logger as APP_LOGGER
except Exception:
    APP_LOGGER = None

logger = APP_LOGGER or logging.getLogger("GVB_Dashboard")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | GVB_Dashboard | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Log-Adapter
try:
    from app import Log as _AppLog
    Log = _AppLog
except Exception:
    class Log:
        @staticmethod
        def scenario_table(msg: str): logger.info(msg)
        @staticmethod
        def info(msg: str): logger.info(msg)
        @staticmethod
        def warning(msg: str): logger.warning(msg)
        @staticmethod
        def error(msg: str): logger.error(msg)
        @staticmethod
        def exception(msg: str): logger.exception(msg)

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


# --- PATCH: Helper zum Sektor-Filter (GVB-Store JSON -> gefiltertes JSON) ---
import pandas as pd  # (falls oben noch nicht importiert)

def _filter_gvb_json_by_sektor(gvb_json: str, sektor_value: str) -> str:
    """Filtert den GVB-Store nach 'sektor' (PH/NFK) und gibt JSON (orient='split') zurück.
       Fällt robust zurück (ungefiltert), wenn Spalte fehlt oder sektor_value None/leer ist."""
    try:
        df = _parse_store_df(gvb_json)
        if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor_value:
            sekt = str(sektor_value).strip().upper()
            df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
        return df.to_json(orient="split", date_format="iso")
    except Exception:
        # Safety fallback: liefere das Original zurück, damit der Forecast nicht bricht
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

    return pd.DataFrame()


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

# ==============================================================================
# GVB-DATENVERARBEITUNG
# ==============================================================================

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
    """Findet ecb_database.xlsx."""
    candidates = [
        Path(__file__).parent / "ecbdata" / "ecb_database.xlsx",
        Path.cwd() / "ecbdata" / "ecb_database.xlsx",
        Path.cwd() / "ecb_database.xlsx"
    ]
    
    for p in candidates:
        if p.exists():
            return p
    
    for pattern in ["ecbdata/ecb_*.xlsx", "ecb_*.xlsx"]:
        matches = list(Path.cwd().glob(pattern))
        if matches:
            return matches[0]
    
    return None


def _load_ecb_options() -> List[Dict]:
    """Lädt ECB-Indikatoren-Optionen mit Serien-ID."""
    db_path = _find_ecb_db()
    if not db_path:
        logger.info("[ECB] Keine ecb_database.xlsx gefunden")
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
            logger.info(f"[ECB] Code-Spalte nicht gefunden. Verfügbar: {df.columns.tolist()}")
            return []

        # Qualitäts-Filter
        before_q = len(df)
        if qc_col:
            qc = df[qc_col].astype(str).str.lower()
            if qc.str.contains("sehr|good|ok|passed").any():
                df = df[qc.str.contains("sehr|good|ok|passed", na=False)]
        logger.info(f"[ECB] Qualität-Filter: {before_q} -> {len(df)} Zeilen")

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

        logger.info(f"[ECB] {len(options)} Indikatoren geladen")
        return options

    except Exception as e:
        logger.exception(f"[ECB] Fehler beim Laden: {e}")
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
            logger.info("[Namen-Mapping] ecb_database.xlsx nicht gefunden")
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
            logger.info(f"[Namen-Mapping] Spalten nicht gefunden. Verfügbar: {list(df.columns)}")
            return {}
        
        name_map = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if code and name and code != 'nan' and name != 'nan' and '.' in code:
                name_map[code] = name
        
        logger.info(f"[Namen-Mapping] {len(name_map)} Serien-Namen geladen")
        return name_map
        
    except Exception as e:
        logger.error(f"[Namen-Mapping] Fehler beim Laden: {e}")
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
        logger.error(f"[HC-Presets] Write error: {e}")


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
        
        logger.debug(
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

def _download_exog_codes(ecb_codes: list) -> pd.DataFrame:
    """Lädt ECB-Codes via Loader."""
    Log.scenario_table(
        f"ECB-Download angefordert für {len(ecb_codes)} Codes: "
        f"{ecb_codes[:6]}{' …' if len(ecb_codes) > 6 else ''}"
    )
    
    if not ecb_codes:
        return pd.DataFrame(columns=["date"])
    
    try:
        import yaml
        import loader
        from tempfile import TemporaryDirectory
        
        # Serie-Definitionen mit sicheren Namen
        series_defs = {re.sub(r'[^A-Za-z0-9_]+', '_', c): c for c in ecb_codes}
        
        config = {
            "start_date": "2000-01",
            "end_date": _current_quarter_end(),
            "prefer_cache": True,
            "cache": {
                "cache_dir": "financial_cache", 
                "cache_max_age_days": 60
            },
            "calendar_index": {
                "freq": "MS", 
                "fill": "none"
            },
            "download_timeout_seconds": 30,
            "series_definitions": series_defs,
            "output_path": "output.xlsx"
        }
        
        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            Log.scenario_table(f"Rufe loader.run_from_config mit Config: {cfg_path}")
            df_raw = loader.run_from_config(str(cfg_path))
            
    except Exception as e:
        Log.scenario_table(f"Loader-Fehler: {e}")
        return pd.DataFrame(columns=["date"])

    if df_raw is None or df_raw.empty:
        Log.scenario_table("Loader lieferte leeres Ergebnis")
        return pd.DataFrame(columns=["date"])

    # Datum normalisieren
    if "Datum" in df_raw.columns and "date" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"Datum": "date"})
    df_raw = _normalize_dates(df_raw)

    # Spalten zurückbenennen zu Original-Codes
    inverse = {re.sub(r'[^A-Za-z0-9_]+', '_', c): c for c in ecb_codes}
    ren = {c: inverse.get(c, c) for c in df_raw.columns if c != "date"}
    df_raw = df_raw.rename(columns=ren)

    # Nur angeforderte Spalten behalten
    keep = ["date"] + [c for c in ecb_codes if c in df_raw.columns]
    df = df_raw[keep].copy()

    # Skalierungs-Korrekturen für bekannte Serien-Typen
    for col in df.columns:
        if col == "date":
            continue
        
        mx = df[col].max()
        CU = col.upper()
        
        # Zinsen (ILM, IRS)
        if any(x in CU for x in ["ILM.", "IRS."]):
            if mx > 100:
                Log.scenario_table(f"Skalierung {col}: /100 (max={mx})")
                df[col] = df[col] / 100
            elif mx > 10:
                Log.scenario_table(f"Skalierung {col}: /10 (max={mx})")
                df[col] = df[col] / 10
        
        # Inflation (ICP, HICP)
        elif any(x in CU for x in ["ICP.", "HICP."]):
            if mx > 100:
                Log.scenario_table(f"Skalierung {col}: /100 (max={mx})")
                df[col] = df[col] / 100

    Log.scenario_table(f"ECB-Daten fertig: shape={df.shape}, Spalten={list(df.columns)}")
    return df


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
# EXPORT-FUNKTIONALITÄT
# ==============================================================================

def _make_export_bytes(
    config_df: Optional[pd.DataFrame],
    main_df: Optional[pd.DataFrame],
    current_view_df: Optional[pd.DataFrame],
    exog_df: Optional[pd.DataFrame]
) -> bytes:
    """Erstellt Excel-Export mit allen relevanten Daten."""
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
                "Gesamt_GVB = Einlagen + Wertpapiere + Versicherungen (ohne Kredite)",
                "Netto_GVB  = Gesamt_GVB − Kredite",
                "Zeitraum stammt aus den verfügbaren GVB-Daten (min..max) bzw. deiner Auswahl."
            ]
        })
        readme.to_excel(writer, index=False, sheet_name="README")
        
        # Config & Daten
        cfg_df.to_excel(writer, index=False, sheet_name="CONFIG")
        main_out.to_excel(writer, sheet_name="GVB_MAIN")
        view_out.to_excel(writer, sheet_name="CURRENT_VIEW")
        exog_out.to_excel(writer, sheet_name="EXOG", index=False)

        # Formatierung
        for sheet_name in writer.sheets.keys():
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 0, 18)
            ws.set_column(1, 50, 14)

    bio.seek(0)
    return bio.getvalue()




# ==============================================================================
# NEU: BACKTEST-VISUALISIERUNG (Option 1: Overlay)
# ==============================================================================
# Diese Funktionen HINZUFÜGEN zu forecaster_main.py (nach den BASIS UTILITIES)

def _add_backtest_to_chart(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_backtest: bool = True
) -> go.Figure:
    """
    Fügt historische Forecasts als Overlay hinzu.
    
    Args:
        fig: Bestehende Figure
        backtest_results: DataFrame mit Spalten ['date', 'actual', 'predicted']
        show_backtest: Toggle für Sichtbarkeit
    
    Returns:
        Erweiterte Figure mit Backtest-Trace
    """
    if not show_backtest or backtest_results is None or backtest_results.empty:
        return fig
    
    try:
        # Validierung der erforderlichen Spalten
        required_cols = ['date', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        
        if missing_cols:
            logger.warning(f"[Backtest-Overlay] Fehlende Spalten: {missing_cols}")
            return fig
        
        # Datum sicherstellen
        bt = backtest_results.copy()
        if not pd.api.types.is_datetime64_any_dtype(bt['date']):
            bt['date'] = pd.to_datetime(bt['date'], errors='coerce')
        
        bt = bt.dropna(subset=['date', 'predicted'])
        
        if bt.empty:
            logger.warning("[Backtest-Overlay] Keine gültigen Daten nach Bereinigung")
            return fig
        
        # Backtest als gestrichelte Linie
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
        
        logger.info(f"[Backtest-Overlay] ✓ {len(bt)} historische Vorhersagen hinzugefügt")
    except Exception as e:
        logger.warning(f"[Backtest-Overlay] Konnte Overlay nicht hinzufügen: {e}")
    
    return fig



def _add_backtest_error_band(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_errors: bool = True
) -> go.Figure:
    """
    Fügt Fehler-Band (Differenz zwischen Actual und Predicted) hinzu.
    
    Visualisiert wo das Modell über- oder unterschätzt hat.
    Grün = Unterschätzung, Rot = Überschätzung
    """
    if not show_errors or backtest_results is None or backtest_results.empty:
        return fig
    
    try:
        # Validierung
        required_cols = ['date', 'actual', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        
        if missing_cols:
            logger.warning(f"[Backtest-ErrorBand] Fehlende Spalten: {missing_cols}")
            return fig
        
        df = backtest_results.copy()
        
        # Datum sicherstellen
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date', 'actual', 'predicted'])
        
        if df.empty:
            logger.warning("[Backtest-ErrorBand] Keine gültigen Daten")
            return fig
        
        # Berechne Fehler
        df['error'] = df['actual'] - df['predicted']
        df['upper_error'] = df['actual']
        df['lower_error'] = df['predicted']
        
        # Positive Fehler (Unterschätzung) - grün
        pos_mask = df['error'] > 0
        if pos_mask.any():
            pos_df = df[pos_mask].copy().sort_values('date')
            
            # Band erstellen
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
        
        # Negative Fehler (Überschätzung) - rot
        neg_mask = df['error'] < 0
        if neg_mask.any():
            neg_df = df[neg_mask].copy().sort_values('date')
            
            # Band erstellen
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
        
        logger.info("[Backtest-ErrorBand] ✓ Fehler-Bänder hinzugefügt")
    except Exception as e:
        logger.warning(f"[Backtest-ErrorBand] Konnte Fehler-Bänder nicht hinzufügen: {e}")
    
    return fig


def _add_backtest_markers(
    fig: go.Figure,
    backtest_results: pd.DataFrame,
    show_markers: bool = True
) -> go.Figure:
    """
    Fügt Marker für große Fehler hinzu.
    
    Hebt Zeitpunkte hervor wo die Vorhersage besonders ungenau war.
    """
    if not show_markers or backtest_results is None or backtest_results.empty:
        return fig
    
    try:
        # Validierung
        required_cols = ['date', 'actual', 'predicted']
        missing_cols = [col for col in required_cols if col not in backtest_results.columns]
        
        if missing_cols:
            logger.warning(f"[Backtest-Markers] Fehlende Spalten: {missing_cols}")
            return fig
        
        df = backtest_results.copy()
        
        # Datum sicherstellen
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date', 'actual', 'predicted'])
        
        if df.empty:
            logger.warning("[Backtest-Markers] Keine gültigen Daten")
            return fig
        
        df['error'] = abs(df['actual'] - df['predicted'])
        
        # Nur große Fehler markieren (z.B. > 75. Perzentil)
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
            
            logger.info(f"[Backtest-Markers] ✓ {len(large_errors)} große Fehler markiert")
    except Exception as e:
        logger.warning(f"[Backtest-Markers] Konnte Marker nicht hinzufügen: {e}")
    
    return fig

def create_backtest_controls():
    """
    Minimalisierte Backtest-Controls:
    - Nur Hauptschalter für historische Vorhersagen.
    - Darstellungsart ist fest auf 'overlay' (gestrichelte Linie) gesetzt.
    """
    if dbc is not None:
        # Bootstrap-Variante: nur ein Switch + dezenter Hinweis
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
        # Fallback ohne Bootstrap: Checkbox + Hinweis
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
    """Trennt Code in Basis, Lag, Unit und Transforms."""
    s = str(code).strip()
    parts = s.split('__')
    base_code = parts[0]
    remainder = '__'.join(parts[1:]) if len(parts) > 1 else ''

    # Lag erkennen
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

    # Transforms extrahieren
    transforms = []
    if remainder:
        for token in filter(None, remainder.split('__')):
            if not re.match(r'^lag[-_\s]*\d+[A-Za-z]?$', token, flags=re.IGNORECASE):
                transforms.append(token)

    return base_code, lag_n, lag_unit, transforms


def _fmt_indicator_label(
    code: str, 
    name_map: dict, 
    width: int = 38
) -> Tuple[str, str]:
    """Formatiert Indikator-Label (kurz und voll)."""
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
    """Erstellt Feature-Importance Bar Chart."""
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
    """Erstellt Icicle-Chart für Feature Importance."""
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

    # Top N Features
    items = sorted(features.items(), key=lambda kv: float(kv[1]), reverse=True)[:max(1, int(top_n))]
    codes_children = [str(k) for k, _ in items]
    values_children = [max(0.0, float(v)) for _, v in items]
    total_value = float(sum(values_children)) if values_children else 0.0

    # Labels erstellen
    name_map = _load_ecb_series_names()
    labels_children = []
    hovers_children = []
    
    for code in codes_children:
        short, full = _fmt_indicator_label(code, name_map, width=28)
        labels_children.append(short)
        hovers_children.append(full)

    # Hierarchie aufbauen
    root_label = f"Features (Top {len(labels_children)})"
    labels = [root_label] + labels_children
    parents = [""] + [root_label] * len(labels_children)
    values = [total_value] + values_children

    # Farbpalette
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

    # Icicle erstellen
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
# CHART-ERSTELLUNG
# ==============================================================================

def _empty_forecast_fig(message: str = 'Klicken Sie auf "Prognose erstellen"') -> go.Figure:
    """Erstellt leeren Forecast-Chart."""
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
    """Erstellt Error-Response mit Chart und HTML."""
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

# ==============================================================================
# GEÄNDERT: CHART-ERSTELLUNG MIT BACKTEST-SUPPORT (Legende oben)
# ==============================================================================
# Diese Funktion ERSETZEN in forecaster_main.py
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
    """
    Erstellt Pipeline-Forecast-Chart mit 80% und 95% Konfidenzintervallen (Original-Stil).
    Backtest-Overlay zeigt NUR die vorhergesagte Linie (gestrichelt).
    Legende wird horizontal OBERHALB des Charts dargestellt.
    """
    import numpy as np  # lokal sichern, falls global nicht importiert

    data_type = 'fluss' if is_fluss_mode else 'bestand'
    value_col = 'fluss' if is_fluss_mode else 'bestand'
    y_title = 'Fluss (Mrd EUR)' if is_fluss_mode else 'Bestand (Mrd EUR)'

    # ------------------------- Historie vorbereiten -------------------------
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
            # Duplikate auf Datum korrekt aggregieren: SUMME
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

    # ------------------------- Forecast vorbereiten -------------------------
    fc = forecast_df.copy() if isinstance(forecast_df, pd.DataFrame) else pd.DataFrame()
    fc_date = None
    
    # Datums-Spalte finden
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

    # Forecast-Wert-Spalte finden
    y_candidates = ['yhat', 'Forecast', 'forecast', 'y_pred', 'yhat_mean', 'pred', 'value']
    fc_val = next((c for c in y_candidates if c in fc.columns), None)

    # Konfidenzband-Spalten definieren
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

    # Forecast bereinigen
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

    # ------------------------- X-Achsen-Bereich -------------------------
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
            logger.debug(f"[Chart] X-Achse fixiert: {x_range[0]} bis {x_range[1]}")
        except Exception as e:
            logger.warning(f"[Chart] Konnte X-Achsen-Bereich nicht berechnen: {e}")
            x_range = None

    # ------------------------- Chart erstellen -------------------------
    fig = go.Figure()

    # Historie
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

    # CI-Bänder zuerst (damit Linien oben liegen)
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

    # Forecast-Linie
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

    # ------------------------- Backtest-Overlay (nur Vorhersage) -------------------------
    if show_backtest and isinstance(metadata, dict):
        backtest_data = metadata.get('backtest_results')
        if backtest_data is not None and not isinstance(backtest_data, pd.DataFrame):
            try:
                backtest_data = pd.DataFrame(backtest_data)
            except Exception as e:
                logger.warning(f"[Chart] Konnte Backtest-Daten nicht in DataFrame konvertieren: {e}")
                backtest_data = None
        
        if isinstance(backtest_data, pd.DataFrame) and not backtest_data.empty:
            bt = backtest_data.copy()
            if 'date' in bt.columns:
                bt['date'] = pd.to_datetime(bt['date'], errors='coerce')
            else:
                logger.info("[Chart] Backtest ohne 'date' Spalte – überspringe Overlay")
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

    # ------------------------- Layout & Legende (oben, horizontal) -------------------------
    title_target = target if target != 'gesamt' else 'Gesamt'

    # Legende OBERHALB des Plots (zentriert)
    legend_cfg = dict(
        title='',
        orientation='h',
        yanchor='bottom',
        y=1.02,            # knapp über dem Plotbereich
        xanchor='center',
        x=0.5,
        traceorder='normal',
        bgcolor='rgba(255,255,255,0.6)'
    )

    # Mehr Kopfzeilen-Padding für die obere Legende
    layout_config = {
        'template': 'plotly_white',
        # 'title': f'Forecast – {title_target}',
        'xaxis_title': 'Datum',
        'yaxis_title': y_title,
        'legend': legend_cfg,
        'hovermode': 'x unified',
        'height': 500,
        'margin': dict(l=50, r=20, t=80, b=40)  # -> t erhöht, damit die Legende Platz hat
    }
    if x_range is not None:
        layout_config['xaxis'] = {'range': x_range, 'fixedrange': False}
    fig.update_layout(**layout_config)

    # ------------------------- Cutoff-Linie & Y-Achse fixieren -------------------------
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

        # Cutoff-Linie (Trennung Historie/Forecast)
        if not hist.empty:
            cutoff = hist['date'].max()
            yaxis_range = fig.layout.yaxis.range
            if yaxis_range:
                y0, y1 = yaxis_range
            else:
                y0, y1 = (y_min - padding, y_max + padding) if y_series else (0, 1)
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
        logger.warning(f"[Chart] Konnte Y-Achse/Cutoff nicht finalisieren: {e}")

    # Sicherheitshalber: überall lineare Linien durchsetzen (gegen Templates/Defaults)
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
    """ECB-Optionen + lokale Spalten aus Store."""
    ecb_opts = _load_ecb_options() or []
    ecb_by_val = {opt.get("value"): opt for opt in ecb_opts if opt.get("value")}

    options_map = {}
    
    # Lokale Spalten aus Store
    exog_df = _safe_load_store(exog_json)
    if isinstance(exog_df, pd.DataFrame) and not exog_df.empty:
        local_cols = [c for c in exog_df.columns if str(c).lower() != "date"]
        for col in sorted(local_cols):
            v = col
            if v not in ecb_by_val:
                options_map[v] = {"label": f"[Lokal] {str(col)[:50]}", "value": v}

    # ECB-Optionen hinzufügen
    for v, opt in ecb_by_val.items():
        options_map[v] = opt

    # Aktuell ausgewählte Werte
    if current_selection:
        existing_values = set(options_map.keys())
        wanted = current_selection if isinstance(current_selection, list) else [current_selection]
        for val in wanted:
            if val not in existing_values:
                options_map[val] = {"label": f"[Ausgewählt] {str(val)[:50]}", "value": val}

    # Sortierung
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
    """Toast-Benachrichtigung beim Hinzufügen von Exog-Variablen."""
    if not n_clicks:
        return True, "Bitte wählen Sie mindestens eine Datenreihe oder geben Sie eine Serien ID ein.", "Keine Auswahl", "danger", no_update

    # Auswahl sammeln
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

    # Duplikate entfernen
    selections = [s for s in selections if s]
    unique_selections = []
    seen = set()
    for s in selections:
        if s not in seen:
            unique_selections.append(s)
            seen.add(s)

    if not unique_selections:
        return True, "Bitte wählen Sie mindestens eine Datenreihe oder geben Sie eine Serien ID ein.", "Keine Auswahl", "danger", no_update

    # Vorhandene Spalten prüfen
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
                cols = obj.get("columns", []) or []
                return set([str(c) for c in cols])
            
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                keys = set()
                for rec in obj:
                    keys.update(list(rec.keys()))
                return set([str(k) for k in keys])
        except Exception:
            pass
        return set()

    existing_cols = _extract_existing_columns(exog_store_json)
    to_add = [v for v in unique_selections if v not in existing_cols]
    duplicates = [v for v in unique_selections if v in existing_cols]

    # Toast-Nachricht erstellen
    if to_add and not duplicates:
        msg = html.Div([
            html.Div("Die folgenden Datenreihen werden hinzugefügt:"),
            html.Ul([html.Li(v) for v in to_add], className="mb-0")
        ])
        return True, msg, "Exogene hinzugefügt", "success", ""

    if to_add and duplicates:
        msg = html.Div([
            html.Div("Einige Reihen sind bereits vorhanden und werden übersprungen:"),
            html.Ul([html.Li(v) for v in duplicates]),
            html.Hr(className="my-2"),
            html.Div("Neu hinzugefügt:"),
            html.Ul([html.Li(v) for v in to_add], className="mb-0"),
        ])
        return True, msg, "Teilweise hinzugefügt", "warning", ""

    msg = html.Div([
        html.Div("Alle ausgewählten Reihen sind bereits vorhanden."),
        html.Ul([html.Li(v) for v in duplicates], className="mb-0")
    ])
    return True, msg, "Bereits vorhanden", "warning", no_update


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
    """Formatiert Horizon-Display."""
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



from typing import Any, Dict, List, Optional
import dash
from dash import html, no_update
from dash.exceptions import PreventUpdate
import logging
import time
import os

# Logger
try:
    from app import logger as APP_LOGGER
    logger = APP_LOGGER
except Exception:
    logger = logging.getLogger("GVB_Dashboard")


# ==========================================================================
# CALLBACK 1: DROPDOWN-OPTIONEN BEFÜLLEN
# ==========================================================================
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
    """
    Befüllt das Preset-Dropdown mit H&C-Presets und User-Presets.
    
    H&C-Presets werden nach aktuellem Target gefiltert (ein Preset pro Target).
    User-Presets werden IMMER frisch von Disk geladen.
    
    Returns:
        (options, updated_store): Dropdown-Optionen und synchronisierter Store
    """
    try:
        # Importiere Helper-Funktionen aus app.py
        from app import (
            _load_user_presets_from_disk,
            merge_hc_and_user_presets_for_dropdown,
            _normalize_target_slug,
        )
        
        # Importiere ECB-Preset-Funktion (falls vorhanden)
        try:
            from forecaster_main import get_ecb_presets_hydrated
        except ImportError:
            logger.warning("[PresetDropdown] get_ecb_presets_hydrated nicht verfügbar")
            def get_ecb_presets_hydrated():
                return {}
        
        # ------------------------------------------------------------------
        # 1) H&C-Presets laden (ein Preset je Target)
        # ------------------------------------------------------------------
        hc_dict: Dict[str, Dict[str, Any]] = {}
        try:
            all_hc_presets = get_ecb_presets_hydrated()
            target_slug = _normalize_target_slug(target_value)
            
            if target_slug and target_slug in all_hc_presets:
                meta = all_hc_presets[target_slug]
                title = meta.get("title") or f"H&C {target_slug}"
                hc_dict[title] = {"id": target_slug}
                logger.info(f"[PresetDropdown] H&C-Preset geladen: {title}")
        except Exception as e:
            logger.warning(f"[PresetDropdown] H&C-Presets nicht verfügbar: {e}")
        
        # ------------------------------------------------------------------
        # 2) User-Presets von Disk laden (Single Source of Truth)
        # ------------------------------------------------------------------
        user_dict_disk = _load_user_presets_from_disk()
        logger.info(f"[PresetDropdown] User-Presets von Disk: {len(user_dict_disk)} Stück")
        
        # ------------------------------------------------------------------
        # 3) Dropdown-Optionen generieren
        # ------------------------------------------------------------------
        options = merge_hc_and_user_presets_for_dropdown(
            hc_presets=hc_dict,
            user_presets=user_dict_disk,
            hc_label="H&C Presets",
            user_label="Eigene Presets",
        )
        
        # "kein Preset"-Option voranstellen
        options.insert(0, {"label": "– kein Preset –", "value": "__none__"})
        
        logger.info(f"[PresetDropdown] Optionen generiert: {len(options)} Einträge")
        
        # Store mit aktuellen Disk-Daten synchronisieren
        return options, user_dict_disk
    
    except Exception as e:
        logger.exception(f"[PresetDropdown] Kritischer Fehler: {e}")
        # Fallback: nur "kein Preset"-Option, Store unverändert
        return [{"label": "– kein Preset –", "value": "__none__"}], no_update



# ==========================================================================
# CALLBACK 3: LOAD-BUTTON AKTIVIEREN/DEAKTIVIEREN
# ==========================================================================
@app.callback(
    dash.Output("load-preset-btn", "disabled"),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=False
)
def toggle_load_preset_button(preset_value):
    # Button deaktivieren, wenn kein Preset oder "__none__" ausgewählt
    return (not preset_value) or (preset_value == "__none__")



# ==========================================================================
# CALLBACK 4: PRESET AUF MODEL-STORE ANWENDEN
# ==========================================================================
@app.callback(
    dash.Output("model-artifact-store", "data", allow_duplicate=True),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_preset_to_model_store(selected_value: Optional[str]):
    """
    Wendet das ausgewählte Preset auf den Model-Artifact-Store an.
    
    Nur relevant für User-Presets, die ein gespeichertes Modell haben.
    H&C-Presets haben in der Regel keine model_path.
    
    Args:
        selected_value: Wert des ausgewählten Presets (z.B. "user_xyz")
    
    Returns:
        dict: {"path": model_path, "exog_snapshot_path": snapshot_path}
    
    Raises:
        PreventUpdate: Wenn kein Preset ausgewählt oder kein Model vorhanden
    """
    if not selected_value or selected_value == "__none__":
        raise PreventUpdate
    
    try:
        from app import _load_user_presets_from_disk
        
        # Nur für User-Presets relevant (nicht für H&C-Presets)
        if not selected_value.startswith("user_"):
            logger.info("[ApplyPreset] Kein User-Preset → Model-Store unverändert")
            raise PreventUpdate
        
        # ID extrahieren (entferne "user_" Präfix)
        user_id = selected_value[5:]
        
        # Lade alle User-Presets von Disk
        user_dict = _load_user_presets_from_disk()
        
        # Suche nach Preset mit passender ID
        for display_name, meta in user_dict.items():
            if not isinstance(meta, dict):
                continue
            
            if meta.get("id") == user_id:
                model_path = meta.get("model_path")
                snapshot_path = meta.get("exog_snapshot_path")
                
                # Nur aktualisieren, wenn mindestens einer der Pfade existiert
                if model_path or snapshot_path:
                    # Prüfe ob Dateien existieren
                    model_exists = model_path and os.path.exists(model_path)
                    snapshot_exists = snapshot_path and os.path.exists(snapshot_path)
                    
                    logger.info(
                        f"[ApplyPreset] Preset '{display_name}' geladen | "
                        f"Model: {model_exists} | Snapshot: {snapshot_exists}"
                    )
                    
                    return {
                        "path": model_path if model_exists else None,
                        "exog_snapshot_path": snapshot_path if snapshot_exists else None
                    }
        
        # Preset-ID nicht gefunden
        logger.warning(f"[ApplyPreset] Preset-ID '{user_id}' nicht in Disk-Presets gefunden")
        raise PreventUpdate
    
    except PreventUpdate:
        raise
    except Exception as e:
        logger.warning(f"[ApplyPreset] Fehler beim Anwenden: {e}")
        raise PreventUpdate




from dash.dependencies import Input, Output, State, ALL

# @app.callback(
#     Output("forecast-horizon-store", "data"),
#     Input({"type":"horizon-btn","value": ALL}, "n_clicks"),
#     State("forecast-horizon-store", "data"),
#     prevent_initial_call=True
# )
# def set_horizon(n_clicks_list, current):
#     if not n_clicks_list or all((c or 0) == 0 for c in n_clicks_list):
#         raise dash.exceptions.PreventUpdate
#     values = [2,4,6]
#     idx = max(range(len(n_clicks_list)), key=lambda i: n_clicks_list[i] or 0)
#     return values[idx]


# ==============================================================================
# ENDE DER FUNKTION
# ==============================================================================


# -*- coding: utf-8 -*-
"""
FEHLENDE CALLBACKS für Modal-basierten Preset-Workflow

PROBLEM: Der save-preset-btn soll ein Modal öffnen, nicht direkt speichern!
LÖSUNG: 2 zusätzliche Callbacks hinzufügen

ANWEISUNG: Füge diese Callbacks in forecaster_main.py NACH den vorhandenen 
           Preset-Callbacks ein (nach Zeile ~2300)
"""

import dash
from dash import html, no_update
from dash.exceptions import PreventUpdate
from typing import Optional, List, Dict, Any
import time

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
    # NEU: Sektor & Modus in der Vorschau anzeigen
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
        # hübsche Labels
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



# ==========================================================================
# CALLBACK 2: PRESET MIT NAMEN SPEICHERN
# ==========================================================================
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
    State("forecast-horizon-store", "data"),   # <- einheitlich: STORE!
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
        
        logger = logging.getLogger("GVB_Dashboard")
        logger.info("[SavePresetModal] Speichervorgang gestartet...")
        
        # ------------------------------------------------------------------
        # 1) Name validieren
        # ------------------------------------------------------------------
        if not preset_name or not preset_name.strip():
            # Kein Name eingegeben → Auto-Name generieren
            target = str(target_value or "Preset").strip()
            exogs = list(exog_values or [])
            horizon = int(horizon_quarters or 0)
            ts = time.strftime("%Y-%m-%d %H:%M")
            preset_name = f"{target} | {len(exogs)} Exog | {horizon}Q @ {ts}"
            logger.info(f"[SavePresetModal] Kein Name → Auto-Name: {preset_name}")
        else:
            preset_name = preset_name.strip()
            logger.info(f"[SavePresetModal] User-Name: {preset_name}")
        
        # ------------------------------------------------------------------
        # 2) Eingaben bereinigen
        # ------------------------------------------------------------------
        target = str(target_value or "Preset").strip()
        exogs = list(exog_values or [])
        
        try:
            horizon = int(horizon_quarters or 0)
        except (TypeError, ValueError):
            horizon = 0
        
        logger.info(f"[SavePresetModal] Target={target}, Exogs={len(exogs)}, Horizon={horizon}Q")
        
        # ------------------------------------------------------------------
        # 3) Preset-Objekt erstellen
        # ------------------------------------------------------------------
        preset_obj = create_user_preset_from_ui_state(
            name=preset_name,
            target=target,
            exog=exogs,
            horizon=horizon,
            is_fluss_mode=bool(is_fluss_mode),
            model_payload=(model_payload or {}),
            extra_ui_options={},
        )
        
        logger.info(f"[SavePresetModal] Preset-Objekt erstellt mit ID: {preset_obj.get('id')}")
        
        # ------------------------------------------------------------------
        # 4) Auf Disk persistieren
        # ------------------------------------------------------------------
        all_presets = upsert_user_preset(preset_name, preset_obj)
        
        logger.info(f"[SavePresetModal] Erfolgreich gespeichert. Gesamt: {len(all_presets)} Presets")
        
        # ------------------------------------------------------------------
        # 5) Dropdown-Value für neues Preset
        # ------------------------------------------------------------------
        new_dropdown_value = f"user_{preset_obj['id']}"
        
        # ------------------------------------------------------------------
        # 6) Toast-Benachrichtigung
        # ------------------------------------------------------------------
        toast_children = html.Div([
            html.Div("✅ Preset erfolgreich gespeichert!", className="fw-bold"),
            html.Small(preset_name, className="text-muted d-block mt-1")
        ])
        
        logger.info(f"[SavePresetModal] ✓ Erfolgreich: {preset_name}")
        
        # Name-Input leeren für nächstes Mal
        return all_presets, new_dropdown_value, True, toast_children, ""
    
    except Exception as e:
        logger = logging.getLogger("GVB_Dashboard")
        logger.exception(f"[SavePresetModal] ✗ Fehler beim Speichern: {e}")
        
        # Fehler-Toast
        error_toast = html.Div([
            html.Div("❌ Speichern fehlgeschlagen", className="fw-bold text-danger"),
            html.Small(str(e), className="text-danger d-block mt-1 font-monospace")
        ])
        
        return no_update, no_update, True, error_toast, no_update



# ==============================================================================
# CALLBACKS - PRESET DELETION
# ==============================================================================


from dash import no_update
from dash.exceptions import PreventUpdate
import dash


@app.callback(
    dash.Output("delete-preset-btn", "disabled"),
    dash.Input("forecast-preset-dropdown", "value"),
    prevent_initial_call=False
)
def toggle_delete_button(value):
    # Button aus, wenn kein Preset oder Platzhalter
    if not value or value == "__none__":
        return True
    # Nur User-Presets löschbar (Konvention: value beginnt mit "user_")
    return not str(value).startswith("user_")


from dash import html, no_update
from dash.exceptions import PreventUpdate


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

    # 1) Aus Store entfernen
    pid = str(dropdown_value).replace("user_", "", 1)
    updated = dict(user_presets or {})
    key_to_remove = None
    for disp_name, meta in (updated or {}).items():
        if str((meta or {}).get("id")) == pid or disp_name == pid:
            key_to_remove = disp_name
            break
    if key_to_remove:
        updated.pop(key_to_remove, None)

    # 2) Optional: Persistenz auf Disk (falls Helper vorhanden)
    try:
        from app import delete_user_preset  # wenn du die Helper-Funktion eingebaut hast
        updated = delete_user_preset(preset_identifier=pid, delete_files=False) or updated
    except Exception:
        # Kein Persist-Helper vorhanden → nur Store wird aktualisiert
        pass

    # 3) Toast
    toast_children = html.Div([html.Div("🗑️ Preset gelöscht", className="fw-bold")])

    # Rückgaben: aktualisierter Store, Dropdown reset, Toast an, Toast-Inhalt
    return updated, "__none__", True, toast_children




# ==============================================================================
# CALLBACKS - PRESET LOADING
# ==============================================================================

@app.callback(
    Output("external-exog-dropdown", "value", allow_duplicate=True),
    Input("forecast-preset-dropdown", "value"),
    State("forecast-target-dropdown", "value"),
    prevent_initial_call=True
)
def apply_preset_to_external_exog(preset_value, target_value):
    """Wendet Preset auf Exog-Dropdown an."""
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
    # NEU: Sektor aus Preset zurück in die UI schreiben
    Output("forecast-sektor-dropdown", "value", allow_duplicate=True),
    Input("load-preset-btn", "n_clicks"),
    State("forecast-preset-dropdown", "value"),
    State("user-presets-store", "data"),
    prevent_initial_call=True
)
def load_selected_preset(n_clicks, preset_value, user_presets):
    """Lädt ausgewähltes Preset vollständig (inkl. Sektor, falls vorhanden)."""
    if not n_clicks or not preset_value or preset_value == "__none__":
        raise dash.exceptions.PreventUpdate

    import os

    presets = get_ecb_presets_hydrated()
    target = dash.no_update
    exogs = dash.no_update
    model_payload = dash.no_update
    exog_store_json = dash.no_update
    sektor_val = dash.no_update  # <- wird gesetzt, wenn im Preset vorhanden

    # ---------- ECB-Preset ----------
    if str(preset_value).startswith("preset_"):
        slug = str(preset_value).replace("preset_", "", 1)
        p = presets.get(slug)
        if not p:
            raise dash.exceptions.PreventUpdate

        # Zielvariable
        target = p.get("target") or dash.no_update

        # Exogs (können als dict {name:code} vorliegen)
        exog_dict = p.get("exog")
        exogs = list(exog_dict.values()) if isinstance(exog_dict, dict) else (exog_dict or [])

        # Sektor (falls im Preset mitgegeben: ui_opts.sektor oder direkt sektor)
        ui_opts = p.get("ui_opts") or {}
        sektor_val = (ui_opts.get("sektor") if isinstance(ui_opts, dict) else None) or p.get("sektor") or dash.no_update

        # Pfade (Modell & Snapshot)
        data_path = p.get("final_dataset_path") or p.get("exog_snapshot_path")
        mdl_path = p.get("model_path")

        if mdl_path or data_path:
            model_payload = {"path": mdl_path, "exog_snapshot_path": data_path}

        if data_path and os.path.exists(data_path):
            exog_store_json = _snapshot_to_store_json(data_path, exogs if isinstance(exogs, list) else [])

        return target, exogs, model_payload, exog_store_json, sektor_val

    # ---------- User-Preset ----------
    if str(preset_value).startswith("user_"):
        pid = str(preset_value).replace("user_", "", 1)
        chosen = None
        for name, meta in (user_presets or {}).items():
            if str(meta.get("id")) == str(pid):
                chosen = meta
                break

        if not chosen:
            raise dash.exceptions.PreventUpdate

        # Zielvariable & Exogs
        target = chosen.get("target")
        exogs = chosen.get("exog") or []

        # Sektor aus ui_opts, falls gespeichert (siehe extra_ui_options={"sektor": ...})
        ui_opts = chosen.get("ui_opts") or {}
        sektor_val = ui_opts.get("sektor") if isinstance(ui_opts, dict) else None
        if not sektor_val:
            sektor_val = chosen.get("sektor")  # Fallback, falls anders gespeichert
        if not sektor_val:
            sektor_val = dash.no_update

        # Pfade (Modell & Snapshot)
        data_path = chosen.get("final_dataset_path") or chosen.get("exog_snapshot_path")
        mdl_path = chosen.get("model_path")

        if mdl_path or data_path:
            model_payload = {"path": mdl_path, "exog_snapshot_path": data_path}

        if data_path and os.path.exists(data_path):
            exog_store_json = _snapshot_to_store_json(data_path, exogs if isinstance(exogs, list) else [])

        return target, exogs, model_payload, exog_store_json, sektor_val

    raise dash.exceptions.PreventUpdate

# ==============================================================================
# CALLBACKS - EXPORT
# ==============================================================================
@app.callback(
    Output("download-rawdata", "data"),
    Input("export-rawdata-btn", "n_clicks"),
    State("forecast-target-dropdown", "value"),
    State("forecast-horizon-store", "data"),
    State("external-exog-dropdown", "value"),
    State("manual-series-input", "value"),
    State("forecast-datenmodus-switch","value"),
    State("gvb-data-store", "data"),
    State("exog-data-store", "data"),
    State("model-artifact-store", "data"),
    # NEU: gewählter Sektor (PH/NFK)
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
    """Exportiert Forecast-Rohdaten als Excel – inklusive Sektor-Filter (PH/NFK) & Modus (Bestand/Fluss)."""
    if not n_clicks:
        return no_update

    import pandas as pd
    import numpy as np

    # Modus ableiten
    modus = "fluss" if bool(forecast_real_switch_value) else "bestand"

    # --- Sektor-Filter (GVB-Store) ---
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
        gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)  # nutzt globalen Helper, falls vorhanden
    except Exception:
        gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

    # Config erstellen (inkl. Sektor & Modus)
    cfg_rows = [{
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
        "forecast_real_switch": forecast_real_switch_value if forecast_real_switch_value is not None else "(n/a)"
    }]
    config_df = pd.DataFrame(cfg_rows)

    # GVB-Daten laden (bereits sektorgefiltert)
    gvb_df = _parse_store_df(gvb_json_filtered)
    if gvb_df.empty or "date" not in gvb_df.columns:
        config_df.loc[0, "status"] = "Fehler"
        config_df.loc[0, "msg"] = "GVB-Daten fehlen (nach Sektor-Filter) oder 'date' ist nicht vorhanden."
        xbytes = _make_export_bytes(config_df, pd.DataFrame(), pd.DataFrame(), _parse_store_df(exog_json))
        return dcc.send_bytes(lambda b: b.write(xbytes), "forecast_export_error.xlsx")

    # Spalten sicherstellen
    for c in ("ebene1", "ebene2", "ebene3", "bestand", "fluss", "sektor"):
        if c not in gvb_df.columns:
            gvb_df[c] = np.nan

    # Datum normalisieren
    gvb_df["date"] = pd.to_datetime(gvb_df["date"], errors="coerce")
    gvb_df = gvb_df.dropna(subset=["date"]).sort_values("date")
    
    if gvb_df.empty:
        config_df.loc[0, "status"] = "Fehler"
        config_df.loc[0, "msg"] = "Alle Datumswerte sind ungültig (nach Sektor-Filter)."
        xbytes = _make_export_bytes(config_df, pd.DataFrame(), pd.DataFrame(), _parse_store_df(exog_json))
        return dcc.send_bytes(lambda b: b.write(xbytes), "forecast_export_error.xlsx")

    # Zeitraum bestimmen
    start_date = gvb_df["date"].min().date()
    end_date = gvb_df["date"].max().date()
    config_df.loc[0, "start_date"] = str(start_date)
    config_df.loc[0, "end_date"] = str(end_date)

    # Haupttabelle erstellen (mit Modus & Sektor)
    main_df = _build_main_e1_table_from_store(
        gvb_df,
        data_type=modus,                   # "bestand" oder "fluss"
        start_date=str(start_date),
        end_date=str(end_date),
        smoothing=1,
        use_log=False,
        sektor=sektor_value               # <- NEU: Sektor durchreichen
    )

    # Current View erstellen (je nach Modus auf 'bestand' oder 'fluss' aggregieren)
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

    # Exog-Daten laden (unverändert)
    exog_df = _parse_store_df(exog_json)
    if not exog_df.empty:
        for c in ["date", "Date", "DATE", "time", "Time"]:
            if c in exog_df.columns:
                if c != "date":
                    exog_df = exog_df.rename(columns={c: "date"})
                cols = ["date"] + [x for x in exog_df.columns if x != "date"]
                exog_df = exog_df[cols]
                break

    # Export erstellen
    xbytes = _make_export_bytes(
        config_df=config_df,
        main_df=main_df,
        current_view_df=current_view_df,
        exog_df=exog_df
    )
    
    filename = f"forecast_export_{sektor_value or 'NA'}_{modus}_{pd.Timestamp.today().strftime('%Y-%m-%d')}.xlsx"
    return dcc.send_bytes(lambda b: b.write(xbytes), filename)

# ==============================================================================
# CALLBACKS - MANUELLE SERIEN & ECB DOWNLOAD
# ==============================================================================

@app.callback(
    Output("external-exog-dropdown", "value", allow_duplicate=True),
    Input("add-manual-series-btn", "n_clicks"),
    State("manual-series-input", "value"),
    State("external-exog-dropdown", "value"),
    prevent_initial_call=True
)
def add_manual_series(n_clicks, manual_input, current_selection):
    """
    Fügt eine manuell eingegebene ECB-Serien-ID zur aktuellen Auswahl hinzu.
    """
    if not n_clicks or not manual_input:
        return dash.no_update

    manual_input = str(manual_input).strip()
    if not _looks_like_ecb_code(manual_input):
        return dash.no_update

    current = current_selection or []
    if not isinstance(current, list):
        current = [current]

    if manual_input not in current:
        current.append(manual_input)

    return current


@app.callback(
    Output("exog-data-store", "data"),
    Input("create-forecast-btn", "n_clicks"),
    State("external-exog-dropdown", "value"),
    State("exog-data-store", "data"),
    prevent_initial_call=True
)
def download_and_merge_exog(n_clicks, selected_codes, current_store):
    """
    Lädt fehlende ECB-Serien (nur die, die im Store noch nicht vorhanden sind),
    korrigiert offensichtliche Skalen und merged sie in den exog-Store.
    """
    if not n_clicks or not selected_codes:
        return dash.no_update

    # Auswahl vereinheitlichen
    if not isinstance(selected_codes, list):
        selected_codes = [selected_codes]

    # Bereits vorhandene Daten
    df_existing = _safe_load_store(current_store)
    if df_existing is not None:
        df_existing = _normalize_dates(df_existing)

    existing_cols = set()
    if df_existing is not None and not df_existing.empty:
        existing_cols = {c for c in df_existing.columns if c.lower() != "date"}

    # Nur fehlende ECB-Codes laden
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

    # ECB-Daten laden
    try:
        df_downloaded = _download_exog_codes(to_download)
        if df_downloaded is None or df_downloaded.empty:
            return dash.no_update
    except Exception:
        return dash.no_update

    # Merge in bestehenden Store
    if df_existing is None or df_existing.empty:
        df_result = df_downloaded.copy()
    else:
        df_result = pd.merge(
            df_existing, df_downloaded, on="date", how="outer", suffixes=("", "__new")
        )
        
        # Neue Spalten übernehmen
        for col in df_downloaded.columns:
            if col == "date":
                continue
            new_col = f"{col}__new"
            if new_col in df_result.columns:
                df_result[col] = df_result[new_col].combine_first(df_result.get(col))
                df_result.drop(columns=[new_col], inplace=True)
        
        df_result = df_result.sort_values("date").reset_index(drop=True)

    # Nur die tatsächlich gewählten Codes
    keep = ["date"]
    for code in selected_codes:
        if code in df_result.columns:
            keep.append(code)
        else:
            # Unterstrich-Variante ggf. rückbenennen
            norm = re.sub(r'[^A-Za-z0-9_]+', '_', str(code))
            if norm in df_result.columns:
                df_result = df_result.rename(columns={norm: code})
                keep.append(code)

    df_final = df_result[[c for c in keep if c in df_result.columns]]
    return df_final.to_json(orient="split", date_format="iso")

# ==============================================================================
# CALLBACKS - PREWARM (H&C Presets)
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
    """Prewarm: H&C Presets vorbereiten und cachen."""
    if not n_clicks:
        raise PreventUpdate

    def _log(msg):
        try:
            Log.scenario_table(msg)
        except Exception:
            try:
                logger.info(msg)
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

    # Presets laden
    try:
        presets = get_ecb_presets_hydrated()
    except Exception as e:
        _log(f"get_ecb_presets_hydrated() Fehler: {e}")
        presets = {}

    _log(f"Anzahl Presets geladen: {len(presets)} → Slugs: {list(presets.keys()) if isinstance(presets, dict) else '(n/a)'}")
    
    if not presets:
        return no_update, True, "⚠️ Keine H&C Presets gefunden"

    # Cache laden
    try:
        cache = _load_hc_preset_cache()
    except Exception as e:
        _log(f"Cache laden fehlgeschlagen, starte leer. Grund: {e}")
        cache = {}
    
    _log(f"Aktueller Cache-Keys: {list(cache.keys())}")

    updated, skipped, failed = {}, 0, 0

    # Presets verarbeiten
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

        # Exog-Daten mergen
        try:
            df_exog, have = _merge_exogs_from_sources(exogs, exog_store_json)
        except Exception as e:
            _log(f"Merge-Fehler: {e}")
            failed += 1
            continue

        if df_exog.empty or not have:
            skipped += 1
            continue

        # Final Dataset schreiben
        try:
            final_path = _write_final_dataset(df_exog, slug)
        except Exception as e:
            _log(f"final_dataset write error: {e}")
            failed += 1
            continue

        if not os.path.exists(final_path):
            failed += 1
            continue

        # Optional: Pipeline-Training
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

        # Cache-Eintrag erstellen
        updated_entry = {
            "exog_snapshot_path": final_path,
            "final_dataset_path": final_path
        }
        if model_path:
            updated_entry["model_path"] = model_path

        updated[slug] = updated_entry

    # Cache speichern
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
# GEÄNDERT: FORECAST CALLBACKS MIT BACKTEST-SUPPORT
# ==============================================================================
# Diese beiden Callbacks ERSETZEN in forecaster_main.py
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
        # NEU: gewählter Sektor (PH/NFK)
        State("forecast-sektor-dropdown", "value"),
    ],
    prevent_initial_call='initial_duplicate',
)
def show_initial_forecast_history(
    pathname, target, is_fluss_mode,
    gvb_json, fc_state, horizon, sektor_value
    ):
        """
        Zeigt historische Daten beim ersten Laden mit reserviertem Platz für Forecast.
        Berücksichtigt den gewählten Sektor (PH/NFK), indem der GVB-Store vorab gefiltert wird.
        X-Achse bleibt stabil, wenn später Forecast hinzugefügt wird.
        Backtest-Einstellungen: initial immer aus, Modus fest auf 'overlay'.
        """
        import pandas as pd
        from dash.exceptions import PreventUpdate

        # Fallback-Filter, falls der globale Helper nicht importiert/definiert ist
        def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
            try:
                df = _parse_store_df(gjson)  # vorhandener App-Helper
                if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                    sekt = str(sektor).strip().upper()
                    df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
                return df.to_json(orient="split", date_format="iso")
            except Exception:
                return gjson

        if pathname != "/forecast":
            raise PreventUpdate

        ctx = dash.callback_context
        if ctx.triggered:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
            # Wenn bereits eine Prognose existiert und das Laden nur durch URL kam → keine Neu-Zeichnung
            if isinstance(fc_state, dict) and fc_state.get("has_forecast") and triggered_id == "url":
                raise PreventUpdate

        if not gvb_json:
            return _empty_forecast_fig("Lade GVB-Daten...")

        try:
            # --- Sektor vor Verarbeitung filtern ---
            try:
                gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)  # nutzt globalen Helper (falls vorhanden)
            except Exception:
                gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

            # Store → DataFrame
            try:
                gvb_df = _parse_store_df(gvb_json_filtered)  # bevorzugter App-Helper
            except Exception:
                # robuster Fallback
                gvb_df = pd.read_json(gvb_json_filtered, orient="split")

            empty_fc = pd.DataFrame()

            # Hinweis: Falls der Filter zu einem leeren Frame führt, zeichnen wir dennoch eine leere Historie
            # (die Chart-Funktion sollte damit robust umgehen); optional könnte man hier eine Meldung setzen.

            fig = _create_pipeline_chart(
                forecast_df=empty_fc,
                metadata={},
                gvb_data=gvb_df,
                target=target or "gesamt",
                is_fluss_mode=bool(is_fluss_mode),
                horizon_quarters=horizon or 6,
                show_backtest=False,            # initial kein Backtest
                backtest_mode="overlay"         # Modus fest verdrahtet
            )
            return fig

        except Exception as e:
            logger.error(f"[InitialHistory] Fehler beim Laden der Historie: {e}")
            return _empty_forecast_fig("Fehler beim Laden der historischen Daten")


def _compute_simple_metrics(metadata: dict) -> dict:
    import numpy as np
    import pandas as pd

    cv = (metadata or {}).get('cv_performance', {}) or {}
    mae = cv.get('cv_mae', cv.get('mae'))
    rmse = cv.get('cv_rmse', cv.get('rmse'))
    r2 = cv.get('cv_r2', cv.get('r2'))

    bt = (metadata or {}).get('backtest_results', []) or []
    bt_df = pd.DataFrame(bt) if isinstance(bt, (list, tuple, dict)) else (bt if isinstance(bt, pd.DataFrame) else pd.DataFrame())

    actual = pd.to_numeric(bt_df.get('actual'), errors='coerce') if 'actual' in bt_df else pd.Series(dtype=float)
    pred   = pd.to_numeric(bt_df.get('predicted'), errors='coerce') if 'predicted' in bt_df else pd.Series(dtype=float)

    bias_pct = None
    if actual.size and pred.size and actual.size == pred.size:
        num = float(np.nanmean(pred - actual))
        denom = float(np.nanmean(np.abs(actual))) or np.nan
        if np.isfinite(denom) and denom != 0:
            bias_pct = 100.0 * num / denom

    smape = None
    if actual.size and pred.size and actual.size == pred.size:
        denom = (np.abs(actual) + np.abs(pred))
        valid = (denom > 0) & np.isfinite(denom)
        if valid.any():
            smape = 100.0 * np.nanmean(2.0 * np.abs(pred[valid] - actual[valid]) / denom[valid])

    directional = None
    if actual.size >= 2 and pred.size >= 2:
        da = np.sign(np.diff(actual))
        dp = np.sign(np.diff(pred))
        directional = 100.0 * float(np.mean(da == dp))

    # Coverage zunächst aus metadata lesen …
    coverage = (((metadata or {}).get('diagnostics', {}) or {}).get('ci', {}) or {}).get('coverage') or {}

    # … ansonsten Fallback aus Backtest-Intervallen berechnen, falls vorhanden
    if not coverage and not bt_df.empty and 'actual' in bt_df.columns:
        cov = {}
        def _cov_for(level):
            lower_cols = [f'yhat_lower_{level}', f'lower_{level}', f'lo{level}', f'y_lower_{level}']
            upper_cols = [f'yhat_upper_{level}', f'upper_{level}', f'hi{level}', f'y_upper_{level}']
            lower = next((c for c in lower_cols if c in bt_df.columns), None)
            upper = next((c for c in upper_cols if c in bt_df.columns), None)
            if lower and upper:
                lo = pd.to_numeric(bt_df[lower], errors='coerce')
                hi = pd.to_numeric(bt_df[upper], errors='coerce')
                act = pd.to_numeric(bt_df['actual'], errors='coerce')
                valid = lo.notna() & hi.notna() & act.notna()
                if valid.any():
                    return 100.0 * float(((act[valid] >= lo[valid]) & (act[valid] <= hi[valid])).mean())
            return None

        cov80 = _cov_for(80)
        cov95 = _cov_for(95)
        if cov80 is not None: cov[80] = cov80
        if cov95 is not None: cov[95] = cov95
        coverage = cov

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias_pct": bias_pct,
        "smape": smape,
        "directional": directional,
        "coverage": coverage or {}
    }

@app.callback(
    Output('forecast-chart', 'figure', allow_duplicate=True),
    Output('forecast-metrics', 'children'),
    Output('feature-importance-table', 'children'),
    Output('feature-importance-icicle', 'figure'),
    Output('model-artifact-store', 'data', allow_duplicate=True),
    Output('forecast-state-store', 'data', allow_duplicate=True),
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
        # NEU: vorab geladenes Modell aus Preset/Store für PKL-Reuse
        State('model-artifact-store', 'data'),
        # NEU: Sektor-Auswahl (PH/NFK)
        State('forecast-sektor-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def create_pipeline_forecast(
    n_clicks, exog_json, target, exog_vars, horizon,
    use_cache, gvb_json, is_fluss_mode, show_backtest,
    model_payload, sektor_value
    ):
        """
        Hauptcallback für Forecast-Erstellung mit 80% und 95% Konfidenzintervallen.
        Backtest-Darstellung ist fest auf 'overlay' (gestrichelte Linie) gesetzt.
        Sektor-Filter (PH/NFK) wird VOR Adapter-Anlage angewandt.
        """
        import numpy as np
        import pandas as pd

        # -------- Helpers für leicht verständliche Metriken --------
        def _compute_simple_metrics(metadata: dict) -> dict:
            cv = (metadata or {}).get('cv_performance', {}) or {}
            mae = cv.get('cv_mae', cv.get('mae'))
            rmse = cv.get('cv_rmse', cv.get('rmse'))
            r2 = cv.get('cv_r2', cv.get('r2'))

            bt = (metadata or {}).get('backtest_results', []) or []
            actual = np.asarray([b.get('actual') for b in bt if b.get('actual') is not None], dtype=float)
            pred   = np.asarray([b.get('predicted') for b in bt if b.get('predicted') is not None], dtype=float)

            bias_pct = None
            if actual.size and pred.size and actual.size == pred.size:
                num = float(np.nanmean(pred - actual))
                denom = float(np.nanmean(np.abs(actual))) or np.nan
                if np.isfinite(denom) and denom != 0:
                    bias_pct = 100.0 * num / denom

            smape = None
            if actual.size and pred.size and actual.size == pred.size:
                denom = (np.abs(actual) + np.abs(pred))
                valid = denom > 0
                if valid.any():
                    smape = 100.0 * np.nanmean(2.0 * np.abs(pred[valid] - actual[valid]) / denom[valid])

            directional = None
            if actual.size >= 2 and pred.size >= 2:
                da = np.sign(np.diff(actual))
                dp = np.sign(np.diff(pred))
                directional = 100.0 * float(np.mean(da == dp))

            coverage = ((metadata or {}).get('diagnostics', {}).get('ci', {}) or {}).get('coverage') or {}

            return {"mae": mae, "rmse": rmse, "r2": r2,
                    "bias_pct": bias_pct, "smape": smape,
                    "directional": directional, "coverage": coverage}

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
        # -----------------------------------------------------------

        # Kleiner Fallback-Filter, falls der globale Helper nicht verfügbar ist
        def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
            try:
                df = _parse_store_df(gjson)  # vorhandener App-Helper
                if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                    sekt = str(sektor).strip().upper()
                    df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
                return df.to_json(orient="split", date_format="iso")
            except Exception:
                return gjson

        ctx = dash.callback_context
        if not ctx.triggered or not n_clicks:
            raise dash.exceptions.PreventUpdate

        if not HAS_PIPELINE or DashboardForecastAdapter is None:
            msg = "Pipeline-Adapter nicht gefunden – bitte Integration prüfen."
            empty_icicle = go.Figure()
            empty_icicle.update_layout(template='plotly_white')
            return (
                _empty_forecast_fig(msg),
                html.Div(msg, className="text-danger"),
                html.Div(),
                empty_icicle,
                dash.no_update,
                dash.no_update
            )

        try:
            # --- Sektor vor dem Adapter filtern ---
            try:
                # bevorzuge globalen Helper, sonst fallback
                gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)  # type: ignore[name-defined]
            except Exception:
                gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

            # --- Adapter mit gefiltertem Store aufbauen ---
            adapter = DashboardForecastAdapter(gvb_json_filtered, exog_json)
            # Sektor & UI-Infos in pipeline_info hinterlegen (u. a. für cache_tag)
            adapter.pipeline_info.update({
                "ui_target": target,
                "use_flows": bool(is_fluss_mode),
                "horizon": int(horizon or 6),
                "sektor": sektor_value
            })

            # --- Forecast ausführen ---
            forecast_df, metadata = adapter.run_forecast(
                target=target,
                selected_exog=exog_vars or [],
                horizon=horizon or 6,
                use_cached=bool(use_cache),
                force_retrain=False,
                use_flows=bool(is_fluss_mode),
                confidence_levels=[80, 95],
                preload_model_path=(model_payload or {}).get('path')
            )

            logger.info(f"[Forecast] DataFrame-Spalten: {forecast_df.columns.tolist()}")
            if 'yhat_lower_80' in forecast_df.columns:
                logger.info("[Forecast] ✓ 80% Konfidenzintervall vorhanden")
            if 'yhat_lower_95' in forecast_df.columns:
                logger.info("[Forecast] ✓ 95% Konfidenzintervall vorhanden")

            # Backtest verfügbar?
            if metadata and 'backtest_results' in metadata:
                logger.info(f"[Forecast] ✓ Backtest-Daten: {len(metadata['backtest_results'])} Punkte")
            else:
                logger.info("[Forecast] ℹ Keine Backtest-Daten in Metadata")

            # Chart erstellen – Backtest-Modus fest auf 'overlay'
            # (Optionales Labeling mit Sektor)
            fig = _create_pipeline_chart(
                forecast_df=forecast_df,
                metadata=metadata,
                gvb_data=adapter.gvb_data,
                target=target,
                is_fluss_mode=bool(is_fluss_mode),
                horizon_quarters=horizon or 6,
                show_backtest=bool(show_backtest),
                backtest_mode="overlay"
            )

            # ---------- Kompakte, verständliche Metriken ----------
            simple = _compute_simple_metrics(metadata)

            mae_txt   = _fmt_num(simple["mae"], nd=2)
            rmse_txt  = _fmt_num(simple["rmse"], nd=2)
            r2_txt    = _fmt_num(simple["r2"], nd=3)
            bias_txt  = _fmt_pct(simple["bias_pct"], nd=1)
            smape_txt = _fmt_pct(simple["smape"], nd=1)
            dir_txt   = _fmt_pct(simple["directional"], nd=0)

            cov80 = _fmt_pct((simple["coverage"] or {}).get(80), nd=1)
            cov95 = _fmt_pct((simple["coverage"] or {}).get(95), nd=1)

            if dbc is not None:
                metrics = dbc.Container([
                    dbc.Row(dbc.Col(html.Div([
                        html.Span("Mittlerer Schätzfehler (MAE)", className="text-muted small"),
                        html.H4(mae_txt, className="mb-0 mt-1"),
                        html.Div(f"Bias (Über-/Unterschätzung): {bias_txt}", className="small text-secondary mt-1")
                    ])), className="mb-3 pb-3", style={"borderBottom": "1px solid #e9ecef"}),

                    dbc.Row(dbc.Col(html.Div([
                        html.Span("Prognosegüte", className="text-muted small"),
                        html.H4(smape_txt, className="mb-0 mt-1"),
                        html.Div(f"Trefferquote Richtung: {dir_txt}", className="small text-secondary mt-1")
                    ])), className="mb-3 pb-3", style={"borderBottom": "1px solid #e9ecef"}),

                    dbc.Row(dbc.Col(html.Div([
                        html.Span("Modellgüte", className="text-muted small"),
                        html.H4(r2_txt, className="mb-0 mt-1"),
                        html.Div(f"Typisches Fehlerband (RMSE): {rmse_txt}", className="small text-secondary mt-1"),
                        html.Div(f"CI-Deckung 80/95: {cov80} / {cov95}", className="small text-secondary mt-1")
                    ])), className="mb-3 pb-1"),
                ])
            else:
                metrics = html.Div([
                    html.Div([html.B("Mittlerer Schätzfehler (MAE): "), html.Span(mae_txt),
                            html.Small(f"  | Bias: {bias_txt}")]),
                    html.Div([html.B("Prognosegüte (sMAPE): "), html.Span(smape_txt),
                            html.Small(f"  | Richtungstreffer: {dir_txt}")]),
                    html.Div([html.B("Modellgüte (R², CV): "), html.Span(r2_txt),
                            html.Small(f"  | Fehlerband (RMSE): {rmse_txt}  | CI-Deckung 80/95: {cov80}/{cov95}")]),
                ], className="p-2")

            feature_bar = _create_feature_importance(metadata)
            features = (metadata or {}).get('model_complexity', {}).get('top_features', {})
            icicle_fig = create_feature_importance_icicle(features, top_n=15)

            model_path = (metadata or {}).get("model_path")
            snapshot_path = (metadata or {}).get("exog_snapshot_path")
            model_payload_out = {
                "path": model_path,
                "exog_snapshot_path": snapshot_path
            } if (model_path or snapshot_path) else dash.no_update

            return (
                fig,
                metrics,
                feature_bar,
                icicle_fig,
                model_payload_out,
                {"has_forecast": True}
            )

        except Exception as e:
            logger.exception(f"[Forecast] Fehler: {e}")
            empty_icicle = go.Figure()
            empty_icicle.update_layout(template='plotly_white')
            fig, error_html, _ = _error_forecast_response(e)
            return (
                fig,
                error_html,
                html.Div(),
                empty_icicle,
                dash.no_update,
                dash.no_update
            )


# ==============================================================================
# NEU: CALLBACK FÜR BACKTEST-TOGGLE
# ==============================================================================
# Diese Callback-Funktion HINZUFÜGEN zu forecaster_main.py (bei den anderen Callbacks)
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
        # NEU: Sektor (PH/NFK)
        State('forecast-sektor-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def toggle_backtest_visualization(
    show_backtest,
    fc_state, gvb_json, target, is_fluss_mode, horizon,
    exog_json, exog_vars, sektor_value
):
    """
    Aktualisiert den Chart, wenn der Backtest-Schalter betätigt wird.
    Modus ist fest auf 'overlay' (gestrichelte Linie) gesetzt.
    Lädt Forecast (aus Cache) neu und fügt ggf. Backtest-Overlay hinzu.
    Berücksichtigt den gewählten Sektor (PH/NFK), indem der GVB-Store vorab gefiltert wird.
    """
    import pandas as pd

    # Fallback-Filter, falls globaler Helper nicht importiert ist
    def _filter_gvb_json_by_sektor_fallback(gjson: str, sektor: str) -> str:
        try:
            df = _parse_store_df(gjson)
            if isinstance(df, pd.DataFrame) and not df.empty and "sektor" in df.columns and sektor:
                sekt = str(sektor).strip().upper()
                df = df[df["sektor"].astype(str).str.upper() == sekt].copy()
            return df.to_json(orient="split", date_format="iso")
        except Exception:
            return gjson

    # Nur wenn bereits ein Forecast existiert
    if not fc_state or not fc_state.get('has_forecast'):
        logger.info("[Backtest-Toggle] Kein Forecast vorhanden, überspringe")
        return dash.no_update

    if not gvb_json:
        logger.warning("[Backtest-Toggle] Keine GVB-Daten")
        return dash.no_update

    try:
        backtest_mode = "overlay"
        logger.info(f"[Backtest-Toggle] show={show_backtest}, mode={backtest_mode}, sektor={sektor_value}")

        if not HAS_PIPELINE or DashboardForecastAdapter is None:
            logger.error("[Backtest-Toggle] Pipeline nicht verfügbar")
            return dash.no_update

        # --- Sektor vor dem Adapter filtern ---
        try:
            gvb_json_filtered = _filter_gvb_json_by_sektor(gvb_json, sektor_value)  # nutzt globalen Helper, wenn vorhanden
        except Exception:
            gvb_json_filtered = _filter_gvb_json_by_sektor_fallback(gvb_json, sektor_value)

        # --- Adapter aufsetzen (Cache nutzen) ---
        adapter = DashboardForecastAdapter(gvb_json_filtered, exog_json)
        adapter.pipeline_info.update({
            "ui_target": target,
            "use_flows": bool(is_fluss_mode),
            "horizon": int(horizon or 6),
            "sektor": sektor_value
        })

        # Forecast erneut durchführen (nutzt Cache, retrain=False)
        forecast_df, metadata = adapter.run_forecast(
            target=target,
            selected_exog=exog_vars or [],
            horizon=horizon or 6,
            use_cached=True,
            force_retrain=False,
            use_flows=bool(is_fluss_mode),
            confidence_levels=[80, 95]
        )

        # Chart neu erstellen MIT (optionalem) Backtest-Overlay
        fig = _create_pipeline_chart(
            forecast_df=forecast_df,
            metadata=metadata,
            gvb_data=adapter.gvb_data,
            target=target or "gesamt",
            is_fluss_mode=bool(is_fluss_mode),
            horizon_quarters=horizon or 6,
            show_backtest=bool(show_backtest),
            backtest_mode=backtest_mode  # immer 'overlay' (gestrichelt)
        )

        logger.info("[Backtest-Toggle] Chart erfolgreich aktualisiert")
        return fig

    except Exception as e:
        logger.exception(f"[Backtest-Toggle] Fehler: {e}")
        return dash.no_update

# ==============================================================================
# CALLBACK-REGISTRIERUNG
# ==============================================================================

def register_forecaster_callbacks(real_app: "dash.Dash", Log):
    """
    Registriert alle gesammelten Callbacks an der echten Dash-App.
    
    Args:
        real_app: Die echte Dash-Applikation
        Log: Logger-Instanz
    """
    # 1. Registriere die gesammelten Standard-Callbacks
    regs = getattr(app, "_registrations", [])
    
    logger.info(f"Registriere {len(regs)} Forecaster-Callbacks")
    
    for args, kwargs, fn in regs:
        real_app.callback(*args, **kwargs)(fn)
    
    logger.info("Forecaster-Callbacks erfolgreich registriert")
    

# geospacial/geospacial_main.py

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union, Callable
import re
import logging
import json
import base64
import io

import pandas as pd
from dash import (
    html,
    dcc,
    Input,
    Output,
    State,
    no_update,
    callback_context,
)
import dash_bootstrap_components as dbc
import dash_ag_grid as dag  
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

from .geospacial_viz import (
    build_map_from_df,
    build_empty_map,
)

from forecaster.forecaster_main import (  # NEU
    _discover_runs,
    _load_run_meta,
    RUNS_DIR,
)





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





# Leere Standardkarte für Initialzustand
EMPTY_GEO_MAP_HTML = build_empty_map(
    level="krs",
    message="Bitte wählen Sie einen Indikator und klicken Sie auf 'Filter anwenden'.",
)


logger = logging.getLogger("GeoSpacial")


GEO_DIR = Path(__file__).resolve().parent  # .../geospacial

try:
    APP_ROOT: Path = GEO_DIR.parent       # Projektroot (da liegt meist app.py)
except Exception:
    APP_ROOT = Path.cwd()

# ============================================================
# Pfade & Dateien
# ============================================================

DATA_INPUT_DIR = GEO_DIR / "data_input"
PROCESSED_DIR = DATA_INPUT_DIR / "deutschlandatlas_processed"
INDICATOR_CSV_PATH = DATA_INPUT_DIR / "deutschlandatlas_services_mit_counts.csv"
DEUTSCHLANDATLAS_XLSX_PATH = DATA_INPUT_DIR / "Deutschlandatlas.xlsx"
DEUTSCHLANDATLAS_EXCEL_PATH = DATA_INPUT_DIR / "Deutschlandatlas.xlsx"


# optional: falls die Sheet-Namen in der Excel anders heißen, hier anpassen
DEUTSCHLANDATLAS_SHEETS_BY_LEVEL: Dict[str, str] = {
    "krs": "KRS",     # Blattname für Kreise
    "gem": "GEM",     # Blattname für Gemeinden
    "vbgem": "VBGEM", # Blattname für Gemeindeverbünde
}


# fachlich korrekte Ebenen
PARQUETS_BY_LEVEL: Dict[str, Path] = {
    "krs": PROCESSED_DIR / "Deutschlandatlas_KRS_merged.parquet",   # Kreis
    "gem": PROCESSED_DIR / "Deutschlandatlas_GEM_merged.parquet",   # Gemeinde
    "vbgem": PROCESSED_DIR / "Deutschlandatlas_VBGEM_merged.parquet",  # Gemeindeverbund
}

LEVEL_LABELS: Dict[str, str] = {
    "krs": "Kreis (KRS)",
    "gem": "Gemeinde (GEM)",
    "vbgem": "Gemeindeverbund (VBGEM)",
}

# ============================================================
# Konstanten
# ============================================================
NON_INDICATOR_COLS = {
    "GKZ",
    "Gemeindename",
    "Kreisname",
    "Gemeindeverbandsname",
    "NAME",
}

MISSING_STRINGS = {
    "",
    "nan",
    "none",
    "-9999",
    "-9999.0",
    "-9999,0",
    "-99999",
    "-99999.0",
    "-99999,0",
    "-999999",
    "-999999.0",
    "-999999,0",
}
MISSING_NUMBERS = {
    -9999,
    -9999.0,
    -99999,
    -99999.0,
    -999999,
    -999999.0,
}

SUFFIX_DROP = {
    "ha2023",
    "ha2022",
    "ha2021",
    "ha2020",
    "za2023",
    "za2022",
    "3857",
}
INDICATOR_PREFIXES = {"p", "v", "pendel", "teilz", "bev", "beschq", "kbetr"}

# Grenze: ab so vielen ausgewählten GKZs rendern wir die Karte nicht neu
HIGHLIGHT_MAX_REBUILD = 25

# mögliche Transformationen
TRANSFORM_OPTIONS = [
    {"label": "Rohwerte", "value": "raw"},
    {"label": "% von Angezeigt", "value": "pct_visible"},
    {"label": "% vom Gesamt", "value": "pct_total"},
]

# ============================================================
# Caches
# ============================================================
_DF_CACHE: Dict[str, pd.DataFrame] = {}
_DF_CACHE_MTIME: Dict[str, int] = {}
_OPTIONS_CACHE: Dict[str, List[dict]] = {}
_NAMECOL_CACHE: Dict[str, Optional[str]] = {}
_INDICATOR_SLICE_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
# WICHTIG: in main cachen wir schon nach transform
_MAP_HTML_CACHE: Dict[Tuple[str, str, str], str] = {}
_STATS_CACHE: Dict[Tuple[str, str, str], Dict[str, float]] = {}
_COLUMNS_CACHE: Dict[str, List[str]] = {}
_REGION_OPTIONS_CACHE: Dict[str, List[dict]] = {}

# ============================================================
# Hilfsfunktionen für RS/GKZ-Normalisierung (RS5)
# ============================================================
_DIGITS_RE = re.compile(r"\D+")


def _only_digits_series(s: pd.Series) -> pd.Series:
    """Lässt nur Ziffern übrig (serienweise)."""
    return s.astype(str).str.replace(_DIGITS_RE, "", regex=True)


def _mk_rs5_from_any_series(series: pd.Series) -> pd.Series:
    """
    Macht aus diversen Formaten einen 5-stelligen Kreis-Schlüssel (RS5):
    - 7/8-stellige ARS (z. B. 6611000) → zfill(8)[:5] → 06611
    - 5-stellige Kreis-RS bleiben 5-stellig
    - 3-stellige (RBZ/Land) werden auf 5 aufgefüllt, matchen aber i.d.R. nichts
    """
    d = _only_digits_series(series)
    # Heuristik: wenn 7/8-stellig (ARS), erst auf 8 auffüllen, dann die ersten 5
    rs5_from_ars = d.where(~d.str.len().between(7, 8), d.str.zfill(8).str[:5])
    # ansonsten auf 5 auffüllen und die ersten 5 nehmen
    return rs5_from_ars.str.zfill(5).str[:5]


# ============================================================
# Label-Mapping
# ============================================================
def _strip_suffixes(parts: list[str]) -> list[str]:
    while parts and parts[-1] in SUFFIX_DROP:
        parts.pop()
    return parts


def canonicalize_indicator_name(name: str) -> str:
    if not name:
        return ""
    parts = str(name).strip().lower().split("_")
    parts = _strip_suffixes(parts)
    if not parts:
        return ""
    if len(parts) >= 2 and parts[0] in INDICATOR_PREFIXES:
        return f"{parts[0]}_{parts[1]}"
    return "_".join(parts)


def extract_long_from_layers(layers_val: str) -> Optional[str]:
    if not isinstance(layers_val, str) or not layers_val:
        return None
    txt = layers_val.strip()
    if ": " in txt:
        txt = txt.split(": ", 1)[1]
    if " (Features:" in txt:
        txt = txt.split(" (Features:", 1)[0]
    return txt.strip() or None


def load_indicator_label_map(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        logger.info(f"[Geo] Indicator CSV nicht gefunden: {csv_path}")
        return {}
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        logger.warning(f"[Geo] Konnte Indicator-CSV nicht laden: {e}")
        return {}
    cols_lower = {c.lower(): c for c in df.columns}
    title_col = cols_lower.get("title")
    layers_col = cols_lower.get("layers")
    tf_col = cols_lower.get("totalfeatures")
    if not title_col:
        logger.warning("[Geo] Indicator-CSV ohne 'title'-Spalte – Mapping leer.")
        return {}
    best: Dict[str, Tuple[str, int]] = {}
    for _, row in df.iterrows():
        short_raw = str(row.get(title_col, "")).strip()
        if not short_raw:
            continue
        canon = canonicalize_indicator_name(short_raw)
        long_name = (
            extract_long_from_layers(row.get(layers_col, "")) if layers_col else None
        )
        long_name = long_name or short_raw
        score = 0
        if tf_col:
            try:
                score = int(float(str(row.get(tf_col, "0")).replace(",", ".")))
            except Exception:
                pass
        if canon not in best or score > best[canon][1]:
            best[canon] = (long_name, score)
    mapping = {k: v[0] for k, v in best.items()}
    logger.info(f"[Geo] Indicator-Label-Mapping geladen: {len(mapping)} Einträge.")
    return mapping


INDICATOR_LABELS = load_indicator_label_map(INDICATOR_CSV_PATH)

# ============================================================
# Datenzugriff & Caching
# ============================================================
def _resolve_level_path(level: str) -> Optional[Path]:
    return PARQUETS_BY_LEVEL.get(level)


def _get_df_cached(level: str) -> Optional[pd.DataFrame]:
    path = _resolve_level_path(level)
    if not path or not path.exists():
        logger.warning(f"[Geo] _get_df_cached: Pfad für Level '{level}' existiert nicht: {path}")
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except Exception:
        mtime = 0
    if level in _DF_CACHE and _DF_CACHE_MTIME.get(level) == mtime:
        return _DF_CACHE[level]
    df = pd.read_parquet(path)
    _DF_CACHE[level] = df
    _DF_CACHE_MTIME[level] = mtime
    logger.info(f"[Geo] Cache erneuert für Ebene '{level}': shape={df.shape}, path={path}")
    # abhängige Caches leeren
    _OPTIONS_CACHE.pop(level, None)
    _NAMECOL_CACHE.pop(level, None)
    _COLUMNS_CACHE.pop(level, None)
    _REGION_OPTIONS_CACHE.pop(level, None)
    _MAP_HTML_CACHE.clear()
    _INDICATOR_SLICE_CACHE.clear()
    _STATS_CACHE.clear()
    return df


def _get_available_columns(level: str) -> List[str]:
    if level in _COLUMNS_CACHE:
        return _COLUMNS_CACHE[level]

    path = _resolve_level_path(level)
    if not path or not path.exists():
        _COLUMNS_CACHE[level] = []
        logger.warning(f"[Geo] _get_available_columns: kein Pfad für Level '{level}': {path}")
        return []

    cols: List[str] = []
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(path)
        cols = pf.schema.names
    except Exception:
        try:
            empty_df = pd.read_parquet(path, columns=[])
            cols = list(empty_df.columns)
        except Exception:
            df = _get_df_cached(level)
            cols = list(df.columns) if df is not None else []

    _COLUMNS_CACHE[level] = cols
    logger.info(f"[Geo] Verfügbare Spalten für Ebene '{level}': {len(cols)} Spalten.")
    return cols


def _detect_name_col_fast(level: str) -> Optional[str]:
    if level in _NAMECOL_CACHE:
        return _NAMECOL_CACHE[level]
    available = _get_available_columns(level)
    for cand in ("Gemeindename", "Kreisname", "Gemeindeverbandsname", "NAME"):
        if cand in available:
            _NAMECOL_CACHE[level] = cand
            return cand
    _NAMECOL_CACHE[level] = None
    return None


def _detect_name_col(df: pd.DataFrame, level: str) -> Optional[str]:
    nc = _detect_name_col_fast(level)
    if nc:
        return nc
    for cand in ("Gemeindename", "Kreisname", "Gemeindeverbandsname", "NAME"):
        if cand in df.columns:
            _NAMECOL_CACHE[level] = cand
            return cand
    _NAMECOL_CACHE[level] = None
    return None


def _column_has_real_data_fast(level: str, column: str, sample_size: int = 60) -> bool:
    path = _resolve_level_path(level)
    if not path or not path.exists():
        return False
    try:
        sample_df = pd.read_parquet(path, columns=[column]).head(sample_size)
        series = sample_df[column]
        if pd.api.types.is_numeric_dtype(series):
            mask = (series.notna()) & (~series.isin(MISSING_NUMBERS))
            return bool(mask.any())
        s = series.astype(str).str.strip().str.lower()
        return bool((~s.isin(MISSING_STRINGS)).any())
    except Exception:
        return False


def _options_for_level(level: str, max_len: int = 40) -> List[dict]:
    if level in _OPTIONS_CACHE:
        return _OPTIONS_CACHE[level]

    available_cols = _get_available_columns(level)
    opts: List[dict] = []
    for c in available_cols:
        if c in NON_INDICATOR_COLS:
            continue
        if not _column_has_real_data_fast(level, c, sample_size=60):
            continue
        canon = canonicalize_indicator_name(c)
        full_label = INDICATOR_LABELS.get(canon, c)
        opts.append(
            {
                "label": full_label,
                "value": c,
                "title": full_label,
            }
        )

    _OPTIONS_CACHE[level] = opts
    logger.info(f"[Geo] _options_for_level('{level}'): {len(opts)} Indikator-Optionen.")
    return opts


def _get_indicator_slice(level: str, indicator: str) -> Optional[pd.DataFrame]:
    cache_key = (level, indicator)
    if cache_key in _INDICATOR_SLICE_CACHE:
        return _INDICATOR_SLICE_CACHE[cache_key]

    base_df = _get_df_cached(level)
    if base_df is None or indicator not in base_df.columns:
        logger.warning(f"[Geo] _get_indicator_slice: Ebene '{level}' ohne Indikator '{indicator}'.")
        return None

    name_col = _detect_name_col(base_df, level)
    cols = ["GKZ", indicator]
    if name_col:
        cols.insert(1, name_col)
    slice_df = base_df[cols].copy()
    _INDICATOR_SLICE_CACHE[cache_key] = slice_df
    logger.info(
        f"[Geo] _get_indicator_slice: level={level}, indicator={indicator}, "
        f"shape={slice_df.shape}, name_col={name_col}"
    )
    return slice_df


def _get_region_options(level: str) -> List[dict]:
    if level in _REGION_OPTIONS_CACHE:
        return _REGION_OPTIONS_CACHE[level]

    df = _get_df_cached(level)
    if df is None:
        _REGION_OPTIONS_CACHE[level] = []
        logger.warning(f"[Geo] _get_region_options: kein DF für Ebene '{level}'.")
        return []

    name_col = _detect_name_col(df, level)
    if name_col:
        df_opts = df[[name_col, "GKZ"]].dropna(subset=["GKZ"]).sort_values(by=name_col)
        opts = [
            {"label": f"{row[name_col]} ({row['GKZ']})", "value": str(row["GKZ"])}
            for _, row in df_opts.iterrows()
        ]
    else:
        df_opts = df[["GKZ"]].dropna(subset=["GKZ"]).sort_values(by="GKZ")
        opts = [
            {"label": str(row["GKZ"]), "value": str(row["GKZ"])}
            for _, row in df_opts.iterrows()
        ]

    _REGION_OPTIONS_CACHE[level] = opts
    logger.info(f"[Geo] _get_region_options('{level}'): {len(opts)} Regionen.")
    return opts


# ============================================================
# Transformation
# ============================================================
# Transformation Options - ERWEITERT
TRANSFORM_OPTIONS = [
    {"label": "Rohwerte", "value": "raw"},
    {"label": "% von Angezeigt", "value": "pct_visible"},
    {"label": "% vom Gesamt", "value": "pct_total"},
]

# ============================================================
# Transformation - ERWEITERT
# ============================================================
def _apply_transform(
    df: pd.DataFrame, 
    indicator: str, 
    transform: str,
    *,
    full_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Wendet die gewünschte Transformation auf eine *KOPIE* des DataFrames an.
    
    Args:
        df: DataFrame mit den Daten (kann gefiltert sein)
        indicator: Name der Indikator-Spalte
        transform: Art der Transformation
            - "raw": Rohwerte (keine Transformation)
            - "pct_visible": % vom Gesamt der SICHTBAREN/ANGEZEIGTEN Daten (df)
            - "pct_total": % vom Gesamt ALLER Daten (full_df erforderlich)
        full_df: Optional - vollständiger DataFrame für pct_total Berechnung
    
    Returns:
        Transformierter DataFrame (Kopie)
    """
    df = df.copy()
    
    if transform == "raw":
        # Keine Transformation
        return df
    
    elif transform == "pct_visible":
        # % von den ANGEZEIGTEN Daten (basierend auf df selbst)
        s_num = pd.to_numeric(df[indicator], errors="coerce")
        s_num = s_num[~s_num.isin(MISSING_NUMBERS)]
        total_visible = s_num.sum()
        
        if total_visible and total_visible != 0:
            df[indicator] = pd.to_numeric(df[indicator], errors="coerce") / total_visible * 100
        else:
            # Falls keine gültigen Werte, setze auf 0
            df[indicator] = 0.0
            
    elif transform == "pct_total":
        # % vom GESAMTEN Datensatz (basierend auf full_df)
        if full_df is None:
            raise ValueError(
                "pct_total Transform benötigt full_df Parameter - "
                "vollständiger DataFrame muss übergeben werden"
            )
        
        # Berechne Summe aus dem VOLLSTÄNDIGEN DataFrame
        s_num_full = pd.to_numeric(full_df[indicator], errors="coerce")
        s_num_full = s_num_full[~s_num_full.isin(MISSING_NUMBERS)]
        total_all = s_num_full.sum()
        
        if total_all and total_all != 0:
            df[indicator] = pd.to_numeric(df[indicator], errors="coerce") / total_all * 100
        else:
            df[indicator] = 0.0
    
    return df


def _get_or_compute_stats(
    level: str, indicator: str, df_like: pd.DataFrame, transform: str
) -> Dict[str, float]:
    cache_key = (level, indicator, transform)
    if cache_key in _STATS_CACHE:
        return _STATS_CACHE[cache_key]

    if indicator not in df_like.columns:
        stats = {"count": float(len(df_like)), "min": 0.0, "max": 0.0, "mean": 0.0}
        _STATS_CACHE[cache_key] = stats
        return stats

    s_num = pd.to_numeric(df_like[indicator], errors="coerce")
    s_num = s_num[~s_num.isin(MISSING_NUMBERS)]
    if s_num.empty:
        stats = {"count": float(len(df_like)), "min": 0.0, "max": 0.0, "mean": 0.0}
    else:
        stats = {
            "count": float(len(df_like)),
            "min": float(s_num.min(skipna=True)),
            "max": float(s_num.max(skipna=True)),
            "mean": float(s_num.mean(skipna=True)),
        }
    _STATS_CACHE[cache_key] = stats
    logger.info(
        f"[Geo] Stats für level={level}, indicator={indicator}, transform={transform}: "
        f"count={stats['count']}, min={stats['min']}, max={stats['max']}, mean={stats['mean']}"
    )
    return stats


# ============================================================
# Hilfsfunktion: schmale Map-Daten
# ============================================================
def _build_minimal_map_df(base_df: pd.DataFrame, indicator_col: str, level: str) -> pd.DataFrame:
    """
    Schneidet das DataFrame auf GKZ + optionalen Namen + konkrete Anzeigespalte zu.
    """
    cols = ["GKZ"]
    name_col = _detect_name_col_fast(level)
    if name_col and name_col in base_df.columns:
        cols.append(name_col)
    if indicator_col not in cols:
        cols.append(indicator_col)
    df_out = base_df[cols].copy()
    logger.info(
        f"[Geo] _build_minimal_map_df: level={level}, indicator={indicator_col}, "
        f"shape={df_out.shape}, columns={list(df_out.columns)}"
    )
    return df_out


# ============================================================
# -------- VGRdL Excel-Import (Blatt 1.1) & Merge (pandas) ----
# ============================================================
def _normalize_rs_to_gkz(value: object, width: int = 8) -> str:
    """
    Regionalschlüssel -> Zero-padded String (Standard: Breite 8).
    Robust gegenüber floats (z.B. '8111.0') und Whitespace.
    (Wird aktuell nicht mehr zentral für RS5 genutzt – RS5 erfolgt über _mk_rs5_from_any_series.)
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        i = int(float(s.replace(" ", "")))
        return str(i).zfill(width)
    except Exception:
        s2 = s.split(".")[0].replace(" ", "")
        return s2.zfill(width)


def _find_header_row_for_regionalschluessel(
    xl_path_or_buffer: Union[str, Path, bytes, bytearray],
    sheet_name: str
) -> int:
    """
    Sucht die Header-Zeile anhand 'Regionalschlüssel' (tolerant ggü. 'Regional-schlüssel').
    """
    df_probe = pd.read_excel(xl_path_or_buffer, sheet_name=sheet_name, header=None, nrows=40)
    for r in range(min(len(df_probe), 40)):
        for c in range(df_probe.shape[1]):
            val = df_probe.iat[r, c]
            if isinstance(val, str):
                norm = val.strip().lower().replace("-", "")
                if norm == "regionalschlüssel":
                    return r
    raise ValueError("Kopfzeile mit 'Regionalschlüssel' im Blatt nicht gefunden.")


def _detect_year_columns(header_row: Iterable[object]) -> Tuple[List[int], Optional[int]]:
    """
    Liefert alle Spaltenindizes, die nach Jahreszahlen aussehen (1900..2100),
    plus das maximale Jahr (int) als Convenience.
    """
    year_cols: List[int] = []
    years: List[int] = []
    for idx, val in enumerate(header_row):
        try:
            if pd.notna(val):
                y = int(float(val))
                if 1900 <= y <= 2100:
                    year_cols.append(idx)
                    years.append(y)
        except Exception:
            pass
    latest_year = max(years) if years else None
    return year_cols, latest_year


def load_excel_kreise_data(
    excel_path_or_file: Union[str, Path, bytes, bytearray],
    sheet_name: str = "1.1",
    name_col_preference: Tuple[str, ...] = ("Gebietseinheit", "Kreisname", "NAME"),
) -> pd.DataFrame:
    """
    Lädt VGRdL Blatt 1.1, erkennt Header & jüngstes Jahr robust, filtert auf NUTS 3,
    und gibt einen schlanken DataFrame zurück:
        ['GKZ', 'Kreisname'?, 'value', 'year']
    - 'GKZ': RS5 (5-stellig) – wie im Deutschlandatlas-Kreise-Parquet
    - 'value': Wert aus dem jüngsten Jahr
    - 'year': Jahr (int)
    """
    logger.info(f"[Geo] load_excel_kreise_data: Lade Excel '{excel_path_or_file}', Sheet '{sheet_name}'.")
    # 1) Header-Zeile finden
    header_row_idx = _find_header_row_for_regionalschluessel(excel_path_or_file, sheet_name)

    # 2) Blatt mit Header lesen
    df = pd.read_excel(excel_path_or_file, sheet_name=sheet_name, header=header_row_idx)
    logger.info(f"[Geo] load_excel_kreise_data: Roh-DF shape={df.shape}")

    # 3) Jahrgangsspalten erkennen
    year_cols, latest_year = _detect_year_columns(df.columns.tolist())
    if not year_cols or latest_year is None:
        raise ValueError("Keine Jahrgangsspalten entdeckt – Struktur prüfen.")

    # 4) Auf Kreisebene (NUTS 3) filtern
    if "NUTS 3" in df.columns:
        df = df[df["NUTS 3"].notna()].copy()

    # 5) Regionalschlüssel-Spalte finden
    rs_col = None
    for cand in ("Regionalschlüssel", "Regional-schlüssel"):
        if cand in df.columns:
            rs_col = cand
            break
    if rs_col is None:
        # generisches Fallback
        rs_col = [c for c in df.columns if str(c).lower().replace("-", "") == "regionalschlüssel"][0]

    # 6) Zielspalte = jüngstes Jahr
    target_year_col = None
    for col in df.columns:
        try:
            if pd.notna(col) and int(float(col)) == int(latest_year):
                target_year_col = col
                break
        except Exception:
            continue
    if target_year_col is None:
        raise ValueError("Spalte des jüngsten Jahres nicht gefunden.")

    # 7) Name-Spalte bestimmen
    name_col = None
    for cand in name_col_preference:
        if cand in df.columns:
            name_col = cand
            break

    # 8) Minimal-Schema
    out_cols = [rs_col, target_year_col]
    if name_col:
        out_cols.insert(1, name_col)
    slim = df[out_cols].copy()

    # 9) GKZ auf RS5 normalisieren (entscheidend für Matches)
    slim["GKZ"] = _mk_rs5_from_any_series(slim[rs_col])

    # 10) benennen & Ordnung
    if name_col:
        slim.rename(columns={name_col: "Kreisname"}, inplace=True)
    slim["year"] = int(latest_year)
    slim.rename(columns={target_year_col: "value"}, inplace=True)
    slim = slim[["GKZ"] + (["Kreisname"] if "Kreisname" in slim.columns else []) + ["value", "year"]]
    slim = slim.dropna(subset=["GKZ"]).drop_duplicates(subset=["GKZ"], keep="first").reset_index(drop=True)

    logger.info(
        f"[Geo] load_excel_kreise_data: Ergebnis shape={slim.shape}, "
        f"year={latest_year}, GKZ unique={slim['GKZ'].nunique()}"
    )
    return slim


def merge_with_kreise_df(
    kreise_df: pd.DataFrame,
    excel_df: pd.DataFrame,
    *,
    kreise_key: str = "GKZ",
    excel_key: str = "GKZ",
    how: str = "left",
    indicator_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merged Excel-Werte (jüngstes Jahr) in das Kreise-DataFrame (reines pandas).
    - how='left' (Standard), damit alle Kreise/Geometrien aus dem Parquet erhalten bleiben.
    - indicator_col_name: Optionaler Spaltenname für den Wert (Default: 'vgrdl_1_1_<year>')
    - gibt ein neues DataFrame zurück.
    """
    if "value" not in excel_df.columns or "year" not in excel_df.columns:
        raise ValueError("excel_df muss Spalten 'value' und 'year' enthalten (siehe load_excel_kreise_data).")

    year = int(excel_df["year"].iloc[0])
    indicator_name = indicator_col_name or f"vgrdl_1_1_{year}"

    left = kreise_df.copy()
    right = excel_df.copy()

    # Join über temporären RS5-Key, GKZ selbst unangetastet lassen
    left["_join"] = _mk_rs5_from_any_series(left[kreise_key])
    right["_join"] = _mk_rs5_from_any_series(right[excel_key])
    right = right[["_join", "value"]].rename(columns={"value": indicator_name})

    merged = left.merge(right, how=how, on="_join")
    merged = merged.drop(columns=["_join"])

    # Kurzprüfung
    total = len(merged)
    matched = merged[indicator_name].notna().sum()
    unmatched = total - matched
    if unmatched > 0:
        logger.warning(
            f"[Geo] merge_with_kreise_df: {unmatched} von {total} Kreisen ohne Excel-Zuordnung "
            f"(indicator={indicator_name})."
        )

    logger.info(
        f"[Geo] merge_with_kreise_df: indicator={indicator_name}, matched={matched}, total={total}, "
        f"shape={merged.shape}"
    )
    return merged


def add_indicator_to_cache(
    level: str,
    df_with_new_indicator: pd.DataFrame,
    indicator_col_name: str,
) -> None:
    """
    Optional: Schreibt ein DataFrame mit neuer Indikatorspalte in den internen Cache.
    Danach werden abhängige Caches geleert, damit der Indikator im Dropdown erscheint.
    """
    if indicator_col_name not in df_with_new_indicator.columns:
        raise ValueError(f"Spalte '{indicator_col_name}' nicht im DataFrame vorhanden.")
    _DF_CACHE[level] = df_with_new_indicator.copy()
    _DF_CACHE_MTIME[level] = _DF_CACHE_MTIME.get(level, 0)  # Dummy, damit Cache-Zugriff ok ist
    _OPTIONS_CACHE.pop(level, None)
    _NAMECOL_CACHE.pop(level, None)
    _COLUMNS_CACHE.pop(level, None)
    _REGION_OPTIONS_CACHE.pop(level, None)
    _MAP_HTML_CACHE.clear()
    _INDICATOR_SLICE_CACHE.clear()
    _STATS_CACHE.clear()
    logger.info(
        f"[Geo] add_indicator_to_cache: level={level}, indicator={indicator_col_name}, "
        f"cached_shape={df_with_new_indicator.shape}"
    )


# ============================================================
# ---- NEU: High-Level Merge/Append-Flow für app.py ----------
# ============================================================
def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"[Geo] _safe_to_parquet: geschrieben nach {path} (snappy).")
    except Exception as e:
        logger.warning(f"[Geo] _safe_to_parquet: snappy fehlgeschlagen ({e}), schreibe ohne Kompression.")
        df.to_parquet(path, index=False)


def _safe_to_excel(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """
    Schreibt ein DataFrame sicher als Excel-Datei.
    Praktisch für Debugging nach dem Merge.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=index)
        logger.info(f"[Geo] _safe_to_excel: geschrieben nach {path}.")
    except Exception as e:
        logger.warning(f"[Geo] _safe_to_excel: Schreiben nach {path} fehlgeschlagen ({e}).")


def _read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    logger.info(f"[Geo] _read_parquet_if_exists: path={path}, shape={df.shape}")
    return df

def update_deutschlandatlas_from_vgrdl_excel(
    excel_path: Path,
    *,
    level: str = "krs",
    sheet_names: Optional[List[str]] = None,
    use_loader: bool = True,
    export_excel: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Union[str, int, Path, List]]:
    """
    Führt den kompletten Flow aus:
      1) Excel (VGRdL) laden → ein oder mehrere Sheets
      2) Processed-Parquet(s) unter PROCESSED_DIR schreiben
      3) In das bestehende Merged-Parquet für die Ebene als neue Spalte(n) anhängen
      4) Cache aktualisieren
      5) Interims-Parquet-Dateien aufräumen (nur finale Merged-Dateien behalten)
      
    Args:
        excel_path: Pfad zur VGRdL-Excel
        level: Geografische Ebene (Standard: "krs")
        sheet_names: Liste der zu ladenden Sheets. None = alle numerischen Sheets
        use_loader: Wenn True, nutze load_vgrdl_sheets_as_krs
        export_excel: Wenn True, werden zusätzlich Excel-Dateien exportiert
        log: Logger-Funktion
        
    Returns:
        Dictionary mit Informationen zu allen erzeugten Indikatoren
    """
    if log is None:
        log = lambda m: logger.info(str(m))

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel nicht gefunden: {excel_path}")

    log(f"[Geo] update_deutschlandatlas_from_vgrdl_excel: starte für {excel_path}, level={level}")

    # 1) Excel laden - alle gewünschten Sheets
    if use_loader:
        try:
            if sheet_names:
                log(f"[Geo] Lade spezifische Sheets: {sheet_names}")
            else:
                log("[Geo] Lade alle numerischen Sheets...")
                
            excel_df = load_vgrdl_sheets_as_krs(excel_path, sheet_names=sheet_names)
            
            if not isinstance(excel_df, pd.DataFrame):
                raise RuntimeError("load_vgrdl_sheets_as_krs lieferte kein DataFrame.")
        except Exception as e:
            log(f"[Geo] load_vgrdl_sheets_as_krs fehlgeschlagen ({e})")
            raise
    else:
        raise ValueError("use_loader=False wird nicht mehr unterstützt für Multi-Sheet Import")

    # Erwartete Struktur sicherstellen
    required_cols = {"GKZ", "value", "year", "sheet", "description"}
    if not required_cols.issubset(excel_df.columns):
        raise RuntimeError(
            f"Excel-DataFrame hat nicht die erwarteten Spalten {required_cols}, "
            f"vorhanden: {list(excel_df.columns)}"
        )

    excel_df = excel_df.copy()
    excel_df["GKZ"] = _mk_rs5_from_any_series(excel_df["GKZ"])

    # 2) Bestehendes Merged-Parquet laden
    merged_path = _resolve_level_path(level)
    if not merged_path:
        raise RuntimeError(f"Unbekannte Ebene: {level}")

    if not merged_path.exists():
        raise FileNotFoundError(
            f"Für level='{level}' wurde kein bestehendes Merged-Parquet gefunden: {merged_path}"
        )

    merged = _get_df_cached(level)
    if merged is None:
        merged = pd.read_parquet(merged_path)

    log(f"[Geo] Bestehendes Merged-Parquet geladen: path={merged_path}, shape={merged.shape}")

    if "GKZ" not in merged.columns:
        raise RuntimeError(f"Das Merged-Parquet für level='{level}' enthält keine 'GKZ'-Spalte.")

    merged = merged.copy()
    original_gkz_dtype = merged["GKZ"].dtype

    # Join-Key vorbereiten
    merged["_merge_key"] = _mk_rs5_from_any_series(merged["GKZ"])

    # 3) Für jedes Sheet einen Indikator erstellen
    indicators_info = []
    
    def sanitize_filename(name: str, sheet: str, max_length: int = 100) -> str:
        """
        Erstellt einen sicheren Dateinamen aus dem Indikator-Namen.
        
        Args:
            name: Indikator-Name (kann lang sein)
            sheet: Sheet-Nummer (z.B. "2.4.1")
            max_length: Maximale Länge des Dateinamens (ohne .parquet)
        
        Returns:
            Sicherer Dateiname
        """
        import re
        import hashlib
        
        # Bereinige den Namen
        safe = name.lower()
        # Ersetze Umlaute
        safe = safe.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        # Ersetze problematische Zeichen durch Unterstriche
        safe = re.sub(r'[^\w\s-]', '_', safe)
        # Mehrfache Unterstriche/Leerzeichen reduzieren
        safe = re.sub(r'[\s_]+', '_', safe)
        # Führende/trailing Unterstriche entfernen
        safe = safe.strip('_')
        
        # Wenn zu lang: Kürze und füge Hash hinzu für Eindeutigkeit
        if len(safe) > max_length:
            # Hash der Originalbeschreibung für Eindeutigkeit
            hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
            # Kürze auf max_length - 9 (8 für hash + 1 für Unterstrich)
            safe = safe[:max_length - 9] + '_' + hash_suffix
        
        # Falls immer noch zu lang (sehr langer Pfad-Präfix), nutze Sheet-Nummer als Fallback
        if len(safe) > max_length:
            safe = f"vgrdl_sheet_{sheet.replace('.', '_')}"
        
        return safe
    
    for sheet_name in excel_df["sheet"].unique():
        sheet_data = excel_df[excel_df["sheet"] == sheet_name].copy()
        
        year_val = int(sheet_data["year"].iloc[0])
        description = sheet_data["description"].iloc[0] if "description" in sheet_data.columns else ""
        unit = sheet_data["unit"].iloc[0] if "unit" in sheet_data.columns else ""
        
        # Indikator-Name aus Beschreibung generieren (wird als Spaltenname verwendet)
        if description:
            indicator_name = description.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        else:
            indicator_name = f"vgrdl_{sheet_name.replace('.', '_')}_{year_val}"

        log(
            f"[Geo] Verarbeite Sheet '{sheet_name}': indicator='{indicator_name}', "
            f"year={year_val}, rows={len(sheet_data)}"
        )

        # KRITISCH: Sicherer Dateiname (gekürzt, ohne problematische Zeichen)
        safe_filename = sanitize_filename(indicator_name, sheet_name, max_length=100)
        processed_file = PROCESSED_DIR / f"{safe_filename}.parquet"
        
        # Processed-Datei wegschreiben
        processed_df = sheet_data[["GKZ", "value", "year"]].copy()
        processed_df["description"] = description
        processed_df["unit"] = unit
        processed_df["sheet"] = sheet_name
        _safe_to_parquet(processed_df, processed_file)

        # In Merged-DF einfügen
        excel_add = sheet_data[["GKZ", "value"]].copy()
        excel_add.rename(columns={"GKZ": "_merge_key", "value": indicator_name}, inplace=True)

        # ggf. alte Spalte entfernen
        if indicator_name in merged.columns:
            log(f"[Geo] Indicator '{indicator_name}' existiert bereits – wird überschrieben.")
            merged.drop(columns=[indicator_name], inplace=True)

        merged = merged.merge(excel_add, on="_merge_key", how="left")
        
        matched = int(merged[indicator_name].notna().sum())
        total = int(len(merged))

        log(
            f"[Geo] Sheet '{sheet_name}' -> Indikator '{indicator_name}': "
            f"matched={matched}/{total}"
        )

        indicators_info.append({
            "sheet": sheet_name,
            "indicator_name": indicator_name,
            "description": description,
            "unit": unit,
            "year": year_val,
            "matched": matched,
            "total": total,
            "processed_file": processed_file,
        })

    # _merge_key entfernen
    merged.drop(columns=["_merge_key"], inplace=True)

    # GKZ-Datentyp wiederherstellen
    if merged["GKZ"].dtype != original_gkz_dtype:
        log(f"[Geo] GKZ-Datentyp wird zurückkonvertiert: {merged['GKZ'].dtype} → {original_gkz_dtype}")
        merged["GKZ"] = merged["GKZ"].astype(original_gkz_dtype)

    # Speichern
    _safe_to_parquet(merged, merged_path)
    
    if export_excel:
        merged_excel_path = merged_path.with_suffix(".xlsx")
        _safe_to_excel(merged, merged_excel_path, index=False)
    else:
        merged_excel_path = None

    log(
        f"[Geo] Merge abgeschlossen: {len(indicators_info)} Indikatoren hinzugefügt, "
        f"output_parquet={merged_path}"
    )

    # 4) Cache aktualisieren - für jeden neuen Indikator
    for info in indicators_info:
        indicator_name = info["indicator_name"]
        for_cache = merged[["GKZ", indicator_name]].copy()
        
        name_col = next(
            (c for c in ("Kreisname", "Gemeindename", "Gemeindeverbandsname", "NAME") if c in merged.columns),
            None,
        )
        if name_col:
            for_cache = merged[["GKZ", name_col, indicator_name]].copy()

        add_indicator_to_cache(level, for_cache, indicator_name)

    # 5) CLEANUP: Lösche alle Interims-Parquet-Dateien außer den drei finalen Merged-Dateien
    _cleanup_interim_parquets(log)

    # Return-Dictionary mit allen Indikatoren
    return {
        "level": level,
        "merged_parquet": merged_path,
        "merged_excel": merged_excel_path,
        "indicators": indicators_info,  # Liste aller Indikatoren
        "count": len(indicators_info),
    }


def _cleanup_interim_parquets(log: Callable[[str], None]) -> None:
    """
    Löscht alle Interims-Parquet-Dateien aus PROCESSED_DIR,
    außer den drei finalen Merged-Dateien.
    
    Behält:
      - Deutschlandatlas_GEM_merged.parquet
      - Deutschlandatlas_KRS_merged.parquet
      - Deutschlandatlas_VBGEM_merged.parquet
    
    Löscht:
      - Alle anderen .parquet Dateien (VGRdL-Interims-Dateien)
    """
    if not PROCESSED_DIR.exists():
        log(f"[Geo] Cleanup: PROCESSED_DIR existiert nicht: {PROCESSED_DIR}")
        return
    
    # Dateien, die behalten werden sollen
    keep_files = {
        "Deutschlandatlas_GEM_merged.parquet",
        "Deutschlandatlas_KRS_merged.parquet",
        "Deutschlandatlas_VBGEM_merged.parquet",
    }
    
    # Optional: Auch Excel-Versionen behalten, falls vorhanden
    keep_files_with_excel = keep_files | {
        "Deutschlandatlas_GEM_merged.xlsx",
        "Deutschlandatlas_KRS_merged.xlsx",
        "Deutschlandatlas_VBGEM_merged.xlsx",
    }
    
    deleted_count = 0
    kept_count = 0
    
    try:
        # Iteriere über alle Parquet- und Excel-Dateien im Verzeichnis
        for file_path in PROCESSED_DIR.glob("*"):
            # Nur Dateien (keine Verzeichnisse)
            if not file_path.is_file():
                continue
            
            # Nur .parquet und .xlsx Dateien
            if file_path.suffix not in [".parquet", ".xlsx"]:
                continue
            
            # Prüfe ob Datei behalten werden soll
            if file_path.name in keep_files_with_excel:
                kept_count += 1
                log(f"[Geo] Cleanup: Behalte {file_path.name}")
            else:
                # Lösche Interims-Datei
                try:
                    file_path.unlink()
                    deleted_count += 1
                    log(f"[Geo] Cleanup: Gelöscht {file_path.name}")
                except Exception as e:
                    log(f"[Geo] Cleanup: Fehler beim Löschen von {file_path.name}: {e}")
        
        log(
            f"[Geo] Cleanup abgeschlossen: {kept_count} Dateien behalten, "
            f"{deleted_count} Interims-Dateien gelöscht"
        )
        
    except Exception as e:
        log(f"[Geo] Cleanup: Fehler beim Aufräumen: {e}")


def _rebuild_deutschlandatlas_base_files(
    *,
    levels: Iterable[str],
    log: callable,
    export_excel: bool = False,
) -> Dict[str, Dict[str, Union[str, int, Path]]]:
    """
    Baut die Basis-Dateien Deutschlandatlas_KRS/GEM/VBGEM_merged.parquet
    direkt aus der Deutschlandatlas.xlsx nach der bestehenden Skriptlogik neu.
    GKZ und Struktur entsprechen dem manuellen Script.
    
    Args:
        export_excel: Wenn True, werden zusätzlich Excel-Dateien exportiert (Standard: False)
    """
    SRC_PATH = DATA_INPUT_DIR / "Deutschlandatlas.xlsx"

    if not SRC_PATH.exists():
        log(
            f"[Geo] _rebuild_deutschlandatlas_base_files: Deutschlandatlas.xlsx nicht gefunden unter {SRC_PATH} – "
            f"Basis-Rebuild nicht möglich."
        )
        return {}

    PREFIX_CONFIG = {
        "Deutschlandatlas_GEM": "Gemeindename",
        "Deutschlandatlas_KRS": "Kreisname",
        "Deutschlandatlas_VBGEM": "Gemeindeverbandsname",
    }

    LEVEL_TO_PREFIX = {
        "gem": "Deutschlandatlas_GEM",
        "krs": "Deutschlandatlas_KRS",
        "vbgem": "Deutschlandatlas_VBGEM",
    }

    active_prefixes = {
        LEVEL_TO_PREFIX[lvl]
        for lvl in levels
        if lvl in LEVEL_TO_PREFIX
    }

    def normalize_gkz(val, width: int = 8) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        try:
            num = int(float(s.replace(" ", "")))
            return str(num).zfill(width)
        except (ValueError, TypeError):
            if "." in s:
                s = s.split(".")[0]
            s = s.replace(" ", "")
            return s.zfill(width)

    def longest_name_per_gkz(df_names: pd.DataFrame, name_col: str) -> pd.DataFrame:
        df = df_names.copy()
        df["__len__"] = df[name_col].fillna("").astype(str).str.len()
        df = (
            df.sort_values(["GKZ", "__len__"], ascending=[True, False])
              .drop_duplicates(subset=["GKZ"], keep="first")
              .drop(columns=["__len__"])
        )
        return df

    def build_vbgem_gkz_canonical_map(all_names: pd.DataFrame, name_col: str) -> dict:
        fix_map = {}
        for name, grp in all_names.groupby(name_col):
            gkz_list = [g for g in grp["GKZ"].dropna().unique()]
            if not gkz_list:
                continue
            zero_gkz = [g for g in gkz_list if str(g).startswith("0")]
            if zero_gkz:
                canonical = min(zero_gkz)
            else:
                canonical = min(gkz_list)
            for g in gkz_list:
                fix_map[g] = canonical
        return fix_map

    log(f"[Geo] _rebuild_deutschlandatlas_base_files: Lese Deutschlandatlas.xlsx von {SRC_PATH}")
    xls = pd.ExcelFile(SRC_PATH)

    grouped = {p: [] for p in PREFIX_CONFIG.keys()}
    for sheet in xls.sheet_names:
        for prefix in PREFIX_CONFIG.keys():
            if sheet.startswith(prefix):
                df = pd.read_excel(xls, sheet_name=sheet)
                grouped[prefix].append((sheet, df))
                break

    results: Dict[str, Dict[str, Union[str, int, Path]]] = {}

    for prefix, sheets in grouped.items():
        if prefix not in active_prefixes:
            continue
        if not sheets:
            log(f"[Geo] _rebuild_deutschlandatlas_base_files: Keine Sheets für Präfix '{prefix}' gefunden – überspringe.")
            continue

        target_name_col = PREFIX_CONFIG[prefix]
        log(f"[Geo] _rebuild_deutschlandatlas_base_files: Bearbeite Präfix '{prefix}' mit Ziel-Namensspalte '{target_name_col}'")

        name_frames = []
        dataframes_for_merge = []

        for sheet_name, df in sheets:
            if df.shape[1] < 1:
                continue

            gkz_col = df.columns[0]
            df[gkz_col] = df[gkz_col].apply(normalize_gkz)

            if df.shape[1] >= 2:
                current_name_col = df.columns[1]
                tmp = df[[gkz_col, current_name_col]].copy()
                tmp = tmp.rename(columns={gkz_col: "GKZ", current_name_col: target_name_col})
                name_frames.append(tmp)

            df = df.rename(columns={gkz_col: "GKZ"})

            if df.shape[1] >= 2:
                second_col = df.columns[1]
                if second_col != "GKZ":
                    df = df.drop(columns=[second_col], errors="ignore")

            dataframes_for_merge.append(df)

        # VBGEM-Sonderfall
        if prefix == "Deutschlandatlas_VBGEM" and name_frames:
            all_names_raw = pd.concat(name_frames, ignore_index=True)
            fix_map = build_vbgem_gkz_canonical_map(all_names_raw, target_name_col)

            fixed_name_frames = []
            for nf in name_frames:
                nf = nf.copy()
                nf["GKZ"] = nf["GKZ"].map(fix_map).fillna(nf["GKZ"])
                fixed_name_frames.append(nf)
            name_frames = fixed_name_frames

            fixed_dataframes_for_merge = []
            for df in dataframes_for_merge:
                df = df.copy()
                df["GKZ"] = df["GKZ"].map(fix_map).fillna(df["GKZ"])
                fixed_dataframes_for_merge.append(df)
            dataframes_for_merge = fixed_dataframes_for_merge

        # doppelte Spaltennamen behandeln
        level_suffix = prefix.rsplit("_", 1)[-1]

        all_cols = []
        for df in dataframes_for_merge:
            all_cols.extend([c for c in df.columns if c != "GKZ"])

        from collections import Counter as _Counter
        counts = _Counter(all_cols)
        duplicate_cols = {col for col, cnt in counts.items() if cnt > 1}

        if duplicate_cols:
            fixed_dfs = []
            for df in dataframes_for_merge:
                rename_dict = {c: f"{level_suffix}_{c}" for c in df.columns if c in duplicate_cols}
                df = df.rename(columns=rename_dict)
                fixed_dfs.append(df)
            dataframes_for_merge = fixed_dfs

        # Namen bereinigen
        if name_frames:
            all_names = pd.concat(name_frames, ignore_index=True)
            all_names = all_names[all_names[target_name_col].notna() & (all_names[target_name_col] != "")]
            unique_names = longest_name_per_gkz(all_names, target_name_col)
        else:
            unique_names = pd.DataFrame(columns=["GKZ", target_name_col])

        # alle Sheets über GKZ mergen
        merged = None
        for df in dataframes_for_merge:
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="GKZ", how="outer")

        if merged is None:
            log(f"[Geo] _rebuild_deutschlandatlas_base_files: Keine Datenframes für Präfix '{prefix}' – überspringe.")
            continue

        final_df = pd.merge(unique_names, merged, on="GKZ", how="right")
        final_df["GKZ"] = final_df["GKZ"].apply(lambda x: normalize_gkz(x, 8)).astype(str)
        final_df = final_df.sort_values(["GKZ"])

        PREFIX_TO_LEVEL = {v: k for k, v in LEVEL_TO_PREFIX.items()}
        level = PREFIX_TO_LEVEL.get(prefix)
        if level is None:
            log(f"[Geo] _rebuild_deutschlandatlas_base_files: Kein Level-Mapping für Präfix '{prefix}' – überspringe Save.")
            continue

        out_parquet = PARQUETS_BY_LEVEL[level]
        _safe_to_parquet(final_df, out_parquet)
        
        if export_excel:
            out_excel = out_parquet.with_suffix(".xlsx")
            _safe_to_excel(final_df, out_excel, index=False)
            log(f"[Geo] _rebuild_deutschlandatlas_base_files: geschrieben: {out_parquet} + Excel (rows={len(final_df)})")
        else:
            out_excel = None
            log(f"[Geo] _rebuild_deutschlandatlas_base_files: geschrieben: {out_parquet} (rows={len(final_df)})")

        results[level] = {
            "level": level,
            "merged_parquet": out_parquet,
            "merged_excel": out_excel,
            "rows": int(len(final_df)),
        }

    return results


def rebuild_deutschlandatlas_files(
    *,
    levels: Iterable[str] = ("krs", "gem", "vbgem"),
    vgrdl_excel_name: str = "vgrdl_r2b1_bs2023.xlsx",
    export_excel: bool = False,
    force_rebuild: bool = False,
    logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
) -> Dict[str, Dict[str, Union[str, int, Path]]]:
    """
    High-Level-Funktion für app.py:

    - Prüft zunächst, ob die Basis-Dateien bereits existieren
    - Baut nur fehlende Dateien neu auf (außer force_rebuild=True)
    - Falls die VGRdL-Excel vorhanden ist, wird der VGRdL-Indikator 
      NUR FÜR DIE KREISEBENE (KRS) angehängt (falls noch nicht vorhanden)
      
    Args:
        levels: Ebenen, die aufgebaut werden sollen (krs, gem, vbgem)
        vgrdl_excel_name: Name der VGRdL-Excel-Datei
        export_excel: Wenn True, werden zusätzlich Excel-Dateien exportiert (Standard: False)
        force_rebuild: Wenn True, werden Dateien auch neu aufgebaut wenn sie existieren
        logger: Logger-Instanz oder Callable für Logging
        
    Returns:
        Dictionary mit Informationen zu den erzeugten Dateien pro Ebene
    """
    if isinstance(logger, logging.Logger):
        log = lambda m: logger.info(str(m))
    elif callable(logger):
        log = lambda m: logger(str(m))
    else:
        log = lambda m: print(str(m))

    log("[Geo] rebuild_deutschlandatlas_files: Starte Check der Deutschlandatlas-Dateien ...")

    # 1) Prüfe welche Dateien bereits existieren
    existing_levels = []
    missing_levels = []
    
    for level in levels:
        parquet_path = PARQUETS_BY_LEVEL.get(level)
        if parquet_path and parquet_path.exists():
            existing_levels.append(level)
            log(f"[Geo] rebuild_deutschlandatlas_files: Level '{level}' existiert bereits: {parquet_path}")
        else:
            missing_levels.append(level)
            log(f"[Geo] rebuild_deutschlandatlas_files: Level '{level}' fehlt: {parquet_path}")
    
    # 2) Entscheide ob Rebuild nötig ist
    if not missing_levels and not force_rebuild:
        log("[Geo] rebuild_deutschlandatlas_files: Alle Dateien existieren bereits, kein Rebuild nötig.")
        
        # Erstelle Result-Dictionary für existierende Dateien
        results = {}
        for level in existing_levels:
            parquet_path = PARQUETS_BY_LEVEL[level]
            try:
                df = pd.read_parquet(parquet_path)
                results[level] = {
                    "level": level,
                    "merged_parquet": parquet_path,
                    "merged_excel": None,
                    "rows": int(len(df)),
                    "status": "existing",
                }
            except Exception as e:
                log(f"[Geo] rebuild_deutschlandatlas_files: Fehler beim Lesen von {parquet_path}: {e}")
                results[level] = {
                    "level": level,
                    "merged_parquet": parquet_path,
                    "merged_excel": None,
                    "rows": 0,
                    "status": "error",
                }
        
        # Prüfe VGRdL nur wenn KRS existiert
        if "krs" in existing_levels:
            _check_and_add_vgrdl_if_missing(
                results=results,
                vgrdl_excel_name=vgrdl_excel_name,
                export_excel=export_excel,
                force_rebuild=force_rebuild,
                log=log,
            )
        
        return results
    
    # 3) Basis-Rebuild nur für fehlende Ebenen (oder alle bei force_rebuild)
    if force_rebuild:
        log("[Geo] rebuild_deutschlandatlas_files: force_rebuild=True, baue alle Dateien neu auf ...")
        levels_to_rebuild = list(levels)
    else:
        log(f"[Geo] rebuild_deutschlandatlas_files: Baue nur fehlende Ebenen neu auf: {missing_levels}")
        levels_to_rebuild = missing_levels
    
    base_results = _rebuild_deutschlandatlas_base_files(
        levels=levels_to_rebuild,
        log=log,
        export_excel=export_excel,
    )

    if not base_results:
        log("[Geo] rebuild_deutschlandatlas_files: Kein Basis-Rebuild möglich (siehe Log oben).")
        return {}

    # Kombiniere mit existierenden Dateien
    results = dict(base_results)
    for level in existing_levels:
        if level not in results:  # Nur hinzufügen wenn nicht neu gebaut
            parquet_path = PARQUETS_BY_LEVEL[level]
            try:
                df = pd.read_parquet(parquet_path)
                results[level] = {
                    "level": level,
                    "merged_parquet": parquet_path,
                    "merged_excel": None,
                    "rows": int(len(df)),
                    "status": "existing",
                }
            except Exception:
                pass

    # 4) VGRdL-Merge für KRS
    _check_and_add_vgrdl_if_missing(
        results=results,
        vgrdl_excel_name=vgrdl_excel_name,
        export_excel=export_excel,
        force_rebuild=force_rebuild,
        log=log,
    )

    # Hinweise für andere Ebenen
    for level in ["gem", "vbgem"]:
        if level in levels and level in results:
            log(
                f"[Geo] rebuild_deutschlandatlas_files: Ebene '{level}' bereit. "
                f"VGRdL-Daten sind NUR auf Kreisebene verfügbar."
            )

    return results


def _check_and_add_vgrdl_if_missing(
    *,
    results: Dict[str, Dict[str, Union[str, int, Path]]],
    vgrdl_excel_name: str,
    export_excel: bool,
    force_rebuild: bool,
    log: Callable[[str], None],
) -> None:
    """
    Hilfsfunktion: Prüft ob VGRdL-Daten fehlen und fügt sie hinzu falls nötig.
    
    Args:
        results: Results-Dictionary (wird in-place modifiziert)
        vgrdl_excel_name: Name der VGRdL-Excel
        export_excel: Excel-Export aktiviert?
        force_rebuild: Erzwinge Neuaufbau?
        log: Logger-Funktion
    """
    vgrdl_excel_path = DATA_INPUT_DIR / vgrdl_excel_name
    
    log(
        f"[Geo] _check_and_add_vgrdl_if_missing: Suche VGRdL-Excel unter "
        f"{vgrdl_excel_path} | exists={vgrdl_excel_path.exists()}"
    )
    
    if not vgrdl_excel_path.exists():
        log("[Geo] _check_and_add_vgrdl_if_missing: VGRdL-Excel nicht gefunden – überspringe.")
        return

    if "krs" not in results:
        log("[Geo] _check_and_add_vgrdl_if_missing: Ebene 'krs' nicht in results – überspringe VGRdL-Merge.")
        return

    # Prüfe ob VGRdL-Indikatoren bereits vorhanden sind
    krs_parquet = PARQUETS_BY_LEVEL["krs"]
    
    if not force_rebuild and krs_parquet.exists():
        try:
            df_krs = pd.read_parquet(krs_parquet)
            # Prüfe ob bereits VGRdL-Spalten vorhanden sind
            # (erkennbar z.B. an "Bruttoinlandsprodukt" im Spaltennamen)
            vgrdl_cols = [c for c in df_krs.columns if "Bruttoinlandsprodukt" in c or "Bruttowertschoepfung" in c]
            
            if vgrdl_cols:
                log(
                    f"[Geo] _check_and_add_vgrdl_if_missing: VGRdL-Indikatoren bereits vorhanden "
                    f"({len(vgrdl_cols)} Spalten gefunden), überspringe Merge."
                )
                # Ergänze Info im Results-Dictionary
                if "indicators" not in results["krs"]:
                    results["krs"]["vgrdl_status"] = "existing"
                    results["krs"]["vgrdl_indicators_count"] = len(vgrdl_cols)
                return
        except Exception as e:
            log(f"[Geo] _check_and_add_vgrdl_if_missing: Fehler beim Prüfen vorhandener VGRdL-Daten: {e}")
    
    # VGRdL-Merge durchführen
    try:
        log("[Geo] _check_and_add_vgrdl_if_missing: Starte VGRdL-Merge für Ebene 'krs' ...")
        info = update_deutschlandatlas_from_vgrdl_excel(
            vgrdl_excel_path,
            level="krs",
            sheet_names=None,  # None = alle numerischen Sheets
            export_excel=export_excel,
            log=log,
        )
        results["krs"].update(info)
        
        # Log für jeden importierten Indikator
        for ind in info.get("indicators", []):
            log(
                f"[Geo] _check_and_add_vgrdl_if_missing: VGRdL Sheet '{ind['sheet']}' -> "
                f"Indikator '{ind['indicator_name']}' "
                f"(matched={ind['matched']}/{ind['total']})."
            )
            
        log(
            f"[Geo] _check_and_add_vgrdl_if_missing: VGRdL-Merge erfolgreich - "
            f"{info['count']} Indikatoren hinzugefügt."
        )
    except Exception as e:
        log(f"[Geo] _check_and_add_vgrdl_if_missing: VGRdL-Merge fehlgeschlagen ({e}).")


def app_preload_vgrdl(
    logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    force_rebuild: bool = False,
) -> Optional[Dict[str, Union[str, int, Path]]]:
    """
    Backwards-kompatible Convenience-Funktion für app.py:
    - Ruft rebuild_deutschlandatlas_files NUR für die KRS-Ebene auf
    - Baut nur neu auf wenn Dateien fehlen (außer force_rebuild=True)
    - Gibt (wie bisher) das Info-Dict der KRS-Ebene zurück, falls vorhanden
    
    Args:
        logger: Logger-Instanz oder Callable
        force_rebuild: Erzwinge Neuaufbau auch wenn Dateien existieren
    """
    if isinstance(logger, logging.Logger):
        log = lambda m: logger.info(str(m))
    elif callable(logger):
        log = lambda m: logger(str(m))
    else:
        log = lambda m: print(str(m))

    log(f"[Geo] app_preload_vgrdl: Starte Preload für Ebene 'krs' (force_rebuild={force_rebuild}).")
    results = rebuild_deutschlandatlas_files(
        logger=log,
        levels=["krs"],
        force_rebuild=force_rebuild,
    )

    info = results.get("krs")
    if info:
        status = info.get("status", "unknown")
        if status == "existing":
            log(f"[Geo] app_preload_vgrdl: KRS bereits vorhanden – keine Neuberechnung nötig.")
        else:
            log(f"[Geo] app_preload_vgrdl: KRS neu aufgebaut.")
        
        if "indicators" in info:
            log(f"[Geo] app_preload_vgrdl: {info['count']} VGRdL-Indikatoren vorhanden.")
    else:
        log("[Geo] app_preload_vgrdl: Ebene 'krs' konnte nicht geladen/erstellt werden.")
    
    return info



def load_vgrdl_sheets_as_krs(excel_path: Path, sheet_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Liest ein oder mehrere VGRdL-Sheets, filtert NUTS3-Zeilen
    und gibt kombinierte Daten zurück.
    
    Für jedes Sheet werden ausgelesen:
    - A1: Hauptkonzept
    - A2: Unterkonzept (mit Nummerierung)
    - A3: Maßeinheit
    
    Args:
        excel_path: Pfad zur VGRdL-Excel
        sheet_names: Liste der zu ladenden Sheets. 
                     None = alle numerischen Sheets (z.B. "1.1", "2.1.1", "4")
    
    Returns:
        DataFrame mit Spalten: ['GKZ', 'Kreisname_raw', 'value', 'year', 'description', 'unit', 'sheet']
    """
    logger.info(f"[Geo] load_vgrdl_sheets_as_krs: Lade '{excel_path}'.")
    
    # Wenn keine Sheets angegeben, finde alle numerischen Sheets
    if sheet_names is None:
        import re
        xls = pd.ExcelFile(excel_path)
        # ERWEITERTE Numerische Sheets: "1", "1.1", "2.1.1", "2.3.1.1", etc.
        # Pattern: Beginnt mit Ziffer, dann beliebig viele ".Ziffer" Kombinationen
        sheet_names = [s for s in xls.sheet_names if re.match(r'^\d+(\.\d+)*$', s.strip())]
        logger.info(f"[Geo] load_vgrdl_sheets_as_krs: Gefundene numerische Sheets: {sorted(sheet_names)}")
    
    if not sheet_names:
        raise ValueError("Keine numerischen Sheets gefunden oder angegeben.")
    
    all_results = []
    
    for sheet_name in sheet_names:
        logger.info(f"[Geo] load_vgrdl_sheets_as_krs: Verarbeite Sheet '{sheet_name}'...")
        
        try:
            raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        except Exception as e:
            logger.warning(f"[Geo] load_vgrdl_sheets_as_krs: Sheet '{sheet_name}' konnte nicht geladen werden: {e}")
            continue

        # Beschreibung aus A1 und A2 kombinieren, Maßeinheit aus A3
        main_concept = None
        sub_concept_clean = None
        unit = None
        description = None
        
        try:
            # A1 (Zeile 0, Spalte 0) = Hauptkonzept
            if len(raw) > 0 and pd.notna(raw.iloc[0, 0]):
                main_concept = str(raw.iloc[0, 0]).strip()
            
            # A2 (Zeile 1, Spalte 0) = Unterkonzept
            if len(raw) > 1 and pd.notna(raw.iloc[1, 0]):
                sub_concept = str(raw.iloc[1, 0]).strip()
                # Entferne führende Nummerierung vom Unterkonzept (flexibler Pattern)
                import re
                sub_concept_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', sub_concept).strip()
            
            # A3 (Zeile 2, Spalte 0) = Maßeinheit
            if len(raw) > 2 and pd.notna(raw.iloc[2, 0]):
                unit = str(raw.iloc[2, 0]).strip()
            
            # Kombiniere Haupt- und Unterkonzept zu "abc - xyz"
            if main_concept and sub_concept_clean:
                description = f"{main_concept} - {sub_concept_clean}"
            elif main_concept:
                description = main_concept
            elif sub_concept_clean:
                description = sub_concept_clean
            else:
                description = f"Sheet {sheet_name}"
                
            logger.info(f"[Geo] Sheet '{sheet_name}': description='{description}', unit='{unit}'")
                
        except Exception as e:
            logger.warning(f"[Geo] Sheet '{sheet_name}': Konnte Beschreibung/Einheit nicht auslesen: {e}")
            description = f"Sheet {sheet_name}"
            unit = ""

        # Header-Zeile finden
        try:
            hdr_idx = raw.index[(raw[0] == "Lfd. Nr.") & (raw[1] == "EU-Code")][0]
        except Exception as e:
            logger.warning(f"[Geo] Sheet '{sheet_name}': Header nicht gefunden, überspringe: {e}")
            continue
            
        df = raw.iloc[hdr_idx+1:].copy()
        df.columns = raw.iloc[hdr_idx].tolist()

        # Nur NUTS3
        if "NUTS 3" not in df.columns:
            logger.warning(f"[Geo] Sheet '{sheet_name}': Keine 'NUTS 3' Spalte gefunden, überspringe.")
            continue
            
        df = df[df["NUTS 3"].notna()].copy()

        # Jahresspalten finden
        year_cols = [c for c in df.columns if isinstance(c, (int, float))]
        if not year_cols:
            logger.warning(f"[Geo] Sheet '{sheet_name}': Keine Jahresspalten gefunden, überspringe.")
            continue
            
        year = int(max(year_cols))

        out = df[["EU-Code", "Regional-schlüssel", "Gebietseinheit", year]].copy()
        out.rename(columns={year: "value"}, inplace=True)

        # KRITISCH: Bereinige non-numerische Werte
        def clean_value(val):
            """Konvertiert Werte zu float, ersetzt ungültige Werte durch NaN."""
            if pd.isna(val):
                return None
            
            # String-Bereinigung
            if isinstance(val, str):
                val = val.strip()
                # Ersetze häufige Platzhalter durch NaN
                if val in ['/', '-', '.', '...', 'x', 'X', '']:
                    return None
                # Entferne Tausendertrennzeichen
                val = val.replace(' ', '').replace('\xa0', '')
                
            # Versuche Konvertierung zu float
            try:
                return float(val)
            except (ValueError, TypeError):
                logger.debug(f"[Geo] Sheet '{sheet_name}': Ungültiger Wert '{val}' -> NaN")
                return None
        
        out["value"] = out["value"].apply(clean_value)
        
        # Entferne Zeilen ohne gültigen Wert
        rows_before = len(out)
        out = out[out["value"].notna()].copy()
        rows_after = len(out)
        
        if rows_before != rows_after:
            logger.info(
                f"[Geo] Sheet '{sheet_name}': {rows_before - rows_after} Zeilen "
                f"mit ungültigen Werten entfernt ({rows_after} verbleiben)"
            )

        def normalize_to_rs5(rs_value):
            rs_str = str(rs_value).strip()
            
            if '.' in rs_str:
                rs_str = rs_str.split('.')[0]
            
            rs_str = ''.join(c for c in rs_str if c.isdigit())
            
            if not rs_str:
                return ''
            
            if len(rs_str) == 2:
                return rs_str + '000'
            
            if len(rs_str) >= 5:
                return rs_str[:5]
            
            return rs_str.zfill(5)
        
        out["GKZ"] = out["Regional-schlüssel"].apply(normalize_to_rs5)
        out["EU-Code"] = out["EU-Code"].astype(str).str.strip()

        # NUTS-1 (Stadtstaaten) UND NUTS-3 (Kreise) akzeptieren
        out = out[out["EU-Code"].str.match(r"^DE[A-Z0-9]{1,3}$", na=False)]

        out = out.rename(columns={"Gebietseinheit": "Kreisname_raw"})
        out = out[["GKZ", "Kreisname_raw", "value"]].copy()
        out["year"] = year
        
        # Metadaten hinzufügen
        out["description"] = description if description else ""
        out["unit"] = unit if unit else ""
        out["sheet"] = sheet_name  # Merke Original-Sheet-Name
        
        # Validierung: Stelle sicher dass value numerisch ist
        if not pd.api.types.is_numeric_dtype(out["value"]):
            logger.warning(
                f"[Geo] Sheet '{sheet_name}': value-Spalte ist nicht numerisch "
                f"(dtype={out['value'].dtype}), konvertiere zu float"
            )
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
        
        all_results.append(out)
        
        logger.info(
            f"[Geo] Sheet '{sheet_name}': {len(out)} Zeilen geladen "
            f"(year={year}, unique_GKZ={out['GKZ'].nunique()}, "
            f"valid_values={out['value'].notna().sum()})"
        )

    if not all_results:
        raise RuntimeError("Keine Daten aus den angegebenen Sheets geladen.")
    
    # Kombiniere alle Sheets
    combined = pd.concat(all_results, ignore_index=True)
    
    logger.info(
        f"[Geo] load_vgrdl_sheets_as_krs: Gesamt {len(combined)} Zeilen aus {len(all_results)} Sheets, "
        f"unique_GKZ={combined['GKZ'].nunique()}, valid_values={combined['value'].notna().sum()}"
    )
    
    return combined



# Backwards-kompatible Alias-Funktion
def load_vgrdl_11_as_krs(excel_path: Path) -> pd.DataFrame:
    """
    Backwards-kompatible Funktion: Lädt nur Sheet "1.1".
    Nutzt intern load_vgrdl_sheets_as_krs.
    """
    return load_vgrdl_sheets_as_krs(excel_path, sheet_names=["1.1"])






# ============================================================
# Hilfsfunktionen für Forecast-Runs (Geo-Analyse)
# ============================================================

def _render_geo_runs_list(runs: list[dict]) -> html.Div:
    """
    Baut die Liste der gespeicherten Forecast-Runs für das Geo-Modal.
    Layout ist bewusst an die Forecast-Seite angelehnt, aber mit eigenem Button-Typ.
    """
    if not runs:
        return html.Div(
            "Keine gespeicherten Prognose-Runs in loader/runs gefunden.",
            className="text-muted",
        )

    # Neueste Runs zuerst
    runs_sorted = sorted(runs, key=lambda r: r.get("timestamp_raw", ""), reverse=True)

    cards = []
    for r in runs_sorted:
        cache_tag = r.get("cache_tag") or "(ohne Cache-Tag)"
        ts_label = r.get("timestamp_label") or r.get("timestamp_raw") or ""
        sektor = r.get("sector") or r.get("sektor") or "-"
        modus = r.get("ui_mode") or r.get("mode") or "-"
        target = r.get("ui_target") or r.get("target") or "-"
        horizon = r.get("forecast_horizon") or r.get("horizon") or "-"
        exogs = r.get("selected_exog") or []

        if isinstance(exogs, (list, tuple)):
            exog_text = ", ".join(map(str, exogs)) if exogs else "– keine –"
        else:
            exog_text = str(exogs)

        cards.append(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        f"{ts_label}",
                                        className="fw-bold",
                                    ),
                                    html.Span(
                                        f"  ·  Cache: {cache_tag}",
                                        className="text-muted ms-2",
                                    ),
                                ],
                                className="mb-1",
                            ),
                            html.Div(
                                [
                                    html.Span(f"Sektor: {sektor}", className="me-3"),
                                    html.Span(f"Ziel: {target}", className="me-3"),
                                    html.Span(f"Modus: {modus}", className="me-3"),
                                    html.Span(f"Horizont: {horizon} Q"),
                                ],
                                className="small text-muted mb-1",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Exogene: ",
                                        className="fw-bold small me-1",
                                    ),
                                    html.Span(exog_text, className="small"),
                                ],
                                className="mb-2",
                            ),
                            dbc.Button(
                                "Diesen Forecast in Geo laden",
                                id={
                                    "type": "geo-load-run-btn",
                                    "cache_tag": cache_tag,
                                    "timestamp": r.get("timestamp_raw"),
                                },
                                color="primary",
                                size="sm",
                            ),
                        ]
                    )
                ],
                className="mb-2",
            )
        )

    return html.Div(cards)

def _load_geo_forecast_from_run(cache_tag: str, ts_raw: str) -> Optional[str]:
    """
    Lädt die Forecast-Daten für die Geo-Seite aus einem gespeicherten Run.

    Priorität:
    1) Pfad aus run_meta["geo_forecast_path"] (geo_forecast.xlsx, Sheet 'FORECAST_TS')
    2) (Optionaler Fallback) production_forecast.csv aus trained_outputs, falls gewünscht

    Rückgabe:
    - JSON-String (orient='split'), passend für geo-forecast-store
    - None, wenn nichts geladen werden konnte
    """
    if not cache_tag or not ts_raw:
        logger.warning("[Geo] _load_geo_forecast_from_run: cache_tag oder ts_raw leer.")
        return None

    run_dir = RUNS_DIR / cache_tag / ts_raw
    if not run_dir.exists():
        logger.warning(f"[Geo] Run-Verzeichnis nicht gefunden: {run_dir}")
        return None

    # 1) run_meta laden
    try:
        meta = _load_run_meta(run_dir)
    except Exception as e:
        logger.error(f"[Geo] _load_geo_forecast_from_run: Konnte run_meta nicht laden: {e}")
        return None

    # 2) Geo-Forecast-Pfad aus Meta ziehen
    geo_path_str = None
    if isinstance(meta, dict):
        geo_path_str = meta.get("geo_forecast_path")

    if geo_path_str:
        geo_path = Path(geo_path_str)
        if not geo_path.is_absolute():
            # falls relativ gespeichert, relativ zu run_dir interpretieren
            geo_path = run_dir / geo_path

        if geo_path.exists():
            try:
                df_forecast = pd.read_excel(geo_path, sheet_name="FORECAST_TS")
            except Exception as e:
                logger.error(
                    f"[Geo] _load_geo_forecast_from_run: Fehler beim Lesen von "
                    f"'FORECAST_TS' aus {geo_path}: {e}"
                )
            else:
                if df_forecast.empty:
                    logger.warning(
                        f"[Geo] _load_geo_forecast_from_run: FORECAST_TS in {geo_path} ist leer."
                    )
                else:
                    try:
                        return df_forecast.to_json(orient="split", date_format="iso")
                    except Exception as e_json:
                        logger.error(
                            f"[Geo] _load_geo_forecast_from_run: Fehler beim Serialisieren des DF: {e_json}"
                        )

        else:
            logger.warning(
                f"[Geo] _load_geo_forecast_from_run: geo_forecast_path existiert nicht: {geo_path}"
            )

    # 3) (Optional) Fallback: nichts gefunden
    logger.warning(
        f"[Geo] _load_geo_forecast_from_run: Keine Geo-Forecast-Daten für Run "
        f"{cache_tag}@{ts_raw} gefunden."
    )
    return None






def create_geo_layout():
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Geo-Analyse Konfiguration"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            "1. Ebene wählen",
                                            className="text-muted small",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            "2. Indikator wählen",
                                            className="text-muted small",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            "3. Anzeigen / Fokus wählen",
                                            className="text-muted small",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            "4. Darstellung",
                                            className="text-muted small",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="mb-2 gx-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Regionale Ebene:",
                                                className="fw-bold mb-1",
                                            ),
                                            dcc.Dropdown(
                                                id="geo-level-dropdown",
                                                className="geo-dropdown",
                                                options=[
                                                    {"label": "Kreise (KRS)", "value": "krs"},
                                                    {"label": "Gemeinden (GEM)", "value": "gem"},
                                                    {
                                                        "label": "Gemeindeverbünde (VBGEM)",
                                                        "value": "vbgem",
                                                    },
                                                ],
                                                value="krs",
                                                clearable=False,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Indikator:",
                                                className="fw-bold mb-1",
                                            ),
                                            dcc.Dropdown(
                                                id="geo-indicator-dropdown",
                                                className="geo-dropdown",
                                                options=[],
                                                placeholder="Indikator wählen...",
                                                clearable=False,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Fokus-Gebiete (optional):",
                                                className="fw-bold mb-1",
                                            ),
                                            dcc.Dropdown(
                                                id="geo-region-dropdown",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="Region(en) auswählen ...",
                                                maxHeight=200,
                                                optionHeight=35,
                                                searchable=True,
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Darstellung:",
                                                className="fw-bold mb-1",
                                            ),
                                            dcc.Dropdown(
                                                id="geo-transform-dropdown",
                                                options=TRANSFORM_OPTIONS,
                                                value="raw",
                                                clearable=False,
                                            ),
                                        ],
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "\u00A0",
                                                className="fw-bold mb-1",
                                                style={"visibility": "hidden"},
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Filter anwenden",
                                                            id="apply-geo-filters-btn",
                                                            color="primary",
                                                            size="sm",
                                                            style={"width": "100%"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Zurücksetzen",
                                                            id="reset-geo-filters-btn",
                                                            color="secondary",
                                                            size="sm",
                                                            style={"width": "100%"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="g-1",
                                            ),
                                            html.Small(
                                                "Apply rendert Karte neu.",
                                                className="text-muted d-block mt-1",
                                                style={"fontSize": "0.7rem"},
                                            ),
                                        ],
                                        width=2,
                                        className="d-flex flex-column justify-content-start",
                                    ),
                                ],
                                className="g-2",
                                align="start",
                            ),
                        ]
                    ),
                ],
                className="settings-panel mb-3",
            ),

            # Hauptbereich
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Regionale Darstellung"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        id="loading-geo-map",
                                        type="circle",
                                        style={
                                            "minHeight": "624px",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                        },
                                        children=html.Div(
                                            html.Iframe(
                                                id="geo-map",
                                                srcDoc=EMPTY_GEO_MAP_HTML,
                                                style={
                                                    "width": "100%",
                                                    "height": "624px",
                                                    "border": "none",
                                                },
                                            ),
                                            style={
                                                "width": "100%",
                                                "height": "624px",
                                            },
                                        ),
                                    )
                                ),
                            ],
                            className="chart-container flex-fill h-100",
                            style={"minHeight": "720px"},
                        ),
                        width=8,
                        className="d-flex",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Details zur Auswahl"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        id="loading-geo-details",
                                        type="circle",
                                        style={
                                            "minHeight": "672px",
                                            "display": "flex",
                                            "AlignItems": "center",
                                            "justifyContent": "center",
                                        },
                                        children=html.Div(
                                            id="geo-detail-panel",
                                            className="vstack gap-2",
                                            style={
                                                "height": "672px",
                                                "overflowY": "auto",
                                                "width": "100%",
                                            },
                                        ),
                                    ),
                                    className="d-flex flex-column",
                                    style={"position": "relative"},
                                ),
                            ],
                            className="chart-container flex-fill h-100",
                            style={"minHeight": "720px"},
                        ),
                        width=4,
                        className="d-flex",
                    ),
                ],
                className="g-2",
                align="stretch",
            ),

            # Tabelle: Regionale Übersicht
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Button(
                                        "Regionale Übersicht anzeigen / ausblenden",
                                        id="toggle-geo-table",
                                        color="link",
                                        className="p-0",
                                    )
                                ),
                                dbc.Collapse(
                                    dbc.CardBody(
                                        [
                                            dag.AgGrid(
                                                id="geo-table",
                                                columnDefs=[],
                                                rowData=[],
                                                className="ag-theme-alpine",
                                                columnSize="sizeToFit",
                                                defaultColDef={
                                                    "resizable": True,
                                                    "sortable": True,
                                                    "filter": True,
                                                    "floatingFilter": True,
                                                },
                                                dashGridOptions={
                                                    "pagination": True,
                                                    "paginationPageSize": 50,
                                                    "domLayout": "autoHeight",
                                                    "suppressHorizontalScroll": True,
                                                },
                                                style={
                                                    "width": "100%",
                                                    "height": "420px",
                                                    "maxWidth": "100%",
                                                    "overflowX": "hidden",
                                                },
                                            ),
                                            html.Small(
                                                "Tipp: Spalten sortieren, filtern oder verbreitern.",
                                                className="text-muted d-block mt-2",
                                            ),
                                        ]
                                    ),
                                    id="geo-table-collapse",
                                    is_open=False,
                                ),
                            ],
                            className="chart-container",
                        ),
                        width=12,
                    )
                ],
                className="mt-2",
            ),

            # Zeitreihen-Analyse (einklappbar)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Button(
                                        "Zeitreihen-Analyse anzeigen / ausblenden",
                                        id="toggle-geo-timeseries",
                                        color="link",
                                        className="p-0",
                                    )
                                ),
                                dbc.Collapse(
                                    [
                                        # Einstellungsleiste inkl. Upload + Marktanteils-Slider
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        # LINKE SPALTE: Marktanteils-Slider
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Marktanteil (%):",
                                                                    className="fw-bold mb-1",
                                                                ),
                                                                dcc.Slider(
                                                                    id="geo-market-share-slider",
                                                                    min=0,
                                                                    max=100,
                                                                    step=1,
                                                                    value=100,
                                                                    marks={
                                                                        0: "0%",
                                                                        25: "25%",
                                                                        50: "50%",
                                                                        75: "75%",
                                                                        100: "100%",
                                                                    },
                                                                ),
                                                                html.Small(
                                                                    "Wird auf die Prognose der Auswahl angewendet.",
                                                                    className="text-muted small",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        # RECHTE SPALTE: Buttons
                                                        dbc.Col(
                                                            [
                                                                dbc.ButtonGroup(
                                                                    [
                                                                        dbc.Button(
                                                                            "Modell laden",
                                                                            id="geo-load-model-btn",
                                                                            color="primary",
                                                                            size="md",
                                                                        ),
                                                                        dcc.Upload(
                                                                            id="geo-forecast-upload",
                                                                            children=dbc.Button(
                                                                                "Daten hochladen",
                                                                                id="geo-upload-data-btn",
                                                                                color="primary",
                                                                                size="md",
                                                                            ),
                                                                            multiple=False,
                                                                            accept=".xlsx,.xls",
                                                                            style={"display": "inline-block"},
                                                                        ),
                                                                        dbc.Button(
                                                                            "Analyse starten",
                                                                            id="geo-start-analysis-btn",
                                                                            color="success",
                                                                            size="md",
                                                                        ),
                                                                    ],
                                                                    className="gap-2",
                                                                ),
                                                                html.Div(
                                                                    id="geo-forecast-upload-status",
                                                                    className="text-muted small text-end mt-1",
                                                                ),
                                                            ],
                                                            width=6,
                                                            className="d-flex flex-column align-items-end",
                                                        ),
                                                    ],
                                                    className="g-2",
                                                    align="center",
                                                ),
                                            ],
                                            className="settings-panel mb-0",
                                            style={"paddingBottom": "10px"},
                                        ),

                                        # Großes Chart
                                        dbc.CardBody(
                                            dcc.Loading(
                                                id="loading-geo-timeseries",
                                                type="circle",
                                                children=dcc.Graph(
                                                    id="geo-timeseries-chart",
                                                    figure={},
                                                    style={
                                                        "height": "600px",
                                                        "width": "100%",
                                                    },
                                                    config={
                                                        "displayModeBar": True,
                                                        "displaylogo": False,
                                                        "modeBarButtonsToRemove": [
                                                            "lasso2d",
                                                            "select2d",
                                                        ],
                                                    },
                                                ),
                                            ),
                                            style={"padding": "0"},
                                        ),

                                        # Numerische Herleitung: Tabelle unterhalb des Charts
                                        dbc.CardBody(
                                            dcc.Loading(
                                                id="loading-geo-forecast-table",
                                                type="circle",
                                                children=dag.AgGrid(
                                                    id="geo-forecast-table",
                                                    columnDefs=[],
                                                    rowData=[],
                                                    className="ag-theme-alpine",
                                                    columnSize="sizeToFit",
                                                    defaultColDef={
                                                        "resizable": True,
                                                        "sortable": True,
                                                        "filter": True,
                                                        "floatingFilter": True,
                                                    },
                                                    dashGridOptions={
                                                        "pagination": False,
                                                        "domLayout": "autoHeight",
                                                        "suppressHorizontalScroll": True,
                                                    },
                                                    style={
                                                        "width": "100%",
                                                        "height": "100%",
                                                        "maxWidth": "100%",
                                                        "overflowX": "hidden",
                                                    },
                                                ),
                                            ),
                                            style={"padding": "10px 0 0 0"},
                                        ),
                                        dbc.CardBody(
                                            html.Small(
                                                "Numerische Herleitung: Gesamtprognose für Deutschland und anteilige Prognose für die ausgewählten Regionen (auf Basis ihres BIP-Anteils).",
                                                className="text-muted d-block",
                                            ),
                                            style={"paddingTop": "4px"},
                                        ),
                                    ],
                                    id="geo-timeseries-collapse",
                                    is_open=False,
                                ),
                            ],
                            className="chart-container",
                        ),
                        width=12,
                    )
                ],
                className="mt-2",
            ),

            # Modal: Forecast-Runs aus loader/runs auswählen
            dbc.Modal(
                [
                    dbc.ModalHeader("Gespeicherte Forecast-Runs"),
                    dbc.ModalBody(
                        html.Div(
                            id="geo-runs-list-body",
                            children="Noch keine Läufe geladen.",
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Schließen",
                            id="geo-close-runs-list",
                            className="ms-auto",
                        )
                    ),
                ],
                id="geo-runs-list-modal",
                is_open=False,
                size="lg",
                scrollable=True,
            ),

            # Stores
            dcc.Store(id="geo-map-input-store", data={}),
            dcc.Store(id="geo-filter-store", data={}),
            dcc.Store(id="geo-region-market-share-store", data={}),
            dcc.Store(id="geo-forecast-store", data=None),

            # Download-Component für CSV-Export
            dcc.Download(id="download-geo-timeseries-csv"),
        ]
    )


# ============================================================
# Callbacks
# ============================================================
def register_geo_callbacks(app):
    # Ebene -> Indikatorliste + Regionenliste
    @app.callback(
        Output("geo-indicator-dropdown", "options"),
        Output("geo-indicator-dropdown", "value"),
        Output("geo-region-dropdown", "options"),
        Input("geo-level-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_indicator_and_regions(level):
        logger.info(f"[Geo] Callback update_indicator_and_regions: level={level}")
        indicator_opts = _options_for_level(level)
        indicator_val = indicator_opts[0]["value"] if indicator_opts else None
        region_opts = _get_region_options(level)
        logger.info(
            f"[Geo] update_indicator_and_regions: level={level}, "
            f"indicator_val={indicator_val}, regions={len(region_opts)}"
        )
        return indicator_opts, indicator_val, region_opts

    # Tabelle einklappen
    @app.callback(
        Output("geo-table-collapse", "is_open"),
        Input("toggle-geo-table", "n_clicks"),
        State("geo-table-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_geo_table(n_clicks, is_open):
        return not is_open


    @app.callback(
        Output("geo-map-input-store", "data"),
        Output("geo-table", "rowData"),
        Output("geo-table", "columnDefs"),
        Output("geo-detail-panel", "children"),
        Input("apply-geo-filters-btn", "n_clicks"),
        State("geo-level-dropdown", "value"),
        State("geo-indicator-dropdown", "value"),
        State("geo-region-dropdown", "value"),
        State("geo-transform-dropdown", "value"),
        prevent_initial_call=True,
    )
    def apply_geo_filters(n_clicks, level, indicator, selected_gkz, transform):
        logger.info(
            f"[Geo] apply_geo_filters: n_clicks={n_clicks}, level={level}, "
            f"indicator={indicator}, selected={len(selected_gkz or [])}, transform={transform}"
        )
        
        # Basis-Slice holen
        slice_df = _get_indicator_slice(level, indicator)
        df_full = None

        if slice_df is None:
            df_full = _get_df_cached(level)
            if df_full is None or not indicator or indicator not in df_full.columns:
                logger.warning(
                    f"[Geo] apply_geo_filters: Keine Daten gefunden für level={level}, indicator={indicator}."
                )
                return (
                    {"has_data": False},
                    [],
                    [],
                    [html.P("Keine Daten gefunden.", className="text-muted")],
                )

        # Name-Spalte bestimmen
        if slice_df is not None:
            name_col = next(
                (
                    c
                    for c in (
                        "Gemeindename",
                        "Kreisname",
                        "Gemeindeverbandsname",
                        "NAME",
                    )
                    if c in slice_df.columns
                ),
                None,
            )
        else:
            name_col = _detect_name_col(df_full, level) if df_full is not None else None

        # Spaltendefinition für Tabelle
        cols_order = ["GKZ"]
        column_defs = [{"field": "GKZ", "headerName": "GKZ"}]
        if name_col:
            cols_order.append(name_col)
            column_defs.append(
                {"field": name_col, "headerName": name_col, "flex": 1}
            )
        cols_order.append(indicator)
        
        # Header-Label basierend auf Transform
        if transform == "raw":
            indicator_header = indicator
            stats_label = "Rohwerte"
        elif transform == "pct_visible":
            indicator_header = f"{indicator} (% v. Angez.)"
            stats_label = "% von Angezeigt"
        elif transform == "pct_total":
            indicator_header = f"{indicator} (% v. Ges.)"
            stats_label = "% vom Gesamt"
        else:
            indicator_header = indicator
            stats_label = "Rohwerte"
            
        column_defs.append(
            {
                "field": indicator,
                "headerName": indicator_header,
                "type": "rightAligned",
                "filter": "agNumberColumnFilter",
            }
        )

        # Basis-Datenquelle (UNGEFILTERT - vollständig)
        if slice_df is not None:
            base_df_full = slice_df
        else:
            base_df_full = df_full[cols_order].copy()

        logger.info(
            f"[Geo] apply_geo_filters: base_df_full shape={base_df_full.shape}, "
            f"columns={list(base_df_full.columns)}"
        )

        # WICHTIG: Auswahl-Filter anwenden für Tabelle
        selected_set = {str(g) for g in (selected_gkz or [])}
        
        if selected_set and len(selected_set) <= HIGHLIGHT_MAX_REBUILD:
            base_df_visible = base_df_full[base_df_full["GKZ"].astype(str).isin(selected_set)].copy()
        else:
            # Keine Auswahl oder zu viele -> alle anzeigen
            base_df_visible = base_df_full.copy()

        # Transformation anwenden
        # KRITISCH: full_df für pct_total übergeben
        if transform == "pct_total":
            base_df = _apply_transform(
                base_df_visible, 
                indicator, 
                transform,
                full_df=base_df_full  # Vollständiger DF für Gesamt-Summe
            )
            # Auch vollständigen DF transformieren für Gesamtstatistik
            base_df_full_transformed = _apply_transform(
                base_df_full.copy(),
                indicator,
                transform,
                full_df=base_df_full
            )
        else:
            # raw oder pct_visible: nur sichtbare Daten
            base_df = _apply_transform(base_df_visible, indicator, transform)
            # Auch vollständigen DF transformieren für Gesamtstatistik
            base_df_full_transformed = _apply_transform(base_df_full.copy(), indicator, transform)

        # Label für Anzeige
        canon = canonicalize_indicator_name(indicator)
        nice_label = INDICATOR_LABELS.get(canon, indicator)
        
        if transform == "raw":
            nice_label_display = nice_label
        elif transform == "pct_visible":
            nice_label_display = f"{nice_label} (% von Angezeigt)"
        elif transform == "pct_total":
            nice_label_display = f"{nice_label} (% vom Gesamt)"
        else:
            nice_label_display = nice_label

        # Stats berechnen auf TRANSFORMIERTEN Daten
        # Für Gesamtstatistik
        s_full_transformed = pd.to_numeric(base_df_full_transformed[indicator], errors="coerce")
        s_full_transformed = s_full_transformed[~s_full_transformed.isin(MISSING_NUMBERS)]
        
        if not s_full_transformed.empty:
            stats_full = {
                "count": int(len(base_df_full_transformed)),
                "min": float(s_full_transformed.min(skipna=True)),
                "max": float(s_full_transformed.max(skipna=True)),
                "mean": float(s_full_transformed.mean(skipna=True)),
                "sum": float(s_full_transformed.sum(skipna=True)),
            }
        else:
            stats_full = {
                "count": int(len(base_df_full_transformed)),
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "sum": 0.0,
            }

        # Tabellen-Daten (begrenzt auf 1000 Zeilen)
        table_data = base_df[cols_order].head(1000).to_dict("records")

        logger.info(
            f"[Geo] apply_geo_filters: table_rows={len(table_data)}, "
            f"selected_set_size={len(selected_set)}, transform={transform}"
        )

        # Format-String basierend auf Transform
        if transform == "raw":
            fmt = "{:,.2f}"
        else:  # Prozent
            fmt = "{:.2f}%"

        # Detail-Panel erstellen
        if not selected_set:
            detail_children = [
                #html.H5("Details", className="mb-1"),
                html.P(f"Indikator: {nice_label_display}", className="text-muted mb-1"),
                html.P(
                    f"Ebene: {LEVEL_LABELS.get(level, level)}",
                    className="text-muted mb-2",
                ),
                html.Hr(className="my-2"),
                html.P(f"Gesamtstatistik ({stats_label})", className="fw-bold mb-1"),
                html.P(f"Anzahl Regionen: {stats_full['count']}", className="mb-1"),
                html.P(f"Summe: {fmt.format(stats_full['sum'])}", className="mb-1"),
                html.P(f"Min: {fmt.format(stats_full['min'])}", className="mb-1"),
                html.P(f"Max: {fmt.format(stats_full['max'])}", className="mb-1"),
                html.P(f"Mittelwert: {fmt.format(stats_full['mean'])}", className="mb-1"),
                html.Hr(className="my-2"),
                html.P("Auswahlstatistik", className="fw-bold mb-1"),
                html.P("Keine Regionen ausgewählt.", className="text-muted"),
            ]
        else:
            # Auswahl-Details (TRANSFORMIERTE Werte für Statistiken)
            s_sel_transformed = pd.to_numeric(base_df[indicator], errors="coerce")
            s_sel_transformed_clean = s_sel_transformed[~s_sel_transformed.isin(MISSING_NUMBERS)]
            
            detail_children = [
                #html.H5("Details", className="mb-1"),
                html.P(f"Indikator: {nice_label_display}", className="text-muted mb-1"),
                html.P(
                    f"Ebene: {LEVEL_LABELS.get(level, level)}",
                    className="text-muted mb-2",
                ),
                html.Hr(className="my-2"),
                html.P(f"Gesamtstatistik ({stats_label})", className="fw-bold mb-1"),
                html.P(f"Anzahl Regionen: {stats_full['count']}", className="mb-1"),
                html.P(f"Summe: {fmt.format(stats_full['sum'])}", className="mb-1"),
                html.P(f"Min: {fmt.format(stats_full['min'])}", className="mb-1"),
                html.P(f"Max: {fmt.format(stats_full['max'])}", className="mb-1"),
                html.P(f"Mittelwert: {fmt.format(stats_full['mean'])}", className="mb-1"),
                html.Hr(className="my-2"),
                html.P(f"Auswahlstatistik ({stats_label})", className="fw-bold mb-1"),
                html.P(f"Ausgewählt: {len(base_df)}", className="mb-1"),
                html.P(
                    f"Ausgeblendet: {stats_full['count'] - len(base_df)}",
                    className="mb-2",
                ),
            ]
            
            if not s_sel_transformed_clean.empty:
                detail_children.extend(
                    [
                        html.P(
                            f"Summe (Auswahl): {fmt.format(s_sel_transformed_clean.sum(skipna=True))}",
                            className="mb-1",
                        ),
                        html.P(
                            f"Min (Auswahl): {fmt.format(s_sel_transformed_clean.min(skipna=True))}",
                            className="mb-1",
                        ),
                        html.P(
                            f"Max (Auswahl): {fmt.format(s_sel_transformed_clean.max(skipna=True))}",
                            className="mb-1",
                        ),
                        html.P(
                            f"Mittelwert (Auswahl): {fmt.format(s_sel_transformed_clean.mean(skipna=True))}",
                            className="mb-1",
                        ),
                    ]
                )

        # Map-Input für zweite Callback-Stufe
        if not selected_set:
            map_input = {
                "has_data": True,
                "level": level,
                "indicator": indicator,
                "transform": transform,
                "selected_mode": "none",
            }
        elif len(selected_set) > HIGHLIGHT_MAX_REBUILD:
            map_input = {
                "has_data": True,
                "level": level,
                "indicator": indicator,
                "transform": transform,
                "selected_mode": "many",
            }
        else:
            map_input = {
                "has_data": True,
                "level": level,
                "indicator": indicator,
                "transform": transform,
                "selected_mode": "few",
                "selected_gkz": list(selected_set),
            }

        logger.info(f"[Geo] apply_geo_filters: map_input={map_input}")
        return map_input, table_data, column_defs, detail_children

    @app.callback(
        Output("geo-map", "srcDoc", allow_duplicate=True),
        Input("geo-map-input-store", "data"),
        prevent_initial_call=True,
    )
    def render_map_from_store(store_data):
        logger.info(f"[Geo] render_map_from_store: store_data={store_data}")
        
        if not store_data or not store_data.get("has_data"):
            level = "krs"
            if isinstance(store_data, dict):
                level = store_data.get("level", "krs") or "krs"
            return build_empty_map(
                level=level,
                message="Bitte wählen Sie einen Indikator und klicken Sie auf 'Filter anwenden'.",
            )

        level = store_data["level"]
        indicator = store_data["indicator"]
        transform = store_data.get("transform", "raw")
        selected_mode = store_data.get("selected_mode", "none")
        selected_list = store_data.get("selected_gkz") or []

        logger.info(
            "[Geo] render_map_from_store: level=%s, indicator=%s, transform=%s, "
            "selected_mode=%s, selected_len=%d",
            level,
            indicator,
            transform,
            selected_mode,
            len(selected_list),
        )

        # Basis-Slice erneut holen
        slice_df = _get_indicator_slice(level, indicator)
        df_full = None
        
        if slice_df is None:
            df_full = _get_df_cached(level)
            if df_full is None or indicator not in df_full.columns:
                logger.warning(
                    f"[Geo] render_map_from_store: Keine Daten oder Karte verfügbar "
                    f"für level={level}, indicator={indicator}."
                )
                return "<p>Keine Daten oder Karte verfügbar.</p>"

        if slice_df is not None:
            base_df_full = slice_df
        else:
            name_col = _detect_name_col(df_full, level) if df_full is not None else None
            cols = ["GKZ", indicator]
            if name_col:
                cols.insert(1, name_col)
            base_df_full = df_full[cols].copy()

        # Auswahl-Filter
        selected_set = set(selected_list)
        
        if selected_mode == "few" and selected_set:
            base_df_visible = base_df_full[base_df_full["GKZ"].astype(str).isin(selected_set)].copy()
        else:
            base_df_visible = base_df_full.copy()

        # Transformation anwenden
        if transform == "pct_total":
            base_df = _apply_transform(
                base_df_visible,
                indicator,
                transform,
                full_df=base_df_full
            )
        else:
            base_df = _apply_transform(base_df_visible, indicator, transform)

        # Debug
        non_null_count = pd.to_numeric(base_df[indicator], errors="coerce").notna().sum()
        logger.info(
            "[Geo] render_map_from_store: base_df rows=%d, non_null_indicator=%d",
            len(base_df),
            non_null_count,
        )

        # Label
        canon = canonicalize_indicator_name(indicator)
        nice_label = INDICATOR_LABELS.get(canon, indicator)
        
        if transform == "raw":
            nice_label_display = nice_label
        elif transform == "pct_visible":
            nice_label_display = f"{nice_label} (% von Angezeigt)"
        elif transform == "pct_total":
            nice_label_display = f"{nice_label} (% vom Gesamt)"
        else:
            nice_label_display = nice_label

        # Map-Spaltenname
        map_indicator = indicator
        if transform != "raw":
            map_indicator = f"{indicator}__{transform}"

        # Schmale DF für Karte
        df_for_map_wide = base_df.copy()
        if map_indicator not in df_for_map_wide.columns:
            df_for_map_wide[map_indicator] = df_for_map_wide[indicator]
        df_for_map = _build_minimal_map_df(df_for_map_wide, map_indicator, level)

        logger.info(
            "[Geo] render_map_from_store: df_for_map columns=%s, rows=%d",
            list(df_for_map.columns),
            len(df_for_map),
        )

        # Cache-Key
        cache_key = (level, indicator, transform)

        # Fall A: keine Auswahl -> Vollkarte aus Cache
        if selected_mode == "none":
            if cache_key in _MAP_HTML_CACHE:
                logger.info("[Geo] render_map_from_store: nutze gecachte Vollkarte.")
                return _MAP_HTML_CACHE[cache_key]
            html_map = build_map_from_df(
                level=level,
                df=df_for_map,
                indicator=map_indicator,
                display_name=nice_label_display,
                indicator_desc=None,
            )
            _MAP_HTML_CACHE[cache_key] = html_map
            logger.info("[Geo] render_map_from_store: Vollkarte neu erzeugt.")
            return html_map

        # Fall B: zu viele -> Vollkarte aus Cache
        if selected_mode == "many":
            if cache_key in _MAP_HTML_CACHE:
                logger.info("[Geo] render_map_from_store: nutze gecachte Vollkarte (many).")
                return _MAP_HTML_CACHE[cache_key]
            html_map = build_map_from_df(
                level=level,
                df=df_for_map,
                indicator=map_indicator,
                display_name=nice_label_display,
                indicator_desc=None,
            )
            _MAP_HTML_CACHE[cache_key] = html_map
            logger.info("[Geo] render_map_from_store: Vollkarte neu erzeugt (many).")
            return html_map

        # Fall C: wenige -> neu mit visible_gkz
        logger.info(
            "[Geo] render_map_from_store: selected_mode=few, selected_gkz_count=%d",
            len(selected_set),
        )
        html_map = build_map_from_df(
            level=level,
            df=df_for_map,
            indicator=map_indicator,
            display_name=nice_label_display,
            indicator_desc=None,
            visible_gkz=selected_set,
        )
        return html_map

    # Zurücksetzen
    @app.callback(
        Output("geo-level-dropdown", "value"),
        Output("geo-indicator-dropdown", "options", allow_duplicate=True),
        Output("geo-indicator-dropdown", "value", allow_duplicate=True),
        Output("geo-map", "srcDoc", allow_duplicate=True),
        Output("geo-table", "rowData", allow_duplicate=True),
        Output("geo-table", "columnDefs", allow_duplicate=True),
        Output("geo-detail-panel", "children", allow_duplicate=True),
        Output("geo-region-dropdown", "options", allow_duplicate=True),
        Output("geo-region-dropdown", "value", allow_duplicate=True),
        Output("geo-transform-dropdown", "value", allow_duplicate=True),
        Output("geo-map-input-store", "data", allow_duplicate=True),
        Input("reset-geo-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_geo_filters(n_clicks):
        logger.info(f"[Geo] reset_geo_filters: n_clicks={n_clicks}")
        level = "krs"
        indicator_opts = _options_for_level(level)
        indicator_val = indicator_opts[0]["value"] if indicator_opts else None
        region_opts = _get_region_options(level)
        _MAP_HTML_CACHE.clear()
        _INDICATOR_SLICE_CACHE.clear()
        _STATS_CACHE.clear()
        _COLUMNS_CACHE.clear()
        logger.info(
            f"[Geo] reset_geo_filters: level={level}, indicator_val={indicator_val}, "
            f"regions={len(region_opts)}"
        )
        return (
            level,
            indicator_opts,
            indicator_val,
            build_empty_map(                # <--- NEU
                level=level,
                message="Bitte wählen Sie einen Indikator und klicken Sie auf 'Filter anwenden'.",
            ),
            [],
            [],
            [html.P("Bitte Filter anwenden.", className="text-muted")],
            region_opts,
            [],
            "raw",
            {},  # geo-map-input-store leeren
        )



    @app.callback(
        Output("geo-timeseries-collapse", "is_open"),
        Input("toggle-geo-timeseries", "n_clicks"),
        State("geo-timeseries-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_timeseries_chart(n_clicks, is_open):
        return not is_open

    # Hilfsfunktion: Spalte anhand von Keywords finden
    def _find_col_by_keywords(columns, keywords):
        cols = [str(c) for c in columns]
        for key in keywords:
            key = key.lower()
            for c in cols:
                if key in c.lower():
                    return c
        return None



    def _compute_region_shares(
        level: Optional[str],
        indicator: Optional[str],
        selected_mode: str,
        selected_gkz: Optional[List[str]],
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """
        Berechnet für die ausgewählten Gebiete ihren Anteil am Gesamtwert Deutschlands
        für den gewählten Indikator.

        Rückgabe:
        - region_shares: DF mit Spalten ['GKZ', 'name', 'share'] oder None
        - selection_share: Summe der shares (Anteil der Auswahl am Gesamtwert)
        """
        if not level or not indicator:
            return None, 1.0

        slice_df = _get_indicator_slice(level, indicator)
        if slice_df is None or indicator not in slice_df.columns:
            logger.warning(
                f"[Geo] _compute_region_shares: kein Slice für level={level}, indicator={indicator}"
            )
            return None, 1.0

        df = slice_df.copy()
        df["GKZ"] = df["GKZ"].astype(str)

        # Alle Rohwerte (Deutschland gesamt)
        vals_all = pd.to_numeric(df[indicator], errors="coerce")
        vals_all = vals_all[~vals_all.isin(MISSING_NUMBERS)]
        total_all = float(vals_all.sum(skipna=True)) if not vals_all.empty else 0.0

        if total_all <= 0:
            return None, 1.0

        # Name-Spalte ermitteln (falls vorhanden)
        name_col = None
        nc = _detect_name_col_fast(level)
        if nc and nc in df.columns:
            name_col = nc

        # Nur bei "few" haben wir die konkreten GKZ im map_input
        if selected_mode == "few" and selected_gkz:
            selected_gkz = [str(g) for g in selected_gkz]
            df_sel = df[df["GKZ"].isin(selected_gkz)].copy()
        elif selected_mode == "none":
            # Keine Auswahl -> keine Zerlegung, nur Gesamt
            return None, 1.0
        else:
            # "many" oder unbekannt -> ebenfalls nur Gesamt
            return None, 1.0

        if df_sel.empty:
            return None, 0.0

        vals_sel = pd.to_numeric(df_sel[indicator], errors="coerce")
        vals_sel = vals_sel[~vals_sel.isin(MISSING_NUMBERS)]
        df_sel = df_sel.loc[vals_sel.index].copy()

        if df_sel.empty:
            return None, 0.0

        df_sel["share"] = vals_sel / total_all
        if name_col:
            df_sel["name"] = df_sel[name_col].astype(str)
        else:
            df_sel["name"] = df_sel["GKZ"].astype(str)

        selection_share = float(df_sel["share"].sum(skipna=True))
        # Nur schlanke Spalten zurückgeben
        region_shares = df_sel[["GKZ", "name", "share"]].copy()

        logger.info(
            f"[Geo] _compute_region_shares: level={level}, indicator={indicator}, "
            f"mode={selected_mode}, selection_share={selection_share:.4f}, "
            f"regions={len(region_shares)}"
        )

        return region_shares, selection_share


    # Hilfsfunktion: Figure aus Forecast-DF bauen (nur Prognose + Konfidenzband)

    def _build_forecast_figure(
        df_forecast: pd.DataFrame,
        region_shares: Optional[pd.DataFrame],
        selection_share: float,
    ) -> go.Figure:
        """
        Baut die Forecast-Grafik:
        - Wenn region_shares vorhanden: gestapelte Prognosen je Gebiet
        - Immer: Konfidenzband für die gesamte Auswahl (selection_share)
        """
        df = df_forecast.copy()

        # Datums-Spalte finden
        date_col = _find_col_by_keywords(df.columns, ["date", "datum", "ds", "zeit", "period"])
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        x = df[date_col]

        # Spalten für Forecast & Konfidenzband finden
        forecast_col = _find_col_by_keywords(
            df.columns, ["forecast", "prognose", "yhat", "vorhersage"]
        )
        lower_col = _find_col_by_keywords(
            df.columns, ["lower", "unter", "low", "lwr", "conf_low", "yhat_lower"]
        )
        upper_col = _find_col_by_keywords(
            df.columns, ["upper", "ober", "high", "upr", "conf_high", "yhat_upper"]
        )

        fig = go.Figure()

        if not forecast_col:
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title="Zeit",
                yaxis_title="Wert",
                annotations=[
                    dict(
                        text="In FORECAST_TS wurde keine Prognosespalte gefunden.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="#888"),
                    )
                ],
            )
            return fig

        y_forecast_total = pd.to_numeric(df[forecast_col], errors="coerce")

        # 1) Konfidenzband für die gesamte Auswahl (falls vorhanden)
        sel_share = float(selection_share or 1.0)
        if lower_col and upper_col:
            y_lower_total = pd.to_numeric(df[lower_col], errors="coerce")
            y_upper_total = pd.to_numeric(df[upper_col], errors="coerce")

            y_lower_sel = y_lower_total * sel_share
            y_upper_sel = y_upper_total * sel_share

            fig.add_trace(
                go.Scatter(
                    x=list(x) + list(x[::-1]),
                    y=list(y_upper_sel) + list(y_lower_sel[::-1]),
                    fill="toself",
                    name="Konfidenzband (Auswahl)",
                    hoverinfo="skip",
                    line=dict(width=0),
                    opacity=0.2,
                )
            )

        # 2) Gestapelte Prognosen je Gebiet (oder aggregiert)
        if region_shares is not None and not region_shares.empty:
            # Gestapelte Linien/Flächen pro Gebiet
            for _, row in region_shares.iterrows():
                share = float(row.get("share") or 0.0)
                if share <= 0:
                    continue
                label = str(row.get("name") or row.get("GKZ"))
                y_region = y_forecast_total * share

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_region,
                        mode="lines",
                        stackgroup="one",  # macht das Chart gestapelt
                        name=label,
                    )
                )
        else:
            # Nur Gesamt-Auswahl
            y_total = y_forecast_total * sel_share
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_total,
                    mode="lines",
                    name="Prognose (Auswahl gesamt)",
                )
            )

        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Zeit",
            yaxis_title="Wert",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        try:
            format_axis_quarters(fig, df[date_col])
        except Exception:
            pass


        return fig


    @app.callback(
        Output("geo-forecast-store", "data", allow_duplicate=True),
        Output("geo-forecast-upload-status", "children"),
        Input("geo-forecast-upload", "contents"),
        State("geo-forecast-upload", "filename"),
        prevent_initial_call=True,
    )
    def handle_forecast_upload(contents, filename):
        if contents is None:
            return no_update, "Keine Datei hochgeladen."

        content_type, content_string = contents.split(",", 1)
        try:
            decoded = base64.b64decode(content_string)
        except Exception as e:
            logger.error(f"[Geo] Forecast-Upload: base64-Decode fehlgeschlagen: {e}")
            return no_update, "Fehler beim Einlesen der Datei (Base64)."

        try:
            with io.BytesIO(decoded) as bio:
                df_forecast = pd.read_excel(bio, sheet_name="FORECAST_TS")
        except Exception as e:
            logger.error(f"[Geo] Forecast-Upload: Konnte Sheet 'FORECAST_TS' nicht laden: {e}")
            msg = f"Fehler: Sheet 'FORECAST_TS' in {filename or 'Datei'} nicht gefunden oder nicht lesbar."
            return no_update, msg

        if df_forecast.empty:
            logger.warning("[Geo] Forecast-Upload: FORECAST_TS ist leer.")
            return no_update, "FORECAST_TS enthält keine Daten."

        try:
            df_json = df_forecast.to_json(orient="split", date_format="iso")
        except Exception as e:
            logger.error(f"[Geo] Forecast-Upload: Fehler beim Serialisieren des DF: {e}")
            return no_update, "Fehler beim Speichern der Prognosedaten."

        status = f"Prognose geladen aus: {filename}" if filename else "Prognose geladen."
        return df_json, status
    

    @app.callback(
        Output("geo-timeseries-chart", "figure"),
        Input("geo-forecast-store", "data"),
        Input("geo-map-input-store", "data"),
        prevent_initial_call=True,
    )
    def update_forecast_chart(forecast_json, map_input):
        # 1) Kein Forecast hochgeladen -> Hinweis anzeigen
        if not forecast_json:
            fig = go.Figure()
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title="Zeit",
                yaxis_title="Wert",
                annotations=[
                    dict(
                        text="Bitte Prognose-Excel (FORECAST_TS) über 'Daten hochladen' auswählen.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="#888"),
                    )
                ],
            )
            return fig

        try:
            df_forecast = pd.read_json(forecast_json, orient="split")
        except Exception as e:
            logger.error(f"[Geo] update_forecast_chart: Fehler beim Lesen des Forecast-JSON: {e}")
            fig = go.Figure()
            fig.update_layout(
                annotations=[
                    dict(
                        text="Fehler beim Laden der Prognosedaten.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="#d9534f"),
                    )
                ]
            )
            return fig

        level = None
        indicator = None
        selected_mode = "none"
        selected_gkz = None

        if map_input:
            level = map_input.get("level")
            indicator = map_input.get("indicator")
            selected_mode = map_input.get("selected_mode", "none")
            selected_gkz = map_input.get("selected_gkz")

        # Anteile je Gebiet berechnen (nur wenn "few")
        region_shares, selection_share = _compute_region_shares(
            level, indicator, selected_mode, selected_gkz
        )

        fig = _build_forecast_figure(
            df_forecast=df_forecast,
            region_shares=region_shares,
            selection_share=selection_share,
        )
        return fig
    
    @app.callback(
        Output("geo-forecast-table", "columnDefs"),
        Output("geo-forecast-table", "rowData"),
        Input("geo-forecast-store", "data"),
        Input("geo-map-input-store", "data"),
        Input("geo-market-share-slider", "value"),          # globaler Default
        Input("geo-region-market-share-store", "data"),     # pro-Region-Overrides
        prevent_initial_call=True,
    )
    def update_forecast_table(
        forecast_json,
        map_input,
        market_share_percent,
        region_market_shares,
    ):
        # Wenn noch keine Prognosedaten vorhanden sind
        if not forecast_json:
            return [], []

        try:
            df_forecast = pd.read_json(forecast_json, orient="split")
        except Exception as e:
            logger.error(f"[Geo] update_forecast_table: Fehler beim Lesen des Forecast-JSON: {e}")
            return [], []

        if df_forecast.empty:
            return [], []

        # Datums- und Forecast-Spalte identifizieren (analog zum Chart)
        date_col = _find_col_by_keywords(df_forecast.columns, ["date", "datum", "ds", "zeit", "period"])
        if date_col is None:
            date_col = df_forecast.columns[0]

        forecast_col = _find_col_by_keywords(
            df_forecast.columns,
            ["forecast", "prognose", "yhat", "vorhersage"],
        )
        if not forecast_col:
            logger.warning("[Geo] update_forecast_table: Keine Prognosespalte gefunden.")
            return [], []

        df = df_forecast.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        # Periodenlabels bauen (Quartale aus dem Datum ableiten)
        period_labels = []
        for idx, dt in enumerate(df[date_col]):
            if pd.isna(dt):
                label = f"Periode {idx + 1}"
            else:
                q = (dt.month - 1) // 3 + 1
                label = f"{dt.year}-Q{q}"
            period_labels.append(label)

        y_forecast_total = pd.to_numeric(df[forecast_col], errors="coerce")

        # Geo-Kontext aus map_input lesen
        level = None
        indicator = None
        selected_mode = "none"
        selected_gkz = None

        if map_input:
            level = map_input.get("level")
            indicator = map_input.get("indicator")
            selected_mode = map_input.get("selected_mode", "none")
            selected_gkz = map_input.get("selected_gkz")

        # Anteile je Gebiet (nur bei few) + selection_share (Summe der Anteile)
        region_shares, selection_share = _compute_region_shares(
            level, indicator, selected_mode, selected_gkz
        )

        # globaler Marktanteil aus Slider (0–100 %) -> 0–1
        try:
            global_market_share_pct = float(market_share_percent or 0.0)
        except Exception:
            global_market_share_pct = 0.0

        global_market_share = global_market_share_pct / 100.0
        region_market_shares = region_market_shares or {}

        rows = []

        # 1) Deutschland gesamt (immer 100 % Deutschland-Prognose, ohne Marktanteil)
        germany_row = {
            "Region": "Deutschland (gesamt)",
            "Anteil_am_BIP": 100.0,
            "Region_MarketShare": None,  # für Deutschland nicht relevant
        }
        for i, label in enumerate(period_labels):
            val = y_forecast_total.iloc[i]
            germany_row[label] = float(val) if pd.notna(val) else None
        rows.append(germany_row)

        # 2) Regionen einzeln: BIP-Anteil + Prognose * Anteil * regionsspezifischen Marktanteil
        region_rows = []
        effective_share_sum = 0.0  # Summe share_i * market_share_i für "Auswahl – Marktanteil"

        if region_shares is not None and not region_shares.empty:
            for _, r in region_shares.iterrows():
                share = float(r.get("share") or 0.0)
                if share <= 0:
                    continue

                region_name = str(r.get("name") or r.get("GKZ"))

                # Marktanteil für diese Region:
                # - Wenn User einen Wert eingegeben hat: diesen verwenden
                # - sonst: globalen Sliderwert
                stored_pct = region_market_shares.get(region_name)
                if stored_pct is not None:
                    region_market_pct = float(stored_pct)
                else:
                    region_market_pct = global_market_share_pct

                # clamp 0–100
                region_market_pct = max(0.0, min(100.0, region_market_pct))
                region_market_share = region_market_pct / 100.0

                # Beitrag zum effektiven Gesamtmarktanteil
                effective_share_sum += share * region_market_share

                row = {
                    "Region": region_name,
                    # "reiner" BIP-Anteil (ohne Marktanteil)
                    "Anteil_am_BIP": round(share * 100.0, 2),
                    # editierbarer Marktanteil dieser Region
                    "Region_MarketShare": round(region_market_pct, 1),
                }

                for i, label in enumerate(period_labels):
                    base_val = y_forecast_total.iloc[i]
                    if pd.notna(base_val):
                        # Bankvolumen in dieser Region = Gesamtmarkt * BIP-Anteil * Region-Marktanteil
                        val = base_val * share * region_market_share
                    else:
                        val = None
                    row[label] = float(val) if val is not None else None

                region_rows.append(row)

        # 3) Auswahl (gesamt): Marktpotential des Gebiets (ohne Marktanteile)
        selection_row = None
        if selection_share is not None:
            try:
                selection_share = float(selection_share)
            except Exception:
                selection_share = 0.0

        if selection_share and selection_share > 0:
            selection_row = {
                "Region": "Auswahl (gesamt)",
                "Anteil_am_BIP": round(selection_share * 100.0, 2),
                "Region_MarketShare": None,  # kein einzelner Wert, es ist die Summe
            }
            for i, label in enumerate(period_labels):
                base_val = y_forecast_total.iloc[i]
                val = base_val * selection_share if pd.notna(base_val) else None
                selection_row[label] = float(val) if val is not None else None

            rows.append(selection_row)

        # 4) Auswahl – Marktanteil: Summe der regionalen Bankvolumina
        if effective_share_sum > 0:
            ms_row = {
                "Region": "Auswahl – Marktanteil",
                # Effektiver Marktanteil am Gesamtmarkt: Summe share_i * market_share_i
                "Anteil_am_BIP": round(effective_share_sum * 100.0, 2),
                "Region_MarketShare": None,
            }
            for i, label in enumerate(period_labels):
                base_val = y_forecast_total.iloc[i]
                if pd.notna(base_val):
                    val = base_val * effective_share_sum
                else:
                    val = None
                ms_row[label] = float(val) if val is not None else None

            rows.append(ms_row)

        # 5) Regionale Einzelzeilen anhängen (bereits mit Marktanteil skaliert)
        rows.extend(region_rows)

        # Spaltendefinitionen für AgGrid
        column_defs = [
            {"field": "Region", "headerName": "Region"},
            {"field": "Anteil_am_BIP", "headerName": "Anteil am BIP (%)"},
            {
                "field": "Region_MarketShare",
                "headerName": "Marktanteil (%)",
                "editable": True,  # HIER gibt der User in der Tabelle ein
            },
        ]
        for label in period_labels:
            column_defs.append(
                {
                    "field": label,
                    "headerName": label,
                }
            )

        return column_defs, rows




    @app.callback(
        Output("geo-runs-list-modal", "is_open"),
        Output("geo-runs-list-body", "children"),
        Output("geo-forecast-store", "data", allow_duplicate=True),
        Input("geo-load-model-btn", "n_clicks"),
        Input("geo-close-runs-list", "n_clicks"),
        Input({"type": "geo-load-run-btn", "cache_tag": ALL, "timestamp": ALL}, "n_clicks"),
        State("geo-runs-list-modal", "is_open"),
        prevent_initial_call=True,
    )
    def handle_geo_runs_modal(open_click, close_click, run_clicks, is_open):
        """
        Steuert das Modal auf der Geo-Seite:
        - Klick auf 'Modell laden' lädt die Runs aus loader/runs und öffnet das Modal.
        - Klick auf 'Schließen' schließt das Modal.
        - Klick auf 'Diesen Forecast in Geo laden' lädt die Daten aus dem gewählten Run
          und schreibt sie in geo-forecast-store.
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # 1) Modal öffnen + Runs laden
        if trigger == "geo-load-model-btn":
            try:
                runs = _discover_runs()
            except Exception as e:
                logger.error(f"[Geo] handle_geo_runs_modal: Fehler bei _discover_runs: {e}")
                body = html.Div(
                    "Fehler beim Laden der gespeicherten Prognosen.",
                    className="text-danger",
                )
                return True, body, no_update

            body = _render_geo_runs_list(runs)
            return True, body, no_update

        # 2) Modal schließen
        if trigger == "geo-close-runs-list":
            return False, no_update, no_update

        # 3) Einer der dynamischen Run-Buttons wurde geklickt
        try:
            trigger_id = json.loads(trigger)
        except json.JSONDecodeError:
            # Kein JSON → ignorieren
            raise PreventUpdate

        if trigger_id.get("type") != "geo-load-run-btn":
            raise PreventUpdate

        cache_tag = trigger_id.get("cache_tag")
        ts_raw = trigger_id.get("timestamp")

        forecast_json = _load_geo_forecast_from_run(cache_tag, ts_raw)
        if not forecast_json:
            logger.warning(
                f"[Geo] handle_geo_runs_modal: Keine Forecast-Daten für Run "
                f"{cache_tag}@{ts_raw} gefunden."
            )
            # Modal schließen, ohne Store zu ändern (optional: offen lassen)
            return False, no_update, no_update

        # Erfolgreich geladen:
        # - Modal schließen
        # - Body unverändert lassen
        # - geo-forecast-store mit Forecast-Daten füllen
        return False, no_update, forecast_json


    @app.callback(
        Output("geo-region-market-share-store", "data"),
        Input("geo-forecast-table", "cellValueChanged"),
        State("geo-region-market-share-store", "data"),
        prevent_initial_call=True,
    )
    def persist_region_market_shares(cell_event, current_shares):
        """
        Aktualisiert die pro-Region-Marktanteile, wenn der User in der Tabelle
        die Spalte 'Marktanteil (%)' bearbeitet.
        """
        if not cell_event:
            raise PreventUpdate

        # cell_event kann ein einzelnes Dict oder eine Liste von Events sein
        if isinstance(cell_event, dict):
            event = cell_event
        elif isinstance(cell_event, list):
            if len(cell_event) == 0:
                raise PreventUpdate
            # letztes Event in der Liste verwenden
            event = cell_event[-1]
        else:
            logger.warning(f"[Geo] persist_region_market_shares: Unerwartiger Typ von cell_event: {type(cell_event)}")
            raise PreventUpdate

        row = event.get("data") or {}
        col_id = event.get("colId") or event.get("field")

        # Nur reagieren, wenn tatsächlich die Marktanteils-Spalte editiert wurde
        if col_id != "Region_MarketShare":
            raise PreventUpdate

        region_name = row.get("Region")
        if not region_name:
            raise PreventUpdate

        # Für Deutschland / Aggregat-Zeilen keine Speicherung
        if region_name in ("Deutschland (gesamt)", "Auswahl (gesamt)", "Auswahl – Marktanteil"):
            raise PreventUpdate

        try:
            value = float(row.get("Region_MarketShare") or 0.0)
        except Exception:
            value = 0.0

        # auf 0–100 begrenzen
        value = max(0.0, min(100.0, value))

        current_shares = current_shares or {}
        current_shares[region_name] = value

        logger.info(f"[Geo] Region-Marktanteil aktualisiert: {region_name} -> {value:.1f}%")
        return current_shares

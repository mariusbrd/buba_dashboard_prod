# === Path Helpers: immer unter scenario/data arbeiten ===
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = _THIS_DIR  # falls die Files schon im Ordner "scenario" liegen
SCENARIO_DATA_DIR = SCENARIO_DIR / "data"
SCENARIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

def scenario_data_path(*parts: str) -> Path:
    return SCENARIO_DATA_DIR.joinpath(*parts)

# Konventionierte Standard-Dateinamen
DEFAULT_ANALYSIS_XLSX = scenario_data_path("analysis_data.xlsx")
DEFAULT_SCENARIO_XLSX = scenario_data_path("scenario_overrides.xlsx")
DEFAULT_OUTPUT_XLSX   = scenario_data_path("output.xlsx")  # falls du eines schreibst


# ---------------------------------------------------------------------------
# scenario_dataloader.py
# Adapter für Dash: Downloader + Hilfsfunktionen für Szenario-Daten
# ---------------------------------------------------------------------------

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union  # <- Union ergänzt

import pandas as pd
import yaml
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

def _now_berlin() -> datetime:
    return datetime.now(ZoneInfo("Europe/Berlin")) if ZoneInfo else datetime.now()


# ---------------------------------------------------------------------------
# ROBUSTER TRANSFORMS-IMPORT
# (deine app.py liefert evtl. nur DataProcessor + get_excel_engine;
#  hier rüsten wir fehlende Teile nach)
# ---------------------------------------------------------------------------
import sys

# Default-Attr-Container
_detect_data_source = None
_parse_index_specification = None
_CacheManager = None
_IndexCreator = None
_get_excel_engine = None

try:
    # normaler Fall: es gibt ein „richtiges“ transforms
    from transforms import (  # type: ignore
        detect_data_source as _detect_data_source,
        parse_index_specification as _parse_index_specification,
        CacheManager as _CacheManager,
        IndexCreator as _IndexCreator,
        get_excel_engine as _get_excel_engine,
    )
except Exception:
    # es gibt zwar evtl. ein schlankes transforms aus app.py – das holen wir
    try:
        import transforms as _t  # type: ignore
    except Exception:
        _t = None

    # 1) detect_data_source – ganz einfache Heuristik
    def _fallback_detect_data_source(code: str) -> str:
        code = (code or "").upper()
        if "." in code:
            # typische ECB-Präfixe
            if code.startswith(("ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", "STS.", "MNA.", "BOP.", "GFS.", "EXR.")):
                return "ECB"
        # alles andere: BuBa
        return "BUBA"

    _detect_data_source = getattr(_t, "detect_data_source", _fallback_detect_data_source)

    # 2) parse_index_specification – wenn nicht vorhanden, einfach None liefern
    def _fallback_parse_index_specification(spec: str):
        # deine scenario-configs nutzen oft INDEX(...),
        # wenn wir das nicht können, behandeln wir es als „kein Index“
        return None

    _parse_index_specification = getattr(_t, "parse_index_specification", _fallback_parse_index_specification)

    # 3) CacheManager – ganz kleine lokale Version
    class _FallbackCacheManager:
        def __init__(self, cache_dir: str, cache_max_age_days: int = 60):
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_max_age_days = cache_max_age_days

        def _cache_path(self, code: str) -> Path:
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in code)
            return self.cache_dir / f"{safe}.parquet"

        def is_fresh(self, code: str) -> bool:
            p = self._cache_path(code)
            if not p.exists():
                return False
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                return age_days <= self.cache_max_age_days
            except Exception:
                return False

        def read_cache(self, code: str):
            p = self._cache_path(code)
            if not self.is_fresh(code):
                return None
            try:
                return pd.read_parquet(p)
            except Exception:
                return None

        def write_cache(self, code: str, df: pd.DataFrame) -> bool:
            if df is None or df.empty:
                return False
            p = self._cache_path(code)
            try:
                df.to_parquet(p, index=False)
                return True
            except Exception:
                return False

    _CacheManager = getattr(_t, "CacheManager", _FallbackCacheManager)

    # 4) IndexCreator – wenn nicht vorhanden, einfache Summen-Variante
    class _FallbackIndexCreator:
        def __init__(self, *, index_base_year: int, index_base_value: float):
            self.index_base_year = index_base_year
            self.index_base_value = index_base_value

        def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
            # super einfache Version: Mittelwert über die verfügbaren Serien
            if "Datum" not in data_df.columns:
                raise ValueError("DataFrame needs 'Datum'")
            available = [c for c in series_codes if c in data_df.columns]
            if not available:
                return pd.Series(name=index_name, dtype=float)
            idx = data_df["Datum"]
            vals = data_df[available].mean(axis=1, skipna=True)
            # auf 100 normieren (einfach)
            base = vals.dropna().iloc[0] if not vals.dropna().empty else 1.0
            base = base if base not in (0, None) else 1.0
            out = (vals / base) * float(self.index_base_value)
            out.name = index_name
            out.index = idx
            return out

    _IndexCreator = getattr(_t, "IndexCreator", _FallbackIndexCreator)

    # 5) get_excel_engine – brauchen wir beim Speichern
    def _fallback_get_excel_engine() -> str:
        try:
            import openpyxl  # noqa
            return "openpyxl"
        except Exception:
            try:
                import xlsxwriter  # noqa
                return "xlsxwriter"
            except Exception:
                return "openpyxl"

    _get_excel_engine = getattr(_t, "get_excel_engine", _fallback_get_excel_engine)

# jetzt benennen wir sie so, wie der Rest der Datei sie erwartet
detect_data_source = _detect_data_source
parse_index_specification = _parse_index_specification
CacheManager = _CacheManager
IndexCreator = _IndexCreator
get_excel_engine = _get_excel_engine


# ---------------------------------------------------------------------------
# ROBUSTER IMPORT DER QUELLEN (du hast sie im Ordner loader als 'sources.*')
# ---------------------------------------------------------------------------
try:
    from data_sources.ecb_client import fetch_ecb_async, fetch_ecb_sync  # type: ignore
    from data_sources.buba_client import fetch_buba_async, fetch_buba_sync  # type: ignore
except Exception:
    # Fallback auf deine tatsächlichen Modulnamen
    try:
        from sources.ecb_client import fetch_ecb_async, fetch_ecb_sync  # type: ignore
        from sources.buba_client import fetch_buba_async, fetch_buba_sync  # type: ignore
    except Exception as e:
        raise ImportError(
            "Weder 'data_sources.*' noch 'sources.*' konnten importiert werden – "
            "prüfe bitte, dass dein loader/loader.py ausgeführt wurde, bevor "
            "scenario_dataloader.py importiert wird."
        ) from e


# -------------------------------------------------------------------
# Monats-Stempel (einmal pro Monat laden)
# -------------------------------------------------------------------
def should_run_this_month(scenario_dir: Union[str, Path], stamp_name: str = ".scenario_month.stamp") -> bool:
    """
    Prüft, ob im aktuellen Monat schon geladen wurde.
    True -> ausführen; False -> überspringen.
    """
    scenario_dir = Path(scenario_dir)
    stamp_path = scenario_dir / stamp_name
    current_token = _now_berlin().strftime("%Y-%m")
    try:
        if stamp_path.exists():
            token = stamp_path.read_text(encoding="utf-8").strip()
            return token != current_token
        return True
    except Exception:
        return True

def mark_ran_this_month(scenario_dir: Union[str, Path], stamp_name: str = ".scenario_month.stamp") -> None:
    """Schreibt/aktualisiert den Monats-Stempel."""
    scenario_dir = Path(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = scenario_dir / stamp_name
    current_token = _now_berlin().strftime("%Y-%m")
    stamp_path.write_text(current_token, encoding="utf-8")

def ensure_analysis_data_on_startup() -> Path:
    """
    Stellt sicher, dass eine gültige analysis_data.xlsx in scenario/data liegt.
    Gibt den Pfad der verwendeten/erzeugten Datei zurück.
    """
    if DEFAULT_ANALYSIS_XLSX.exists():
        return DEFAULT_ANALYSIS_XLSX

    SCENARIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_hist = pd.DataFrame(columns=["Date", "Gesamt GVB"])
    with pd.ExcelWriter(DEFAULT_ANALYSIS_XLSX, engine=get_excel_engine()) as xw:
        df_hist.to_excel(xw, sheet_name="history", index=False)

    return DEFAULT_ANALYSIS_XLSX

# -------------------------------------------------------------------
# Logger-Utility
# -------------------------------------------------------------------
def _log(logger, msg: str) -> None:
    if logger is None:
        print(msg)
    else:
        try:
            logger(msg)
        except Exception:
            print(msg)


# -------------------------------------------------------------------
# Konfiguration (YAML)
# -------------------------------------------------------------------
@dataclass
class DashDownloadConfig:
    # Zeitfenster
    start_date: str
    end_date: str

    # Cache
    prefer_cache: bool = True
    # Standard: Cache unterhalb scenario/ (relative Angaben werden darunter aufgelöst)
    cache_dir: str = str(SCENARIO_DIR / "cache")
    cache_max_age_days: int = 60

    # Ziel-/Serien-Definitionen
    anchor_var: Optional[str] = None
    series_definitions: Optional[Dict[str, str]] = None

    # Index-Einstellungen
    enable_index: bool = True
    index_aggregate: str = "base"
    index_base_year: int = 2015
    index_base_value: float = 100.0

    # Download-Parameter
    download_timeout_seconds: int = 30
    min_response_size: int = 100
    source_overrides: Optional[Dict[str, str]] = None
    min_populated_vars: int = 2

    # Kalender-Ausrichtung
    calendar_freq: str = "MS"           # "MS" (Monatsanfang) oder "M" (Monatsende)
    calendar_fill: str = "none"         # "none" | "ffill" | "bfill"
    calendar_fill_limit: Optional[int] = None

    # Output (Default IMMER in scenario/data)
    output_path: str = str(DEFAULT_OUTPUT_XLSX)

    @staticmethod
    def _resolve_current_month() -> str:
        """Aktueller Monat (Europe/Berlin) als 'YYYY-MM'."""
        now = _now_berlin()
        return f"{now.year:04d}-{now.month:02d}"

    @staticmethod
    def _resolve_under_scenario_dir(p: str | Path) -> str:
        """
        Relative Pfade robust unterhalb von SCENARIO_DIR auflösen.
        """
        p = Path(p)
        if p.is_absolute():
            return str(p)
        return str((SCENARIO_DIR / p).resolve())

    @staticmethod
    def _resolve_under_scenario_data(p: str | Path) -> str:
        """
        Relative Pfade robust unterhalb von SCENARIO_DATA_DIR auflösen.
        """
        p = Path(p)
        if p.is_absolute():
            return str(p)
        return str((SCENARIO_DATA_DIR / p).resolve())

    @staticmethod
    def from_yaml(path: str | Path) -> "DashDownloadConfig":
        """
        Lädt YAML. Für 'end_date' sind zulässig:
        - 'auto' | 'current_month' | 'heute' | 'aktuell' -> aktueller Monat
        - oder feste Werte 'YYYY-MM' / 'YYYY-MM-DD'
        Außerdem:
        - output_path: relative Pfade werden nach scenario/data umgebogen.
        - cache_dir:  relative Pfade werden nach scenario/… umgebogen.
        """
        path = Path(path)
        try:
            print(f"[Loader] YAML laden: {path.resolve()}")
        except Exception:
            print(f"[Loader] YAML laden: {path}")

        cfg_raw = path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(cfg_raw) or {}

        cal = cfg.get("calendar_index") or {}
        cache_cfg = cfg.get("cache") or {}

        start_date = cfg.get("start_date")

        raw_end = (cfg.get("end_date") or "")
        raw_end_norm = str(raw_end).strip().lower()
        if (not raw_end_norm) or raw_end_norm in {"auto", "current_month", "heute", "aktuell"}:
            end_date_resolved = DashDownloadConfig._resolve_current_month()
        else:
            end_date_resolved = cfg.get("end_date")

        # cache_dir: relative Pfade nach scenario/
        yaml_cache_dir = cache_cfg.get("cache_dir", str(SCENARIO_DIR / "cache"))
        cache_dir_resolved = DashDownloadConfig._resolve_under_scenario_dir(yaml_cache_dir)

        # output_path: fehlend/leer -> DEFAULT_OUTPUT_XLSX; relative Pfade nach scenario/data
        yaml_output_path = cfg.get("output_path")
        if not yaml_output_path:
            output_path_resolved = str(DEFAULT_OUTPUT_XLSX)
        else:
            output_path_resolved = DashDownloadConfig._resolve_under_scenario_data(yaml_output_path)

        inst = DashDownloadConfig(
            start_date=start_date,
            end_date=end_date_resolved,
            prefer_cache=bool(cfg.get("prefer_cache", True)),

            cache_dir=cache_dir_resolved,
            cache_max_age_days=int(cache_cfg.get("cache_max_age_days", 60)),

            anchor_var=cfg.get("anchor_var"),
            series_definitions=cfg.get("series_definitions") or {},

            enable_index=bool(cfg.get("enable_index", True)),
            index_aggregate=str(cfg.get("index_aggregate", "base")),  # NEU: Aggregationsmodus
            index_base_year=int(cfg.get("index_base_year", 2015)),
            index_base_value=float(cfg.get("index_base_value", 100.0)),

            download_timeout_seconds=int(cfg.get("download_timeout_seconds", 30)),
            min_response_size=int(cfg.get("min_response_size", 100)),
            source_overrides=cfg.get("source_overrides") or {},
            min_populated_vars=int(cfg.get("min_populated_vars", 2)),

            calendar_freq=cal.get("freq", "MS"),
            calendar_fill=cal.get("fill", "none"),
            calendar_fill_limit=cal.get("fill_limit", None),

            output_path=output_path_resolved,
        )

        # Debug-Ausgabe der wichtigsten Parameter
        print("[Loader] YAML-Konfiguration geladen:")
        print(f"  start_date={inst.start_date}  end_date={inst.end_date}")
        print(f"  prefer_cache={inst.prefer_cache}  cache_dir={inst.cache_dir}  cache_max_age_days={inst.cache_max_age_days}")
        print(f"  enable_index={inst.enable_index}  index_aggregate={inst.index_aggregate}  "
              f"index_base_year={inst.index_base_year}  index_base_value={inst.index_base_value}")
        print(f"  calendar: freq={inst.calendar_freq}  fill={inst.calendar_fill}  fill_limit={inst.calendar_fill_limit}")
        print(f"  min_populated_vars={inst.min_populated_vars}  anchor_var={inst.anchor_var}")
        try:
            print(f"  output_path={Path(inst.output_path).resolve()}")
        except Exception:
            print(f"  output_path={inst.output_path}")
        try:
            print(f"  series_definitions={list((inst.series_definitions or {}).keys())} (n={len(inst.series_definitions or {})})")
        except Exception:
            pass

        return inst



# -------------------------------------------------------------------
# Parsing/Align-Helfer (wie im Original)
# -------------------------------------------------------------------
def _parse_date_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # YYYY-MM-DD
    mask_ymd = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if mask_ymd.all():
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # YYYY-MM -> -01
    mask_ym = s.str.match(r"^\d{4}-\d{2}$")
    if mask_ym.all():
        return pd.to_datetime(s + "-01", format="%Y-%m-%d", errors="coerce")

    # YYYY-Qn -> Quartalsende
    mask_yq = s.str.match(r"^\d{4}-Q[1-4]$")
    if mask_yq.all():
        q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
        mapped = s.str.replace(
            r"^(\d{4})-(Q[1-4])$",
            lambda m: f"{m.group(1)}-{q_map[m.group(2)]}",
            regex=True
        )
        return pd.to_datetime(mapped, format="%Y-%m-%d", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


_ECB_PREFIXES = {
    "ICP", "RESR", "BP6", "MNA", "LFSI", "FM", "BSI", "BLS", "MIR",
    "PSS", "TRD", "STS", "SEC", "TRI", "EI", "CPI", "HICP", "ILM",
}
_BUBA_PREFIXES = {"BBAF3", "BBK", "BBEX", "BBK01", "BBK01U", "BB", "BBK0"}

def _prefix(code: str) -> str:
    return code.split(".", 1)[0].upper().strip()

def _resolve_source(code: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """
    ECB/BUBA-Auflösung – exakt wie im Original:
      1) exakte Overrides
      2) Prefix-Overrides mit '.*'
      3) bekannte Präfixe
      4) Heuristik via detect_data_source
      5) Fallback auf BB-Präfix => BUBA sonst ECB
    """
    c_up = code.upper().strip()
    pfx = _prefix(c_up)

    if overrides and c_up in (overrides or {}):
        cand = overrides[c_up].upper()
        src = cand if cand in {"ECB", "BUBA"} else "ECB"
        print(f"[Loader] Source-Override (exact) für {code}: {src}")
        return src

    if overrides:
        for k, v in overrides.items():
            ku = k.upper().strip()
            if ku.endswith(".*") and pfx == ku[:-2]:
                cand = v.upper()
                src = cand if cand in {"ECB", "BUBA"} else "ECB"
                print(f"[Loader] Source-Override (prefix) für {code} via {ku}: {src}")
                return src

    if pfx in _ECB_PREFIXES:
        print(f"[Loader] Source-Prefix (ECB) für {code}")
        return "ECB"
    if pfx in _BUBA_PREFIXES:
        print(f"[Loader] Source-Prefix (BUBA) für {code}")
        return "BUBA"

    try:
        ds = detect_data_source(c_up)
        src = "ECB" if (ds and str(ds).upper().startswith("ECB")) else "BUBA"
        print(f"[Loader] Source-Heuristik für {code}: {src}")
        return src
    except Exception:
        if pfx.startswith(("BBA", "BB")):
            print(f"[Loader] Source-Fallback (BUBA) für {code}")
            return "BUBA"
        print(f"[Loader] Source-Fallback (ECB) für {code}")
        return "ECB"



SIMPLE_TARGET_FALLBACKS = {
    "PH_KREDITE": "BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B",
    "PH_EINLAGEN": "BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B",
    "PH_WERTPAPIERE": "BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B",
    "PH_VERSICHERUNGEN": "BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B",
    "NF_KG_EINLAGEN": "BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_WERTPAPIERE": "BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_VERSICHERUNGEN": "BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_KREDITE": "BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B",
}

def _merge_series_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    print(f"[Loader] _merge_series_data: Eingang n={len(data_dict)}")
    all_series = []
    for code, df in data_dict.items():
        if df is None or df.empty:
            print(f"[Loader]   - {code}: leer")
            continue
        cols = {c.lower(): c for c in df.columns}
        if "datum" not in cols or "value" not in cols:
            print(f"[Loader]   - {code}: fehlende Spalten (erwartet 'Datum'+'value'), cols={list(df.columns)}")
            continue
        dcol = cols["datum"]
        series_df = df.copy()
        series_df[dcol] = _parse_date_column(series_df[dcol])
        series_df = series_df.set_index(dcol)[["value"]].rename(columns={"value": code})
        print(f"[Loader]   - {code}: shape={series_df.shape}, range={series_df.index.min()}..{series_df.index.max()}")
        all_series.append(series_df)

    if not all_series:
        print("[Loader] _merge_series_data: keine gültigen Serien")
        return pd.DataFrame()

    merged_df = pd.concat(all_series, axis=1, sort=True)
    merged_df = merged_df.reset_index().rename(columns={"index": "Datum"})
    merged_df = merged_df.sort_values("Datum").reset_index(drop=True)
    print(f"[Loader] _merge_series_data: merged shape={merged_df.shape}, Datum={merged_df['Datum'].min()}..{merged_df['Datum'].max()}")
    return merged_df


def _build_calendar_index(start: str, end: str, freq: str = "MS") -> pd.DataFrame:
    s = pd.to_datetime(start + "-01" if len(start) == 7 else start)
    e = pd.to_datetime(end + "-01" if len(end) == 7 else end)
    rng = pd.date_range(s, e, freq=freq, inclusive="both")
    return pd.DataFrame({"Datum": rng})

def _align_to_calendar(
    merged: pd.DataFrame,
    start: str,
    end: str,
    *,
    freq: str,
    fill: str,
    fill_limit: Optional[int],
) -> pd.DataFrame:
    cal = _build_calendar_index(start, end, freq=freq)
    print(f"[Loader] _align_to_calendar: freq={freq}, fill={fill}, fill_limit={fill_limit}, "
          f"calendar_len={len(cal)}, range={cal['Datum'].min()}..{cal['Datum'].max()}")
    out = cal.merge(merged, on="Datum", how="left")
    if fill in {"ffill", "bfill"}:
        method = {"ffill": "ffill", "bfill": "bfill"}[fill]
        value_cols = [c for c in out.columns if c != "Datum"]
        before_na = out[value_cols].isna().sum().sum()
        out[value_cols] = out[value_cols].fillna(method=method, limit=fill_limit)
        after_na = out[value_cols].isna().sum().sum()
        print(f"[Loader]   Fill '{method}': total NaNs {before_na} → {after_na}")
    print(f"[Loader] _align_to_calendar: out shape={out.shape}")
    return out



# -------------------------------------------------------------------
# Downloader-Klasse (Fetch-Logik identisch zur Referenz)
# -------------------------------------------------------------------
class DashDataDownloader:
    def __init__(self, cfg: DashDownloadConfig, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.cache = CacheManager(
            cache_dir=cfg.cache_dir,
            cache_max_age_days=cfg.cache_max_age_days,
        )

    def _fetch_sync(self, codes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Synchrone Fetch-Pipeline. Ruft die ORIGINALEN fetch_*_sync Funktionen auf,
        ohne deren Signaturen zu verändern. Für jeden Code wird zuvor die Quelle
        (ECB/BUBA) ermittelt und dann die passende Sync-Funktion verwendet.
        """
        out: Dict[str, pd.DataFrame] = {}
        print(f"[Loader] _fetch_sync: n={len(codes)}")
        for code in codes:
            try:
                source = _resolve_source(code, self.cfg.source_overrides)
                if source == "ECB":
                    df = fetch_ecb_sync(
                        code,
                        self.cfg.start_date,
                        self.cfg.end_date,
                        min_response_size=self.cfg.min_response_size,
                        timeout_seconds=self.cfg.download_timeout_seconds,
                    )
                else:
                    # *** WICHTIG: BuBa synchron laden, um Async-Signaturprobleme zu vermeiden ***
                    df = fetch_buba_sync(
                        code,
                        self.cfg.start_date,
                        self.cfg.end_date,
                        min_response_size=self.cfg.min_response_size,
                        timeout_seconds=self.cfg.download_timeout_seconds,
                    )
                if "Datum" in df.columns:
                    df["Datum"] = _parse_date_column(df["Datum"])
                out[code] = df
                print(f"[Loader]   ✓ {code} [{source}]: {len(df)} observations, "
                      f"date={df['Datum'].min() if 'Datum' in df else 'n/a'}.."
                      f"{df['Datum'].max() if 'Datum' in df else 'n/a'}")
            except Exception as e:
                print(f"[Loader]   ✗ {code}: {e}")
            # leichte Pause, um Rate-Limits zu schonen
            import time; time.sleep(0.4)
        return out

    async def _fetch_async(self, codes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Asynchrone Fetch-Pipeline NUR für ECB-Codes.
        BuBa-Codes dürfen aufgrund der sensiblen Signaturen NICHT async angefasst werden
        und werden separat synchron geladen (_fetch_sync).
        """
        # Split in ECB- und BuBa-Codes
        ecb_codes: List[str] = []
        buba_codes: List[str] = []
        for code in codes:
            src = _resolve_source(code, self.cfg.source_overrides)
            (ecb_codes if src == "ECB" else buba_codes).append(code)

        results: Dict[str, pd.DataFrame] = {}
        if ecb_codes:
            print(f"[Loader] _fetch_async (ECB): n={len(ecb_codes)}")
            try:
                import aiohttp  # type: ignore
            except Exception:
                print("[Loader] _fetch_async: aiohttp fehlt -> ECB Sync-Fallback")
                results.update(self._fetch_sync(ecb_codes))
            else:
                async with aiohttp.ClientSession() as session:
                    for code in ecb_codes:
                        try:
                            # *** WICHTIG: ORIGINAL-FUNKTION verwenden, aber nur für ECB ***
                            df = await fetch_ecb_async(
                                session,
                                code,
                                self.cfg.start_date,
                                self.cfg.end_date,
                                min_response_size=self.cfg.min_response_size,
                                timeout_seconds=self.cfg.download_timeout_seconds,
                            )
                            if "Datum" in df.columns:
                                df["Datum"] = _parse_date_column(df["Datum"])
                            results[code] = df
                            print(f"[Loader]   ✓ {code} [ECB-async]: {len(df)} observations, "
                                  f"date={df['Datum'].min() if 'Datum' in df else 'n/a'}.."
                                  f"{df['Datum'].max() if 'Datum' in df else 'n/a'}")
                        except Exception as e:
                            print(f"[Loader]   ✗ {code} [ECB-async]: {e}")
                        # kleine Pause zwischen den Requests
                        import asyncio as _asyncio
                        await _asyncio.sleep(0.2)

        if buba_codes:
            # *** BuBa NIE async anfassen – stattdessen synchron laden ***
            print(f"[Loader] _fetch_async: delegiere BuBa synchron (n={len(buba_codes)})")
            results.update(self._fetch_sync(buba_codes))

        return results


    def _collect_codes(self) -> Tuple[Dict[str, str], Dict[str, List[str]], List[str]]:
        """
        Zerlegt series_definitions in:
        - regular_codes: var_name -> series_code
        - index_defs:    var_name -> [series_codes] (nur wenn enable_index=True)
        - all_codes:     tatsächlich zu ladende Codes
        """
        defs = (self.cfg.series_definitions or {})
        print(f"[Loader] _collect_codes: series_definitions n={len(defs)} enable_index={self.cfg.enable_index}")

        regular_codes: Dict[str, str] = {}
        index_defs: Dict[str, List[str]] = {}

        if not defs:
            return {}, {}, []

        if not self.cfg.enable_index:
            for var_name, definition in defs.items():
                regular_codes[var_name] = definition
            all_codes = list({code for code in regular_codes.values()})
            print(f"[Loader]   regular-only n={len(all_codes)}")
            return regular_codes, index_defs, all_codes

        for var_name, definition in defs.items():
            idx_codes = parse_index_specification(definition)
            if idx_codes:
                index_defs[var_name] = idx_codes
            else:
                regular_codes[var_name] = definition

        all_codes = set(regular_codes.values())
        for codes in index_defs.values():
            all_codes.update(codes)

        print(f"[Loader]   regular={list(regular_codes.keys())}, index={list(index_defs.keys())}")
        print(f"[Loader]   total series to request n={len(all_codes)}")
        return regular_codes, index_defs, list(all_codes)

    def _load_from_cache(self, all_codes: List[str]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        cached_data: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []
        if self.cfg.prefer_cache:
            print(f"[Loader] _load_from_cache: prefer_cache=True, cache_dir={self.cfg.cache_dir}")
            for code in all_codes:
                dfc = self.cache.read_cache(code)
                if dfc is not None and not dfc.empty:
                    if "Datum" in dfc.columns:
                        dfc["Datum"] = _parse_date_column(dfc["Datum"])
                    cached_data[code] = dfc
                    print(f"[Loader]   cache HIT: {code} ({len(dfc)} rows)")
                else:
                    missing.append(code)
                    print(f"[Loader]   cache MISS: {code}")
        else:
            print("[Loader] _load_from_cache: prefer_cache=False (skip cache)")
            missing = all_codes[:]
        print(f"[Loader]   cached={len(cached_data)}  missing={len(missing)}")
        return cached_data, missing


    def _download_missing(self, missing: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Lädt fehlende Serien nach:
        - ECB-Codes: asynchron (falls möglich)
        - BuBa-Codes: **immer synchron**
        So bleiben die ORIGINALEN Downloader-Funktionen intakt.
        """
        if not missing:
            print("[Loader] _download_missing: nothing missing")
            return {}

        # Split in ECB/BuBa
        ecb_codes: List[str] = []
        buba_codes: List[str] = []
        for code in missing:
            src = _resolve_source(code, self.cfg.source_overrides)
            (ecb_codes if src == "ECB" else buba_codes).append(code)

        print(f"[Loader] _download_missing: ECB={len(ecb_codes)} BuBa={len(buba_codes)}")

        results: Dict[str, pd.DataFrame] = {}

        # 1) ECB async (mit sauberem Fallback)
        if ecb_codes:
            print("[Loader] _download_missing: starte ECB async...")
            try:
                import asyncio
                results.update(asyncio.run(self._fetch_async(ecb_codes)))
            except RuntimeError as e:
                # z. B. "cannot be called from a running event loop"
                if "cannot be called from a running event loop" in str(e):
                    print("[Loader] _download_missing: running loop -> nest_asyncio + async")
                    import nest_asyncio  # type: ignore
                    nest_asyncio.apply()
                    import asyncio as _asyncio
                    results.update(_asyncio.run(self._fetch_async(ecb_codes)))
                else:
                    print(f"[Loader] _download_missing: ECB async failed -> ECB sync fallback ({e})")
                    results.update(self._fetch_sync(ecb_codes))
            except Exception as ex:
                print(f"[Loader] _download_missing: ECB async exception -> ECB sync fallback ({ex})")
                results.update(self._fetch_sync(ecb_codes))

        # 2) BuBa sync
        if buba_codes:
            print("[Loader] _download_missing: starte BuBa sync...")
            results.update(self._fetch_sync(buba_codes))

        return results

    def _postprocess_final(
        self,
        merged: pd.DataFrame,
        regular_codes: List[str],
        index_defs: Dict[str, List[str]]
    ) -> pd.DataFrame:
        print(f"[Loader] _postprocess_final: merged shape={merged.shape}, columns={len(merged.columns)}")

        # 1) Kalender-Ausrichtung gemäß Konfiguration
        merged = _align_to_calendar(
            merged,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            freq=self.cfg.calendar_freq,
            fill=self.cfg.calendar_fill,
            fill_limit=self.cfg.calendar_fill_limit,
        )

        # 2) Ziel-Tabellenaufbau
        final_data: Dict[str, Any] = {}
        final_data["Datum"] = merged["Datum"].values

        # 2a) Reguläre Codes direkt übernehmen
        for code in regular_codes:
            if code in merged.columns:
                final_data[code] = merged[code].values
                print(f"[Loader]   regular: {code} ✓")
            else:
                print(f"[Loader]   regular: {code} ✗ (nicht gefunden)")

        # 2b) Index-Definitionen: je nach Modus ("base" -> IndexCreator, "sum" -> reine Summe)
        if self.cfg.enable_index and index_defs:
            print("[Loader]   index: enabled – begin calculation")
            use_sum = (self.cfg.index_aggregate or "base").lower() == "sum"
            if not use_sum:
                indexer = IndexCreator(
                    index_base_year=self.cfg.index_base_year,
                    index_base_value=self.cfg.index_base_value,
                )
            print(f"[Loader]   index: enabled, mode={'SUM' if use_sum else 'BASE'}, "
                  f"base_year={self.cfg.index_base_year}, base_value={self.cfg.index_base_value}")

            for var_name, idx_codes in index_defs.items():
                try:
                    available = [c for c in idx_codes if c in merged.columns]
                    print(f"[Loader]     index var={var_name}: available={len(available)}/{len(idx_codes)}")
                    if len(available) >= max(1, int(len(idx_codes) * 0.3)):
                        if use_sum:
                            summed = merged[available].sum(axis=1, min_count=1)
                            final_data[var_name] = summed.values
                            print(f"[Loader]     -> SUM[{var_name}] from {available}")
                        else:
                            index_series = indexer.create_index(merged, available, var_name)
                            aligned_index = index_series.reindex(pd.to_datetime(merged["Datum"]))
                            final_data[var_name] = aligned_index.values
                            print(f"[Loader]     -> BASE[{var_name}] from {available}")
                    else:
                        fb = SIMPLE_TARGET_FALLBACKS.get(var_name)
                        if fb and fb in merged.columns:
                            final_data[var_name] = merged[fb]
                            print(f"[Loader]     -> FALLBACK[{var_name}] = {fb}")
                        else:
                            print(f"[Loader]     WARN: insufficient data for INDEX {var_name}")
                except Exception as e:
                    print(f"[Loader]     ERR index {var_name}: {e}")
                    fb = SIMPLE_TARGET_FALLBACKS.get(var_name)
                    if fb and fb in merged.columns and var_name not in final_data:
                        final_data[var_name] = merged[fb]
                        print(f"[Loader]     -> FALLBACK-after-failure[{var_name}] = {fb}")

        # 2c) DataFrame erstellen + sortieren
        final_df = pd.DataFrame(final_data)
        final_df["Datum"] = pd.to_datetime(final_df["Datum"])
        final_df = final_df.sort_values("Datum").reset_index(drop=True)

        # 2d) Leading-Trim (auf Basis min_populated_vars)
        value_cols = [c for c in final_df.columns if c != "Datum"]
        if value_cols:
            non_na_count = final_df[value_cols].notna().sum(axis=1)
            required = self.cfg.min_populated_vars if len(value_cols) >= self.cfg.min_populated_vars else 1
            keep_mask = non_na_count >= required
            if keep_mask.any():
                first_keep = keep_mask.idxmax()
                if first_keep > 0:
                    before = len(final_df)
                    final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                    print(f"[Loader]   Trim leading rows (<{required} vars): {before} → {len(final_df)} (first_keep={first_keep})")

        # 2e) Anchor-Fenster + weitere Trims
        if self.cfg.anchor_var and self.cfg.anchor_var in final_df.columns:
            mask_anchor = final_df[self.cfg.anchor_var].notna()
            if mask_anchor.any():
                start_anchor = final_df.loc[mask_anchor, "Datum"].min()
                end_anchor   = final_df.loc[mask_anchor, "Datum"].max()
                before_rows  = len(final_df)
                final_df = final_df[(final_df["Datum"] >= start_anchor) & (final_df["Datum"] <= end_anchor)].copy()
                final_df.reset_index(drop=True, inplace=True)
                print(f"[Loader]   Anchor window '{self.cfg.anchor_var}': {start_anchor.date()}.{end_anchor.date()} "
                      f"({before_rows} → {len(final_df)})")

            exog_cols = [c for c in final_df.columns if c not in ("Datum", self.cfg.anchor_var)]
            if exog_cols:
                tgt_notna = final_df[self.cfg.anchor_var].notna().values
                all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
                keep_start = 0
                for i in range(len(final_df)):
                    if not (tgt_notna[i] and all_exog_nan[i]):
                        keep_start = i
                        break
                if keep_start > 0:
                    before = len(final_df)
                    final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                    print(f"[Loader]   Trim target-only rows: {before} → {len(final_df)} (keep_start={keep_start})")

        print(f"[Loader] Final dataset: rows={final_df.shape[0]} cols(without Datum)={final_df.shape[1]-1}")
        try:
            tail_preview = final_df.tail(3)
            with pd.option_context('display.max_columns', None):
                print("[Loader] Final tail-preview (3 Zeilen):")
                print(tail_preview)
        except Exception:
            pass

        # 3) 'Gesamt GVB' – additives Derivat (vor Imputation initial bilden)
        gvb_components = [c for c in ["Wertpapiere", "Kredite", "Versicherungen", "Einlagen"] if c in final_df.columns]
        if gvb_components:
            final_df["Gesamt GVB"] = final_df[gvb_components].sum(axis=1, min_count=1)
            print(f"[Loader] 'Gesamt GVB' erzeugt aus {gvb_components}")
        else:
            print("[Loader] Hinweis: Keine GVB-Komponenten gefunden – 'Gesamt GVB' nicht angelegt.")

        # 4) Quartals-Imputation der GVB-Komponenten + Flag-Spalte
        try:
            import numpy as np

            candidate_exog = ["lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"]
            exog_cols_present = [c for c in candidate_exog if c in final_df.columns]

            is_quarter = final_df["Datum"].dt.month.isin([1, 4, 7, 10])
            targets = [c for c in ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite"] if c in final_df.columns]
            imputed_any = pd.Series(False, index=final_df.index)

            def _ols_impute(df: pd.DataFrame, y_col: str) -> int:
                if not exog_cols_present:
                    return 0
                train_mask = df[y_col].notna() & is_quarter
                if train_mask.sum() < 6:
                    return 0
                X_train = df.loc[train_mask, exog_cols_present].dropna(axis=1, how="all")
                train_row_mask = ~X_train.isna().any(axis=1)
                X_train = X_train.loc[train_row_mask]
                y_train = df.loc[X_train.index, y_col]
                if X_train.empty or X_train.shape[1] == 0 or y_train.empty:
                    return 0

                X_mat = np.column_stack([np.ones(len(X_train)), X_train.values])
                try:
                    beta, *_ = np.linalg.lstsq(X_mat, y_train.values, rcond=None)
                except Exception:
                    return 0

                test_mask = df[y_col].isna() & is_quarter
                X_test = df.loc[test_mask, X_train.columns].dropna(axis=0, how="any")
                if X_test.empty:
                    return 0
                X_mat_test = np.column_stack([np.ones(len(X_test)), X_test.values])
                y_pred = X_mat_test @ beta

                df.loc[X_test.index, y_col] = y_pred
                imputed_any.loc[X_test.index] = True
                return len(X_test)

            total_imputed = 0
            for y in targets:
                n_imp = _ols_impute(final_df, y)
                if n_imp:
                    print(f"[Loader]   Imputation[{y}]: {n_imp} Quartalswerte ergänzt (OLS auf {exog_cols_present})")
                    total_imputed += n_imp

            # Nach Imputation: Komponenten erneut summieren → 'Gesamt GVB' aktualisieren
            if targets:
                comps = [c for c in ["Wertpapiere", "Kredite", "Versicherungen", "Einlagen"] if c in final_df.columns]
                if comps:
                    final_df["Gesamt GVB"] = final_df[comps].sum(axis=1, min_count=1)

            final_df["gvb_imputed"] = imputed_any.astype(bool)
            print(f"[Loader]   Imputation-Flag gesetzt: {final_df['gvb_imputed'].sum()} Zeilen mit Imputation (quartalsweise).")

        except Exception as ex:
            print(f"[Loader]   WARN: Imputation übersprungen ({ex})")

        # 5) Tail-Cut – nach letztem exogenen Datum und letztem vollständigen GVB-Quartal
        try:
            exog_present_for_cut = [c for c in ["lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"] if c in final_df.columns]
            last_exog_date = None
            if exog_present_for_cut:
                exog_complete_mask = final_df  # ggf. enger fassen, falls „alle exog vorhanden“ gewünscht
                if exog_complete_mask.any():
                    last_exog_date = final_df.loc[exog_complete_mask.index, "Datum"].max()

            comps_for_cut = [c for c in ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite"] if c in final_df.columns]
            last_target_q_date = None
            if comps_for_cut:
                is_q = final_df["Datum"].dt.month.isin([1, 4, 7, 10])
                tgt_complete_mask = is_q & final_df[comps_for_cut].notna().all(axis=1)
                if tgt_complete_mask.any():
                    last_target_q_date = final_df.loc[tgt_complete_mask, "Datum"].max()

            cutoff = None
            if (last_exog_date is not None) and (last_target_q_date is not None):
                cutoff = min(last_exog_date, last_target_q_date)
            elif last_exog_date is not None:
                cutoff = last_exog_date
            elif last_target_q_date is not None:
                cutoff = last_target_q_date

            if cutoff is not None:
                before = len(final_df)
                final_df = final_df[final_df["Datum"] <= cutoff].copy()
                after = len(final_df)
                print(f"[Loader]   Tail cut at {cutoff.date()}: {before} → {after} rows "
                      f"(last_exog={last_exog_date.date() if last_exog_date is not None else 'n/a'}, "
                      f"last_target_q={last_target_q_date.date() if last_target_q_date is not None else 'n/a'})")
            else:
                print("[Loader]   Tail cut skipped (no definitive cutoff could be determined)")
        except Exception as ex:
            print(f"[Loader]   WARN: Tail cut skipped due to error ({ex})")

        # Abschlussvorschau
        try:
            with pd.option_context('display.max_columns', None):
                print("[Loader] Final after cut (tail 3):")
                print(final_df.tail(3))
        except Exception:
            pass

        return final_df


    def _save_outputs(self, final_df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> Path:
        out_path = Path(self.cfg.output_path)
        # Sicherstellen, dass der Zielordner existiert (z. B. scenario/data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[Loader] Speichere Output nach: {out_path.resolve()}")
        except Exception:
            print(f"[Loader] Speichere Output nach: {out_path}")

        raw_ecb = {c: df for c, df in all_data.items() if _resolve_source(c, self.cfg.source_overrides) == "ECB"}
        raw_buba = {c: df for c, df in all_data.items() if _resolve_source(c, self.cfg.source_overrides) != "ECB"}

        with pd.ExcelWriter(out_path, engine=get_excel_engine()) as writer:
            # Sheets: Rohdaten pro Quelle (kalender-ausgerichtet, ohne Fills)
            if raw_ecb:
                ecb_merged = _merge_series_data(raw_ecb)
                ecb_merged = _align_to_calendar(
                    ecb_merged,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    freq=self.cfg.calendar_freq,
                    fill="none",
                    fill_limit=None,
                )
                ecb_merged.to_excel(writer, index=False, sheet_name="raw_ecb")
                print(f"[Loader]   Sheet 'raw_ecb': shape={ecb_merged.shape}")
            else:
                pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_ecb")
                print("[Loader]   Sheet 'raw_ecb': leer")

            if raw_buba:
                buba_merged = _merge_series_data(raw_buba)
                buba_merged = _align_to_calendar(
                    buba_merged,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    freq=self.cfg.calendar_freq,
                    fill="none",
                    fill_limit=None,
                )
                buba_merged.to_excel(writer, index=False, sheet_name="raw_buba")
                print(f"[Loader]   Sheet 'raw_buba': shape={buba_merged.shape}")
            else:
                pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_buba")
                print("[Loader]   Sheet 'raw_buba': leer")

            # Sheet: final_dataset
            final_df.to_excel(writer, index=False, sheet_name="final_dataset")
            print(f"[Loader]   Sheet 'final_dataset': shape={final_df.shape}")

            # Sheet: metadata (erweitert um index_aggregate)
            meta = pd.DataFrame({
                "param": [
                    "start_date", "end_date", "prefer_cache",
                    "index_base_year", "index_base_value",
                    "index_aggregate",                  # NEU
                    "anchor_var", "min_populated_vars",
                    "calendar_freq", "calendar_fill", "calendar_fill_limit",
                    "enable_index",
                ],
                "value": [
                    self.cfg.start_date, self.cfg.end_date, self.cfg.prefer_cache,
                    self.cfg.index_base_year, self.cfg.index_base_value,
                    self.cfg.index_aggregate,           # NEU
                    self.cfg.anchor_var, self.cfg.min_populated_vars,
                    self.cfg.calendar_freq, self.cfg.calendar_fill, self.cfg.calendar_fill_limit,
                    self.cfg.enable_index,
                ]
            })
            meta.to_excel(writer, index=False, sheet_name="metadata")
            print(f"[Loader]   Sheet 'metadata': shape={meta.shape}")

        print(f"[Loader] Wrote Excel: {out_path.resolve() if out_path.exists() else out_path}")
        return out_path


    def run(self, *, save: bool = True) -> Tuple[pd.DataFrame, Optional[Path]]:
        # 1) Codes sammeln
        regular_codes, index_defs, all_codes = self._collect_codes()
        _log(self.logger, f"Downloading {len((self.cfg.series_definitions or {}))} variables "
                          f"from {self.cfg.start_date} to {self.cfg.end_date}")
        _log(self.logger, f"Total series to download: {len(all_codes)}")

        # 2) Cache
        cached_data, missing = self._load_from_cache(all_codes)

        # 3) Download (ECB async, BuBa sync – siehe _download_missing)
        downloaded_data = self._download_missing(missing) if missing else {}
        print(f"[Loader] run: downloaded={len(downloaded_data)} cached={len(cached_data)}")

        # 4) Cache schreiben (nur neue)
        for code, df in downloaded_data.items():
            self.cache.write_cache(code, df)

        all_data = {**cached_data, **downloaded_data}
        if not all_data:
            raise RuntimeError("No series loaded successfully")

        # 5) Mergen + Postprocess
        merged = _merge_series_data(all_data)
        final_df = self._postprocess_final(merged, regular_codes, index_defs)

        # 6) Optional speichern
        out_path = self._save_outputs(final_df, all_data) if save else None

        # 7) Zusammenfassung der letzten Werte je Variable
        try:
            value_cols = [c for c in final_df.columns if c != "Datum"]
            if value_cols and not final_df.empty:
                last_row = final_df.iloc[-1][["Datum"] + value_cols]
                with pd.option_context('display.max_columns', None):
                    print("[Loader] Letzte Zeile (final_df.tail(1)) -> für UI/Vergleich:")
                    print(last_row.to_frame().T)
        except Exception:
            pass

        return final_df, out_path


# -------------------------------------------------------------------
# Dash-Helfer
# -------------------------------------------------------------------
def dash_download_and_merge_exog(config_path: str, *, logger=None, save: bool = True) -> pd.DataFrame:
    """
    Lädt YAML, führt Download aus, gibt finalen DataFrame zurück.
    Event-Loop-sicher (Async mit Sync-Fallback).
    """
    try:
        print(f"[Loader] dash_download_and_merge_exog: config={Path(config_path).resolve()}")
    except Exception:
        print(f"[Loader] dash_download_and_merge_exog: config={config_path}")

    cfg = DashDownloadConfig.from_yaml(config_path)
    runner = DashDataDownloader(cfg, logger=logger)
    df, out = runner.run(save=save)
    try:
        print(f"[Loader] dash_download_and_merge_exog: rows={len(df)} output={(out.resolve() if out else None)}")
    except Exception:
        print(f"[Loader] dash_download_and_merge_exog: rows={len(df)} output={out}")
    return df


def dataframe_to_dash_json(df: pd.DataFrame) -> Dict[str, Any]:
    """Konvertiert DataFrame für dcc.Store (orient='split') – Datum safe."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    out = df.to_dict(orient="split")
    try:
        # kurze Vorschau, damit man sieht, welche Spalten im Store landen
        print(f"[Loader] dataframe_to_dash_json: cols={out.get('columns')}, nrows={len(out.get('data') or [])}")
        if out.get("data"):
            print(f"[Loader] dataframe_to_dash_json: last-row={out['data'][-1]}")
    except Exception:
        pass
    return out

# instructor.py
# Liest variable_defs.xlsx, lädt alle Reihen über loader.py
# und schreibt eine dashboard-kompatible gvb_output.xlsx
# mit den 4 GVB-Sheets + final_dataset ins Projekt-Root.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import tempfile
import importlib
import importlib.util
import warnings
from datetime import date

import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ---------------------------------------------------------------------
# Basiskonfiguration
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
LOADER_DIR = THIS_FILE.parent               # .../loader

try:
    APP_ROOT: Path = LOADER_DIR.parent
except Exception:
    APP_ROOT = Path.cwd()

HERE = APP_ROOT

# liegt physisch in loader/variable_defs.xlsx
EXCEL_DEFS = "loader/variable_defs.xlsx"    # Definitions-Excel
OUTPUT_PARQUET = "loader/gvb_output.parquet"
OUTPUT_XLSX = "loader/gvb_output.xlsx"

LOADER_PATH: str | None = None     # falls loader.py woanders liegt
WRITE_FILES = True

# ---------------------------------------------------------------------
# kleine Helfer
# ---------------------------------------------------------------------
def _compute_current_quarter_end_str(today: date | None = None) -> str:
    if today is None:
        today = date.today()
    q_end_month = ((today.month - 1) // 3 + 1) * 3
    return f"{today.year}-{q_end_month:02d}"

# standard-config für den loader
CONFIG_DEFAULTS: dict = {
    "start_date": "2000-01",
    # Dynamisch auf aktuelles Quartal setzen
    "end_date": _compute_current_quarter_end_str(),
    "prefer_cache": True,
    "cache": {
        # Cache direkt im loader-Unterordner
        "cache_dir": str((HERE / "loader" / "financial_cache").resolve()),
        "cache_max_age_days": 60,
    },
    "anchor_var": None,
    "index_base_year": 2015,
    "index_base_value": 100.0,
    "download_timeout_seconds": 30,
    "min_response_size": 100,
    "source_overrides": {},
    "min_populated_vars": 2,
    "calendar_index": {
        "freq": "MS",
        "fill": "none",
        "fill_limit": None,
    },
    # loader.py ist bereits so gepatcht, dass .parquet + .xlsx geschrieben wird
    "output_path": str((HERE / "loader" / "gvb_output.parquet").resolve()),
}

# ---------------------------------------------------------------------
# weitere Helfer
# ---------------------------------------------------------------------

def _import_loader(loader_path: Optional[str | Path] = None):
    """Versucht zuerst 'import loader', sonst aus Datei laden."""
    try:
        return importlib.import_module("loader")
    except Exception:
        pass

    candidates: List[Path] = []
    if loader_path:
        p = Path(loader_path)
        candidates.append(p if p.is_absolute() else (HERE / p))
    candidates.extend([
        HERE / "loader.py",
        HERE / "loader" / "loader.py",
        HERE.parent / "loader.py",
        HERE.parent / "loader" / "loader.py",
    ])

    target = next((c for c in candidates if c.exists()), None)
    if not target:
        raise RuntimeError("loader.py nicht gefunden. Lege ihn ins Projekt oder setze LOADER_PATH.")

    spec = importlib.util.spec_from_file_location("loader", str(target))
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Kann loader nicht laden von {target}")
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules["loader"] = mod
    return mod


# Excel-Definitions-Sheets ------------------------------------------------
REQUIRED_COLS = [
    "ID Zeitreihe",
    "Clusterebene 1",
    "Clusterebene 3",
    "Clusterebene 4",
    "Clusterebene 5",
    "Zeitreihenschlüssel - Clean",
]


def _load_defs_sheet(xlsx: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = xlsx.parse(sheet_name)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' fehlt Spalten: {missing}")

    df = df[~df["Zeitreihenschlüssel - Clean"].isna()].copy()

    # doppelte IDs entschärfen
    if df["ID Zeitreihe"].duplicated().any():
        counts: Dict[str, int] = {}
        new_ids: List[str] = []
        for v in df["ID Zeitreihe"].astype(str):
            counts[v] = counts.get(v, 0) + 1
            new_ids.append(f"{v}__{counts[v]}" if counts[v] > 1 else v)
        df["ID Zeitreihe"] = new_ids

    return df


def _read_variable_defs(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #xls = pd.ExcelFile(xlsx_path)
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = {s.lower(): s for s in xls.sheet_names}
    sheet_fluss = next((v for k, v in sheets.items() if "fluss" in k), None)
    sheet_bestand = next((v for k, v in sheets.items() if "bestand" in k), None)
    if not sheet_fluss or not sheet_bestand:
        raise ValueError("variable_defs.xlsx braucht zwei Sheets mit 'Fluss' und 'Bestand' im Namen.")
    return _load_defs_sheet(xls, sheet_fluss), _load_defs_sheet(xls, sheet_bestand)


def _make_config(series_map: Dict[str, str], end_date: Optional[str] = None) -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    cfg["cache"] = dict(CONFIG_DEFAULTS["cache"])
    cfg["calendar_index"] = dict(CONFIG_DEFAULTS["calendar_index"])
    cfg["series_definitions"] = dict(series_map)
    cfg["end_date"] = end_date or _compute_current_quarter_end_str()
    return cfg


def _normalize_calendar_freq_for_loader(cfg: dict) -> dict:
    cal = dict(cfg.get("calendar_index", {}) or {})
    freq = (cal.get("freq") or "").upper()
    if freq.startswith("Q") or freq not in {"MS", "M", ""}:
        cal["freq"] = "MS"
        print("[INFO] calendar_index.freq auf 'MS' gesetzt (Loader-Kompatibilität).")
    cfg["calendar_index"] = cal
    return cfg


# Wide → Long -------------------------------------------------------------
def _align_wide_columns_to_ids(final_wide: pd.DataFrame, defs_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = final_wide.copy()

    # Datumsspalte bestimmen
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("datum", "date", "time_period", "period"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    ids = defs_df["ID Zeitreihe"].astype(str).tolist()
    codes = defs_df["Zeitreihenschlüssel - Clean"].astype(str).tolist()
    id_set = set(ids)
    code_to_id = dict(zip(codes, ids))

    cols_as_id = [c for c in df.columns if c in id_set]
    cols_as_code = [c for c in df.columns if c in code_to_id]

    print(f"[DEBUG] Loader-Wide Spalten (ohne Datum): {len([c for c in df.columns if c != date_col])}")
    print(f"[DEBUG] Gefunden als ID: {len(cols_as_id)} | als Code: {len(cols_as_code)}")

    # Codes auf IDs umbenennen
    rename_map = {c: code_to_id[c] for c in cols_as_code}
    df = df.rename(columns=rename_map)

    present_ids = [c for c in df.columns if c in id_set]
    print(f"[DEBUG] Präsente IDs nach Umbenennung: {len(present_ids)}")

    out = df[[date_col] + present_ids].copy()

    # komplett leere Spalten raus
    drop_cols = [c for c in out.columns if c != date_col and out[c].notna().sum() == 0]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out, date_col


def _make_long_gvb(final_wide: pd.DataFrame, defs_df: pd.DataFrame, value_col_name: str) -> pd.DataFrame:
    df_ids, date_col = _align_wide_columns_to_ids(final_wide, defs_df)
    if df_ids.shape[1] <= 1:
        return pd.DataFrame(columns=["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss", "sektor", "datatype"])

    m = df_ids.melt(id_vars=[date_col], var_name="ID Zeitreihe", value_name="value")

    meta_cols = ["ID Zeitreihe", "Clusterebene 1", "Clusterebene 3", "Clusterebene 4", "Clusterebene 5"]
    merged = m.merge(defs_df[meta_cols], on="ID Zeitreihe", how="left")

    merged = merged.rename(
        columns={
            date_col: "date",
            "Clusterebene 3": "ebene1",
            "Clusterebene 4": "ebene2",
            "Clusterebene 5": "ebene3",
        }
    )

    # value in die richtige Spalte
    if value_col_name == "fluss":
        merged["fluss"] = merged["value"]
        merged["bestand"] = pd.NA
    else:
        merged["bestand"] = merged["value"]
        merged["fluss"] = pd.NA

    merged["sektor"] = merged["Clusterebene 1"]
    merged["datatype"] = value_col_name

    out = merged[["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss", "sektor", "datatype"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # nur Zeilen mit Datum und der jeweiligen Kennzahl
    out = out[out["date"].notna() & out[value_col_name].notna()].reset_index(drop=True)
    return out


def _split_by_sector(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Teilt nach den echten Texten aus der Excel."""
    if df_long.empty:
        return df_long.copy(), df_long.copy()

    s = df_long["sektor"].astype(str)
    ph = df_long[s.str.contains("Private Haushalte", case=False, na=False)].copy()
    nfk = df_long[s.str.contains("Nichtfinanzielle Kapitalgesellschaften", case=False, na=False)].copy()

    return ph.reset_index(drop=True), nfk.reset_index(drop=True)


def _write_dashboard_excel(
    out_path: Path,
    final_wide: pd.DataFrame,
    fluss_ph: pd.DataFrame,
    fluss_nfk: pd.DataFrame,
    bestand_ph: pd.DataFrame,
    bestand_nfk: pd.DataFrame,
    loader_mod,
):
    # Excel-Engine möglichst aus loader (der hat bereits die Helper-Funktion)
    try:
        engine = loader_mod.get_excel_engine()
    except Exception:
        engine = "openpyxl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine=engine) as writer:
        final_wide.to_excel(writer, index=False, sheet_name="final_dataset")
        fluss_ph.to_excel(writer, index=False, sheet_name="fluss_ph")
        fluss_nfk.to_excel(writer, index=False, sheet_name="fluss_nfk")
        bestand_ph.to_excel(writer, index=False, sheet_name="bestand_ph")
        bestand_nfk.to_excel(writer, index=False, sheet_name="bestand_nfk")

    print(f"[INFO] Wrote Excel for dashboard: {out_path}")


# ---------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------
def run_gvb_from_excel(
    excel_defs: str = EXCEL_DEFS,
    loader_path: Optional[str | Path] = LOADER_PATH,
    write_files: bool = WRITE_FILES,
):
    # 1) Definitions-Excel
    defs_fluss, defs_bestand = _read_variable_defs(HERE / excel_defs)

    # 2) Codes einsammeln (Schlüssel = Code)
    series_map: Dict[str, str] = {}
    for df in (defs_fluss, defs_bestand):
        for _, row in df.iterrows():
            code = str(row["Zeitreihenschlüssel - Clean"]).strip()
            if not code or code.lower() == "nan":
                continue
            series_map[code] = code

    if not series_map:
        raise ValueError("Keine gültigen Serien in variable_defs.xlsx gefunden.")

    # 3) Config bauen
    cfg_inline = _make_config(series_map, end_date=None)  # None = use current quarter
    cfg_inline = _normalize_calendar_freq_for_loader(cfg_inline)

    # 4) Loader ausführen
    loader = _import_loader(loader_path)
    with tempfile.TemporaryDirectory() as td:
        tmp_yaml = Path(td) / "config_tmp.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_inline, f, sort_keys=False, allow_unicode=True)
        print("[INFO] Running loader with (inline config → temp YAML):", tmp_yaml)
        final_wide = loader.run_from_config(str(tmp_yaml))

    if not isinstance(final_wide, pd.DataFrame) or final_wide.empty:
        raise RuntimeError("Loader lieferte kein Wide-DataFrame oder ein leeres Ergebnis.")

    print(f"[INFO] Loader-Wide shape: {final_wide.shape}")

    # 5) Wide → Long
    fluss_long = _make_long_gvb(final_wide, defs_fluss, "fluss")
    bestand_long = _make_long_gvb(final_wide, defs_bestand, "bestand")

    print(f"[INFO] Rows (fluss_long):   {len(fluss_long)}")
    print(f"[INFO] Rows (bestand_long): {len(bestand_long)}")

    # 5b) nach Sektor splitten – diesmal nach langem Text!
    fluss_ph, fluss_nfk = _split_by_sector(fluss_long)
    bestand_ph, bestand_nfk = _split_by_sector(bestand_long)

    print(
        f"[INFO] Split -> fluss_ph: {len(fluss_ph)}, fluss_nfk: {len(fluss_nfk)}, "
        f"bestand_ph: {len(bestand_ph)}, bestand_nfk: {len(bestand_nfk)}"
    )

    # 6) Dateien schreiben
    if write_files:
        # kombiniertes Parquet
        combined = pd.concat([fluss_long, bestand_long], ignore_index=True)
        out_parquet = (HERE / OUTPUT_PARQUET).resolve()
        combined.to_parquet(out_parquet, index=False)
        print(f"[INFO] Wrote {out_parquet}")

        # dashboard-xlsx genau dort, wo app.py zuerst hinschaut
        out_xlsx = (HERE / OUTPUT_XLSX).resolve()
        _write_dashboard_excel(
            out_xlsx,
            final_wide,
            fluss_ph,
            fluss_nfk,
            bestand_ph,
            bestand_nfk,
            loader,
        )

    return {
        "fluss_ph": fluss_ph,
        "fluss_nfk": fluss_nfk,
        "bestand_ph": bestand_ph,
        "bestand_nfk": bestand_nfk,
        "wide": final_wide,
    }


if __name__ == "__main__":
    run_gvb_from_excel()

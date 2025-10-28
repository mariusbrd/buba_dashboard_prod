# Notebook Controller: GVB aus Excel-Definitionen bauen (ohne externe YAML)
# - Liest variable_defs.xlsx (Sheets mit "Fluss" und "Bestand")
# - Importiert loader.py (Downloader-Logik bleibt 1:1)
# - Erzeugt vier DataFrames in GVB-Form: fluss_ph, fluss_nfk, bestand_ph, bestand_nfk
# - Schreibt Excel + separate CSV/Parquet je Sheet
# - Zusätzlich kombinierte Parquetdatei (gvb_output.parquet)
# - NEU: Alle Konfigurationen (früher config.yaml) sind inline konfigurierbar

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys, tempfile, importlib, importlib.util
import warnings
import pandas as pd
import yaml  # nur zum Schreiben der temporären Datei

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ======================
# User-Parameter
# ======================
EXCEL_DEFS   = "variable_defs.xlsx"   # Excel mit "Fluss" & "Bestand"
OUTPUT_XLSX  = "gvb_output.xlsx"      # Ziel-Excel mit 4 Sheets
LOADER_PATH  = None                   # Optionaler Pfad zu loader.py, sonst Auto-Suche
WRITE_FILES  = True                   # True -> Excel + CSV/Parquet schreiben


# --- Inline-Defaults ---
CONFIG_DEFAULTS: dict = {
    "start_date": "2000-01",
    "end_date":   "2025-09",

    "prefer_cache": True,
    "cache": {
        "cache_dir": "financial_cache",
        "cache_max_age_days": 60,
    },

    "anchor_var": None,

    "index_base_year": 2015,
    "index_base_value": 100.0,

    "download_timeout_seconds": 30,
    "min_response_size": 100,

    "source_overrides": {
        # "BBAF3.*": "BUBA",
        # "BSI.*":   "ECB",
    },

    "min_populated_vars": 2,

    # ACHTUNG: loader._build_calendar_index unterstützt nur "MS" / "M".
    # Wir mappen Q-Frequenzen unten auf "MS".
    "calendar_index": {
        "freq": "MS",          # "MS" | "M"
        "fill": "none",        # "none" | "ffill" | "bfill"
        "fill_limit": None,
    },

    "output_path": OUTPUT_XLSX,
}

HERE = Path.cwd()



from datetime import date

def _compute_current_quarter_end_str(today: date | None = None) -> str:
    """
    Liefert das aktuelle Quartalsende als 'YYYY-MM' (Monatsende des Quartals: 03, 06, 09, 12).
    Beispiel: 2025-09 für Q3/2025.
    """
    if today is None:
        today = date.today()
    q_end_month = ((today.month - 1) // 3 + 1) * 3
    return f"{today.year}-{q_end_month:02d}"



# ============================================
# Loader robust importieren
# ============================================
def _import_loader(loader_path: Optional[str|Path] = None):
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
        raise RuntimeError("loader.py nicht gefunden. Lege es in den Notebook-Ordner oder setze LOADER_PATH.")

    spec = importlib.util.spec_from_file_location("loader", str(target))
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Kann loader nicht laden von {target}")
    spec.loader.exec_module(mod)
    sys.modules["loader"] = mod
    return mod

# ============================================
# Excel-Definitionen laden
# ============================================
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
    if df["ID Zeitreihe"].duplicated().any():
        counts: Dict[str,int] = {}
        new_ids: List[str] = []
        for v in df["ID Zeitreihe"].astype(str):
            counts[v] = counts.get(v, 0) + 1
            new_ids.append(f"{v}__{counts[v]}" if counts[v] > 1 else v)
        df["ID Zeitreihe"] = new_ids
    return df

def _read_variable_defs(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path)
    sheets = {s.lower(): s for s in xls.sheet_names}
    sheet_fluss   = next((v for k,v in sheets.items() if "fluss" in k), None)
    sheet_bestand = next((v for k,v in sheets.items() if "bestand" in k), None)
    if not sheet_fluss or not sheet_bestand:
        raise ValueError("Excel benötigt zwei Sheets mit 'Fluss' und 'Bestand' im Namen.")
    return _load_defs_sheet(xls, sheet_fluss), _load_defs_sheet(xls, sheet_bestand)

# ============================================
# Config bauen
# ============================================
def _make_config(series_map: Dict[str, str],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> dict:
    """
    Baut die Loader-Config. end_date wird standardmäßig auf das aktuelle Quartalsende gesetzt.
    Optional kann start_date/end_date über Parameter überschrieben werden.
    """
    cfg = dict(CONFIG_DEFAULTS)  # shallow copy
    # tiefe Copies für verschachtelte Dicts
    cfg["cache"] = dict(CONFIG_DEFAULTS.get("cache", {}))
    cfg["calendar_index"] = dict(CONFIG_DEFAULTS.get("calendar_index", {}))

    cfg["series_definitions"] = dict(series_map)
    cfg["output_path"] = OUTPUT_XLSX

    # Datumslogik
    if start_date is not None:
        cfg["start_date"] = start_date
    # Dynamisches Quartalsende, falls nichts explizit übergeben
    dyn_end = end_date or _compute_current_quarter_end_str()
    cfg["end_date"] = dyn_end
    print(f"[INFO] Verwende end_date = {cfg['end_date']} (aktuelles Quartalsende)")

    return cfg


def _normalize_calendar_freq_for_loader(cfg: dict) -> dict:
    """
    Mappe jegliche Q-Frequenzen auf 'MS', weil loader._build_calendar_index
    nur 'MS'/'M' versteht. (Damit vermeiden wir Invalid frequency: QE-DEC.)
    """
    cal = dict(cfg.get("calendar_index", {}) or {})
    freq = (cal.get("freq") or "").strip().upper()
    if freq.startswith("Q"):
        cal["freq"] = "MS"
        print("[INFO] calendar_index.freq war Quartal – auf 'MS' gemappt, damit der Loader läuft.")
    elif freq not in {"MS","M",""}:
        cal["freq"] = "MS"
        print(f"[INFO] calendar_index.freq '{freq}' nicht unterstützt – auf 'MS' gemappt.")
    cfg["calendar_index"] = cal
    return cfg

# ============================================
# Wide → Long in GVB-Form (mit Sektor-Split)
# ============================================
def _align_wide_columns_to_ids(final_wide: pd.DataFrame, defs_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Bringt das Wide-DF auf einheitliche Spaltennamen (IDs).
    Erkennt Spalten entweder über 'ID Zeitreihe' ODER über 'Zeitreihenschlüssel - Clean' (Code).
    """
    df = final_wide.copy()

    # Datumsspalte
    date_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("datum", "date", "time_period", "period"):
            date_col = c
            break
    if date_col is None:
        if df.index.name and str(df.index.name).lower() in ("datum","date","time_period","period"):
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            date_col = df.columns[0]

    id_list   = defs_df["ID Zeitreihe"].astype(str).tolist()
    code_list = defs_df["Zeitreihenschlüssel - Clean"].astype(str).tolist()
    id_set    = set(id_list)
    code_set  = set(code_list)

    cols_as_id   = [c for c in df.columns if c in id_set]
    cols_as_code = [c for c in df.columns if c in code_set]

    # Debug: Mapping-Statistik
    print(f"[DEBUG] Loader-Wide Spalten gesamt (ohne Datum): {len([c for c in df.columns if c != date_col])}")
    print(f"[DEBUG] Gefunden als ID: {len(cols_as_id)} | als Code: {len(cols_as_code)}")

    if not cols_as_id and not cols_as_code:
        sample_cols = [x for x in df.columns if x != date_col]
        raise ValueError(
            "Keine Spalten aus dem Loader-Output passen zu den 'ID Zeitreihe' ODER den 'Zeitreihenschlüssel - Clean'.\n"
            f"- Beispiel Wide-Spalten: {sample_cols[:8]}\n"
            f"- Beispiel IDs (aus defs): {id_list[:8]}\n"
            f"- Beispiel Codes (aus defs): {code_list[:8]}"
        )

    # Code → ID umbenennen
    code_to_id = dict(zip(code_list, id_list))
    rename_map = {c: code_to_id[c] for c in cols_as_code}
    df = df.rename(columns=rename_map)

    present_ids = [c for c in df.columns if c in id_set]
    print(f"[DEBUG] Präsente IDs nach Umbenennung: {len(present_ids)}")

    out = df[[date_col] + present_ids].copy()

    # Zusätzliche Sicherheit: entferne Spalten, die komplett NaN sind
    nunique_non_nan = out.drop(columns=[date_col]).count(axis=0)
    drop_these = [c for c, cnt in nunique_non_nan.items() if cnt == 0]
    if drop_these:
        print(f"[DEBUG] Entferne komplett leere Spalten: {drop_these[:8]}{'...' if len(drop_these)>8 else ''}")
        out = out.drop(columns=drop_these)

    return out, date_col

def _make_long_gvb(final_wide: pd.DataFrame, defs_df: pd.DataFrame, value_col_name: str) -> pd.DataFrame:
    df_ids, date_col = _align_wide_columns_to_ids(final_wide, defs_df)

    if df_ids.shape[1] <= 1:
        return pd.DataFrame(columns=["date","ebene1","ebene2","ebene3","bestand","fluss","__sektor"])

    m = df_ids.melt(id_vars=[date_col], var_name="ID Zeitreihe", value_name="value")

    meta_cols = ["ID Zeitreihe", "Clusterebene 1", "Clusterebene 3", "Clusterebene 4", "Clusterebene 5"]
    merged = m.merge(defs_df[meta_cols].copy(), on="ID Zeitreihe", how="left")

    merged.rename(columns={
        date_col: "date",
        "Clusterebene 3": "ebene1",
        "Clusterebene 4": "ebene2",
        "Clusterebene 5": "ebene3",
    }, inplace=True)

    # KORREKTUR: Setze beide auf NA, dann befülle nur die richtige
    merged["fluss"]   = pd.NA
    merged["bestand"] = pd.NA
    merged[value_col_name] = merged["value"]  # Das ist korrekt!
    merged["__sektor"] = merged["Clusterebene 1"]

    out = merged[["date","ebene1","ebene2","ebene3","bestand","fluss","__sektor"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # WICHTIG: Filtere basierend auf dem richtigen value_col_name!
    out = out[out["date"].notna() & out[value_col_name].notna()].reset_index(drop=True)
    return out

def _split_by_sector(df_long_with_sector: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_long_with_sector.empty:
        return (df_long_with_sector.copy(), df_long_with_sector.copy())
    ph  = df_long_with_sector[df_long_with_sector["__sektor"].astype(str).str.contains("Private Haushalte", na=False)].drop(columns="__sektor")
    nfk = df_long_with_sector[df_long_with_sector["__sektor"].astype(str).str.contains("Nichtfinanzielle Kapitalgesellschaften", na=False)].drop(columns="__sektor")
    return ph.reset_index(drop=True), nfk.reset_index(drop=True)

# ============================================
# Hauptfunktion
# ============================================
def run_gvb_from_excel(
    excel_defs: str = EXCEL_DEFS,
    loader_path: Optional[str|Path] = LOADER_PATH,
    write_files: bool = WRITE_FILES
):
    # 1) Excel-Definitionen lesen
    defs_fluss, defs_bestand = _read_variable_defs(HERE / excel_defs)

    # 2) series_definitions bauen – **Schlüssel = CODE**, nicht ID!
    #    So vermeiden wir Überschreiben gleicher IDs aus Fluss/Bestand.
    series_map: Dict[str, str] = {}
    dup_codes: set = set()
    all_codes_seen: set = set()

    for df, tag in ((defs_fluss, "FLUSS"), (defs_bestand, "BESTAND")):
        for _, row in df.iterrows():
            code = str(row["Zeitreihenschlüssel - Clean"]).strip()
            if not code or code.lower() == "nan":
                continue
            if code in all_codes_seen:
                dup_codes.add(code)
            all_codes_seen.add(code)
            # var_name == code → Loader-Spalten heißen nach dem Code
            series_map[code] = code

    if not series_map:
        raise ValueError("Keine gültigen Serien in variable_defs.xlsx gefunden (Codes leer?).")

    if dup_codes:
        print(f"[WARN] Dieselben Codes in beiden Sheets gefunden (werden einmal geladen): {list(sorted(dup_codes))[:8]}{' ...' if len(dup_codes) > 8 else ''}")

    # 3) Config + Frequenz-Mapping passend für loader
    #    (nutzt weiter dein Quartalsende-Default u. a.)
    cfg_inline = _make_config(series_map, end_date="2024-12")
    cfg_inline = _normalize_calendar_freq_for_loader(cfg_inline)

    # 4) Loader ausführen → Wide mit Spaltennamen = Codes
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

    # 5) Wide → Long in GVB-Form (getrennt je Sheet)
    #    _align_wide_columns_to_ids erkennt Codes u. mappt sie auf IDs je Sheet.
    fluss_long   = _make_long_gvb(final_wide, defs_fluss,   value_col_name="fluss")
    bestand_long = _make_long_gvb(final_wide, defs_bestand, value_col_name="bestand")

    print(f"[INFO] Rows (fluss_long):   {len(fluss_long)}")
    print(f"[INFO] Rows (bestand_long): {len(bestand_long)}")

    fluss_ph, fluss_nfk     = _split_by_sector(fluss_long)
    bestand_ph, bestand_nfk = _split_by_sector(bestand_long)

    print(f"[INFO] Split -> fluss_ph: {len(fluss_ph)}, fluss_nfk: {len(fluss_nfk)}, "
          f"bestand_ph: {len(bestand_ph)}, bestand_nfk: {len(bestand_nfk)}")

    # 6) Dateien schreiben (unverändert)
    if write_files:
        out_xlsx = (HERE / OUTPUT_XLSX).resolve()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            fluss_ph.to_excel(writer, index=False, sheet_name="fluss_ph")
            fluss_nfk.to_excel(writer, index=False, sheet_name="fluss_nfk")
            bestand_ph.to_excel(writer, index=False, sheet_name="bestand_ph")
            bestand_nfk.to_excel(writer, index=False, sheet_name="bestand_nfk")

        base = out_xlsx.with_suffix("")
        try:
            fluss_ph.to_csv(str(base) + "_fluss_ph.csv", index=False)
            fluss_nfk.to_csv(str(base) + "_fluss_nfk.csv", index=False)
            bestand_ph.to_csv(str(base) + "_bestand_ph.csv", index=False)
            bestand_nfk.to_csv(str(base) + "_bestand_nfk.csv", index=False)

            fluss_ph.to_parquet(str(base) + "_fluss_ph.parquet", index=False)
            fluss_nfk.to_parquet(str(base) + "_fluss_nfk.parquet", index=False)
            bestand_ph.to_parquet(str(base) + "_bestand_ph.parquet", index=False)
            bestand_nfk.to_parquet(str(base) + "_bestand_nfk.parquet", index=False)
        except Exception as e:
            print(f"[WARN] Konnte Einzel-CSV/Parquet nicht schreiben: {e}")

        try:
            combined = []
            for df, name in [(fluss_ph, "fluss_ph"), (fluss_nfk, "fluss_nfk"),
                             (bestand_ph, "bestand_ph"), (bestand_nfk, "bestand_nfk")]:
                tmp = df.copy()
                tmp.insert(0, "sheet", name)
                combined.append(tmp)
            combined_df = pd.concat(combined, ignore_index=True)
            combined_df.to_parquet(str(base) + ".parquet", index=False)
            print(f"[INFO] Wrote {out_xlsx} (+ CSV/Parquet je Sheet) und kombinierte Parquet: {str(base)}.parquet")
        except Exception as e:
            print(f"[WARN] Konnte kombinierte Parquetdatei nicht schreiben: {e}")

    return {
        "fluss_ph": fluss_ph,
        "fluss_nfk": fluss_nfk,
        "bestand_ph": bestand_ph,
        "bestand_nfk": bestand_nfk,
        "wide": final_wide,
    }

if __name__ == "__main__":
    res = run_gvb_from_excel()

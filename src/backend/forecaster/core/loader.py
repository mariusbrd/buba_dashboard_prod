# =============================================================================
# Datei- und Pfad-Utilities (Schritt 8)
# =============================================================================
import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

import logging

_logger = logging.getLogger("forecaster_pipeline")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: Union[str, Path], label: str = "CSV") -> None:
    path = str(path)
    try:
        ensure_dir(Path(path).parent)
        df.to_csv(path, index=False)
        _logger.debug(f"{label} geschrieben: {path}")
    except Exception as e:
        _logger.error(f"{label} konnte nicht geschrieben werden: {e}")


def safe_write_json(obj, path: Union[str, Path], label: str = "JSON") -> None:
    path = str(path)
    try:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        _logger.debug(f"{label} geschrieben: {path}")
    except Exception as e:
        _logger.error(f"{label} konnte nicht geschrieben werden: {e}")


def harvest_exogs_from_downloader_output(path: str) -> List[str]:
    """
    Liest die vom Downloader erzeugte Datei ein und gibt die Spaltennamen
    der exogenen Reihen zurück (ohne Zeitspalten).
    Unterstützt: .xlsx/.xls/.csv/.parquet
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Downloader-Output nicht gefunden: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unbekanntes Downloader-Output-Format: {ext}")

    date_like = {"date", "datum", "time", "quarter", "q", "period"}
    cols = [c for c in df.columns if str(c).strip() and str(c).strip().lower() not in date_like]
    return cols


def autodetect_downloader_output(candidates_dirs: List[str]) -> Optional[str]:
    """
    Sucht heuristisch nach der zuletzt geschriebenen Downloader-Datei in den
    angegebenen Verzeichnissen.
    """
    patterns = ["output.xlsx", "output.csv", "output.parquet", "*.xlsx", "*.csv", "*.parquet"]
    found: List[str] = []
    for d in filter(None, candidates_dirs):
        try:
            for pat in patterns:
                found.extend(glob.glob(os.path.join(d, pat)))
        except Exception:
            pass
    if not found:
        return None
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found[0]


def _to_jsonable(obj):
    """Wandelt verschachtelte Objekte (pandas/numpy/Period/Timestamp etc.) in JSON-kompatible Typen um."""
    import numpy as _np
    import pandas as _pd
    from datetime import date, datetime

    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (not _np.isfinite(obj)):
            return None
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, _pd.Series):
        return {str(k): _to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, _pd.DataFrame):
        return [
            {str(k): _to_jsonable(v) for k, v in rec.items()}
            for rec in obj.to_dict(orient="records")
        ]
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return f"<<unserializable:{type(obj).__name__}>>"
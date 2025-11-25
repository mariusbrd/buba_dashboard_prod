# forecaster_pipeline.py
# Aufgabe: Kernpipeline für Forecast. Einlesen, Aufbereiten, Modell verwalten, Prognose erstellen.
# -*- coding: utf-8 -*-
"""
Decision-Tree-Forecasting-Pipeline mit robustem Logging, UTF-8-Umlaute
und schlanker, wartbarer Struktur.

Schritt 7: Pipeline in drei sichtbare Abschnitte geteilt

Abschnitt S1 – Daten laden & vorbereiten
Abschnitt S2 – Modell laden oder trainieren (mit Cache)
Abschnitt S3 – Zukunfts-Design bauen, Forecast rechnen, alles exportieren

Schritt 8: Dateibezogene Funktionen zusammengelegt
- zentrales Schreiben von CSV/JSON
- gemeinsamer Ort für Downloader-Helper
- weniger duplizierte Try/Except-Blöcke in S1/S3

Schritt 9: Zukunftsaufbau zusammengelegt
- Future-Design-Bau + Auffüllen auf X_cols in einer Funktion
- S3 ruft nur noch einen Einstiegspunkt für den Zukunftsaufbau

Weitere Änderungen/Refactor:
- UTF-8 Fix + Symbol-Fallback
- Konsistentes Logging statt print()
- Hilfsfunktionen aus run_production_pipeline herausgezogen
- Caching- und Export-Pfade bleiben kompatibel
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor

# Pandas Anzeigeoptionen (nur für lokale Diagnose)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)


FORECASTER_DIR = Path(__file__).resolve().parent  

try:
    APP_ROOT: Path = FORECASTER_DIR.parent    
except Exception:
    APP_ROOT = Path.cwd()



# =============================================================================
# Logging
# =============================================================================
def _supports_utf8() -> bool:
    # wir lassen die schönen Symbole grundsätzlich drin
    return True


def _sym(text: str) -> str:
    if _supports_utf8():
        return text
    repl = {"✓": "OK", "→": "->", "—": "-", "–": "-", "…": "...", "•": "*", "⚠": "WARN", "ℹ": "i"}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


LOGGER = logging.getLogger("forecaster_pipeline")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)
LOGGER.setLevel(logging.INFO)


# =============================================================================
# KONFIGURATION
# =============================================================================
@dataclass
class Config:
    # =========================
    # Datenquelle
    # =========================
    excel_path: str = "transformed_output.xlsx"
    sheet_name: str = "final_dataset"
    date_col: str = "Datum"
    target_col: str = "PH_EINLAGEN"

    # =========================
    # Aggregation
    # =========================
    agg_methods_exog: List[str] = field(default_factory=lambda: ["last"])
    agg_method_target: str = "mean"

    # =========================
    # Lags
    # =========================
    exog_month_lags: List[int] = field(default_factory=lambda: [-24,
                                                                -12,
                                                                -6,
                                                                -3,
                                                                -1])
    target_lags_q: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # =========================
    # Backtest / Validierung
    # =========================
    min_train_quarters: int = 24
    test_horizon_quarters: int = 2
    gap_quarters: int = 1

    # =========================
    # Param-Grid
    # =========================
    param_grid: Union[Dict, List[Dict]] = field(
        default_factory=lambda: [
            {
                "criterion": ["squared_error"],
                "max_depth": [8, 10, 12, 14, 16, 20, 50],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2],
                "max_features": [None],
                "ccp_alpha": [0.0],
            },
        ]
    )

    # =========================
    # Deterministische Features
    # =========================
    add_trend_features: bool = True
    trend_degree: int = 1
    add_seasonality: bool = True
    seasonality_mode: str = "dummies"

    # =========================
    # Transformation
    # =========================
    target_transform: str = "none"  # "none" | "yeo-johnson"
    target_standardize: bool = True

    # =========================
    # Forecast / Future-Exogs
    # =========================
    forecast_horizon: int = 6
    future_exog_strategy: str = "mixed"  # "mixed" | "arima" | "drift"
    future_exog_drift_window_q: int = 8
    future_exog_seasonal_period_q: int = 4

    # =========================
    # Output & Persistierung
    # =========================

    output_dir: str = field(
        default_factory=lambda: str((FORECASTER_DIR / "trained_outputs").resolve())
    )
    model_dir: str = field(
        default_factory=lambda: str((FORECASTER_DIR / "trained_models").resolve())
    )
    use_cached_model: bool = True
    random_state: int = 42

    # =========================
    # ARIMA-Optionen
    # =========================
    use_arima_extrapolation: bool = True
    arima_for_important_vars: bool = True
    arima_importance_threshold: float = 0.10

    # =========================
    # Debug / Diagnose
    # =========================
    debug_exog: bool = True
    debug_design: bool = True
    debug_recur: bool = True
    diag_max_cols: int = 20
    diag_show_heads: int = 2

    # =========================
    # Dumps für Offline-Check
    # =========================
    dump_future_design_csv: bool = True
    dump_future_design_path: Optional[str] = None

    dump_quarterly_dataset_csv: bool = True
    dump_quarterly_dataset_path: Optional[str] = None

    dump_train_design_csv: bool = True
    dump_train_design_path: Optional[str] = None

    # =========================
    # Seeds/Fallbacks für Rekursion
    # =========================
    stable_exog_cols: List[str] = field(default_factory=list)
    last_train_row: Optional[Dict[str, float]] = None
    train_feature_medians: Dict[str, float] = field(default_factory=dict)
    last_target_value: Optional[float] = None

    # =========================
    # Adapter/Cache
    # =========================
    cache_tag: str = ""
    selected_exog: List[str] = field(default_factory=list)
    data_signature: str = ""

    # =========================
    # Convenience
    # =========================
    def ensure_paths(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if self.dump_future_design_csv and self.dump_future_design_path is None:
            self.dump_future_design_path = str(Path(self.output_dir) / "future_design_debug.csv")

        if self.dump_quarterly_dataset_csv and self.dump_quarterly_dataset_path is None:
            self.dump_quarterly_dataset_path = str(Path(self.output_dir) / "train_quarterly_debug.csv")

        if self.dump_train_design_csv and self.dump_train_design_path is None:
            self.dump_train_design_path = str(Path(self.output_dir) / "train_design_debug.csv")

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        for k in [
            "output_dir",
            "model_dir",
            "dump_future_design_path",
            "dump_quarterly_dataset_path",
            "dump_train_design_path",
        ]:
            if isinstance(d.get(k), Path):
                d[k] = str(d[k])
        return d


# stabiler Alias
PipelineConfig = Config

__all__ = [
    "Config",
    "PipelineConfig",
    "ModelArtifact",
    "get_model_filepath",
    "run_production_pipeline",
    "list_saved_models",
    "delete_model",
    "compare_model_performance",
]


# =============================================================================
# Datei- und Pfad-Utilities (Schritt 8)
# =============================================================================
def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: Union[str, Path], label: str = "CSV") -> None:
    path = str(path)
    try:
        ensure_dir(Path(path).parent)
        df.to_csv(path, index=False)
        LOGGER.debug(f"{label} geschrieben: {path}")
    except Exception as e:
        LOGGER.error(f"{label} konnte nicht geschrieben werden: {e}")


def safe_write_json(obj, path: Union[str, Path], label: str = "JSON") -> None:
    path = str(path)
    try:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        LOGGER.debug(f"{label} geschrieben: {path}")
    except Exception as e:
        LOGGER.error(f"{label} konnte nicht geschrieben werden: {e}")


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


# =============================================================================
# MODELL-PERSISTIERUNG
# =============================================================================
class ModelArtifact:
    """Container für alle Modell-Artefakte."""

    def __init__(self, model, tj, X_cols, best_params, metadata, config_dict):
        self.model = model
        self.tj = tj
        self.X_cols = X_cols
        self.best_params = best_params
        self.metadata = metadata
        self.config_dict = config_dict

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "tj": self.tj,
            "X_cols": self.X_cols,
            "best_params": self.best_params,
            "metadata": self.metadata,
            "config_dict": self.config_dict,
        }
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        LOGGER.debug(_sym(f"✓ Modell gespeichert: {filepath}"))

    @staticmethod
    def load(filepath: str) -> "ModelArtifact":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modell nicht gefunden: {filepath}")
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)
        LOGGER.info(_sym(f"✓ Modell geladen: {filepath}"))
        return ModelArtifact(
            model=artifact["model"],
            tj=artifact["tj"],
            X_cols=artifact["X_cols"],
            best_params=artifact["best_params"],
            metadata=artifact["metadata"],
            config_dict=artifact["config_dict"],
        )

    @staticmethod
    def exists(filepath: str) -> bool:
        return os.path.exists(filepath)

    def is_compatible(self, current_config: Config) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        cfg_dict = asdict(current_config)

        critical_params = [
            "target_col",
            "agg_method_target",
            "exog_month_lags",
            "target_lags_q",
            "add_trend_features",
            "trend_degree",
            "add_seasonality",
            "seasonality_mode",
            "target_transform",
        ]
        for param in critical_params:
            if cfg_dict.get(param) != self.config_dict.get(param):
                issues.append(f"Config-Mismatch: {param} ({cfg_dict.get(param)} vs {self.config_dict.get(param)})")

        # Hinweise sind möglich, aber nicht blocking
        return len(issues) == 0, issues


def get_model_filepath(cfg: Config) -> str:
    """Generiert eindeutigen Modell-Pfad basierend auf Config + Exog-Signatur."""
    exog_list = getattr(cfg, "selected_exog", []) or []
    exog_sig = hashlib.md5(",".join(sorted(map(str, exog_list))).encode()).hexdigest()[:8]

    tag = (getattr(cfg, "cache_tag", "") or cfg.target_col or "model").lower()
    payload = {
        "tag": tag,
        "target_col": cfg.target_col,
        "agg_method_target": cfg.agg_method_target,
        "exog_month_lags": cfg.exog_month_lags,
        "target_lags_q": cfg.target_lags_q,
        "target_transform": cfg.target_transform,
        "forecast_horizon": getattr(cfg, "forecast_horizon", 4),
        "exog_sig": exog_sig,
    }
    config_hash = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    filename = f"model_{tag}_{config_hash}.pkl"
    return os.path.join(cfg.model_dir, filename)


# =============================================================================
# Daten-Layer
# =============================================================================
def read_excel(cfg: Config) -> pd.DataFrame:
    """Liest Excel und parst Zeitstempel."""
    df = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_name)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col).reset_index(drop=True)
    return df


def aggregate_to_quarter(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aggregiert Monatsdaten auf Quartal – leakage-arm.
    """
    df = df.copy()
    base_cols = [c for c in df.columns if c != cfg.date_col]
    exogs = [c for c in base_cols if c != cfg.target_col and pd.api.types.is_numeric_dtype(df[c])]

    df["Q"] = df[cfg.date_col].dt.to_period("Q")
    df = df.sort_values(cfg.date_col)

    if cfg.agg_method_target == "mean":
        yq = df.groupby("Q")[cfg.target_col].mean()
    elif cfg.agg_method_target == "last":
        yq = df.groupby("Q")[cfg.target_col].apply(lambda s: s.ffill().iloc[-1])
    else:
        raise ValueError(f"Unbekannte agg_method_target: {cfg.agg_method_target}")

    parts = []
    for method in cfg.agg_methods_exog:
        if method == "last":
            part = df.groupby("Q")[exogs].apply(lambda g: g.ffill().iloc[-1])
            if isinstance(part.index, pd.MultiIndex):
                part.index = part.index.get_level_values(0)
        elif method == "mean":
            part = df.groupby("Q")[exogs].mean(numeric_only=True)
        elif method == "median":
            part = df.groupby("Q")[exogs].median(numeric_only=True)
        else:
            continue

        new_cols = []
        for c in part.columns:
            suffix = f"__{method}"
            new_cols.append(c if c.endswith(suffix) else f"{c}{suffix}")
        part.columns = new_cols
        parts.append(part)

    Xq = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=yq.index)
    out = pd.concat([yq.rename(cfg.target_col), Xq], axis=1).reset_index()
    out["Q_end"] = out["Q"].dt.to_timestamp(how="end")
    return out


def add_deterministic_features(df_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_q.copy().sort_values("Q").reset_index(drop=True)
    n = len(df)

    if cfg.add_trend_features:
        t = np.arange(n, dtype=float)
        df["DET_trend_t"] = t
        for d in range(2, max(2, cfg.trend_degree) + 1):
            df[f"DET_trend_t{d}"] = t ** d

    if cfg.add_seasonality and cfg.seasonality_mode.lower() == "dummies":
        qnum = df["Q"].dt.quarter.astype(int)
        for q in [1, 2, 3]:
            df[f"SEAS_Q{q}"] = (qnum == q).astype(int)

    return df


def month_lags_to_quarter_lags(month_lags: List[int]) -> List[int]:
    q_lags = []
    for m in month_lags:
        q = -int(np.ceil(abs(int(m)) / 3))
        q_lags.append(q)
    return sorted(set(q_lags))


def build_quarterly_lags(df_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df_q = df_q.sort_values("Q").reset_index(drop=True)

    det_cols = [c for c in df_q.columns if c.startswith("DET_") or c.startswith("SEAS_")]
    ignore = ["Q", "Q_end", cfg.target_col] + det_cols
    exog_base_cols = [c for c in df_q.columns if c not in ignore]

    exog_q_lags = month_lags_to_quarter_lags(cfg.exog_month_lags)

    out = df_q[["Q", "Q_end", cfg.target_col]].copy()

    for c in det_cols:
        out[c] = df_q[c]

    if exog_q_lags and exog_base_cols:
        for col in exog_base_cols:
            for ql in exog_q_lags:
                out[f"{col}__lag{ql}Q"] = df_q[col].shift(abs(ql))

    if cfg.target_lags_q:
        for L in sorted(set(int(abs(x)) for x in cfg.target_lags_q if x >= 1)):
            out[f"TARGET__lag-{L}Q"] = df_q[cfg.target_col].shift(L)

    return out.dropna().reset_index(drop=True)


# =============================================================================
# Exog-Namen auflösen
# =============================================================================
def _canonical_exog_name(name: str) -> str:
    n = str(name).strip().lower()
    for suf in ["__last", "__first", "__mean"]:
        n = n.replace(suf, "")
    n = n.split("__lag-")[0]
    return n


def resolve_exogs(requested: list[str], available_cols: list[str]) -> dict[str, str]:
    avail_map = {c: _canonical_exog_name(c) for c in available_cols}
    resolved: dict[str, str] = {}
    for req in requested:
        can = _canonical_exog_name(req)
        if req in available_cols:
            resolved[req] = req
            continue
        hit = next((col for col, ccan in avail_map.items() if ccan == can), None)
        if hit:
            resolved[req] = hit
    return resolved


# =============================================================================
# Zieltransformation (Yeo-Johnson)
# =============================================================================
class TargetYJ:
    def __init__(self, standardize: bool = True):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
        self.fitted = False

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1, 1)
        self.pt.fit(y)
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.pt.transform(np.asarray(y).reshape(-1, 1)).ravel()

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.pt.inverse_transform(np.asarray(y_t).reshape(-1, 1)).ravel()


# =============================================================================
# CV / Metriken
# =============================================================================
def expanding_splits(n: int, min_train: int, gap: int, horizon: int):
    for test_end in range(min_train + gap, n - horizon + 1):
        train_end = test_end - gap
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_end, test_end + horizon))
        yield train_idx, test_idx


def calculate_cv_metrics(df_feats: pd.DataFrame, cfg: Config, model, tj, X_cols: List[str]) -> Dict[str, float]:
    from sklearn.base import clone

    y_all = df_feats[cfg.target_col].astype(float).values
    X_all = df_feats[X_cols].astype(float).values
    n = len(df_feats)

    min_train = getattr(cfg, "min_train_quarters", max(12, n // 3))
    gap = getattr(cfg, "gap_quarters", 0)
    horizon = 1

    y_true_all = []
    y_pred_all = []

    for test_end in range(min_train + gap, n - horizon + 1):
        train_end = test_end - gap
        tr_idx = list(range(0, train_end))
        te_idx = list(range(test_end, test_end + horizon))

        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        m = clone(model)

        if tj is not None and hasattr(tj, "fit") and hasattr(tj, "transform"):
            from copy import deepcopy

            tj_local = deepcopy(tj)
            tj_local.fit(y_tr.reshape(-1, 1))
            y_tr_t = tj_local.transform(y_tr.reshape(-1, 1)).ravel()

            m.fit(X_tr, y_tr_t)
            y_hat_t = m.predict(X_te)

            if hasattr(tj_local, "inverse_transform"):
                y_hat = tj_local.inverse_transform(y_hat_t.reshape(-1, 1)).ravel()
            elif hasattr(tj_local, "inverse"):
                y_hat = tj_local.inverse(y_hat_t)
            else:
                m = clone(model)
                m.fit(X_tr, y_tr)
                y_hat = m.predict(X_te)
        else:
            m.fit(X_tr, y_tr)
            y_hat = m.predict(X_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(np.asarray(y_hat, dtype=float).tolist())

    if len(y_true_all) == 0:
        return {
            "cv_rmse": float("nan"),
            "cv_mae": float("nan"),
            "cv_r2": float("nan"),
            "n_oos": 0,
            "residuals": [],
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    residuals = y_true_all - y_pred_all

    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))

    return {
        "cv_rmse": rmse,
        "cv_mae": mae,
        "cv_r2": r2,
        "n_oos": int(len(y_true_all)),
        "residuals": residuals.tolist(),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def calculate_insample_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def calculate_model_diagnostics(model, X_cols: List[str]):
    return {
        "n_features": len(X_cols),
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "n_splits": int(model.tree_.node_count),
        "top_features": {
            feat: float(imp)
            for feat, imp in zip(X_cols, model.feature_importances_)
            if imp > 0.01
        },
    }


def create_comprehensive_metadata(model, tj, X_cols, best_params, df_feats, cfg):
    def _to_1d(a):
        a = np.asarray(a)
        return a.ravel()

    def _safe_inverse_transform(_tj, y_hat):
        if _tj is None:
            return y_hat
        try:
            y_hat_2d = np.asarray(y_hat, dtype=float).reshape(-1, 1)
            if hasattr(_tj, "inverse"):
                return _to_1d(_tj.inverse(y_hat_2d))
            if hasattr(_tj, "inverse_transform"):
                return _to_1d(_tj.inverse_transform(y_hat_2d))
        except Exception:
            pass
        return _to_1d(y_hat)

    def _safe_predict_original_scale(_model, _tj, X):
        y_hat = _model.predict(X)
        return _safe_inverse_transform(_tj, y_hat)

    cv_metrics = calculate_cv_metrics(df_feats, cfg, model, tj, X_cols) or {}
    cv_scale = str(cv_metrics.get("scale") or "").lower() or "original"

    y_train = _to_1d(df_feats[cfg.target_col].values)
    X_train = np.asarray(df_feats[X_cols].values, dtype=float)
    y_train_pred = _safe_predict_original_scale(model, tj, X_train)
    insample = calculate_insample_metrics(y_train, y_train_pred) or {}
    diagnostics = calculate_model_diagnostics(model, X_cols) or {}

    with np.errstate(all="ignore"):
        y_summary = {
            "mean": float(np.nanmean(y_train)) if len(y_train) else None,
            "std": float(np.nanstd(y_train)) if len(y_train) else None,
            "min": float(np.nanmin(y_train)) if len(y_train) else None,
            "max": float(np.nanmax(y_train)) if len(y_train) else None,
            "n": int(len(y_train)),
        }

    if "Q" in df_feats.columns:
        try:
            date_range = f"{df_feats['Q'].iloc[0]} to {df_feats['Q'].iloc[-1]}"
        except Exception:
            date_range = None
    else:
        date_range = None

    metadata = {
        "model_type": type(model).__name__,
        "timestamp": pd.Timestamp.now().isoformat(),
        "cv_performance": {
            "cv_rmse": cv_metrics.get("cv_rmse"),
            "cv_mae": cv_metrics.get("cv_mae"),
            "cv_r2": cv_metrics.get("cv_r2"),
            "n_oos": cv_metrics.get("n_oos"),
            "scale": cv_scale,
            "rmse": cv_metrics.get("cv_rmse"),
            "mae": cv_metrics.get("cv_mae"),
            "r2": cv_metrics.get("cv_r2"),
        },
        "insample_performance": {
            **insample,
            "_warning": "Optimistisch – für Generalisierung cv_performance verwenden.",
        },
        "hyperparameters": best_params,
        "model_complexity": diagnostics,
        "training_data": {
            "n_train_quarters": int(len(df_feats)),
            "date_range": date_range,
            "target_variable": str(cfg.target_col),
            "feature_count": int(len(X_cols)),
        },
        "forecast_config": {
            "horizon_quarters": int(getattr(cfg, "forecast_horizon", None) or 0),
            "future_exog_strategy": getattr(cfg, "future_exog_strategy", None),
            "target_transform": getattr(cfg, "target_transform", "none"),
            "target_standardize": bool(getattr(cfg, "target_standardize", False)),
            "exog_lags_months": list(getattr(cfg, "exog_month_lags", []) or []),
            "ar_lags_quarters": list(getattr(cfg, "target_lags_q", []) or []),
            "seasonality_mode": getattr(cfg, "seasonality_mode", None),
            "add_trend_features": bool(getattr(cfg, "add_trend_features", False)),
            "trend_degree": int(getattr(cfg, "trend_degree", 1) or 1),
            "selected_exog": list(getattr(cfg, "selected_exog", []) or []),
            "cache_tag": getattr(cfg, "cache_tag", None),
            "random_state": int(getattr(cfg, "random_state", 0) or 0),
        },
        "y_train_summary": y_summary,
        "cv_metrics_scale": cv_scale,
        "X_cols": list(map(str, X_cols)),
    }

    try:
        res = cv_metrics.get("residuals", [])
        if isinstance(res, (list, tuple, np.ndarray)) and len(res) > 0:
            res_arr = np.asarray(res, dtype=float)
            res_arr = res_arr[np.isfinite(res_arr)]
            if res_arr.size > 0:
                metadata["cv_residuals"] = res_arr.tolist()
    except Exception:
        pass

    return metadata


# =============================================================================
# Extrapolationen
# =============================================================================
def extrapolate_with_arima(series: pd.Series, horizon: int, var_name: str = "") -> np.ndarray:
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    s_clean = series.dropna()
    if len(s_clean) < 20:
        LOGGER.debug(f"[ARIMA] {var_name[:50]}: Zu wenig Daten ({len(s_clean)}), Fallback")
        return _fallback_extrapolation(series, horizon)

    series_train = s_clean.iloc[-60:]
    best_aic = np.inf
    best_order = (1, 0, 1)
    best_model = None

    orders_to_try = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (2, 0, 1),
        (1, 0, 2),
        (0, 1, 1),
        (1, 1, 0),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for order in orders_to_try:
            try:
                model = ARIMA(series_train, order=order)
                fitted = model.fit(method_kwargs={"warn_convergence": False})
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
                    best_model = fitted
            except Exception:
                continue

    if best_model is None:
        LOGGER.debug(f"[ARIMA] {var_name[:50]}: Alle Orders fehlgeschlagen, Fallback")
        return _fallback_extrapolation(series, horizon)

    forecast = best_model.forecast(steps=horizon)
    hist_mean = series_train.mean()
    hist_std = series_train.std()
    vals = forecast.values

    lo = hist_mean - 3 * hist_std
    hi = hist_mean + 3 * hist_std
    clipped = np.clip(vals, lo, hi)

    if np.std(clipped) < max(1e-12, 0.1 * hist_std):
        hist_changes = series_train.diff().dropna()
        hist_change_std = float(hist_changes.std() if len(hist_changes) else 0.0)
        np.random.seed(hash(var_name) % (2**32))
        noise = np.cumsum(np.random.normal(0, 0.4 * hist_change_std, horizon))
        clipped = np.clip(clipped + noise, lo, hi)

    LOGGER.debug(f"[ARIMA] {var_name[:50]}: Order={best_order}, AIC={best_aic:.1f}")
    return clipped


def _fallback_extrapolation(series: pd.Series, horizon: int) -> np.ndarray:
    s = series.dropna()
    if len(s) < 4:
        return np.full(horizon, s.iloc[-1] if len(s) else np.nan)

    recent = s.iloc[-8:] if len(s) >= 8 else s
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent.values, deg=1)
    drift_per = coeffs[0]

    seasonal = np.zeros(4)
    if len(s) >= 16:
        for q in range(4):
            q_values = s.iloc[q::4].iloc[-4:]
            if len(q_values) > 0:
                seasonal[q] = q_values.mean() - s.mean()

    max_seas = s.std() * 0.5
    seasonal = np.clip(seasonal, -max_seas, max_seas)

    last_value = s.iloc[-1]
    base = np.zeros(horizon)
    for h in range(horizon):
        quarter_index = (len(s) + h) % 4
        base[h] = last_value + drift_per * (h + 1) + seasonal[quarter_index]

    noise_std = s.std() * 0.3
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, horizon)

    hist_min, hist_max = s.min(), s.max()
    r = hist_max - hist_min
    soft_min, soft_max = hist_min - 0.2 * r, hist_max + 0.2 * r

    return np.clip(base + noise, soft_min, soft_max)


def _extrapolate_drift_seasonal(
    series: pd.Series,
    horizon: int,
    drift_window_q: int,
    seasonal_period_q: int,
    last_q: pd.Period,
    verbose: bool,
    var_name: str,
) -> np.ndarray:
    recent = series.iloc[-drift_window_q:] if len(series) >= drift_window_q else series
    if len(recent) >= 2:
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent.values, deg=1)
        drift_per_q = coeffs[0]
    else:
        drift_per_q = 0.0

    seasonal_dict: Dict[int, float] = {}
    if len(series) >= seasonal_period_q * 4:
        for q_num in range(1, seasonal_period_q + 1):
            q_mask = [i for i, idx in enumerate(series.index) if idx.quarter == q_num]
            q_values = series.iloc[q_mask].iloc[-4:] if len(q_mask) >= 4 else series.iloc[q_mask]
            seasonal_dict[q_num] = (q_values.mean() - series.mean()) if len(q_values) > 0 else 0.0
    else:
        for q_num in range(1, seasonal_period_q + 1):
            seasonal_dict[q_num] = 0.0

    max_seas = series.std() * 0.5
    for q_num in seasonal_dict:
        seasonal_dict[q_num] = float(np.clip(seasonal_dict[q_num], -max_seas, max_seas))

    last_val = series.iloc[-1] if len(series) else 0.0
    forecast = np.zeros(horizon)
    for h in range(horizon):
        future_q_num = (last_q + h + 1).quarter
        forecast[h] = last_val + drift_per_q * (h + 1) + seasonal_dict.get(future_q_num, 0.0)

    hist_std = float(series.std() if len(series) else 0.0)
    noise_std = hist_std * 0.2
    np.random.seed(hash(var_name) % (2**32))
    forecast = forecast + np.random.normal(0, noise_std, horizon)

    hist_min, hist_max = float(series.min()), float(series.max())
    r = hist_max - hist_min
    soft_min, soft_max = hist_min - 0.2 * r, hist_max + 0.2 * r
    forecast = np.clip(forecast, soft_min, soft_max)

    if verbose:
        LOGGER.info(f"[FUT-EXOG] {var_name[:50]}: last={last_val:.3f}, drift={drift_per_q:.4f}, seas(*)…")
    return forecast


# =============================================================================
# Zukunft: exogene Variablen (Quartal)
# =============================================================================
def impute_future_exog_quarterly(
    hist_quarter_df: pd.DataFrame,
    future_quarters: pd.PeriodIndex,
    exog_var_names: List[str],
    strategy: str = "mixed",
    drift_window_q: int = 8,
    seasonal_period_q: int = 4,
    deterministic_cols: Optional[List[str]] = None,
    use_arima: bool = True,
    arima_threshold_importance: float = 0.15,
    verbose: bool = True,
    debug_exog: bool = False,
) -> pd.DataFrame:
    if deterministic_cols is None:
        deterministic_cols = []

    H = len(future_quarters)
    requested = list(exog_var_names or [])

    if verbose:
        LOGGER.debug(f"[FUT-EXOG] Angefordert ({len(requested)}): {requested}")

    if "Quarter" in hist_quarter_df.columns:
        q_idx = pd.PeriodIndex(hist_quarter_df["Quarter"], freq="Q")
    elif "Q" in hist_quarter_df.columns:
        q_idx = pd.PeriodIndex(hist_quarter_df["Q"], freq="Q")
    elif isinstance(hist_quarter_df.index, pd.PeriodIndex):
        q_idx = hist_quarter_df.index.asfreq("Q")
    else:
        q_idx = pd.PeriodIndex(hist_quarter_df.index, freq="Q")

    hist_clean = hist_quarter_df.copy()
    hist_clean.index = q_idx
    hist_clean = hist_clean.sort_index()

    last_q = q_idx.max()
    if verbose:
        LOGGER.debug(f"[FUT-EXOG] Input-DF (hist): {hist_quarter_df.shape} | horizon: {H}")
        LOGGER.debug(f"[FUT-EXOG] Letztes Q: {last_q} | Forecast-Q: {list(future_quarters)}")
    if debug_exog:
        LOGGER.debug("[FUT-EXOG|Diag] Verfügbare Hist-Spalten (Top 25): %s", list(hist_clean.columns)[:25])

    available_in_hist: List[str] = []
    not_found: List[str] = []
    resolved_map: Dict[str, str] = {}

    for req in requested:
        candidates = [req]
        if not req.endswith("__last__"):
            candidates.append(f"{req}__last__")
        if req.endswith("__last__"):
            candidates.append(req[: -len("__last__")])
        if "__last" in req and not req.endswith("__"):
            candidates.append(req.replace("__last", "__last__"))

        picked = None
        for cand in candidates:
            if cand in hist_clean.columns and cand not in deterministic_cols:
                picked = cand
                break

        if picked is None:
            not_found.append(req)
        else:
            available_in_hist.append(picked)
            resolved_map[req] = picked

    if not_found and verbose:
        LOGGER.warning(f"[FUT-EXOG] Nicht exakt gefunden (übersprungen): {not_found}")
    if verbose:
        LOGGER.debug(f"[FUT-EXOG] Aufgelöst ({len(available_in_hist)}): {available_in_hist}")
    if debug_exog:
        LOGGER.debug("[FUT-EXOG|Resolve] Mapping: %s", resolved_map)

    fut_df = pd.DataFrame(index=future_quarters)

    important_keywords = {"ILM", "IRS", "ICP", "HICP", "GDP", "BIP"}

    for var in available_in_hist:
        series = pd.to_numeric(hist_clean[var], errors="coerce").dropna()
        if series.empty:
            if verbose:
                LOGGER.warning(f"[FUT-EXOG] {var}: Keine historischen Werte – Variable wird ausgelassen.")
            continue

        n_hist = len(series)
        last_val = float(series.iloc[-1])

        if n_hist < 4:
            if verbose:
                LOGGER.debug(f"[FUT-EXOG] {var}: Zu wenig Daten (n={n_hist}), nutze LOCF")
            fut_df[var] = np.full(H, last_val, dtype=float)
            continue

        series_recent = series.iloc[-max(12, seasonal_period_q):]
        try:
            trend_slope = (
                float(np.polyfit(np.arange(len(series_recent)), series_recent.values, 1)[0])
                if len(series_recent) >= 2
                else 0.0
            )
        except Exception:
            trend_slope = 0.0
        std_recent = float(series_recent.std(ddof=1)) if len(series_recent) > 1 else 0.0
        mean_recent = float(series_recent.mean()) if len(series_recent) > 0 else 0.0
        variability = (std_recent / abs(mean_recent)) if mean_recent != 0 else 0.0

        use_arima_for_this = False
        method = "drift+seasonal"
        if use_arima and strategy in ("arima", "mixed"):
            is_important = any(kw in var.upper() for kw in important_keywords)
            has_trend = abs(trend_slope) > (std_recent / max(len(series_recent), 12) if std_recent else 0.0)
            is_volatile = variability > 0.2
            use_arima_for_this = bool(is_important and (has_trend or is_volatile))

        if use_arima_for_this:
            try:
                vals = extrapolate_with_arima(series, H, var_name=var)
                method = "arima"
            except Exception as e:
                if verbose:
                    LOGGER.warning(f"[FUT-EXOG] {var}: ARIMA fehlgeschlagen ({e}) – fallback zu Drift+Seasonal")
                vals = _extrapolate_drift_seasonal(
                    series, H, drift_window_q, seasonal_period_q, last_q, verbose, var
                )
                method = "drift+seasonal"
        else:
            vals = _extrapolate_drift_seasonal(
                series,
                H,
                drift_window_q,
                seasonal_period_q,
                last_q,
                verbose,
                var,
            )

        vals = np.asarray(vals, dtype=float)
        fut_df[var] = vals

        if debug_exog:
            head = vals[: min(3, H)]
            LOGGER.debug("[FUT-EXOG|%s] Methode=%s, Fut-Head=%s", var, method, np.asarray(head).round(6).tolist())

    if fut_df.empty and verbose:
        LOGGER.warning("[FUT-EXOG] Keine exogenen Zukunftswerte generiert (alle angeforderten Variablen fehlen).")

    return fut_df


# =============================================================================
# Zukunfts-Designmatrix (Schritt 9: zusammengelegt)
# =============================================================================
def build_future_design(
    df_q: pd.DataFrame,
    cfg: Config,
    df_feats: Optional[pd.DataFrame] = None,
    X_cols: Optional[List[str]] = None,
    debug_design: bool = False,
) -> pd.DataFrame:
    """
    Baut das Zukunfts-Design in einem Schritt:
    1. Historie in Quartale legen
    2. Future-Exogs imputen
    3. Deterministische Features für Zukunft bauen
    4. Lags der Exogs bauen
    5. Target-Lags bauen
    6. (neu in Schritt 9) optional auf vom Modell erwartete X_cols auffüllen
    """
    df_hist = df_q.copy()

    # 1) Quartalsachse aufbauen
    if "Q" in df_hist.columns:
        q_hist = pd.PeriodIndex(df_hist["Q"], freq="Q")
    elif "Quarter" in df_hist.columns:
        q_hist = pd.PeriodIndex(df_hist["Quarter"], freq="Q")
    elif isinstance(df_hist.index, pd.PeriodIndex):
        q_hist = df_hist.index.asfreq("Q")
    else:
        raise TypeError("build_future_design: Keine gültige Quartalsachse ('Q'/'Quarter'/PeriodIndex).")

    df_hist = df_hist.set_index(q_hist).sort_index()
    last_q = q_hist.max()
    H = int(getattr(cfg, "forecast_horizon", 4))
    fut_Q = pd.period_range(last_q + 1, periods=H, freq=last_q.freq or "Q")

    LOGGER.debug(_sym("\n[Design] Impute future exog …"))
    LOGGER.debug(f"[Design] Strategy: {getattr(cfg, 'future_exog_strategy', 'mixed')}")

    # 2) Exog-Wunschliste und deterministische Spalten definieren
    det_cols_hist = [c for c in df_hist.columns if c.startswith(("DET_", "SEAS_"))]
    exog_wishlist_raw = getattr(cfg, "selected_exog", None) or getattr(cfg, "exog_cols", None)

    exog_resolved = None
    if exog_wishlist_raw:
        try:
            res_map = resolve_exogs(exog_wishlist_raw, df_hist.columns.tolist())
            mapped = []
            unresolved = []
            for req in exog_wishlist_raw:
                if req in res_map:
                    mapped.append(res_map[req])
                else:
                    unresolved.append(req)
            exog_resolved = list(dict.fromkeys(mapped))
            if unresolved:
                LOGGER.warning(
                    f"[Design] Exog nicht in Historie gefunden (werden übersprungen/fallen auf Fallbacks): {unresolved}"
                )
        except Exception as e:
            LOGGER.warning(f"[Design] resolve_exogs fehlgeschlagen ({e}) – nutze Wunschliste roh.")
            exog_resolved = None

    exog_wishlist = exog_resolved if (exog_resolved is not None and len(exog_resolved) > 0) else exog_wishlist_raw

    # 3) Historie ggf. für Imputation anpassen
    hist_for_impute = df_hist.copy()
    if "Q" in hist_for_impute.columns:
        hist_for_impute = hist_for_impute.rename(columns={"Q": "Quarter"})

    debug_exog = bool(getattr(cfg, "debug_exog", False))

    # 4) Zukunftswerte exogener Variablen schätzen
    fut_exog_base = impute_future_exog_quarterly(
        hist_quarter_df=hist_for_impute,
        future_quarters=fut_Q,
        exog_var_names=exog_wishlist,
        strategy=getattr(cfg, "future_exog_strategy", "mixed"),
        drift_window_q=getattr(cfg, "future_exog_drift_window_q", 8),
        seasonal_period_q=getattr(cfg, "future_exog_seasonal_period_q", 4),
        deterministic_cols=det_cols_hist + [getattr(cfg, "target_col", "PH_EINLAGEN")],
        use_arima=getattr(cfg, "use_arima_extrapolation", True),
        verbose=True,
        debug_exog=debug_exog,
    )

    # 5) Spaltennamen ggf. auf __last__ normalisieren
    rename_to_last = {}
    target_list_for_normalize = exog_wishlist or []
    if not fut_exog_base.empty and target_list_for_normalize:
        for nm in target_list_for_normalize:
            last_nm = f"{nm}__last__"
            if last_nm in fut_exog_base.columns:
                continue
            if nm in fut_exog_base.columns and last_nm in df_hist.columns:
                rename_to_last[nm] = last_nm
    if rename_to_last:
        fut_exog_base = fut_exog_base.rename(columns=rename_to_last)
        LOGGER.debug(f"[Design] Renamed future exog to __last__: {rename_to_last}")

    # 6) Deterministische Zukunftsfeatures
    fut_det = pd.DataFrame(index=fut_Q)

    if getattr(cfg, "add_trend_features", True):
        if "DET_trend_t" in df_hist.columns and np.issubdtype(df_hist["DET_trend_t"].dtype, np.number):
            t0 = float(df_hist["DET_trend_t"].iloc[-1])
            fut_det["DET_trend_t"] = np.arange(t0 + 1, t0 + 1 + H, dtype=float)
        else:
            fut_det["DET_trend_t"] = np.arange(len(df_hist) + 1, len(df_hist) + 1 + H, dtype=float)
        if int(getattr(cfg, "trend_degree", 2)) >= 2:
            fut_det["DET_trend_t2"] = fut_det["DET_trend_t"] ** 2

    if getattr(cfg, "add_seasonality", True):
        qnum = pd.Series([q.quarter for q in fut_Q], index=fut_Q)
        for q_val in [1, 2, 3]:
            fut_det[f"SEAS_Q{q_val}"] = (qnum == q_val).astype(int)
        LOGGER.debug(f"[Design] Saisondummies Q1-3 für {len(fut_Q)} Quartale gesetzt")

    # 7) Lags für Exogs der Zukunft
    lags_q = sorted(set(getattr(cfg, "target_lags_q", [1, 2, 4])))

    exog_lag_blocks = []
    if not fut_exog_base.empty:
        for base_col in fut_exog_base.columns:
            if base_col in df_hist.columns:
                hist_series = pd.to_numeric(df_hist[base_col], errors="coerce")
                hist_src = base_col
            else:
                if base_col.endswith("__last__"):
                    alt = base_col[: -len("__last__")]
                else:
                    alt = f"{base_col}__last__"
                hist_series = pd.to_numeric(df_hist.get(alt), errors="coerce")
                hist_src = alt

            fut_series = pd.to_numeric(fut_exog_base[base_col], errors="coerce")
            series_full = pd.concat([hist_series, fut_series])
            series_full.index = pd.PeriodIndex(series_full.index, freq=last_q.freq or "Q")

            lag_block = pd.DataFrame(index=fut_Q)
            for k in lags_q:
                col_lag = f"{base_col}__lag-{k}Q"
                vals = []
                for q in fut_Q:
                    prev = q - k
                    vals.append(series_full.get(prev, np.nan))
                lag_block[col_lag] = vals

            exog_lag_blocks.append(lag_block)

    fut_exog_lags = pd.concat(exog_lag_blocks, axis=1) if exog_lag_blocks else pd.DataFrame(index=fut_Q)

    # 8) Target-Lags für Zukunft
    tgt_name = None
    for cand in [getattr(cfg, "target_col", None), "TARGET", "PH_EINLAGEN", "target", "y", "y_q"]:
        if cand and cand in df_hist.columns:
            tgt_name = cand
            break

    tgt_lag_block = pd.DataFrame(index=fut_Q)
    if tgt_name is not None:
        tgt_hist = pd.to_numeric(df_hist[tgt_name], errors="coerce").dropna()
        tgt_hist.index = pd.PeriodIndex(tgt_hist.index, freq=last_q.freq or "Q")
        last_obs = float(tgt_hist.iloc[-1]) if len(tgt_hist) else np.nan
        for k in lags_q:
            col = f"TARGET__lag-{k}Q"
            vals = []
            for q in fut_Q:
                prev = q - k
                val = tgt_hist.get(prev, np.nan)
                if pd.isna(val):
                    val = last_obs
                vals.append(float(val))
            tgt_lag_block[col] = vals
    else:
        LOGGER.warning("[Design] Konnte Target-Serie für TARGET__lag-* nicht finden.")

    # 9) Alles zusammenführen
    fut_designs = pd.concat([fut_det, fut_exog_lags, tgt_lag_block], axis=1)
    fut_designs.index.name = "Quarter"

    num_cols = fut_designs.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        fut_designs[num_cols] = (
            fut_designs[num_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        )

    LOGGER.debug(f"[Design] Zukunfts-Design (roh): {fut_designs.shape[0]} Quartale, {fut_designs.shape[1]} Features")

    # 10) Schritt 9: direkt hier sicherstellen, dass alle Modell-Features vorhanden sind
    if X_cols is not None and df_feats is not None:
        fut_designs = _ensure_future_has_all_X(
            fut=fut_designs,
            X_cols=X_cols,
            cfg=cfg,
            hist_feats=df_feats,
        )
        LOGGER.debug(f"[Design] Zukunfts-Design (final): {fut_designs.shape[0]} Quartale, {fut_designs.shape[1]} Features")

    return fut_designs


# =============================================================================
# Training
# =============================================================================
def train_best_model_h1(df_feats: pd.DataFrame, cfg: Config):
    y = df_feats[cfg.target_col].values
    X_all = [c for c in df_feats.columns if c not in ["Q", "Q_end", cfg.target_col]]

    def is_valid_for_h1(name: str) -> bool:
        if name.startswith(("DET_", "SEAS_")):
            return True
        if "__lag" in name and name.endswith("Q"):
            if name.startswith("TARGET__lag-"):
                try:
                    L = int(name.split("TARGET__lag-")[1].replace("Q", ""))
                    return L >= 1
                except Exception:
                    return False
            try:
                _ = int(name.split("__lag")[1].replace("Q", ""))
                return True
            except Exception:
                return False
        return False

    X_cols = [c for c in X_all if is_valid_for_h1(c)]
    X = df_feats[X_cols].values
    n = len(df_feats)

    best_params, best_score = None, np.inf
    combos = list(ParameterGrid(cfg.param_grid))
    LOGGER.info(f"  Starte Grid-Search mit {len(combos)} Kombinationen…")

    for params in combos:
        preds = np.full(n, np.nan, dtype=float)
        for tr, te in expanding_splits(n, cfg.min_train_quarters, cfg.gap_quarters, 1):
            model = DecisionTreeRegressor(random_state=cfg.random_state, **params)
            if cfg.target_transform.lower() == "yeo-johnson":
                tj = TargetYJ(standardize=cfg.target_standardize).fit(y[tr])
                y_tr_t = tj.transform(y[tr])
                model.fit(X[tr], y_tr_t)
                yhat_t = model.predict(X[te])
                preds[te] = tj.inverse(yhat_t)
            else:
                model.fit(X[tr], y[tr])
                preds[te] = model.predict(X[te])

        mask = ~np.isnan(preds)
        rmse = float(np.sqrt(mean_squared_error(y[mask], preds[mask])))
        if rmse < best_score:
            best_score = rmse
            best_params = params

    model = DecisionTreeRegressor(random_state=cfg.random_state, **best_params)
    tj = None
    if cfg.target_transform.lower() == "yeo-johnson":
        tj = TargetYJ(standardize=cfg.target_standardize).fit(y)
        y_t = tj.transform(y)
        model.fit(X, y_t)
    else:
        model.fit(X, y)

    return model, tj, X_cols, best_params, best_score


# =============================================================================
# Rekursive Vorhersage
# =============================================================================
def recursive_forecast(model, tj, fut_designs: pd.DataFrame, X_cols: List[str], cfg: Config) -> np.ndarray:
    H = int(len(fut_designs))
    if H == 0:
        return np.array([], dtype=float)

    df = fut_designs.copy()
    missing_cols = [c for c in X_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[Recur] Fehlende Features im Future-Design: {missing_cols}")

    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors="coerce")

    target_lag_info: Dict[str, int] = {}
    for col in X_cols:
        if isinstance(col, str) and col.startswith("TARGET__lag-") and col.endswith("Q"):
            try:
                target_lag_info[col] = int(col.replace("TARGET__lag-", "").replace("Q", ""))
            except Exception:
                pass

    LOGGER.debug(f"[Recur] Starte rekursive Prognose über {H} Quartale")
    LOGGER.debug(f"[Recur] Target-Lags: {target_lag_info}")

    stable_cols: List[str] = list(getattr(cfg, "stable_exog_cols", []) or [])
    stable_cols = [c for c in stable_cols if c in X_cols]
    last_train_row = getattr(cfg, "last_train_row", None)
    train_medians: Dict[str, float] = dict(getattr(cfg, "train_feature_medians", {}) or {})
    last_target_value = getattr(cfg, "last_target_value", None)

    if H > 0 and last_train_row is not None and len(stable_cols) > 0:
        first_idx = df.index[0]
        for c in stable_cols:
            if pd.isna(df.at[first_idx, c]) and c in last_train_row and np.isfinite(last_train_row[c]):
                df.at[first_idx, c] = float(last_train_row[c])

    if len(stable_cols) > 0:
        df.loc[:, stable_cols] = df[stable_cols].ffill()

    for seas in ("SEAS_Q1", "SEAS_Q2", "SEAS_Q3"):
        if seas in df.columns:
            df[seas] = df[seas].fillna(0).astype(int)

    non_stable = [c for c in X_cols if c not in stable_cols]
    for c in non_stable:
        if c in df.columns and df[c].isna().any():
            if c in train_medians and np.isfinite(train_medians[c]):
                df[c] = df[c].fillna(float(train_medians[c]))
            else:
                n_nan = int(df[c].isna().sum())
                if n_nan > 0:
                    LOGGER.warning(
                        f"[Recur] Feature '{c}' hat {n_nan} NaN in Zukunft – Fallback 0 (Median/Drift bereitstellen)."
                    )
                    df[c] = df[c].fillna(0.0)

    df = df[X_cols].astype(float)

    if not np.isfinite(df.to_numpy()).all():
        nan_before = int(np.isnan(df.to_numpy()).sum())
        inf_before = int(np.isinf(df.to_numpy()).sum())
        if nan_before or inf_before:
            LOGGER.warning(
                f"[Recur] Nichtendliche Werte vor Start (NaN={nan_before}, Inf={inf_before}) – sichere Füllung."
            )
            df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            col_meds = df.median(numeric_only=True).to_dict()
            for c in df.columns:
                if df[c].isna().any():
                    fallback = train_medians.get(c, col_meds.get(c, 0.0))
                    df[c] = df[c].fillna(float(fallback))

    y_pred: List[float] = []

    for h in range(H):
        row = df.iloc[[h]].copy()

        for lag_col in target_lag_info:
            if lag_col in row.columns and row[lag_col].isna().any():
                if last_target_value is not None and np.isfinite(last_target_value):
                    LOGGER.warning(
                        f"[Recur] {lag_col} NaN in Schritt {h+1} – setze Fallback last_target_value={last_target_value}."
                    )
                    row[lag_col] = row[lag_col].fillna(float(last_target_value))
                else:
                    raise ValueError(f"[Recur] Target-Lag {lag_col} ist NaN in Quartal {h+1}")

        if not np.isfinite(row.to_numpy()).all():
            for c in row.columns:
                if not np.isfinite(row[c].values).all():
                    v = row[c].values.astype(float)
                    if np.isnan(v).any():
                        fill_val = train_medians.get(c, 0.0)
                        row[c] = np.nan_to_num(v, nan=float(fill_val), posinf=float(fill_val), neginf=float(fill_val))
                    else:
                        row[c] = np.nan_to_num(v, nan=0.0)

        x = row.values.astype(float)
        try:
            y_hat_t = float(model.predict(x)[0])
        except Exception as e:
            raise RuntimeError(f"[Recur] Modellvorhersage fehlgeschlagen in Schritt {h+1}: {e}")

        if tj is not None:
            try:
                if hasattr(tj, "inverse_transform"):
                    y_hat = float(tj.inverse_transform([[y_hat_t]])[0][0])
                elif hasattr(tj, "inverse"):
                    y_hat = float(tj.inverse(y_hat_t))
                else:
                    y_hat = y_hat_t
            except Exception:
                y_hat = y_hat_t
        else:
            y_hat = y_hat_t

        if not np.isfinite(y_hat):
            LOGGER.warning(f"[Recur] Nichtendliche Prognose in Schritt {h+1} – setze 0.0 als Fallback.")
            y_hat = 0.0

        y_pred.append(float(y_hat))

        for lag_col, k in target_lag_info.items():
            tpos = h + k
            if tpos < H:
                try:
                    df.iloc[tpos, df.columns.get_loc(lag_col)] = y_hat
                except Exception:
                    pass

        if stable_cols and h < H - 1:
            blk = df.loc[df.index[h + 1] :, stable_cols]
            if blk.isna().any().any():
                df.loc[df.index[h + 1] :, stable_cols] = blk.ffill()

    y_arr = np.array(y_pred, dtype=float)

    try:
        deg_fix_enabled = bool(getattr(cfg, "degenerate_fix", True))
        var_y = float(np.nanstd(y_arr)) if len(y_arr) else 0.0
        if deg_fix_enabled and (var_y < 1e-8 or np.allclose(y_arr, y_arr[0], atol=1e-8)):
            LOGGER.warning("[Recur] Prognose ist (nahezu) konstant – wende sanftes Post-Processing an.")
            sigma = getattr(cfg, "cv_residual_std", None)
            if sigma is None:
                sigma = getattr(cfg, "fallback_sigma", None)
            if sigma is None:
                sigma = max(1.0, abs(y_arr[0]) * 0.05)

            frac = float(getattr(cfg, "degenerate_max_frac_sigma", 0.2))
            amp = float(frac * sigma)

            rng = np.random.default_rng(42)
            ramp = np.linspace(0.0, amp, H)
            noise = rng.normal(0.0, amp * 0.15, H) if H > 1 else np.array([0.0])
            y_arr = y_arr + ramp + noise
    except Exception as e:
        LOGGER.warning(f"[Recur] Degeneracy-Post-Processing übersprungen: {e}")

    LOGGER.debug(f"[Recur] Done. Prognosen: {y_arr}")
    return y_arr


# =============================================================================
# run_production_pipeline: jetzt in 3 Abschnitte aufgeteilt
# =============================================================================
# Gemeinsame kleine Helper für S2+S3
def _fmt(x, ndigits=2, default="n/a"):
    try:
        x = float(x)
        if np.isnan(x):
            return default
        return f"{x:.{ndigits}f}"
    except Exception:
        return default


def _cv_vals(cv: dict) -> tuple[object, object, object, object]:
    cv = cv or {}
    rmse_v = cv.get("cv_rmse", cv.get("rmse", float("nan")))
    mae_v = cv.get("cv_mae", cv.get("mae", float("nan")))
    r2_v = cv.get("cv_r2", cv.get("r2", float("nan")))
    n_v = cv.get("n_samples", cv.get("n", None))
    return rmse_v, mae_v, r2_v, n_v


def _parse_target_lag_q(colname: str) -> Optional[int]:
    if not (isinstance(colname, str) and colname.startswith("TARGET__lag-") and colname.endswith("Q")):
        return None
    try:
        return int(colname.replace("TARGET__lag-", "").replace("Q", ""))
    except Exception:
        return None


def _is_season_dummy(col: str) -> bool:
    return col in ("SEAS_Q1", "SEAS_Q2", "SEAS_Q3")


def _extend_deterministic_column(name: str, hist_df: pd.DataFrame, H: int) -> Optional[np.ndarray]:
    s = pd.to_numeric(hist_df.get(name, pd.Series([], dtype=float)), errors="coerce").dropna()
    if s.empty:
        return None
    last = float(s.iloc[-1])

    if name == "DET_trend_t":
        start = int(round(last)) + 1
        return np.arange(start, start + H, dtype=float)

    if name == "DET_trend_t2":
        if "DET_trend_t" in hist_df.columns:
            t_hist = pd.to_numeric(hist_df["DET_trend_t"], errors="coerce").dropna()
            if not t_hist.empty:
                t_last = int(round(t_hist.iloc[-1]))
                t_future = np.arange(t_last + 1, t_last + 1 + H, dtype=int)
                return (t_future ** 2).astype(float)
        diff2 = float(s.diff().dropna().iloc[-1]) if len(s) >= 2 else 0.0
        return np.array([last + (i + 1) * diff2 for i in range(H)], dtype=float)

    diff = float(s.diff().dropna().iloc[-1]) if len(s) >= 2 else 0.0
    if diff == 0.0:
        return np.full(H, last, dtype=float)
    return np.array([last + (i + 1) * diff for i in range(H)], dtype=float)


def _ensure_future_has_all_X(
    fut: pd.DataFrame,
    X_cols: List[str],
    cfg: Config,
    hist_feats: pd.DataFrame,
) -> pd.DataFrame:
    out = fut.copy()
    H = len(out)

    y_hist = pd.to_numeric(hist_feats.get(cfg.target_col, pd.Series([], dtype=float)), errors="coerce").dropna()

    if "Q" in out.columns:
        fut_Q = pd.PeriodIndex(out["Q"], freq="Q")
    elif "Quarter" in out.columns:
        fut_Q = pd.PeriodIndex(out["Quarter"], freq="Q")
    else:
        if "Q" in hist_feats.columns:
            last_q = pd.PeriodIndex(hist_feats["Q"], freq="Q").max()
        else:
            last_q = pd.period_range(start=pd.Period("2000Q1", freq="Q"), periods=len(hist_feats), freq="Q").max()
        fut_Q = pd.period_range(last_q + 1, periods=H, freq="Q")
        out["Q"] = fut_Q

    for seas in ("SEAS_Q1", "SEAS_Q2", "SEAS_Q3"):
        if seas not in out.columns:
            out[seas] = 0
    seas_map = {"SEAS_Q1": 1, "SEAS_Q2": 2, "SEAS_Q3": 3}
    for seas, qnum in seas_map.items():
        out[seas] = (fut_Q.quarter == qnum).astype(int)

    for col in X_cols:
        if col in out.columns:
            if out[col].isna().any():
                if _is_season_dummy(col):
                    out[col] = out[col].fillna(0)
                else:
                    if col in hist_feats.columns:
                        last_vals = pd.to_numeric(hist_feats[col], errors="coerce").dropna()
                        if not last_vals.empty:
                            out[col] = out[col].fillna(float(last_vals.iloc[-1]))
                        else:
                            out[col] = out[col].fillna(0.0)
                    else:
                        out[col] = out[col].fillna(0.0)
            continue

        lag_k = _parse_target_lag_q(col)
        if lag_k is not None:
            if y_hist.empty:
                out[col] = 0.0
            else:
                idx = -lag_k
                if abs(idx) <= len(y_hist):
                    init_val = float(y_hist.iloc[idx])
                else:
                    init_val = float(y_hist.iloc[-1])
                out[col] = init_val
            continue

        if _is_season_dummy(col):
            qnum = {"SEAS_Q1": 1, "SEAS_Q2": 2, "SEAS_Q3": 3}[col]
            out[col] = (fut_Q.quarter == qnum).astype(int)
            continue

        if col.startswith("DET_"):
            vals = _extend_deterministic_column(col, hist_feats, H)
            if vals is None:
                out[col] = 0.0
            else:
                out[col] = vals
            continue

        if col in hist_feats.columns:
            s_hist = pd.to_numeric(hist_feats[col], errors="coerce").dropna()
            if not s_hist.empty:
                out[col] = float(s_hist.iloc[-1])
                continue

        out[col] = 0.0

    out = out.reindex(columns=list(dict.fromkeys(list(X_cols) + list(out.columns))), fill_value=0.0)
    out = out[X_cols].copy()

    if out.isna().values.any():
        out = out.ffill().bfill().fillna(0.0)

    return out


# =============================================================================
# Abschnitt S1 – Daten vorbereiten
# =============================================================================
def _section_1_prepare_data(cfg: Config) -> dict:
    LOGGER.info("=" * 80)
    LOGGER.info("[S1] DATEN LADEN UND AUFBEREITEN")
    LOGGER.info("=" * 80)

    df_m = read_excel(cfg)
    df_q = aggregate_to_quarter(df_m, cfg)
    df_q = add_deterministic_features(df_q, cfg)
    df_feats = build_quarterly_lags(df_q, cfg)

    LOGGER.debug(_sym(f"[S1] → {len(df_feats)} Quartale, {df_feats.shape[1]} Features"))

    # Dumps über zentralen Writer
    if getattr(cfg, "dump_quarterly_dataset_csv", False):
        q_path = cfg.dump_quarterly_dataset_path or str(Path(cfg.output_dir) / "train_quarterly_debug.csv")
        safe_write_csv(df_q, q_path, "[S1] Quartals-Datensatz")
    if getattr(cfg, "dump_train_design_csv", False):
        d_path = cfg.dump_train_design_path or str(Path(cfg.output_dir) / "train_design_debug.csv")
        safe_write_csv(df_feats, d_path, "[S1] Trainings-Designmatrix")

    # optional: Exogs aus Downloader ernten
    try:
        if getattr(cfg, "use_downloader_exog", False):
            resolved_exogs: List[str] = []
            out_path = getattr(cfg, "downloader_output_path", None)
            if not out_path:
                candidates = [
                    getattr(cfg, "output_dir", None),
                    str(FORECASTER_DIR),  # Ordner forecaster
                    str(APP_ROOT),        # Projektroot
                    os.getcwd(),          # letzter Fallback
                ]
                out_path = autodetect_downloader_output(candidates)
            if out_path and os.path.exists(out_path):
                resolved_exogs = harvest_exogs_from_downloader_output(out_path)
                if getattr(cfg, "debug_exog", False):
                    LOGGER.debug(f"[S1|Exog] harvested from: {out_path}")
                    LOGGER.debug(f"[S1|Exog] columns={resolved_exogs}")
            else:
                if getattr(cfg, "debug_exog", False):
                    LOGGER.debug("[S1|Exog] Keine Downloader-Output-Datei gefunden.")
            if resolved_exogs:
                cfg.selected_exog = resolved_exogs
                if getattr(cfg, "debug_exog", False):
                    LOGGER.debug(f"[S1|Exog] cfg.selected_exog gesetzt (n={len(cfg.selected_exog)})")
    except Exception as e:
        LOGGER.warning(f"[S1|Exog] Konnte Exogs nicht aus Downloader-Output übernehmen: {e}")

    return {
        "cfg": cfg,
        "df_m": df_m,
        "df_q": df_q,
        "df_feats": df_feats,
    }


# =============================================================================
# Abschnitt S2 – Modell laden oder trainieren
# =============================================================================
def _section_2_train_or_load(ctx: dict, *, force_retrain: bool) -> dict:
    cfg: Config = ctx["cfg"]
    df_feats: pd.DataFrame = ctx["df_feats"]

    LOGGER.info("=" * 80)
    LOGGER.info("[S2] MODELL LADEN ODER TRAINIEREN")
    LOGGER.info("=" * 80)

    model_path = get_model_filepath(cfg)

    artifact: Optional[ModelArtifact] = None
    skip_training = False

    if cfg.use_cached_model and not force_retrain and ModelArtifact.exists(model_path):
        try:
            artifact = ModelArtifact.load(model_path)
            compatible, issues = artifact.is_compatible(cfg)
            if compatible:
                LOGGER.info(
                    _sym(f"[S2] ✓ Verwende gecachtes Modell (trainiert: {artifact.metadata.get('timestamp', 'n/a')})")
                )
                rmse_v, mae_v, r2_v, _ = _cv_vals(artifact.metadata.get("cv_performance", {}))
                LOGGER.info(f"[S2] CV-RMSE: {_fmt(rmse_v, 2)}")
                model = artifact.model
                tj = artifact.tj
                X_cols = artifact.X_cols
                best_params = artifact.best_params
                metadata = artifact.metadata
                skip_training = True
            else:
                LOGGER.warning("[S2] Gecachtes Modell inkompatibel mit aktueller Config:")
                for issue in issues:
                    LOGGER.warning("  - " + str(issue))
                LOGGER.info(_sym("[S2] → Neues Training wird durchgeführt"))
                artifact = None
        except Exception as e:
            LOGGER.warning(f"[S2] Fehler beim Laden: {e}")
            LOGGER.info(_sym("[S2] → Neues Training wird durchgeführt"))
    else:
        if force_retrain:
            LOGGER.info(_sym("[S2] → force_retrain=True: Neues Training erzwungen"))

    if not skip_training:
        LOGGER.info("[S2] Trainiere Modell (Grid-Search)…")
        model, tj, X_cols, best_params, best_rmse = train_best_model_h1(df_feats, cfg)
        LOGGER.info(f"[S2] → Beste Parameter: {best_params}")
        LOGGER.info(f"[S2] → CV-RMSE: {_fmt(best_rmse, 2)}")
        LOGGER.info(f"[S2] → {len(X_cols)} Features")

        LOGGER.debug("[S2] Berechne Metriken…")
        metadata = create_comprehensive_metadata(model, tj, X_cols, best_params, df_feats, cfg)

        LOGGER.debug("[S2] Speichere Modell…")
        artifact = ModelArtifact(
            model=model,
            tj=tj,
            X_cols=X_cols,
            best_params=best_params,
            metadata=metadata,
            config_dict=asdict(cfg),
        )
        artifact.save(model_path)
    else:
        model = artifact.model
        tj = artifact.tj
        X_cols = artifact.X_cols
        best_params = artifact.best_params
        metadata = artifact.metadata
        LOGGER.info("[S2] Training übersprungen (Cache)")

    # σ für spätere CIs sichern
    try:
        if isinstance(metadata, dict):
            cv_resid = np.asarray(metadata.get("cv_residuals", []), dtype=float)
            if cv_resid.size > 0 and np.isfinite(cv_resid).any():
                sigma = float(np.nanstd(cv_resid, ddof=1))
                if np.isfinite(sigma) and sigma > 0:
                    cfg.cv_residual_std = sigma
            else:
                rmse_v, _, _, _ = _cv_vals(metadata.get("cv_performance", {}))
                if np.isfinite(rmse_v):
                    cfg.cv_residual_std = float(rmse_v)
    except Exception as _e_sigma:
        LOGGER.warning(f"[S2] Konnte cv_residual_std nicht ableiten: {_e_sigma}")

    ctx.update(
        {
            "model_path": model_path,
            "model": model,
            "tj": tj,
            "X_cols": X_cols,
            "metadata": metadata,
            "best_params": best_params,
            "skip_training": skip_training,
        }
    )
    return ctx


# =============================================================================
# Abschnitt S3 – Zukunfts-Design + Forecast + Export
# =============================================================================
def _section_3_forecast_and_export(ctx: dict) -> tuple[pd.DataFrame, dict]:
    cfg: Config = ctx["cfg"]
    df_q: pd.DataFrame = ctx["df_q"]
    df_feats: pd.DataFrame = ctx["df_feats"]
    model = ctx["model"]
    tj = ctx["tj"]
    X_cols: List[str] = ctx["X_cols"]
    metadata: dict = ctx["metadata"]
    model_path: str = ctx["model_path"]
    skip_training: bool = ctx["skip_training"]

    LOGGER.info("=" * 80)
    LOGGER.info("[S3] ZUKUNFTS-DESIGN ERSTELLEN & FORECAST RECHNEN")
    LOGGER.info("=" * 80)

    if getattr(cfg, "debug_exog", False):
        LOGGER.debug(f"[S3] Effektive Exog-Wunschliste: {list(getattr(cfg, 'selected_exog', []) or [])}")

    # Schritt 9: nur noch ein Aufruf für den Zukunftsaufbau
    fut_designs = build_future_design(
        df_q=df_q,
        cfg=cfg,
        df_feats=df_feats,
        X_cols=X_cols,
        debug_design=getattr(cfg, "debug_design", False),
    )
    LOGGER.debug(_sym(f"[S3] → {len(fut_designs)} Zukunftsquartale (final)"))

    if getattr(cfg, "dump_future_design_csv", True):
        dbg_path = cfg.dump_future_design_path or str(Path(cfg.output_dir) / "future_design_debug.csv")
        safe_write_csv(fut_designs.reset_index(drop=True), dbg_path, "[S3] Future-Design dump")

    LOGGER.info("[S3] Rekursive Prognose …")
    forecasts = recursive_forecast(model, tj, fut_designs, X_cols, cfg)

    fut_Q = pd.period_range(df_q["Q"].iloc[-1] + 1, periods=cfg.forecast_horizon, freq="Q")
    df_results = pd.DataFrame({"Quarter": [str(q) for q in fut_Q], "Forecast": forecasts})

    output_path = os.path.join(cfg.output_dir, "production_forecast.csv")
    safe_write_csv(df_results, output_path, "[S3] Forecast")

    metadata_export = (metadata or {}).copy()
    metadata_export["forecast_timestamp"] = pd.Timestamp.now().isoformat()
    metadata_export["model_source"] = "cached" if skip_training else "fresh_training"
    metadata_export["exog_source"] = {
        "mode": "downloader" if getattr(cfg, "use_downloader_exog", False) else "manual",
        "selected_exog": list(getattr(cfg, "selected_exog", []) or []),
        "downloader_output_path": getattr(cfg, "downloader_output_path", None),
    }

    metadata_export["train_quarterly_csv"] = cfg.dump_quarterly_dataset_path or str(
        Path(cfg.output_dir) / "train_quarterly_debug.csv"
    )
    metadata_export["train_design_csv"] = cfg.dump_train_design_path or str(
        Path(cfg.output_dir) / "train_design_debug.csv"
    )
    metadata_export["future_design_csv"] = cfg.dump_future_design_path or str(
        Path(cfg.output_dir) / "future_design_debug.csv"
    )

    try:
        if "cv_residuals" in metadata_export and isinstance(metadata_export["cv_residuals"], list):
            sigma_unscaled = float(np.nanstd(np.asarray(metadata_export["cv_residuals"], dtype=float), ddof=1))
            if np.isfinite(sigma_unscaled):
                metadata_export["cv_residual_std_unscaled"] = sigma_unscaled
                metadata_export["ci_std_error"] = sigma_unscaled
    except Exception:
        pass

    metadata_sanitized = _to_jsonable(metadata_export)
    metadata_path = os.path.join(cfg.output_dir, "production_forecast_metadata.json")
    safe_write_json(metadata_sanitized, metadata_path, "[S3] Metadata")

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("MODELL-DIAGNOSE")
    LOGGER.info("=" * 80)
    top_features = (metadata or {}).get("model_complexity", {}).get("top_features", {})
    if top_features:
        sorted_feats = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]
        LOGGER.info("\n[FEATURE IMPORTANCE] Top 5:")
        for feat, imp in sorted_feats:
            LOGGER.info(f"  • {feat[:60]}: {imp:.3f} ({imp*100:.1f}%)")

    cv_perf = (metadata or {}).get("cv_performance", {})
    rmse_v, mae_v, r2_v, n_v = _cv_vals(cv_perf)
    LOGGER.info("\n[PERFORMANCE-METRIKEN]")
    LOGGER.info(f"  CV-RMSE: {_fmt(rmse_v, 2)}")
    LOGGER.info(f"  CV-MAE:  {_fmt(mae_v, 2)}")
    LOGGER.info(f"  CV-R²:   {_fmt(r2_v, 3)}")
    if n_v is not None:
        LOGGER.info(f"  OOS-Samples: {n_v}")

    last_hist = float(df_feats[cfg.target_col].iloc[-1])
    std_fc = float(np.nanstd(np.asarray(forecasts, dtype=float))) if len(forecasts) else float("nan")
    LOGGER.info("\n[KONTEXT]")
    LOGGER.info(f"  Letzter hist. Wert: {last_hist:.1f}")
    LOGGER.info(f"  Erste Prognose:     {float(forecasts[0]):.1f}")
    denom = last_hist if last_hist != 0 else np.nan
    if np.isfinite(denom):
        LOGGER.info(f"  Abweichung:         {_fmt(((forecasts[0] - last_hist) / denom * 100), 1)}%")
    LOGGER.info(f"  Forecast-STD (Horizont): {_fmt(std_fc, 3)}")

    LOGGER.info("=" * 80)
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ERGEBNISSE")
    LOGGER.info("=" * 80)
    LOGGER.info("\n" + df_results.to_string(index=False))
    LOGGER.info(f"\nExportiert nach: {output_path}")
    LOGGER.info(f"Metadata:        {metadata_path}")
    LOGGER.info(f"Modell:          {model_path}")

    return df_results, metadata_export


# =============================================================================
# Öffentliche Pipeline-Funktion
# =============================================================================
def run_production_pipeline(cfg: Config, force_retrain: bool = False):
    """
    Produktions-Pipeline in 3 sichtbaren Abschnitten:
    S1 – Daten laden & vorbereiten
    S2 – Modell laden oder trainieren
    S3 – Zukunfts-Design, Forecast, Export
    """
    cfg.ensure_paths()

    ctx1 = _section_1_prepare_data(cfg)
    ctx2 = _section_2_train_or_load(ctx1, force_retrain=force_retrain)
    df_results, metadata_export = _section_3_forecast_and_export(ctx2)

    return df_results, metadata_export


# =============================================================================
# Modell-Management
# =============================================================================
def list_saved_models(model_dir: str = "models"):
    if not os.path.exists(model_dir):
        LOGGER.info(f"Kein Modell-Verzeichnis gefunden: {model_dir}")
        return []

    models = []
    for f in os.listdir(model_dir):
        if f.endswith(".pkl"):
            filepath = os.path.join(model_dir, f)
            try:
                artifact = ModelArtifact.load(filepath)
                models.append(
                    {
                        "filename": f,
                        "path": filepath,
                        "target": artifact.config_dict.get("target_col"),
                        "trained": artifact.metadata.get("timestamp"),
                        "cv_rmse": artifact.metadata["cv_performance"]["rmse"],
                        "n_features": artifact.metadata["model_complexity"]["n_features"],
                    }
                )
            except Exception as e:
                LOGGER.warning(f"Warnung: Konnte {f} nicht laden: {e}")

    if models:
        df = pd.DataFrame(models)
        LOGGER.info("\nGespeicherte Modelle:\n" + df.to_string(index=False))
    else:
        LOGGER.info("Keine gespeicherten Modelle gefunden.")
    return models


def delete_model(cfg: Config):
    model_path = get_model_filepath(cfg)
    if os.path.exists(model_path):
        os.remove(model_path)
        LOGGER.info(_sym(f"✓ Modell gelöscht: {model_path}"))
    else:
        LOGGER.info(f"Kein Modell gefunden: {model_path}")


def compare_model_performance(model_paths: List[str]):
    results = []
    for path in model_paths:
        try:
            artifact = ModelArtifact.load(path)
            results.append(
                {
                    "model": os.path.basename(path),
                    "target": artifact.config_dict.get("target_col"),
                    "cv_rmse": artifact.metadata["cv_performance"]["rmse"],
                    "cv_mae": artifact.metadata["cv_performance"]["mae"],
                    "cv_r2": artifact.metadata["cv_performance"]["r2"],
                    "n_features": artifact.metadata["model_complexity"]["n_features"],
                    "tree_depth": artifact.metadata["model_complexity"]["tree_depth"],
                }
            )
        except Exception as e:
            LOGGER.warning(f"Fehler bei {path}: {e}")

    if results:
        df = pd.DataFrame(results)
        LOGGER.info("\nModell-Vergleich:\n" + df.to_string(index=False))
        return df
    return None
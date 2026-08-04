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

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.backend.forecaster.core.config import Config
from src.backend.forecaster.core.data import read_excel
from src.backend.forecaster.core.data import aggregate_to_quarter
from src.backend.forecaster.core.data import add_deterministic_features
from src.backend.forecaster.core.data import build_quarterly_lags
from src.backend.forecaster.core.forecast import recursive_forecast
from src.backend.forecaster.core.helper import _fmt
from src.backend.forecaster.core.helper import build_future_design
from src.backend.forecaster.core.helper import _export_prediction
from src.backend.forecaster.core.loader import safe_write_csv
from src.backend.forecaster.core.loader import harvest_exogs_from_downloader_output
from src.backend.forecaster.core.loader import autodetect_downloader_output
from src.backend.forecaster.core.metrics import create_comprehensive_metadata
from src.backend.forecaster.core.metrics import _cv_vals
from src.backend.forecaster.core.model.model import ModelArtifact
from src.backend.forecaster.core.model.model_management import get_model_filepath
from src.backend.forecaster.core.training import train_best_model_h1

# Pandas Anzeigeoptionen (nur für lokale Diagnose)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)


FORECASTER_DIR = Path(__file__).resolve().parent  

try:
    APP_ROOT: Path = FORECASTER_DIR.parent    
except Exception:
    APP_ROOT = Path.cwd()

_logger = logging.getLogger("forecaster_pipeline")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


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
]


def _prepare_data(cfg: Config) -> dict:
    _logger.info("=" * 80)
    _logger.info("[S1] DATEN LADEN UND AUFBEREITEN")
    _logger.info("=" * 80)

    df_m = read_excel(cfg)
    df_q = aggregate_to_quarter(df_m, cfg)
    df_q = add_deterministic_features(df_q, cfg)
    df_feats = build_quarterly_lags(df_q, cfg)

    _logger.debug(f"[S1] → {len(df_feats)} Quartale, {df_feats.shape[1]} Features")

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
                    _logger.debug(f"[S1|Exog] harvested from: {out_path}")
                    _logger.debug(f"[S1|Exog] columns={resolved_exogs}")
            else:
                if getattr(cfg, "debug_exog", False):
                    _logger.debug("[S1|Exog] Keine Downloader-Output-Datei gefunden.")
            if resolved_exogs:
                cfg.selected_exog = resolved_exogs
                if getattr(cfg, "debug_exog", False):
                    _logger.debug(f"[S1|Exog] cfg.selected_exog gesetzt (n={len(cfg.selected_exog)})")
    except Exception as e:
        _logger.warning(f"[S1|Exog] Konnte Exogs nicht aus Downloader-Output übernehmen: {e}")

    return {
        "cfg": cfg,
        "df_m": df_m,
        "df_q": df_q,
        "df_feats": df_feats,
    }

def _load_model(ctx: dict, *, force_retrain: bool) -> Optional[dict]:
    """
    Tries to load a compatible model from the cache.
    If it exists, it returns an ctx-update, otherwise None.
    """
    cfg = ctx["cfg"]
    model_path = get_model_filepath(cfg)

    # Explizit: force_retrain setzt Laden außer Kraft
    if force_retrain:
        _logger.info("[S2] → force_retrain=True: Cache-Laden übersprungen")
        return None

    # Cache-Policy
    if not cfg.use_cached_model:
        _logger.info("[S2] → use_cached_model=False: Cache-Laden übersprungen")
        return None

    if not ModelArtifact.exists(model_path):
        _logger.info("[S2] → Kein gecachtes Modell gefunden")
        return None

    # Laden + Kompatibilität
    try:
        artifact = ModelArtifact.load(model_path)
        compatible, issues = artifact.is_compatible(cfg)

        if not compatible:
            _logger.warning("[S2] Gecachtes Modell inkompatibel mit aktueller Config:")
            for issue in issues:
                _logger.warning("  - " + str(issue))
            _logger.info("[S2] → Trainieren statt Laden")
            return None

        _logger.info(
            f"[S2] ✓ Verwende gecachtes Modell (trainiert: {artifact.metadata.get('timestamp', 'n/a')})"
        )
        rmse_v, mae_v, r2_v, _ = _cv_vals(artifact.metadata.get("cv_performance", {}))
        _logger.info(f"[S2] CV-RMSE: {_fmt(rmse_v, 2)}")

        return {
            "model_path": model_path,
            "model": artifact.model,
            "tj": artifact.tj,
            "X_cols": artifact.X_cols,
            "best_params": artifact.best_params,
            "metadata": artifact.metadata,
            "skip_training": True,
        }

    except Exception as e:
        _logger.warning(f"[S2] Fehler beim Laden: {e}")
        _logger.info("[S2] → Trainieren statt Laden")
        return None


def _train_model(ctx: dict) -> dict:
    """
    Trains a new model, creates metadata and saves the artifact.
    Returns a ctx-Update-Dict.
    """
    cfg, df_feats = ctx["cfg"], ctx["df_feats"]
    model_path = get_model_filepath(cfg)

    _logger.info("[S2] Trainiere Modell (Grid-Search)…")
    model, tj, X_cols, best_params, best_rmse = train_best_model_h1(df_feats, cfg)

    _logger.info(f"[S2] → Beste Parameter: {best_params}")
    _logger.info(f"[S2] → CV-RMSE: {_fmt(best_rmse, 2)}")
    _logger.info(f"[S2] → {len(X_cols)} Features")

    _logger.debug("[S2] Berechne Metriken…")
    metadata = create_comprehensive_metadata(model, tj, X_cols, best_params, df_feats, cfg)

    _logger.debug("[S2] Speichere Modell…")
    artifact = ModelArtifact(
        model=model,
        tj=tj,
        X_cols=X_cols,
        best_params=best_params,
        metadata=metadata,
        config_dict=asdict(cfg),
    )
    artifact.save(model_path)

    return {
        "model_path": model_path,
        "model": model,
        "tj": tj,
        "X_cols": X_cols,
        "best_params": best_params,
        "metadata": metadata,
        "skip_training": False,
    }


def _set_cv_residual_std(cfg, metadata: Dict[str, Any]) -> None:
    """
    σ für spätere CIs sichern (Side-effect: cfg.cv_residual_std).
    """
    try:
        if isinstance(metadata, dict):
            cv_resid = np.asarray(metadata.get("cv_residuals", []), dtype=float)
            if cv_resid.size > 0 and np.isfinite(cv_resid).any():
                sigma = float(np.nanstd(cv_resid, ddof=1))
                if np.isfinite(sigma) and sigma > 0:
                    cfg.cv_residual_std = sigma
                    return

            rmse_v, _, _, _ = _cv_vals(metadata.get("cv_performance", {}))
            if np.isfinite(rmse_v):
                cfg.cv_residual_std = float(rmse_v)

    except Exception as _e_sigma:
        _logger.warning(f"[S2] Konnte cv_residual_std nicht ableiten: {_e_sigma}")

# =============================================================================
# Abschnitt S3 – Zukunfts-Design + Forecast + Export
# =============================================================================

def _predict_future(ctx: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Creates recursive predictions for the future.

    Returns:
        df_results: Quarter table
        preds:      numpy array mit 
        fut_designs: Future-Design 
    """
    cfg: Config = ctx["cfg"]

    _logger.info("=" * 80)
    _logger.info("[S3] ZUKUNFTS-DESIGN ERSTELLEN & PREDICT RECHNEN")
    _logger.info("=" * 80)

    if getattr(cfg, "debug_exog", False):
        _logger.debug(f"[S3] Effektive Exog-Wunschliste: {list(getattr(cfg, 'selected_exog', []) or [])}")

    fut_designs = build_future_design(
        df_q=ctx["df_q"],
        cfg=cfg,
        df_feats=ctx["df_feats"],
        X_cols=ctx["X_cols"],
        debug_design=getattr(cfg, "debug_design", False),
    )
    _logger.debug(f"[S3] → {len(fut_designs)} Zukunftsquartale (final)")

    if getattr(cfg, "dump_future_design_csv", True):
        dbg_path = cfg.dump_future_design_path or str(Path(cfg.output_dir) / "future_design_debug.csv")
        safe_write_csv(fut_designs.reset_index(drop=True), dbg_path, "[S3] Future-Design dump")

    _logger.info("[S3] Rekursive Prediction …")
    preds = recursive_forecast(ctx["model"], ctx["tj"], fut_designs, ctx["X_cols"], cfg)

    fut_Q = pd.period_range(ctx["df_q"]["Q"].iloc[-1] + 1, periods=cfg.forecast_horizon, freq="Q")
    df_results = pd.DataFrame({"Quarter": [str(q) for q in fut_Q], "Prediction": preds})

    return df_results, np.asarray(preds)


def run_production_pipeline(cfg: Config, force_retrain: bool = False):
    """
    Produktions-Pipeline in 3 sichtbaren Abschnitten:
    S1 – Daten laden & vorbereiten
    S2 – Modell laden oder trainieren
    S3 – Zukunfts-Design, Forecast, Export
    """
    cfg.ensure_paths()

    # Prepare Data
    prep_data = _prepare_data(cfg)

    # Load Model or train new one
    loaded_model = _load_model(prep_data, force_retrain=force_retrain)
    if loaded_model is None:
        if force_retrain:
            _logger.info("[S2] → Neues Training erzwungen")
        loaded_model = _train_model(prep_data)
    _set_cv_residual_std(cfg, loaded_model.get("metadata", {}))
    prep_data.update(loaded_model)

    # Make Predictions with model
    df_results, predictions = _predict_future(prep_data)
    metadata_export = _export_prediction(prep_data, df_results, predictions)

    return df_results, metadata_export








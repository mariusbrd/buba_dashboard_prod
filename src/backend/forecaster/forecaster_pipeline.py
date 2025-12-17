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

from backend.forecaster.core.config import Config
from backend.forecaster.core.data import read_excel
from backend.forecaster.core.data import aggregate_to_quarter
from backend.forecaster.core.data import add_deterministic_features
from backend.forecaster.core.data import build_quarterly_lags
from backend.forecaster.core.forecast import recursive_forecast
from backend.forecaster.core.helper import _fmt
from backend.forecaster.core.helper import build_future_design
from backend.forecaster.core.loader import _to_jsonable
from backend.forecaster.core.loader import safe_write_csv
from backend.forecaster.core.loader import safe_write_json
from backend.forecaster.core.loader import harvest_exogs_from_downloader_output
from backend.forecaster.core.loader import autodetect_downloader_output
from backend.forecaster.core.metrics import create_comprehensive_metadata
from backend.forecaster.core.metrics import _cv_vals
from backend.forecaster.core.model.model import ModelArtifact
from backend.forecaster.core.model.model_management import get_model_filepath
from backend.forecaster.core.training import train_best_model_h1

# Pandas Anzeigeoptionen (nur für lokale Diagnose)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)


FORECASTER_DIR = Path(__file__).resolve().parent  

try:
    APP_ROOT: Path = FORECASTER_DIR.parent    
except Exception:
    APP_ROOT = Path.cwd()

LOGGER = logging.getLogger("forecaster_pipeline")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)
LOGGER.setLevel(logging.INFO)


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


# =============================================================================
# Abschnitt S1 – Daten vorbereiten
# =============================================================================
def _prepare_data(cfg: Config) -> dict:
    LOGGER.info("=" * 80)
    LOGGER.info("[S1] DATEN LADEN UND AUFBEREITEN")
    LOGGER.info("=" * 80)

    df_m = read_excel(cfg)
    df_q = aggregate_to_quarter(df_m, cfg)
    df_q = add_deterministic_features(df_q, cfg)
    df_feats = build_quarterly_lags(df_q, cfg)

    LOGGER.debug(f"[S1] → {len(df_feats)} Quartale, {df_feats.shape[1]} Features")

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
def _train_or_load(ctx: dict, *, force_retrain: bool) -> dict:
    cfg, df_feats  = ctx["cfg"], ctx["df_feats"]

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
                    f"[S2] ✓ Verwende gecachtes Modell (trainiert: {artifact.metadata.get('timestamp', 'n/a')})"
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
                LOGGER.info("[S2] → Neues Training wird durchgeführt")
                artifact = None
        except Exception as e:
            LOGGER.warning(f"[S2] Fehler beim Laden: {e}")
            LOGGER.info("[S2] → Neues Training wird durchgeführt")
    else:
        if force_retrain:
            LOGGER.info("[S2] → force_retrain=True: Neues Training erzwungen")

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
def _forecast_and_export(ctx: dict) -> tuple[pd.DataFrame, dict]:
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
    LOGGER.debug(f"[S3] → {len(fut_designs)} Zukunftsquartale (final)")

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

    ctx1 = _prepare_data(cfg)
    ctx2 = _train_or_load(ctx1, force_retrain=force_retrain)
    df_results, metadata_export = _forecast_and_export(ctx2)

    return df_results, metadata_export








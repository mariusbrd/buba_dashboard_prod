from backend.forecaster.core.config import Config
from backend.forecaster.core.model.model import ModelArtifact
from backend.forecaster.forecaster_pipeline import _logger


import pandas as pd


import os
from typing import List


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
            _logger.warning(f"Fehler bei {path}: {e}")

    if results:
        df = pd.DataFrame(results)
        _logger.info("\nModell-Vergleich:\n" + df.to_string(index=False))
        return df
    return None


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


def delete_model(cfg: Config):
    model_path = get_model_filepath(cfg)
    if os.path.exists(model_path):
        os.remove(model_path)
        _logger.info(f"✓ Modell gelöscht: {model_path}")
    else:
        _logger.info(f"Kein Modell gefunden: {model_path}")


def list_saved_models(model_dir: str = "models"):
    if not os.path.exists(model_dir):
        _logger.info(f"Kein Modell-Verzeichnis gefunden: {model_dir}")
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
                _logger.warning(f"Warnung: Konnte {f} nicht laden: {e}")

    if models:
        df = pd.DataFrame(models)
        _logger.info("\nGespeicherte Modelle:\n" + df.to_string(index=False))
    else:
        _logger.info("Keine gespeicherten Modelle gefunden.")
    return models
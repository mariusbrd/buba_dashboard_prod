from backend.forecaster.core.config import Config
from backend.forecaster.forecaster_pipeline import LOGGER


import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple


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
        LOGGER.debug(f"✓ Modell gespeichert: {filepath}")

    @staticmethod
    def load(filepath: str) -> "ModelArtifact":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modell nicht gefunden: {filepath}")
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)
        LOGGER.info(f"✓ Modell geladen: {filepath}")
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
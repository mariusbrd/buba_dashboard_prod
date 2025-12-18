from src.backend.forecaster.core.config import Config
from src.backend.forecaster.core.metrics import expanding_splits
from src.backend.forecaster.core.transform import TargetYJ


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeRegressor

import logging

_logger = logging.getLogger("forecaster_pipeline")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


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
    _logger.info(f"  Starte Grid-Search mit {len(combos)} Kombinationen…")

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
from src.backend.forecaster.core.config import Config


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from typing import Dict, List


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


def _cv_vals(cv: dict) -> tuple[object, object, object, object]:
    cv = cv or {}
    rmse_v = cv.get("cv_rmse", cv.get("rmse", float("nan")))
    mae_v = cv.get("cv_mae", cv.get("mae", float("nan")))
    r2_v = cv.get("cv_r2", cv.get("r2", float("nan")))
    n_v = cv.get("n_samples", cv.get("n", None))
    return rmse_v, mae_v, r2_v, n_v
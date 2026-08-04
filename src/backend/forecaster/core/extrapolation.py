from typing import Dict


import numpy as np
import pandas as pd

import logging

_logger = logging.getLogger("forecaster_pipeline")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


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


def extrapolate_with_arima(series: pd.Series, horizon: int, var_name: str = "") -> np.ndarray:
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    s_clean = series.dropna()
    if len(s_clean) < 20:
        _logger.debug(f"[ARIMA] {var_name[:50]}: Zu wenig Daten ({len(s_clean)}), Fallback")
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
        _logger.debug(f"[ARIMA] {var_name[:50]}: Alle Orders fehlgeschlagen, Fallback")
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

    _logger.debug(f"[ARIMA] {var_name[:50]}: Order={best_order}, AIC={best_aic:.1f}")
    return clipped


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
        _logger.info(f"[FUT-EXOG] {var_name[:50]}: last={last_val:.3f}, drift={drift_per_q:.4f}, seas(*)…")
    return forecast
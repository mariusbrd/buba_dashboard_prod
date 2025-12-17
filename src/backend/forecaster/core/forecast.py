from backend.forecaster.core.config import Config
from backend.forecaster.core.extrapolation import _extrapolate_drift_seasonal, extrapolate_with_arima
from backend.forecaster.forecaster_pipeline import LOGGER


import numpy as np
import pandas as pd


from typing import Dict, List


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
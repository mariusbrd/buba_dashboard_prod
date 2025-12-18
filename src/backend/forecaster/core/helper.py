import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from src.backend.forecaster.core.config import Config
from src.backend.forecaster.core.forecast import impute_future_exog_quarterly
from src.backend.forecaster.core.loader import _to_jsonable, safe_write_csv, safe_write_json
from src.backend.forecaster.core.metrics import _cv_vals

import logging

_logger = logging.getLogger("forecaster_pipeline")
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


def _fmt(x, ndigits=2, default="n/a"):
    try:
        x = float(x)
        if np.isnan(x):
            return default
        return f"{x:.{ndigits}f}"
    except Exception:
        return default


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

    _logger.debug("\n[Design] Impute future exog …")
    _logger.debug(f"[Design] Strategy: {getattr(cfg, 'future_exog_strategy', 'mixed')}")

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
                _logger.warning(
                    f"[Design] Exog nicht in Historie gefunden (werden übersprungen/fallen auf Fallbacks): {unresolved}"
                )
        except Exception as e:
            _logger.warning(f"[Design] resolve_exogs fehlgeschlagen ({e}) – nutze Wunschliste roh.")
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
        _logger.debug(f"[Design] Renamed future exog to __last__: {rename_to_last}")

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
        _logger.debug(f"[Design] Saisondummies Q1-3 für {len(fut_Q)} Quartale gesetzt")

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
        _logger.warning("[Design] Konnte Target-Serie für TARGET__lag-* nicht finden.")

    # 9) Alles zusammenführen
    fut_designs = pd.concat([fut_det, fut_exog_lags, tgt_lag_block], axis=1)
    fut_designs.index.name = "Quarter"

    num_cols = fut_designs.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        fut_designs[num_cols] = (
            fut_designs[num_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        )

    _logger.debug(f"[Design] Zukunfts-Design (roh): {fut_designs.shape[0]} Quartale, {fut_designs.shape[1]} Features")

    # 10) Schritt 9: direkt hier sicherstellen, dass alle Modell-Features vorhanden sind
    if X_cols is not None and df_feats is not None:
        fut_designs = _ensure_future_has_all_X(
            fut=fut_designs,
            X_cols=X_cols,
            cfg=cfg,
            hist_feats=df_feats,
        )
        _logger.debug(f"[Design] Zukunfts-Design (final): {fut_designs.shape[0]} Quartale, {fut_designs.shape[1]} Features")

    return fut_designs


def _canonical_exog_name(name: str) -> str:
    n = str(name).strip().lower()
    for suf in ["__last", "__first", "__mean"]:
        n = n.replace(suf, "")
    n = n.split("__lag-")[0]
    return n


def _export_prediction(ctx: dict, df_results: pd.DataFrame, preds: np.ndarray) -> dict:
    """
    Exports prediction CSV + Metadata JSON and persists results.

    Returns:
        metadata_export 
    """
    cfg: Config = ctx["cfg"]

    output_path = os.path.join(cfg.output_dir, "production_prediction.csv")
    safe_write_csv(df_results, output_path, "[S3] Prediction")

    metadata_export = (ctx["metadata"] or {}).copy()
    metadata_export["prediction_timestamp"] = pd.Timestamp.now().isoformat()
    metadata_export["model_source"] = "cached" if ctx.get("skip_training") else "fresh_training"
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

    # Optional: aus CV-Residuals ableiten
    try:
        if "cv_residuals" in metadata_export and isinstance(metadata_export["cv_residuals"], list):
            sigma_unscaled = float(np.nanstd(np.asarray(metadata_export["cv_residuals"], dtype=float), ddof=1))
            if np.isfinite(sigma_unscaled):
                metadata_export["cv_residual_std_unscaled"] = sigma_unscaled
                metadata_export["ci_std_error"] = sigma_unscaled
    except Exception:
        pass

    metadata_sanitized = _to_jsonable(metadata_export)
    metadata_path = os.path.join(cfg.output_dir, "production_prediction_metadata.json")
    safe_write_json(metadata_sanitized, metadata_path, "[S3] Metadata")

    # Logging Diagnose/Ergebnisse (wie bisher, nur Namen angepasst)
    _logger.info("\n" + "=" * 80)
    _logger.info("MODELL-DIAGNOSE")
    _logger.info("=" * 80)

    top_features = (ctx["metadata"] or {}).get("model_complexity", {}).get("top_features", {})
    if top_features:
        sorted_feats = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]
        _logger.info("\n[FEATURE IMPORTANCE] Top 5:")
        for feat, imp in sorted_feats:
            _logger.info(f"  • {feat[:60]}: {imp:.3f} ({imp*100:.1f}%)")

    cv_perf = (ctx["metadata"] or {}).get("cv_performance", {})
    rmse_v, mae_v, r2_v, n_v = _cv_vals(cv_perf)
    _logger.info("\n[PERFORMANCE-METRIKEN]")
    _logger.info(f"  CV-RMSE: {_fmt(rmse_v, 2)}")
    _logger.info(f"  CV-MAE:  {_fmt(mae_v, 2)}")
    _logger.info(f"  CV-R²:   {_fmt(r2_v, 3)}")
    if n_v is not None:
        _logger.info(f"  OOS-Samples: {n_v}")

    last_hist = float(ctx["df_feats"][cfg.target_col].iloc[-1])
    std_pred = float(np.nanstd(np.asarray(preds, dtype=float))) if len(preds) else float("nan")
    _logger.info("\n[KONTEXT]")
    _logger.info(f"  Letzter hist. Wert: {last_hist:.1f}")
    _logger.info(f"  Erste Prediction:   {float(preds[0]):.1f}")
    denom = last_hist if last_hist != 0 else np.nan

    if np.isfinite(denom):
        _logger.info(f"  Abweichung:         {_fmt(((preds[0] - last_hist) / denom * 100), 1)}%")
    _logger.info(f"  Prediction-STD (Horizont): {_fmt(std_pred, 3)}")

    _logger.info("=" * 80)
    _logger.info("\n" + "=" * 80)
    _logger.info("ERGEBNISSE")
    _logger.info("=" * 80)
    _logger.info("\n" + df_results.to_string(index=False))
    _logger.info(f"\nExportiert nach: {output_path}")
    _logger.info(f"Metadata:        {metadata_path}")
    _logger.info(f"Modell:          {ctx['model_path']}")

    return metadata_export



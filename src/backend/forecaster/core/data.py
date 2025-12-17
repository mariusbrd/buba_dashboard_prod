from typing import List
import numpy as np
from backend.forecaster.core.config import Config


import pandas as pd


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
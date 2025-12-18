# =============================================================================
# KONFIGURATION
# =============================================================================


import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


FORECASTER_DIR = Path(__file__).resolve().parent.parent  

@dataclass
class Config:
    # =========================
    # Datenquelle
    # =========================
    excel_path: str = "transformed_output.xlsx"
    sheet_name: str = "final_dataset"
    date_col: str = "Datum"
    target_col: str = "PH_EINLAGEN"

    # =========================
    # Aggregation
    # =========================
    agg_methods_exog: List[str] = field(default_factory=lambda: ["last"])
    agg_method_target: str = "mean"

    # =========================
    # Lags
    # =========================
    exog_month_lags: List[int] = field(default_factory=lambda: [-24,
                                                                -12,
                                                                -6,
                                                                -3,
                                                                -1])
    target_lags_q: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # =========================
    # Backtest / Validierung
    # =========================
    min_train_quarters: int = 24
    test_horizon_quarters: int = 2
    gap_quarters: int = 1

    # =========================
    # Param-Grid
    # =========================
    param_grid: Union[Dict, List[Dict]] = field(
        default_factory=lambda: [
            {
                "criterion": ["squared_error"],
                "max_depth": [8, 10, 12, 14, 16, 20, 50],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2],
                "max_features": [None],
                "ccp_alpha": [0.0],
            },
        ]
    )

    # =========================
    # Deterministische Features
    # =========================
    add_trend_features: bool = True
    trend_degree: int = 1
    add_seasonality: bool = True
    seasonality_mode: str = "dummies"

    # =========================
    # Transformation
    # =========================
    target_transform: str = "none"  # "none" | "yeo-johnson"
    target_standardize: bool = True

    # =========================
    # Forecast / Future-Exogs
    # =========================
    forecast_horizon: int = 6
    future_exog_strategy: str = "mixed"  # "mixed" | "arima" | "drift"
    future_exog_drift_window_q: int = 8
    future_exog_seasonal_period_q: int = 4

    # =========================
    # Output & Persistierung
    # =========================

    output_dir: str = field(
        default_factory=lambda: str((FORECASTER_DIR / "trained_outputs").resolve())
    )
    model_dir: str = field(
        default_factory=lambda: str((FORECASTER_DIR / "trained_models").resolve())
    )
    use_cached_model: bool = True
    random_state: int = 42

    # =========================
    # ARIMA-Optionen
    # =========================
    use_arima_extrapolation: bool = True
    arima_for_important_vars: bool = True
    arima_importance_threshold: float = 0.10

    # =========================
    # Debug / Diagnose
    # =========================
    debug_exog: bool = True
    debug_design: bool = True
    debug_recur: bool = True
    diag_max_cols: int = 20
    diag_show_heads: int = 2

    # =========================
    # Dumps für Offline-Check
    # =========================
    dump_future_design_csv: bool = True
    dump_future_design_path: Optional[str] = None

    dump_quarterly_dataset_csv: bool = True
    dump_quarterly_dataset_path: Optional[str] = None

    dump_train_design_csv: bool = True
    dump_train_design_path: Optional[str] = None

    # =========================
    # Seeds/Fallbacks für Rekursion
    # =========================
    stable_exog_cols: List[str] = field(default_factory=list)
    last_train_row: Optional[Dict[str, float]] = None
    train_feature_medians: Dict[str, float] = field(default_factory=dict)
    last_target_value: Optional[float] = None

    # =========================
    # Adapter/Cache
    # =========================
    cache_tag: str = ""
    selected_exog: List[str] = field(default_factory=list)
    data_signature: str = ""

    # =========================
    # Convenience
    # =========================
    def ensure_paths(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if self.dump_future_design_csv and self.dump_future_design_path is None:
            self.dump_future_design_path = str(Path(self.output_dir) / "future_design_debug.csv")

        if self.dump_quarterly_dataset_csv and self.dump_quarterly_dataset_path is None:
            self.dump_quarterly_dataset_path = str(Path(self.output_dir) / "train_quarterly_debug.csv")

        if self.dump_train_design_csv and self.dump_train_design_path is None:
            self.dump_train_design_path = str(Path(self.output_dir) / "train_design_debug.csv")

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        for k in [
            "output_dir",
            "model_dir",
            "dump_future_design_path",
            "dump_quarterly_dataset_path",
            "dump_train_design_path",
        ]:
            if isinstance(d.get(k), Path):
                d[k] = str(d[k])
        return d
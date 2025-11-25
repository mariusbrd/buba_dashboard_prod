import os
import json
import pickle
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger("GVB_Dashboard")
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)

# =============================================================================
# KONFIGURATION
# =============================================================================
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from pathlib import Path

# -------------------------------------------------------------------
# Pfad-Logik: Scenario-Ordner + App-Root
# -------------------------------------------------------------------
SCENARIO_DIR = Path(__file__).resolve().parent  # .../scenario

try:
    APP_ROOT: Path = SCENARIO_DIR.parent        # Projektroot (dort liegt typischerweise app.py)
except Exception:
    APP_ROOT = Path.cwd()


@dataclass
class Config:
    # Datenquelle
    excel_path: str = "output.xlsx"
    sheet_name: str = "final_dataset"
    date_col: str = "Datum"

    # Zielvariable
    target_col: str = "Einlagen"

    # Targets nie als Exogene verwenden
    target_candidates: List[str] = field(default_factory=lambda: [
        "Einlagen", "Wertpapiere", "Versicherungen", "Kredite"
    ])

    # Feste Exogenenliste
    exog_fixed_list: List[str] = field(default_factory=lambda: [
        "lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"
    ])

    # Aggregation (Monat -> Quartal)
    agg_methods_exog: List[str] = field(default_factory=lambda: ["last"])
    agg_method_target: str = "mean"

    # Lags
    exog_month_lags: List[int] = field(default_factory=lambda: [-3, -1])
    target_lags_q: List[int] = field(default_factory=lambda: [1, 2])  # weniger Plateau als [1,2,4]

    # Backtest/Hyperparameter (Expanding Window, h=0 Nowcast)
    min_train_quarters: int = 24
    gap_quarters: int = 1

    # Debug/Transparenz
    debug_decision_path: bool = True
    debug_perturbation: bool = False
    debug_perturbation_pct: float = 0.05  # Größe des Tests

    # Exog-Nutzung erzwingen
    min_exog_share: float = 0.35
    exog_penalty_alpha: float = 1.5
    top_k_exog_check: int = 50

    # Modell-Grid
    param_grid: Dict = field(default_factory=lambda: {
       "max_depth": [8, 10, 12, 14, 16],
       "min_samples_leaf": [0.01, 0.02, 0.03],
       "min_samples_split": [0.02, 0.05, 0.1],
       "max_features": [None, "sqrt"],
       "criterion": ["squared_error"],
       "ccp_alpha": [0.0, 5e-4],
       "min_impurity_decrease": [0.0, 1e-4],
    })


    # Modell-Grid - Slow Speed
    # param_grid: Dict = field(default_factory=lambda: {
    #     "max_depth": [8, 10],
    #     "min_samples_leaf": [0.01],
    #     "min_samples_split": [0.02],
    #     "max_features": [None, "sqrt"],
    #     "criterion": ["squared_error"],
    #     "ccp_alpha": [0.0, 5e-4],
    #     "min_impurity_decrease": [0.0],
    # })

    
    # Deterministische Features
    add_trend_features: bool = True
    trend_degree: int = 1
    add_seasonality: bool = True
    seasonality_mode: str = "dummies"

    # Horizon-Features (unterstützt unterschiedliche Leaves je H)
    add_horizon_features: bool = True
    horizon_mode: str = "onehot"       # "onehot" oder "index"
    max_horizon: int = 8
    horizon_prefix: str = "HORIZON__"

    # Transformation
    target_transform: str = "none"
    target_standardize: bool = True

    # Zukunfts-Extrapolation
    forecast_horizon_quarters: int = 4
    future_exog_strategy: str = "mixed"
    future_exog_drift_window_q: int = 8
    future_exog_seasonal_period_q: int = 4

    # Output & Persistierung
    output_dir: str = field(
        default_factory=lambda: str((SCENARIO_DIR / "outputs").resolve())
    )
    model_dir: str = field(
        default_factory=lambda: str((SCENARIO_DIR / "models_scenario_horizon").resolve())
    )
    use_cached_model: bool = False              # Neu trainieren, damit Änderungen wirken
    random_state: int = 42


    # Guardrails (Level-Bound)
    target_lower_bound: Optional[float] = None

    # --------------------------------------------------------------------------------
    # Szenario-/Plateau-Handling & Local Sensitivity (Leaf-Linear)
    # --------------------------------------------------------------------------------

    # Outlier-Cap nur auf Baumantwort anwenden
    # gültige Werte: "none", "tree_clip"
    outlier_cap_mode: str = "tree_clip"
    scenario_tree_only: bool = False  # False = Baum normal; True = nur Tree-Clip

    # Plateau-Guard: greift, wenn rekursive Baseline über H (nahezu) konstant ist
    plateau_guard_enable: bool = True
    plateau_guard_rel_tol: float = 1e-4
    plateau_guard_abs_tol: float = 0.05

    # Relax-Schwelle für Outlier-Cap (Forecast-Funktion nutzt Fallback auf min_exog_share)
    cap_relax_exog_share_threshold: float = 0.35

    # Leaf-Linear Fallback (wenn ΔX vorhanden, aber Leaf gleich bleibt)
    enable_leaf_linear: bool = True
    leaf_linear_method: str = "knn"       # "knn" oder "ols"
    leaf_linear_k: int = 85
    leaf_linear_exog_only: bool = True
    leaf_linear_use_zscore: bool = True
    leaf_linear_cap_rmse_mult: float = 2.0
    leaf_linear_cap_pct_of_level: float = 0.10
    leaf_linear_min_abs_z: float = 0.1

# =============================================================================
# MODELL-PERSISTIERUNG
# =============================================================================

class ModelArtifact:
    """Container für alle Modell-Artefakte (Nowcast)."""

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
        LOGGER.info("✓ Modell gespeichert: %s", filepath)

    @staticmethod
    def load(filepath: str) -> "ModelArtifact":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modell nicht gefunden: {filepath}")
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)
        LOGGER.info("✓ Modell geladen: %s", filepath)
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
        """Prüft Kompatibilität für Nowcast-Setup."""
        issues: List[str] = []
        cfg_dict = asdict(current_config)
        critical_params = [
            "target_col", "target_candidates", "exog_fixed_list",
            "agg_method_target", "agg_methods_exog",
            "exog_month_lags", "target_lags_q",
            "add_trend_features", "trend_degree",
            "add_seasonality", "seasonality_mode",
            "target_transform"
        ]
        for p in critical_params:
            if cfg_dict.get(p) != self.config_dict.get(p):
                issues.append(f"Config-Mismatch: {p} ({cfg_dict.get(p)} vs {self.config_dict.get(p)})")
        return len(issues) == 0, issues
# =============================================================================
# KLASSE: SCENARIO ANALYSIS
# =============================================================================
class ScenarioAnalysis:
    """Dash-freundlicher Wrapper für Nowcast + Szenario-Forecast."""

    def _log(self, msg: str, level: Optional[str] = None):
        """
        Zentrale Logging-Funktion.

        - schreibt (bei info/warning/error) bevorzugt in die Scenario-Table
        - nutzt Python-Logging mit Level-Mapping
        - reduziert Info-Noise, indem Detail-Diagnostik automatisch als DEBUG klassifiziert wird
        """
        try:
            lvl = (level or "").lower()
            upper = str(msg).upper()

            # Auto-Level, falls kein Level explizit übergeben wurde
            if not lvl:
                if "ERROR" in upper:
                    lvl = "error"
                elif "WARN" in upper or "⚠" in msg:
                    lvl = "warning"
                elif any(tag in upper for tag in [
                    "DEBUG", "LAGGUARD", "DECISIONPATH",
                    "BASELINEDIAG", "LEAFDIAG", "LEAFLINEAR",
                    "SICHTPRÜFUNG"
                ]):
                    lvl = "debug"
                else:
                    lvl = "info"

            # 1) Scenario-Table nur für info/warning/error
            if lvl in ("info", "warning", "error"):
                try:
                    Log.scenario_table(msg)  # noqa: F821
                except Exception:
                    pass

            # 2) Python-Logger mit Level-Mapping
            try:
                logger = LOGGER
                if lvl == "debug":
                    logger.debug(msg)
                elif lvl == "warning":
                    logger.warning(msg)
                elif lvl == "error":
                    logger.error(msg)
                else:
                    logger.info(msg)
            except Exception:
                pass

            # 3) Fallback stdout nur für warning/error
            if lvl in ("warning", "error"):
                try:
                    print(msg)
                except Exception:
                    pass
        except Exception:
            # Hard-Fail des Loggers vermeiden
            try:
                print(msg)
            except Exception:
                pass


    # ---------- Konstruktor & Utilities ----------

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._df_m: Optional[pd.DataFrame] = None
        self._df_q: Optional[pd.DataFrame] = None
        self._df_det: Optional[pd.DataFrame] = None
        self._df_feats: Optional[pd.DataFrame] = None

        self.artifact: Optional[ModelArtifact] = None
        self.model: Optional[DecisionTreeRegressor] = None
        self.tj: Optional["TargetYJ"] = None
        self.X_cols: Optional[List[str]] = None
        self.metadata: Optional[Dict] = None

    # ---------- Öffentliche API (für Dash) ----------

    def _exog_mask_and_share(self) -> tuple[list[bool], float]:
        """
        Liefert (Maske für EXOG-Features, Anteil der EXOG-Importance).
        EXOG = alles, was NICHT mit 'DET_', 'SEAS_' oder 'TARGET__lag-' beginnt.
        """
        try:
            if self.model is None or self.X_cols is None or not hasattr(self.model, "feature_importances_"):
                return [], 0.0
            X_cols = list(self.X_cols)
            fi = np.asarray(getattr(self.model, "feature_importances_", np.zeros(len(X_cols))), dtype=float)
            exog_mask = [not c.startswith(("DET_", "SEAS_", "TARGET__lag-")) for c in X_cols]
            exog_sum = float(fi[np.array(exog_mask)].sum()) if fi.size else 0.0
            total = float(fi.sum()) if fi.size else 0.0
            share = (exog_sum / total) if total > 0 else 0.0
            return exog_mask, share
        except Exception:
            return [], 0.0

    def _log_exog_usage(self, top_k: int = 12, *, log_once: bool = True) -> None:
        """
        Sichtprüfung: Wie stark nutzt das Modell exogene Features?
        - Anteil exogener Importance vs. deterministische/lags
        - Top-K Features mit Typ-Kennzeichnung

        Parameter
        ---------
        top_k : int
            Anzahl der Top-Features, die geloggt werden.
        log_once : bool
            Wenn True, wird pro (Modell, X_cols, Summe der Importances) nur einmal geloggt.
            Verhindert doppelte Log-Blöcke bei mehrfachen Aufrufen im selben Pipeline-Durchlauf.
        """
        import numpy as np
        import pandas as pd

        def _log(msg: str):
            # Detail-Diagnostik zu EXOG-Nutzung → DEBUG-Level
            self._log(msg, level="debug")

        X_cols = list(getattr(self, "X_cols", []) or [])
        fi = getattr(self.model, "feature_importances_", None)

        # Early exit / "log once" Guard (verhindert doppelte Blöcke)
        if fi is not None and len(X_cols) and log_once:
            try:
                fi_arr = np.asarray(fi, dtype=float)
                # Signatur basiert auf: Modell-Identität, Anzahl Features, Summe der Importances
                curr_sig = (id(self.model), len(X_cols), float(np.nansum(fi_arr)))
                last_sig = getattr(self, "_last_exog_log_sig", None)
                if last_sig == curr_sig:
                    return  # bereits geloggt – überspringen
                # Signatur speichern (auch wenn später ein Fehler auftritt, um Stürme zu vermeiden)
                self._last_exog_log_sig = curr_sig
            except Exception:
                # Wenn die Signaturbildung fehlschlägt, nicht blockieren – einfach weiter loggen.
                pass

        if fi is None or not len(X_cols):
            _log("[ScenarioForecast] Keine Feature-Importances verfügbar (Modell ohne .feature_importances_?).")
            return

        fi = np.asarray(fi, dtype=float)
        if fi.size != len(X_cols):
            _log(f"[ScenarioForecast] WARN: Länge feature_importances_ ({fi.size}) != len(X_cols) ({len(X_cols)}).")
            # robust: auf gemeinsame Länge trimmen
            k = min(fi.size, len(X_cols))
            fi, X_cols = fi[:k], X_cols[:k]

        def _typ(c: str) -> str:
            if c.startswith("DET_"): return "DET"
            if c.startswith("SEAS_"): return "SEAS"
            if c.startswith("TARGET__lag-"): return "LAG"
            return "EXOG"

        types = [_typ(c) for c in X_cols]
        fi_total = float(np.nansum(fi)) if np.isfinite(fi).all() else 0.0
        if fi_total <= 0:
            _log("[ScenarioForecast] Alle Importances sind 0.")
            return

        exog_mask = np.array([t == "EXOG" for t in types], dtype=bool)
        exog_sum = float(np.nansum(fi[exog_mask])) if fi.size else 0.0
        exog_share = (exog_sum / fi_total) if fi_total > 0 else 0.0

        _log(
            f"[ScenarioForecast] Feature-Importance: exog_sum={exog_sum:.4f}, "
            f"total={fi_total:.4f}, share={exog_share:.2%}"
        )

        # Top-K nach Importance
        order = np.argsort(-fi)
        rows = []
        for idx in order[:top_k]:
            rows.append({
                "rank": len(rows) + 1,
                "feature": X_cols[idx],
                "type": types[idx],
                "importance": float(fi[idx])
            })

        df = pd.DataFrame(rows, columns=["rank", "feature", "type", "importance"])
        _log("[ScenarioForecast] Top-Features (Importance, Top-K):")
        for _, r in df.iterrows():
            _log(f"  #{int(r['rank']):2d} | {r['feature']:<40s} | {r['type']:<4s} | imp={r['importance']:.5f}")


    def adjustable_features(self) -> List[str]:
        """Aktuelle, verstellbare EXOG-Spaltennamen (für Slider)."""
        if self._df_feats is None or self.X_cols is None:
            return []
        return self._list_current_scenario_features(self._df_feats, self.X_cols)

    def load_and_prepare(self):
        """Daten laden + Feature Engineering (bis Designmatrix)."""
        cfg = self.cfg
        df_m = self._read_excel(cfg)#.ffill().bfill()
        if cfg.target_col not in df_m.columns:
            raise ValueError(f"target_col '{cfg.target_col}' nicht in Daten gefunden.")
        _ = self._exog_base_columns(df_m, cfg)  # validiert feste Exogen

        df_q = self._aggregate_to_quarter(df_m, cfg)
        df_det = self._add_deterministic_features(df_q, cfg)
        df_feats = self._build_nowcast_design(df_det, cfg)

        self._df_m, self._df_q, self._df_det, self._df_feats = df_m, df_q, df_det, df_feats
        return df_feats

    def train_or_load(self, force_retrain: bool = False):
        """Modell aus Cache laden oder neu trainieren."""
        if self._df_feats is None:
            self.load_and_prepare()

        cfg = self.cfg
        model_path = self._get_model_filepath(cfg)
        artifact = None
        skip = False

        if cfg.use_cached_model and not force_retrain and ModelArtifact.exists(model_path):
            try:
                artifact = ModelArtifact.load(model_path)
                ok, issues = artifact.is_compatible(cfg)
                if ok:
                    self.artifact = artifact
                    self.model = artifact.model
                    self.tj = artifact.tj
                    self.X_cols = artifact.X_cols
                    self.metadata = artifact.metadata
                    skip = True
                    LOGGER.info("    ✓ Verwende gecachtes Modell (%s)", artifact.metadata.get('timestamp'))
                else:
                    LOGGER.warning("    ⚠ Cache inkompatibel → neu trainieren:")
                    for s in issues:
                        LOGGER.warning("      - %s", s)
            except Exception as e:
                LOGGER.error("    ⚠ Laden fehlgeschlagen → neu trainieren: %s", e)

        if not skip:
            model, tj, X_cols, best_params, best_rmse = self._train_best_model_nowcast(self._df_feats, cfg)
            metadata = self._create_comprehensive_metadata_nowcast(model, tj, X_cols, best_params, self._df_feats, cfg)
            self.artifact = ModelArtifact(model=model, tj=tj, X_cols=X_cols,
                                          best_params=best_params, metadata=metadata,
                                          config_dict=asdict(cfg))
            Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
            self.artifact.save(model_path)

            self.model = model
            self.tj = tj
            self.X_cols = X_cols
            self.metadata = metadata

        # Sichtprüfung direkt nach Train/Laden
        try:
            self._log_exog_usage(top_k=12)
        except Exception as _e:
            self._log(f"[ExogUsage] ERROR: {_e}")

        return self.model, self.tj, self.X_cols, self.metadata

    def set_target(self, target_col: str):
        """Target wechseln (eine der 4 Kandidaten), invalidiert Caches im Objekt."""
        if target_col not in self.cfg.target_candidates:
            raise ValueError(f"'{target_col}' ist kein gültiges Target. Erlaubt: {self.cfg.target_candidates}")
        self.cfg.target_col = target_col
        # invalidate in-memory Artefakte
        self._df_m = self._df_q = self._df_det = self._df_feats = None
        self.artifact = None
        self.model = None
        self.tj = None
        self.X_cols = None
        self.metadata = None

    def nowcast_t0(self,
                scenario_adjustments: Optional[Dict[str, float]] = None,
                persist: bool = True,
                force_retrain: bool = False) -> Dict:
        """
        t0-Nowcast (Baseline) und optional t0-Szenario anwenden (nur Decision Tree, kein Linear-Fallback).
        Rückgabe ist JSON-freundlich (ideal für dcc.Store).
        """
        LOGGER.info("=" * 80)
        LOGGER.info("NOWCAST PIPELINE (t0) – Szenario-Analyse")
        LOGGER.info("=" * 80)

        # 1) Daten + Modell
        LOGGER.info("[1/3] Daten vorbereiten …")
        df_feats = self.load_and_prepare()
        LOGGER.info("    → %s Quartale, %s Features", len(df_feats), df_feats.shape[1])

        LOGGER.info("[2/3] Modell laden/trainieren …")
        self.train_or_load(force_retrain=force_retrain)

        # --- Exogen-Nutzung des Baums messen ---
        X_cols = list(self.X_cols or [])
        _, exog_importance_share = self._exog_mask_and_share()
        self._log(f"[ScenarioForecast] EXOG-Importance-Share: {exog_importance_share:.1%}")
        self._log_exog_usage(top_k=12)

        if exog_importance_share < 0.05:
            self._log("[ScenarioForecast] Hinweis: EXOG-Share <5% – Szenarioeffekte im Tree könnten gering ausfallen.")

        # 3) t0-Nowcast & (optional) Szenario
        LOGGER.info("[3/3] Nowcast & Szenario …")
        X_row_t0 = df_feats[self.X_cols].iloc[-1].copy()
        yhat_baseline = self._predict_nowcast_t0(self.model, self.tj, X_row_t0)

        yhat_scenario = yhat_baseline
        scenario_method = "none"

        if scenario_adjustments:
            # Szenario auf die aktuelle X-Zeile anwenden und Tree-Vorhersage holen
            X_row_scn = self._apply_scenario_to_Xrow(X_row_t0, scenario_adjustments)
            yhat_tree = self._predict_nowcast_t0(self.model, self.tj, X_row_scn)
            yhat_scenario = yhat_tree
            scenario_method = "tree_only"

        delta_abs = yhat_scenario - yhat_baseline
        delta_pct = (delta_abs / yhat_baseline * 100.0) if yhat_baseline else 0.0

        results = {
            "Q_t0": str(self._df_q["Q"].iloc[-1]),
            "target": self.cfg.target_col,
            "yhat_baseline_t0": float(yhat_baseline),
            "yhat_scenario_t0": float(yhat_scenario),
            "delta_abs": float(delta_abs),
            "delta_pct": float(delta_pct),
            "scenario_applied_to_cols": list(scenario_adjustments.keys()) if scenario_adjustments else [],
            "scenario_method": scenario_method,
            "adjustable_feature_names": self.adjustable_features(),
        }

        if persist:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            path = os.path.join(self.cfg.output_dir, f"nowcast_t0_{self.cfg.target_col}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            meta_path = os.path.join(self.cfg.output_dir, f"nowcast_metadata_{self.cfg.target_col}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        return results

    def _debug_decision_path_for_row(
        self,
        X_row: pd.Series,
        horizon: Optional[int] = None,
        top_k: int = 8
    ) -> None:
        """
        Loggt die vom Decision Tree genutzten Splits für einen einzelnen Forecast-Zeilenvektor.
        Zeigt Feature-Name, aktuellen Wert und den Threshold im Knoten.
        """
        try:
            if self.model is None or self.X_cols is None:
                return

            X = (
                X_row[self.X_cols]
                .to_frame()
                .T
                .infer_objects(copy=False)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            tree = self.model.tree_
            feature_names = np.array(self.X_cols)

            node_indicator = self.model.decision_path(X)
            leaf_id = self.model.apply(X)[0]
            node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]

            used = []
            for node_id in node_index:
                if leaf_id == node_id:
                    continue
                feat_id = tree.feature[node_id]
                thr = tree.threshold[node_id]
                if feat_id >= 0:
                    fname = feature_names[feat_id]
                    fval = float(X[0, feat_id])
                    used.append((fname, fval, float(thr)))

            if not used:
                return

            used = used[-top_k:]
            hdr = f"[DecisionPath] H={horizon}" if horizon is not None else "[DecisionPath]"
            self._log(f"{hdr} – genutzte Splits ({len(used)}):")
            for (fname, fval, thr) in used:
                self._log(f"  - {fname}: value={fval:.4f} vs threshold={thr:.4f}")
            uniq = list(dict.fromkeys([u[0] for u in used]))
            self._log(f"{hdr} – Features im Pfad: {', '.join(uniq)}")

        except Exception as e:
            self._log(f"[DecisionPath] ERROR: {e}")


    def forecast(self,
            H: Optional[int] = None,
            scenario_future: Optional[Dict[str, Union[float, List[float], Dict[int, float]]]] = None,
            persist: bool = True,
            force_retrain: bool = False) -> Dict:
        """
        Rekursive Extrapolation t1..tH (Baseline & Szenario) mit Robustheit:
        - Clipping der Szenario-EXOG auf den gelernten Bereich (+15% Puffer)
        - Sichtprüfung: EXOG-Anteil, Top-Features, Base-vs-Szenario im Design (H1/H2)
        - LagGuard: prüft Exog-Lag-Ausrichtung (H1)
        - Plateau-Guard: wechselt bei flacher rekursiver Vorhersage auf Batch
        - Identity-Guard: Baseline-Preset → Szenario == Baseline (nicht konstant, aber identisch)
        - Outlier-Cap (Tree-only) + optionaler Leaf-Linear-Fallback (KNN/OLS)
        """
        import os, json, numpy as np, pandas as pd

        cfg = self.cfg  # Kurzalias
        H = int(H or cfg.forecast_horizon_quarters)

        # --- Konfiguration / Flags -------------------------------------------------
        tree_only_flag: bool = bool(getattr(cfg, "scenario_tree_only", False))
        outlier_cap_mode: str = str(getattr(cfg, "outlier_cap_mode", "tree_clip")).lower()

        # Leaf-Linear: Abgleich mit deiner Config (mit Fallbacks auf ältere Schlüsselnamen)
        enable_leaf_linear: bool = bool(getattr(cfg, "enable_leaf_linear",
                                        getattr(cfg, "enable_leaf_linear_fallback", True)))
        leaf_linear_method: str = str(getattr(cfg, "leaf_linear_method", "knn")).lower()  # "knn" | "ols"
        leaf_linear_k: int = int(getattr(cfg, "leaf_linear_k", 85))
        leaf_linear_exog_only: bool = bool(getattr(cfg, "leaf_linear_exog_only", True))
        leaf_linear_use_zscore: bool = bool(getattr(cfg, "leaf_linear_use_zscore", True))
        leaf_linear_cap_rmse_mult: float = float(getattr(cfg, "leaf_linear_cap_rmse_mult", 2.0))
        leaf_linear_cap_pct_of_level: float = float(getattr(cfg, "leaf_linear_cap_pct_of_level", 0.10))
        leaf_linear_min_abs_z: float = float(getattr(cfg, "leaf_linear_min_abs_z", 0.10))

        # Schwelle, ab der Outlier-Cap gelockert wird (nutze deine min_exog_share als Default)
        cap_relax_thr: float = float(getattr(cfg, "cap_relax_exog_share_threshold",
                                    getattr(cfg, "min_exog_share", 0.10)))

        def log(msg: str):
            # Szenario-spezifische Meldungen konsistent über zentrale Logger-Logik führen
            self._log(msg)

        # --------------------------------------------------------------------------
        # A) Daten & Modell vorbereiten
        # --------------------------------------------------------------------------
        df_feats = self.load_and_prepare()
        self.train_or_load(force_retrain=force_retrain)

        X_cols = list(self.X_cols or [])
        fi = (self.model.feature_importances_ if hasattr(self.model, "feature_importances_") else None)

        exog_mask = [not c.startswith(("DET_", "SEAS_", "TARGET__lag-")) for c in X_cols]
        exog_importance = float(np.sum((fi if fi is not None else np.zeros(len(X_cols)))[np.array(exog_mask)]))
        total_importance = float(np.sum(fi)) if fi is not None else 0.0
        exog_importance_share = (exog_importance / total_importance) if total_importance > 0 else 0.0
        self._log(f"[ScenarioForecast] Feature-Importance: exog_sum={exog_importance:.4f}, share={exog_importance_share:.3%}")

        try:
            if hasattr(self, "_log_exog_usage"):
                self._log_exog_usage(top_k=12)
        except Exception as _e:
            self._log(f"[ScenarioForecast] _log_exog_usage WARN: {_e}")

        # Historie
        df_hist_det = self._df_det.copy()
        df_hist_det = df_hist_det[df_hist_det[cfg.target_col].notna()].copy()
        if df_hist_det.empty:
            raise ValueError(f"Keine gültigen {cfg.target_col}-Werte in Historie")
        last_Q = df_hist_det["Q"].iloc[-1]

        # --------------------------------------------------------------------------
        # B) Baseline-EXOG (imputiert) + Designs
        # --------------------------------------------------------------------------
        ignore = {"Q", "Q_end", cfg.target_col}
        exog_cols = [c for c in df_hist_det.columns
                    if c not in ignore and "__" in c and not c.startswith(("DET_", "SEAS_", "TARGET__lag"))]
        if not exog_cols:
            raise ValueError("Keine exogenen Aggregat-Spalten gefunden (…__last/…__mean)")

        fut_exog_baseline = self._impute_future_exog_quarterly_fixed(
            df_hist_det, exog_cols, H,
            strategy=cfg.future_exog_strategy,
            window_q=cfg.future_exog_drift_window_q,
            seas_p=cfg.future_exog_seasonal_period_q
        )

        # Deterministiken + Design (Baseline)
        fut_det_base = self._extend_deterministics_for_future(
            df_hist_det, fut_exog_baseline[["Q", "Q_end"]].copy(), cfg
        )
        fut_base_baseline = fut_det_base.merge(
            fut_exog_baseline.drop(columns=["Q", "Q_end"]),
            left_index=True, right_index=True
        )
        comb_base = pd.concat([df_hist_det, fut_base_baseline], ignore_index=True)
        comb_design = self._build_nowcast_design_no_drop(comb_base, cfg)
        fut_designs_base = comb_design.loc[comb_design["Q"] > last_Q].reset_index(drop=True)

        # --------------------------------------------------------------------------
        # C) Szenario anwenden (+ Clip) + Designs
        # --------------------------------------------------------------------------
        fut_exog_scn = self._apply_future_scenario_on_exogs(
            fut_exog_baseline.copy(), scenario_future, H
        )

        # Clip (+15% vom historischen Spannmaß)
        ranges = {}
        for c in exog_cols:
            s = df_hist_det[c].astype(float)
            if s.dropna().empty:
                continue
            mn, mx = float(s.min()), float(s.max())
            span = mx - mn
            pad = 0.15 * span if span > 0 else max(1.0, abs(mx) * 0.15)
            ranges[c] = (mn - pad, mx + pad)
        for c, (lo, hi) in ranges.items():
            if c in fut_exog_scn.columns:
                before = fut_exog_scn[c].copy()
                fut_exog_scn[c] = fut_exog_scn[c].astype(float).clip(lower=lo, upper=hi)
                if not np.allclose(before.values, fut_exog_scn[c].values, equal_nan=True):
                    log(f"[ScenarioForecast] Clip {c} auf [{lo:.3f}, {hi:.3f}]")

        # Szenario-Design
        fut_det_scn = self._extend_deterministics_for_future(
            df_hist_det, fut_exog_scn[["Q", "Q_end"]].copy(), cfg
        )
        fut_base_scn = fut_det_scn.merge(
            fut_exog_scn.drop(columns=["Q", "Q_end"]),
            left_index=True, right_index=True
        )
        comb_scn = pd.concat([df_hist_det, fut_base_scn], ignore_index=True)
        comb_design_scn = self._build_nowcast_design_no_drop(comb_scn, cfg)
        fut_designs_scn = comb_design_scn.loc[comb_design_scn["Q"] > last_Q].reset_index(drop=True)

        # --------------------------------------------------------------------------
        # LagGuard (H1)
        # --------------------------------------------------------------------------
        try:
            Ls_cfg = ScenarioAnalysis._month_lags_to_quarter_lags(cfg.exog_month_lags)
            Ls = sorted({int(abs(l)) for l in Ls_cfg if int(abs(l)) != 0}) or [1, 2, 4]
            base_exog = [c for c in df_hist_det.columns
                        if c not in {"Q", "Q_end", cfg.target_col} and "__" in c and
                            not c.startswith(("DET_", "SEAS_", "TARGET__lag-"))]
            if not fut_designs_base.empty:
                ok_cnt = warn_cnt = 0
                for col in base_exog[:12]:
                    for L in Ls:
                        lagc = f"{col}__lag-{L}Q"
                        if lagc in fut_designs_base.columns and col in df_hist_det.columns:
                            got = fut_designs_base.loc[0, lagc]
                            ref_q = last_Q - (L - 1)
                            ref_series = df_hist_det.loc[df_hist_det["Q"] == ref_q, col]
                            if len(ref_series):
                                refv = float(ref_series.iloc[0])
                                ok = (pd.notna(got) and np.isfinite(got) and np.isclose(float(got), refv))
                                if ok:
                                    if ok_cnt < 4:
                                        self._log(f"[LagGuard] OK  {lagc}: H1={float(got):.6g} == hist@{ref_q}={refv:.6g}")
                                    ok_cnt += 1
                                else:
                                    self._log(f"[LagGuard] WARN {lagc}: H1={float(got) if pd.notna(got) else np.nan:.6g} "
                                            f"≠ hist@{ref_q}={refv:.6g}")
                                    warn_cnt += 1
                if warn_cnt == 0 and ok_cnt == 0:
                    self._log("[LagGuard] Keine passenden Kandidaten gefunden (nichts zu prüfen).")
        except Exception as _e:
            self._log(f"[LagGuard] skipped ({_e})")

        # --------------------------------------------------------------------------
        # Sichtprüfung (wirkt Szenario?)
        # --------------------------------------------------------------------------
        any_diff_used = False
        try:
            fi_arr = getattr(self.model, "feature_importances_", None)
            used_cols = (
                [c for c, imp in zip(X_cols, (fi_arr if fi_arr is not None else np.zeros(len(X_cols)))) if float(imp) > 1e-4]
                or X_cols[:min(24, len(X_cols))]
            )
            exog_in_X = [c for c in X_cols if "__" in c and not c.startswith(("DET_", "SEAS_", "TARGET__lag-"))]
            inter = sorted(set(exog_in_X) & set(fut_designs_base.columns) & set(fut_designs_scn.columns))

            self._log(f"[ScenarioForecast][Sichtprüfung] X_cols={len(X_cols)}, used≈{len(used_cols)}, EXOG_in_X={len(exog_in_X)}")
            self._log(f"[ScenarioForecast][Sichtprüfung] EXOG∩Design={len(inter)} → {inter[:12]}")

            def _v(df, r, c):
                try: return float(df.loc[r, c])
                except Exception: return np.nan

            self._log("[ScenarioForecast][Sichtprüfung] Diff der genutzten X-Spalten (H1/H2):")
            for c in used_cols:
                b1, s1 = _v(fut_designs_base, 0, c), _v(fut_designs_scn, 0, c)
                b2 = _v(fut_designs_base, 1, c) if len(fut_designs_base) > 1 else np.nan
                s2 = _v(fut_designs_scn, 1, c) if len(fut_designs_scn) > 1 else np.nan
                flag = "≠" if (not np.isclose(b1, s1, equal_nan=True) or not np.isclose(b2, s2, equal_nan=True)) else "=="
                if flag == "≠": any_diff_used = True
                self._log(f"  {c:40s} | H1 base={b1:.6g} scen={s1:.6g} | H2 base={b2:.6g} scen={s2:.6g}  {flag}")
        except Exception as e:
            self._log(f"[ScenarioForecast][Sichtprüfung] DEBUG-Block Fehler: {e}")

        # Optional: Perturbation-Test
        if getattr(cfg, "debug_perturbation", False) and fi is not None and len(fi):
            try:
                exog_pairs = [(imp, c) for c, imp in zip(self.X_cols, fi)
                            if not c.startswith(("DET_", "SEAS_", "TARGET__lag-"))]
                if exog_pairs:
                    top_exog = max(exog_pairs)[1]
                    row = fut_designs_base.iloc[0][self.X_cols].copy()
                    eps = (abs(row[top_exog]) or 1.0) * 0.05
                    row2 = row.copy(); row2[top_exog] = row[top_exog] + eps
                    y0 = self._predict_nowcast_t0(self.model, self.tj, row)
                    y1 = self._predict_nowcast_t0(self.model, self.tj, row2)
                    self._log(f"[DEBUG] Perturb {top_exog} +5% → ΔY={y1 - y0:+.3f}")
            except Exception as _e:
                self._log(f"[DEBUG] Perturbation ERROR: {_e}")

        # --------------------------------------------------------------------------
        # D) Vorhersagen (Baum) – rekursiv & Batch + Plateau-Guard (Baseline)
        # --------------------------------------------------------------------------
        y_fut_base = self._recursive_forecast_multi(
            self.model, self.tj, fut_designs_base, self.X_cols, df_hist_det, cfg
        )

        # Baseline-Diagnostik
        y_batch = None
        Xb = None
        try:
            Xb = fut_designs_base[self.X_cols]
            y_batch = self.model.predict(Xb.to_numpy())  # reine Batch-Prediction
            self._log(f"[BaselineDiag] batch_pred unique (rounded 3): {np.unique(np.round(y_batch, 3))[:10]}")
            self._log(f"[BaselineDiag] rec_pred  unique (rounded 3): {np.unique(np.round(y_fut_base, 3))[:10]}")
            self._log(f"[BaselineDiag] batch_vs_rec allclose(1e-9): {np.allclose(y_batch, y_fut_base, atol=1e-9)}")
            self._log(f"[BaselineDiag] rec std={np.std(y_fut_base):.6f} | "
                    f"diffs={np.array2string(np.diff(y_fut_base), precision=6)}")
        except Exception as e:
            self._log(f"[BaselineDiag] skipped ({e})")

        # Leaf-Werte sichtbar machen
        try:
            if Xb is None:
                Xb = fut_designs_base[self.X_cols]
            if hasattr(self.model, "apply") and hasattr(self.model, "tree_"):
                leaves_b = self.model.apply(Xb.to_numpy())
                leaf_vals = self.model.tree_.value[leaves_b, 0, 0].astype(float)
                self._log(f"[BaselineDiag] leaf_ids (first 8): {leaves_b[:8].tolist()}")
                self._log(f"[BaselineDiag] leaf_vals (first 8, 3dp): {np.round(leaf_vals[:8],3).tolist()}")
                self._log(f"[BaselineDiag] unique leaf_vals (3dp): {np.unique(np.round(leaf_vals,3))[:10]}")
        except Exception as e:
            self._log(f"[BaselineDiag] leaf dump skipped ({e})")

        # Intra-Baseline-Feature-Drift
        try:
            top_cols = []
            fi_local = getattr(self.model, "feature_importances_", None)
            if fi_local is not None:
                order = np.argsort(fi_local)[::-1][:10]
                top_cols = [self.X_cols[i] for i in order]
            else:
                top_cols = self.X_cols[:10]
            var_report = {c: float(np.nanstd(fut_designs_base[c].values[:H])) for c in top_cols if c in fut_designs_base}
            self._log(f"[BaselineDiag] Feature std over H (Top10): { {k: round(v,6) for k,v in var_report.items()} }")
        except Exception as e:
            self._log(f"[BaselineDiag] feature drift skipped ({e})")

        # Plateau-Guard (Baseline): bei flacher rekursiver Spur auf Batch wechseln
        try:
            if y_batch is None:
                Xb_pg = fut_designs_base[self.X_cols]
                y_batch = self.model.predict(Xb_pg.to_numpy())

            abs_span_rec = float(np.nanmax(y_fut_base) - np.nanmin(y_fut_base))
            abs_span_batch = float(np.nanmax(y_batch) - np.nanmin(y_batch))
            mean_level = float(np.nanmean(y_fut_base))

            rel_flat = (abs_span_rec <= max(1e-12, cfg.plateau_guard_rel_tol * max(1.0, abs(mean_level))))
            abs_flat = (abs_span_rec <= cfg.plateau_guard_abs_tol)

            if cfg.plateau_guard_enable and (rel_flat or abs_flat) and (abs_span_batch > abs_span_rec):
                self._log("[PlateauGuard] Rec ist (nahezu) flach → nutze Batch-Vorhersage (Baseline).")
                y_fut_base = np.asarray(y_batch, dtype=float).copy()
        except Exception as e:
            self._log(f"[PlateauGuard] skipped ({e})")

        # --------------------------------------------------------------------------
        # E) Szenario – rekursiv
        # --------------------------------------------------------------------------
        y_fut_scn = self._recursive_forecast_multi(
            self.model, self.tj, fut_designs_scn, self.X_cols, df_hist_det, cfg
        )

        # Identity-Guard: Ist das Szenario identisch zur Baseline?
        try:
            scn_is_baseline = (
                not scenario_future  # Preset=baseline → Szenario leer
                and set(self.X_cols).issubset(fut_designs_base.columns)
                and set(self.X_cols).issubset(fut_designs_scn.columns)
                and np.allclose(
                    fut_designs_base[self.X_cols].to_numpy(dtype=float),
                    fut_designs_scn[self.X_cols].to_numpy(dtype=float),
                    equal_nan=True, atol=1e-12, rtol=0.0,
                )
            )
        except Exception:
            scn_is_baseline = False

        if scn_is_baseline:
            # Wichtig: Nach Plateau-Guard die Baseline übernehmen,
            # damit beide Spuren identisch (und nicht konstant) sind.
            y_fut_scn = np.asarray(y_fut_base, dtype=float).copy()
            self._log("[ScenarioForecast] Szenario ≡ Baseline → kopiere Baseline (inkl. PlateauGuard) auf Szenario.")
        else:
            # Plateau-Guard (Szenario) nur anwenden, wenn Szenario ≠ Baseline
            if cfg.plateau_guard_enable:
                try:
                    y_batch_scn = self.model.predict(fut_designs_scn[self.X_cols].to_numpy())
                    span_rec = float(np.nanmax(y_fut_scn) - np.nanmin(y_fut_scn))
                    span_batch = float(np.nanmax(y_batch_scn) - np.nanmin(y_batch_scn))
                    mean_lvl  = float(np.nanmean(y_fut_scn))
                    rel_flat = (span_rec <= max(1e-12, cfg.plateau_guard_rel_tol * max(1.0, abs(mean_lvl))))
                    abs_flat = (span_rec <= cfg.plateau_guard_abs_tol)
                    if (rel_flat or abs_flat) and (span_batch > span_rec):
                        self._log("[PlateauGuard] Szenario ist (nahezu) flach → nutze Batch-Vorhersage (Szenario).")
                        y_fut_scn = np.asarray(y_batch_scn, dtype=float).copy()
                except Exception as e:
                    self._log(f"[PlateauGuard][Szenario] skipped ({e})")

            # Leaf-ID-Diagnose (zeigt, ob Szenario den Leaf wechselt)
            try:
                if hasattr(self.model, "apply"):
                    leaf_base = self.model.apply(fut_designs_base[self.X_cols].to_numpy())
                    leaf_scn  = self.model.apply(fut_designs_scn[self.X_cols].to_numpy())
                    same = (leaf_base == leaf_scn)
                    sw = int(np.sum(~same))
                    self._log(f"[LeafDiag] unterschiedliche Leaves (Szenario vs. Base): {sw}/{len(same)}")
                    for i in range(min(len(same), 4)):
                        self._log(f"[LeafDiag] H{i+1}: base={int(leaf_base[i])} | scen={int(leaf_scn[i])} | "
                                f"{'gleich' if same[i] else '≠'}")
            except Exception as _e:
                self._log(f"[LeafDiag] skipped ({_e})")

            # Outlier-Cap (Tree-only), ggf. relaxen
            meta = self.metadata or {}
            cv_rmse = ((meta.get("cv_performance") or {}).get("cv_rmse")) or None
            try:
                cv_rmse = float(cv_rmse) if cv_rmse is not None else None
            except Exception:
                cv_rmse = None
            if outlier_cap_mode not in {"none", "tree_clip"}:
                outlier_cap_mode = "tree_clip"

            if (scenario_future and any_diff_used and exog_importance_share >= cap_relax_thr
                    and outlier_cap_mode == "tree_clip"):
                self._log("[ScenarioForecast] Outlier-Cap deaktiviert (Δ in genutzten EXOG + ausreichender EXOG-Share).")
                outlier_cap_mode = "none"

            if outlier_cap_mode != "none":
                for i in range(H):
                    base_i = float(y_fut_base[i]); scn_i = float(y_fut_scn[i])
                    delta_tree = scn_i - base_i
                    thr_candidates = []
                    if cv_rmse and cv_rmse > 0:
                        thr_candidates.append(2.0 * cv_rmse)
                    thr_candidates.append(0.10 * max(1.0, abs(base_i)))
                    thr = min(thr_candidates) if thr_candidates else 0.10 * max(1.0, abs(base_i))
                    if abs(delta_tree) > thr and scenario_future:
                        y_old = y_fut_scn[i]
                        y_fut_scn[i] = float(np.clip(scn_i, base_i - thr, base_i + thr))
                        log(f"[ScenarioForecast] OUTLIER CAP (Tree-Only) @H{i + 1}: Δ_tree={delta_tree:+.3f} "
                            f"> thr={thr:.3f} → clip to [{base_i - thr:.3f}, {base_i + thr:.3f}]; "
                            f"y {y_old:.3f} → {y_fut_scn[i]:.3f}")

            # Leaf-Linear-Fallback: nur wenn Tree stückweise konstant bleibt
            def _is_flat(y):
                if y is None or len(y) == 0: return True
                return np.allclose(y, np.full_like(y, y[0]), atol=1e-9)

            if enable_leaf_linear and scenario_future and (
                np.allclose(y_fut_scn, y_fut_base, atol=1e-9) or (_is_flat(y_fut_base) and _is_flat(y_fut_scn))
            ):
                try:
                    # Trainingsdesign
                    df_train = getattr(self, "_df_feats", None)
                    if df_train is None or df_train.empty:
                        df_train = comb_design.loc[comb_design["Q"] <= last_Q].copy()

                    # Regressions-Features wählen
                    if fi is not None:
                        used_cols_full = [c for c, imp in zip(X_cols, fi) if float(imp) > 1e-4]
                    else:
                        used_cols_full = X_cols

                    if leaf_linear_exog_only:
                        reg_cols = [c for c in used_cols_full
                                    if "__" in c and not c.startswith(("DET_","SEAS_","TARGET__lag-"))]
                        if not reg_cols:
                            reg_cols = [c for c in X_cols if "__" in c and not c.startswith(("DET_","SEAS_","TARGET__lag-"))]
                    else:
                        reg_cols = used_cols_full

                    # Limit auf sinnvolle Größe
                    reg_cols = reg_cols[:min(len(reg_cols), 16)]
                    df_tr = df_train.dropna(subset=reg_cols + [cfg.target_col]).copy()
                    if df_tr.empty:
                        self._log("[LeafLinear] Kein Trainingsfenster verfügbar – skip.")
                    else:
                        X_tr_full = df_train.dropna(subset=X_cols + [cfg.target_col])
                        X_trA = X_tr_full[X_cols].to_numpy()
                        y_trA = X_tr_full[cfg.target_col].to_numpy()

                        # Leaf-IDs Training (falls möglich)
                        leaf_tr = None
                        if hasattr(self.model, "apply"):
                            try:
                                leaf_tr = self.model.apply(X_trA)
                            except Exception:
                                leaf_tr = None

                        adj = np.zeros(H, dtype=float)

                        for i in range(H):
                            row_b = fut_designs_base.iloc[i][X_cols]
                            row_s = fut_designs_scn.iloc[i][X_cols]

                            Z = df_tr[reg_cols].to_numpy(dtype=float)
                            yZ = df_tr[cfg.target_col].to_numpy(dtype=float)

                            # Lokales Fenster
                            idx = None; leaf_b = None
                            if leaf_tr is not None:
                                try:
                                    leaf_b = int(self.model.apply(row_b.to_numpy(dtype=float).reshape(1, -1))[0])
                                    idx = (leaf_tr == leaf_b)
                                except Exception:
                                    idx = None

                            if leaf_linear_method == "knn":
                                if Z.shape[0] < 5:
                                    self._log("[LeafLinear] Zu wenige Trainingspunkte – skip.")
                                    continue
                                xb = row_b[reg_cols].to_numpy(dtype=float)
                                d = np.linalg.norm(Z - xb, axis=1)
                                k = min(leaf_linear_k, Z.shape[0])
                                nn = np.argpartition(d, k-1)[:k]
                                Zloc = Z[nn]; yloc = yZ[nn]
                                source = f"KNN{k}"
                            else:  # "ols" (oder Leaf-Cluster sofern genug Punkte)
                                if idx is not None and np.sum(idx) >= max(leaf_linear_k, 25):
                                    Zloc = X_trA[idx][:, [X_cols.index(c) for c in reg_cols]]
                                    yloc = y_trA[idx]
                                    source = f"leaf#{leaf_b} ({np.sum(idx)} obs)"
                                else:
                                    Zloc = Z; yloc = yZ
                                    source = "OLS(all)"

                            # Standardisierung/Z-Score falls gewünscht
                            mu = Zloc.mean(axis=0); sd = Zloc.std(axis=0)
                            sd[sd == 0] = 1.0
                            Zs = (Zloc - mu) / sd if leaf_linear_use_zscore else Zloc
                            dx_raw = (row_s[reg_cols].to_numpy(dtype=float) - row_b[reg_cols].to_numpy(dtype=float))
                            dxs = (dx_raw / sd) if leaf_linear_use_zscore else dx_raw

                            # Trigger nur bei ausreichendem Signal (z-score)
                            if leaf_linear_use_zscore and np.nanmax(np.abs(dxs)) < leaf_linear_min_abs_z:
                                self._log(f"[LeafLinear] H{i+1}: |ΔX|_z<{leaf_linear_min_abs_z:.3f} → skip.")
                                continue

                            # Lineare Projektion (pseudoinverse)
                            try:
                                beta = np.linalg.pinv(Zs) @ yloc
                            except Exception:
                                beta = np.linalg.lstsq(Zs, yloc, rcond=None)[0]
                            delta_y = float(beta @ dxs)

                            # Cap stabilisieren (RMSE & %-Level)
                            base_pred = float(y_fut_base[i])
                            thr_list = []
                            meta = self.metadata or {}
                            cv_rmse = ((meta.get("cv_performance") or {}).get("cv_rmse")) or None
                            try:
                                cv_rmse = float(cv_rmse) if cv_rmse is not None else None
                            except Exception:
                                cv_rmse = None
                            if cv_rmse and cv_rmse > 0:
                                thr_list.append(leaf_linear_cap_rmse_mult * cv_rmse)
                            thr_list.append(leaf_linear_cap_pct_of_level * max(1.0, abs(base_pred)))
                            thr = min(thr_list) if thr_list else leaf_linear_cap_pct_of_level * max(1.0, abs(base_pred))

                            delta_y_c = float(np.clip(delta_y, -thr, +thr))
                            adj[i] = delta_y_c
                            self._log(f"[LeafLinear] H{i+1}: src={source} | Δy_raw={delta_y:+.3f} → capped={delta_y_c:+.3f} (thr={thr:.3f})")

                        y_fut_scn = (y_fut_base + adj)
                        self._log(f"[LeafLinear] angewandt. Mean(Δ)={np.mean(adj):+.3f}")
                except Exception as _e:
                    self._log(f"[LeafLinear] skipped ({_e})")

        # --------------------------------------------------------------------------
        # F) Ausgabe
        # --------------------------------------------------------------------------
        fut_Q = comb_design.loc[comb_design["Q"] > last_Q, "Q"].iloc[:H].astype(str).tolist()
        out = pd.DataFrame({
            "Quarter": fut_Q,
            "Forecast_Baseline": y_fut_base,
            "Forecast_Scenario": y_fut_scn,
            "Delta_abs": y_fut_scn - y_fut_base,
            "Delta_pct": np.where(y_fut_base != 0,
                                (y_fut_scn - y_fut_base) / y_fut_base * 100.0,
                                np.nan)
        })

        results = {
            "Q_t0": str(last_Q),
            "target": cfg.target_col,
            "yhat_baseline_t0": float(self._predict_nowcast_t0(
                self.model, self.tj, df_feats[self.X_cols].iloc[-1].copy()
            )),
            "horizon_Q": H,
            "table": out.to_dict(orient="records"),
            "scenario_cols": list((scenario_future or {}).keys()),
        }

        if persist:
            os.makedirs(cfg.output_dir, exist_ok=True)
            path = os.path.join(cfg.output_dir, f"forecast_{cfg.target_col}_{H}Q.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results




    # ---------- (Optionale) Helfer für UI ----------

    @staticmethod
    def decay_path(initial: float, H: int = 4, half_life_q: float = 2.0) -> List[float]:
        """Geometrischer Abbau (hilfreich für realistische Szenario-Pfade)."""
        lam = 0.5 ** (1.0 / half_life_q)
        v = float(initial)
        out = []
        for _ in range(H):
            v *= lam
            out.append(round(v, 3))
        return out

    # ---------- Interne Helfer (Feature Engineering) ----------

    @staticmethod
    def _read_excel(cfg: Config) -> pd.DataFrame:
        df = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_name)
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        return df.sort_values(cfg.date_col).reset_index(drop=True)

    @staticmethod
    def _exog_base_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
        present = [c for c in cfg.exog_fixed_list if c in df.columns]
        if len(present) < len(cfg.exog_fixed_list):
            missing = [c for c in cfg.exog_fixed_list if c not in df.columns]
            raise ValueError(f"Folgende Exogen-Spalten fehlen im Sheet '{cfg.sheet_name}': {missing}")
        non_num = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            raise ValueError(f"Diese Exogen-Spalten sind nicht numerisch: {non_num}")
        return present

    @staticmethod
    def _aggregate_to_quarter(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        df = df.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

        exogs = ScenarioAnalysis._exog_base_columns(df, cfg)
        df["Q"] = df[cfg.date_col].dt.to_period("Q").astype("period[Q]")
        df = df.sort_values(cfg.date_col)

        if cfg.agg_method_target == "mean":
            yq = df.groupby("Q", as_index=True)[cfg.target_col].mean()
        elif cfg.agg_method_target == "last":
            yq = df.groupby("Q", as_index=True)[cfg.target_col].apply(lambda s: s.ffill().iloc[-1])
        else:
            raise ValueError("Unsupported agg_method_target")

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
                raise ValueError(f"Unsupported agg method: {method}")
            part.columns = [f"{c}__{method}" for c in part.columns]
            parts.append(part)

        Xq = pd.concat(parts, axis=1)
        out = pd.concat([yq.rename(cfg.target_col), Xq], axis=1).reset_index()
        out["Q_end"] = out["Q"].dt.to_timestamp(how="end")
        return out.sort_values("Q").reset_index(drop=True)

    @staticmethod
    def _add_deterministic_features(df_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        df = df_q.copy().sort_values("Q").reset_index(drop=True)
        n = len(df)

        if cfg.add_trend_features:
            t = np.arange(n, dtype=float)
            df["DET_trend_t"] = t
            for d in range(2, max(2, cfg.trend_degree) + 1):
                df[f"DET_trend_t{d}"] = t ** d

        if cfg.add_seasonality:
            if cfg.seasonality_mode.lower() != "dummies":
                raise ValueError("Aktuell wird nur seasonality_mode='dummies' unterstützt.")
            qnum = df["Q"].dt.quarter.astype(int)
            for q in [1, 2, 3]:
                df[f"SEAS_Q{q}"] = (qnum == q).astype(int)
        return df

    @staticmethod
    def _month_lags_to_quarter_lags(month_lags: List[int]) -> List[int]:
        q_lags = []
        for m in month_lags:
            q = -int(np.ceil(abs(int(m)) / 3))
            q_lags.append(q)
        return sorted(set(q_lags))

    @staticmethod
    def _build_nowcast_design(df_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        df_q = df_q.sort_values("Q").reset_index(drop=True)
        # Guard: Zeitachse muss streng aufsteigend sein
        try:
            assert df_q["Q"].is_monotonic_increasing
        except Exception:
            raise AssertionError("_build_nowcast_design: 'Q' ist nicht strikt aufsteigend – bitte prüfen!")

        det_cols = [c for c in df_q.columns if c.startswith(("DET_", "SEAS_"))]
        ignore = {"Q", "Q_end", cfg.target_col}
        exog_base_cols = [c for c in df_q.columns
                          if c not in ignore and "__" in c and not c.startswith(("DET_", "SEAS_"))]

        out = df_q[["Q", "Q_end", cfg.target_col]].copy()
        for c in det_cols:
            out[c] = df_q[c]

        # Monat→Quartal Lags: immer als positive L interpretieren, Name = __lag-{L}Q
        exog_q_lags_raw = ScenarioAnalysis._month_lags_to_quarter_lags(cfg.exog_month_lags)
        exog_q_lags = sorted(set(int(abs(x)) for x in exog_q_lags_raw if x != 0))  # L ∈ {1,2,4,…}

        for col in exog_base_cols:
            out[col] = df_q[col]
            for L in exog_q_lags:
                out[f"{col}__lag-{L}Q"] = df_q[col].shift(L)  # eindeutige Lag-Richtung

        if cfg.target_lags_q:
            for L in sorted(set(int(abs(x)) for x in cfg.target_lags_q if x >= 1)):
                out[f"TARGET__lag-{L}Q"] = df_q[cfg.target_col].shift(L)

        return out.dropna().reset_index(drop=True)


    # ---------- Interne Helfer (Modell & Metriken) ----------

    class _TargetYJ:
        def __init__(self, standardize: bool = True):
            self.pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
            self.fitted = False

        def fit(self, y: np.ndarray):
            y = np.asarray(y).reshape(-1, 1)
            self.pt.fit(y)
            self.fitted = True
            return self

        def transform(self, y: np.ndarray) -> np.ndarray:
            assert self.fitted
            return self.pt.transform(np.asarray(y).reshape(-1, 1)).ravel()

        def inverse(self, y_t: np.ndarray) -> np.ndarray:
            assert self.fitted
            return self.pt.inverse_transform(np.asarray(y_t).reshape(-1, 1)).ravel()

    @staticmethod
    def _expanding_splits(n: int, min_train: int, gap: int):
        for test_end in range(min_train + gap, n):
            train_end = test_end - gap
            train_idx = list(range(0, train_end))
            test_idx = [test_end]
            yield train_idx, test_idx
            
    @staticmethod
    def _train_best_model_nowcast(df_feats: pd.DataFrame, cfg: Config):
        """
        Nur DecisionTreeRegressor, aber exog-bewusste Auswahl:
        - Zuerst CV-RMSE über Grid (expanding split).
        - Dann Full-Fit der besten Kandidaten, Messung EXOG-Share.
        - Auswahl via score = CV_RMSE * (1 + alpha * max(0, min_exog_share - exog_share)).
        → Bevorzugt Modelle, die Exogene wirklich nutzen.
        """
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import ParameterGrid
        import numpy as np

        # -------------------- Daten vorbereiten --------------------
        y = df_feats[cfg.target_col].values
        X_cols = [c for c in df_feats.columns if c not in ["Q", "Q_end", cfg.target_col]]
        X = df_feats[X_cols].values
        n = len(df_feats)

        # Falls param_grid leer/fehlt → sinnvolle Defaults nur für DT
        default_grid = {
            "max_depth": [3, 4, 5, 6, 8, None],
            "min_samples_leaf": [3, 5, 8, 12],
            "min_impurity_decrease": [0.0, 1e-4, 1e-3],
            "splitter": ["best"],  # "random" optional, meist schwächer bei TimeSeries
            # "max_features" existiert bei DT; meist None gut für kleine n
        }
        user_grid = getattr(cfg, "param_grid", None)
        if not user_grid:
            grid = list(ParameterGrid(default_grid))
        else:
            grid = list(ParameterGrid(user_grid))
        LOGGER.info("  Grid-Search mit %s Kombinationen …", len(grid))

        # -------------------- CV mit expanding split --------------------
        def _expanding_splits(n_: int, min_train: int, gap: int):
            for test_end in range(min_train + gap, n_):
                train_end = test_end - gap
                yield list(range(0, train_end)), [test_end]

        results = []  # sammelt (cv_rmse, params)

        for params in grid:
            preds = np.full(n, np.nan, dtype=float)
            for tr, te in ScenarioAnalysis._expanding_splits(n, cfg.min_train_quarters, cfg.gap_quarters):
                model = DecisionTreeRegressor(random_state=cfg.random_state, **params)
                if str(cfg.target_transform).lower() == "yeo-johnson":
                    tj_temp = ScenarioAnalysis._TargetYJ(standardize=cfg.target_standardize).fit(y[tr])
                    y_tr_t = tj_temp.transform(y[tr])
                    model.fit(X[tr], y_tr_t)
                    yhat_t = model.predict(X[te])
                    preds[te] = tj_temp.inverse(yhat_t)
                else:
                    model.fit(X[tr], y[tr])
                    preds[te] = model.predict(X[te])

            mask = ~np.isnan(preds)
            if mask.sum() < 3:
                continue
            rmse = float(np.sqrt(mean_squared_error(y[mask], preds[mask])))
            results.append((rmse, params))

        if not results:
            # robuster Fallback
            best_params = {"max_depth": 3, "min_samples_leaf": 8}
            model = DecisionTreeRegressor(random_state=cfg.random_state, **best_params)
            tj = None
            if str(cfg.target_transform).lower() == "yeo-johnson":
                tj = ScenarioAnalysis._TargetYJ(standardize=cfg.target_standardize).fit(y)
                y_t = tj.transform(y); model.fit(X, y_t)
            else:
                model.fit(X, y)
            return model, tj, X_cols, best_params, float("inf")

        # -------------------- Exog-bewusste Auswahl --------------------
        # sortiert nach CV-RMSE
        results.sort(key=lambda t: t[0])

        # Tuning der Exog-Gewichtung (über Config steuerbar)
        min_exog_share = float(getattr(cfg, "min_exog_share", 0.08))       # Ziel: ≥8% Exog-Importance
        exog_penalty_alpha = float(getattr(cfg, "exog_penalty_alpha", 0.5))# Gewicht der Penalty
        top_k_exog_check = int(getattr(cfg, "top_k_exog_check", 25))       # wie viele Kandidaten genauer prüfen

        def _fit_full_and_exog_share(params):
            """Fit auf Full-Sample (für Importance) + EXOG-Share messen."""
            m = DecisionTreeRegressor(random_state=cfg.random_state, **params)
            tj_local = None
            if str(cfg.target_transform).lower() == "yeo-johnson":
                tj_local = ScenarioAnalysis._TargetYJ(standardize=cfg.target_standardize).fit(y)
                y_t = tj_local.transform(y); m.fit(X, y_t)
            else:
                m.fit(X, y)

            fi = getattr(m, "feature_importances_", None)
            if fi is None:
                return m, tj_local, 0.0
            fi = np.asarray(fi, dtype=float)
            if fi.size != len(X_cols):
                # auf Gleichlänge trimmen (defensiv)
                k = min(fi.size, len(X_cols))
                fi = fi[:k]

            exog_mask = np.array([not c.startswith(("DET_", "SEAS_", "TARGET__lag-")) for c in X_cols[:fi.size]], dtype=bool)
            exog_sum = float(fi[exog_mask].sum()) if fi.size else 0.0
            total = float(fi.sum()) if fi.size else 0.0
            share = (exog_sum / total) if total > 0 else 0.0
            return m, tj_local, float(share)

        best_tuple = None  # (score_penalized, cv_rmse, params, model, tj, exog_share)
        for cv_rmse, params in results[:max(1, top_k_exog_check)]:
            m, tj_local, exog_share = _fit_full_and_exog_share(params)
            penalty = 1.0 + exog_penalty_alpha * max(0.0, (min_exog_share - exog_share))
            score = cv_rmse * penalty
            if (best_tuple is None) or (score < best_tuple[0]):
                best_tuple = (score, cv_rmse, params, m, tj_local, exog_share)

        score_pen, best_rmse, best_params, model, tj, exog_share = best_tuple

        try:
            LOGGER.info(
                "  → Gewählt: DT | CV_RMSE=%.3f | EXOG-Share=%.1f%% | PenaltyScore=%.3f | params=%s",
                best_rmse,
                exog_share * 100.0,
                score_pen,
                best_params,
            )
        except Exception:
            pass

        return model, tj, X_cols, best_params, float(best_rmse)


    @staticmethod
    def _create_comprehensive_metadata_nowcast(model, tj, X_cols, best_params, df_feats, cfg):
        y = df_feats[cfg.target_col].values
        X = df_feats[X_cols].values
        n = len(df_feats)
        preds = np.full(n, np.nan, dtype=float)

        params = model.get_params()
        params["random_state"] = cfg.random_state

        for tr, te in ScenarioAnalysis._expanding_splits(n, cfg.min_train_quarters, cfg.gap_quarters):
            m = DecisionTreeRegressor(**params)
            if cfg.target_transform.lower() == "yeo-johnson":
                tj_temp = ScenarioAnalysis._TargetYJ(standardize=cfg.target_standardize).fit(y[tr])
                y_tr_t = tj_temp.transform(y[tr])
                m.fit(X[tr], y_tr_t)
                yhat_t = m.predict(X[te])
                preds[te] = tj_temp.inverse(yhat_t)
            else:
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])

        mask = ~np.isnan(preds)
        y_true, y_pred = y[mask], preds[mask]

        denom = (np.abs(y_true) + np.abs(y_pred))
        denom = np.where(denom == 0, 1e-8, denom)
        smape = 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

        insample_pred = tj.inverse(model.predict(X)) if tj is not None else model.predict(X)

        diagnostics = {
            "n_features": len(X_cols),
            "tree_depth": int(model.get_depth()),
            "n_leaves": int(model.get_n_leaves()),
            "n_splits": int(model.tree_.node_count),
            "top_features": {
                feat: float(imp)
                for feat, imp in zip(X_cols, model.feature_importances_)
                if imp > 0.01
            }
        }

        return {
            "model_type": "DecisionTreeRegressor",
            "timestamp": pd.Timestamp.now().isoformat(),
            "cv_performance": {
                "cv_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if mask.any() else None,
                "cv_mae": float(mean_absolute_error(y_true, y_pred)) if mask.any() else None,
                "cv_smape": float(smape) if mask.any() else None,
                "cv_r2": float(r2_score(y_true, y_pred)) if mask.any() else None,
                "cv_n_folds": int(mask.sum())
            },
            "insample_performance": {
                "rmse": float(np.sqrt(mean_squared_error(y, insample_pred))),
                "mae": float(mean_absolute_error(y, insample_pred)),
                "r2": float(r2_score(y, insample_pred)),
                "n_samples": int(len(y)),
                "_warning": "Optimistisch; CV für realistische Einschätzung nutzen"
            },
            "hyperparameters": best_params,
            "model_complexity": diagnostics,
            "training_data": {
                "n_quarters": len(df_feats),
                "date_range": f"{df_feats['Q'].iloc[0]} to {df_feats['Q'].iloc[-1]}",
                "target_variable": cfg.target_col,
                "exog_fixed": cfg.exog_fixed_list,
            },
            "nowcast_config": {
                "target_transform": cfg.target_transform,
                "exog_lags_months": cfg.exog_month_lags,
                "ar_lags_quarters": cfg.target_lags_q
            }
        }

    # ---------- Interne Helfer (Prediction & Szenario) ----------

    def _predict_nowcast_t0(self, model, tj, X_row: pd.Series) -> float:
        X = (X_row.to_frame().T
             .infer_objects(copy=False)
             .fillna(0.0)
             .to_numpy(dtype=float))
        yhat = model.predict(X)
        if tj is not None:
            yhat = tj.inverse(yhat)
        yhat = float(yhat[0])

        lower = getattr(self.cfg, "target_lower_bound", None)
        if lower is not None:
            yhat = max(yhat, float(lower))
        return yhat

    @staticmethod
    def _apply_scenario_to_Xrow(X_row: pd.Series, adjustments: Dict[str, float]) -> pd.Series:
        X_new = X_row.copy()
        for key, delta in (adjustments or {}).items():
            if key in X_new.index:
                X_new[key] = X_new[key] + float(delta)
        return X_new

    @staticmethod
    def _list_current_scenario_features(df_feats: pd.DataFrame, X_cols: List[str]) -> List[str]:
        feats = []
        for c in X_cols:
            if c.startswith(("DET_", "SEAS_", "TARGET__lag-")):
                continue
            if "__lag" in c:
                continue
            feats.append(c)
        return feats

    @staticmethod
    def _future_quarters(last_Q: pd.Period, H: int) -> pd.DataFrame:
        Q_fut = [last_Q + i for i in range(1, H + 1)]
        df = pd.DataFrame({"Q": Q_fut})
        df["Q_end"] = pd.PeriodIndex(df["Q"], freq="Q").to_timestamp(how="end")
        return df

    @staticmethod
    def _linear_drift_forecast(s: pd.Series, H: int, window_q: int = 8) -> pd.Series:
        s = s.dropna()
        if len(s) < 2:
            return pd.Series([np.nan] * H)
        s_tail = s.tail(window_q)
        y = s_tail.values.astype(float)
        x = np.arange(len(y), dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        try:
            beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        except Exception:
            return pd.Series([np.nan] * H)
        x_fut = np.arange(len(y), len(y) + H, dtype=float)
        y_fut = beta * x_fut + alpha
        return pd.Series(y_fut, index=range(H), dtype=float)

    def _impute_future_exog_quarterly_fixed(
        self,
        df_q_hist: pd.DataFrame,
        exog_cols: List[str],
        H: int,
        strategy: str = "mixed",
        window_q: int = 8,
        seas_p: int = 4,
    ) -> pd.DataFrame:
        """
        Exogene Pfade für H zukünftige Quartale, ausschließlich auf Basis der Historie.
        Strategien:
        - 'last'     : Konstanthalten des letzten beobachteten Werts.
        - 'seasonal' : Replikation der Vorjahresquartale (additive Saisonannahme).
        - 'drift'    : Linearer Drift (Deterministischer Trend aus Historie).
        - 'mixed'    : Saison (wie 'seasonal') + Drift der saisonalen Residuen (additiv, geschrumpft).

        Hinweise:
        * Saisonalität ist so indexiert, dass Q+1 dem Wert des gleichen Quartals im Vorjahr entspricht:
            bei seas_p=4 → Q+1 = hist.iloc[-3], Q+4 = hist.iloc[0].
        * NaN-Forecasts werden deterministisch mit dem letzten gültigen Level gefüllt.
        * 'mixed' addiert den gedrifteten Residuenpfad zur Saison – mit Shrinkage, um Überschwingen zu vermeiden.
        * Anschließend variablenabhängiges Soft-Clipping (Raten vs. Level).
        """
        assert H >= 1, "H (Forecast-Horizont) muss >= 1 sein."
        fut = self._future_quarters(df_q_hist["Q"].iloc[-1], H)

        # defensiv
        strategy = (strategy or "mixed").lower()
        if strategy not in {"last", "seasonal", "drift", "mixed"}:
            strategy = "mixed"

        # Shrinkage-Faktor für Residuen-Drift (senkt Überziehen bei Raten)
        # per Config überschreibbar, Standard konservativ
        try:
            resid_drift_shrink = float(getattr(self.cfg, "future_exog_resid_drift_shrink", 0.35))
        except Exception:
            resid_drift_shrink = 0.35
        resid_drift_shrink = max(0.0, min(1.0, resid_drift_shrink))

        for c in exog_cols:
            if c not in df_q_hist.columns:
                fut[c] = [np.nan] * H
                continue

            hist = pd.to_numeric(df_q_hist[c], errors="coerce")
            hn = hist.dropna()
            lastv = float(hn.iloc[-1]) if hn.size else np.nan

            if strategy == "last":
                fut[c] = [lastv] * H

            elif strategy == "seasonal":
                vals = []
                for h in range(1, H + 1):
                    src_idx = -seas_p + h  # bei seas_p=4: Q+1→-3, Q+4→0
                    try:
                        vals.append(float(hist.iloc[src_idx]))
                    except Exception:
                        vals.append(lastv)
                fut[c] = vals

            elif strategy == "drift":
                fc = self._linear_drift_forecast(hist, H, window_q=window_q)
                fut[c] = fc.fillna(lastv).astype(float).values

            else:  # "mixed" = Saison + (geschrumpfter) Drift der saisonalen Residuen
                # 1) Saison-Komponente
                seas_vals = []
                for h in range(1, H + 1):
                    src_idx = -seas_p + h
                    try:
                        seas_vals.append(float(hist.iloc[src_idx]))
                    except Exception:
                        seas_vals.append(np.nan)
                seas_vals = pd.Series(seas_vals, index=range(H), dtype=float).fillna(lastv)

                # 2) Residuen (Level - Vorjahresquartal) und deren Drift
                resid = (hist.astype(float) - hist.astype(float).shift(seas_p)).dropna()
                if resid.size >= 2:
                    drift_resid = self._linear_drift_forecast(
                        resid, H, window_q=min(window_q, len(resid))
                    ).fillna(0.0)
                else:
                    drift_resid = pd.Series([0.0] * H, index=range(H), dtype=float)

                # Shrinkage anwenden
                drift_resid = drift_resid * resid_drift_shrink
                fut[c] = (seas_vals.values + drift_resid.values)

            # ---- Soft-Clipping: variable-spezifische Plausibilitätsgrenzen ----
            # Heuristik anhand Basisnamen (vor "__")
            base = c.split("__", 1)[0] if "__" in c else c
            series = pd.Series(fut[c], index=range(H), dtype=float)

            if any(k in base.lower() for k in ["rate", "zins", "interest", "inflation", "arbeitslosen", "unemploy"]):
                # Raten in Prozentpunkten: beschneide moderat um den letzten Wert
                if pd.notna(lastv):
                    lo, hi = lastv - 1.5, lastv + 1.5
                    # nicht-negative Grenzen für u.a. Arbeitslosenquote
                    if "arbeitslosen" in base.lower() or "unemploy" in base.lower():
                        lo = max(0.0, lo)
                    series = series.clip(lower=lo, upper=hi)
            else:
                # Level-/Indexgrößen: konservativ um den letzten Level
                if pd.notna(lastv):
                    # BIP/Preisindizes: ±5 % (kann über cfg angepasst werden)
                    pct = float(getattr(self.cfg, "future_exog_level_clip_pct", 0.05))
                    lo, hi = (1.0 - pct) * lastv, (1.0 + pct) * lastv
                    # Level nie negativ
                    lo = min(lo, lastv) if lastv >= 0 else lo
                    series = series.clip(lower=0.0 if lastv >= 0 else lo, upper=hi)

            fut[c] = series.values

        return fut

    @staticmethod
    def _extend_deterministics_for_future(df_hist_det: pd.DataFrame, fut: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        out = fut.copy()
        if cfg.add_seasonality:
            qnum = out["Q"].dt.quarter.astype(int)
            for q in [1, 2, 3]:
                out[f"SEAS_Q{q}"] = (qnum == q).astype(int)
        if cfg.add_trend_features:
            t0 = len(df_hist_det)
            t = np.arange(t0, t0 + len(out), dtype=float)
            out["DET_trend_t"] = t
            for d in range(2, max(2, cfg.trend_degree) + 1):
                out[f"DET_trend_t{d}"] = t ** d
        return out

    @staticmethod
    def _apply_future_scenario_on_exogs(
        fut_exog: pd.DataFrame,
        adjustments: Optional[Dict[str, object]],
        H: int
    ) -> pd.DataFrame:
        if not adjustments:
            return fut_exog
        fut = fut_exog.copy()
        for col, adj in adjustments.items():
            if col not in fut.columns:
                continue
            if isinstance(adj, (int, float, np.floating)):
                fut[col] = fut[col].astype(float) + float(adj)
            elif isinstance(adj, (list, tuple)) and len(adj) == H:
                fut[col] = fut[col].astype(float) + pd.Series(adj, index=fut.index, dtype=float).values
            elif isinstance(adj, dict):
                # saubere, sparsame Belegung nur an angegebenen Horizonten (1-basiert)
                s_adj = pd.Series(0.0, index=fut.index, dtype=float)
                for i, v in adj.items():
                    j = int(i) - 1
                    if 0 <= j < len(s_adj):
                        s_adj.iloc[j] = float(v)
                fut[col] = fut[col].astype(float) + s_adj.values
        return fut


    @staticmethod
    def _build_nowcast_design_no_drop(df_q: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        df_q = df_q.sort_values("Q").reset_index(drop=True)
        try:
            assert df_q["Q"].is_monotonic_increasing
        except Exception:
            raise AssertionError("_build_nowcast_design_no_drop: 'Q' ist nicht strikt aufsteigend – bitte prüfen!")

        det_cols = [c for c in df_q.columns if c.startswith(("DET_", "SEAS_"))]
        ignore = {"Q", "Q_end", cfg.target_col}
        exog_base_cols = [c for c in df_q.columns if c not in ignore and "__" in c and not c.startswith(("DET_", "SEAS_"))]

        out = df_q[["Q", "Q_end", cfg.target_col]].copy()
        for c in det_cols:
            out[c] = df_q[c]

        exog_q_lags_raw = ScenarioAnalysis._month_lags_to_quarter_lags(cfg.exog_month_lags)
        exog_q_lags = sorted(set(int(abs(x)) for x in exog_q_lags_raw if x != 0))

        for col in exog_base_cols:
            out[col] = df_q[col]
            for L in exog_q_lags:
                out[f"{col}__lag-{L}Q"] = df_q[col].shift(L)

        if cfg.target_lags_q:
            for L in sorted(set(int(abs(x)) for x in cfg.target_lags_q if x >= 1)):
                out[f"TARGET__lag-{L}Q"] = df_q[cfg.target_col].shift(L)

        return out  # kein Drop

    def _recursive_forecast_multi(
        self,
        model,
        tj,
        fut_designs: pd.DataFrame,
        X_cols: list,
        df_q_hist: pd.DataFrame,
        cfg: Config
    ) -> np.ndarray:
        """
        Rekursive Vorhersage über H Horizonte.
        - Übergibt 1-Zeilen-DataFrame mit Spaltennamen an model.predict (Pipeline-sicher).
        - Setzt TARGET-Lags aus Rolling-Buffer.
        - (Neu) Setzt Horizon-Feature(s), falls vorhanden.
        - Optional: Leaf-Debug auch für Pipelines.
        """
        import numpy as np
        import pandas as pd
        from collections import deque

        H = len(fut_designs)
        if H == 0:
            return np.array([])

        target_lags = getattr(cfg, "target_lags_q", []) or []
        max_L = max([int(abs(L)) for L in (target_lags if len(target_lags) else [1])])

        y_hist = df_q_hist[cfg.target_col].dropna().astype(float).values
        buf = deque(y_hist[-max_L:].tolist(), maxlen=max_L)

        # Spaltenordnung beibehalten (wenn Modell welche kennt)
        model_cols = list(getattr(model, "feature_names_in_", []))
        predict_cols = model_cols if (model_cols and set(model_cols) <= set(fut_designs.columns)) else X_cols[:]

        # Hilfsfunktionen für Leaf-Debug (funktioniert mit Pipeline oder Estimator)
        def _apply_leaf(mdl, Xdf: pd.DataFrame):
            est = mdl
            Xt = Xdf
            try:
                if hasattr(mdl, "steps"):
                    # alles außer letztem Schritt = Preprocessor
                    pre = mdl[:-1]
                    est = mdl[-1]
                    Xt = pre.transform(Xdf)
                if hasattr(est, "apply"):
                    leaf_id = int(est.apply(Xt)[0])
                    # Leaf-Wert (Durchschnitt im Leaf)
                    try:
                        val = float(est.tree_.value[leaf_id, 0, 0])
                    except Exception:
                        val = np.nan
                    return leaf_id, val
            except Exception:
                pass
            return None, None

        preds: list[float] = []
        lower = getattr(cfg, "target_lower_bound", None)

        for h in range(H):
            # Reihe kopieren (nur die erwarteten Spalten)
            row = fut_designs.iloc[h][predict_cols].copy()

            # (NEU) Horizon-Feature(s) befüllen, wenn vorhanden
            # numerischer Index: HORIZON__idx ∈ {1,2,...}
            if "HORIZON__idx" in row.index:
                row["HORIZON__idx"] = float(h + 1)
            # one-hots: HORIZON__is_1, ..., HORIZON__is_32 (beliebig viele)
            for col in row.index:
                if col.startswith("HORIZON__is_"):
                    try:
                        k = int(col.split("_")[-1])
                        row[col] = 1.0 if (h + 1) == k else 0.0
                    except Exception:
                        pass

            # TARGET-Lags aus Buffer einsetzen
            for L in sorted({int(abs(x)) for x in target_lags if x >= 1}):
                col = f"TARGET__lag-{L}Q"
                if col in row.index:
                    try:
                        row[col] = float(buf[-L])
                    except Exception:
                        # falls Buffer noch nicht voll
                        pass

            # 1-Zeilen-DataFrame in exakt der Spaltenreihenfolge
            X_df = pd.DataFrame([row], columns=predict_cols)
            X_df = X_df.infer_objects(copy=False).fillna(0.0).astype(float)

            # Debug Decision Path (deine bestehende Routine, jetzt mit mutierter row)
            if getattr(cfg, "debug_decision_path", False):
                try:
                    self._debug_decision_path_for_row(row, horizon=h + 1, top_k=8)
                except Exception as _e:
                    self._log(f"[DecisionPath] WARN: {str(_e)}")

            # Optional: Leaf-Debug (zeigt, ob wir im selben Leaf landen)
            try:
                leaf_id, leaf_val = _apply_leaf(model, X_df)
                if leaf_id is not None:
                    self._log(f"[Rec] H={h+1} → leaf={leaf_id} | leaf_val={leaf_val:.3f}")
            except Exception:
                pass

            # Vorhersage
            yhat_arr = model.predict(X_df)
            yhat_arr = np.asarray(yhat_arr).reshape(-1)

            if tj is not None:
                try:
                    yhat_arr = tj.inverse(yhat_arr)
                except Exception as _e:
                    self._log(f"[Transform] inverse() failed → raw prediction. err={_e}")

            yhat = float(yhat_arr[0])

            if lower is not None:
                yhat = max(yhat, float(lower))

            preds.append(yhat)
            buf.append(yhat)  # wichtig für die nächsten Lags

        return np.array(preds, dtype=float)




    @staticmethod
    def _estimate_local_betas(df_feats: pd.DataFrame, X_cols: list, cfg: Config, window_q: int = 24) -> Tuple[pd.Series, Dict[str, np.ndarray]]:
        """
        Liefert standardisierte Betas und Skalen (Mittelwerte/Std) für X und y.
        Rückgabe:
          - betas_std: pd.Series (in std-Einheiten von X → std-Einheiten von y)
          - scales: Dict mit 'X_mean','X_std','y_mean','y_std' (np.ndarrays / float)
        """
        z = df_feats.tail(max(8, window_q)).copy()
        z = z.dropna(subset=[cfg.target_col])
        if len(z) < 8:
            return pd.Series(0.0, index=X_cols, dtype=float), {
                "X_mean": np.zeros(len(X_cols)), "X_std": np.ones(len(X_cols)),
                "y_mean": 0.0, "y_std": 1.0
            }

        X = (z[X_cols]
             .infer_objects(copy=False)
             .fillna(0.0)
             .to_numpy(dtype=float))
        y = z[cfg.target_col].to_numpy(dtype=float)

        X_mean = X.mean(0)
        X_std = X.std(0, ddof=1)
        y_mean = float(y.mean())
        y_std = float(y.std(ddof=1))
        X_std_safe = np.where(X_std == 0, 1.0, X_std)
        y_std_safe = y_std if y_std != 0 else 1.0

        Xz = (X - X_mean) / X_std_safe
        yz = (y - y_mean) / y_std_safe

        X_ext = np.c_[np.ones(len(Xz)), Xz]
        try:
            beta_ext, *_ = np.linalg.lstsq(X_ext, yz, rcond=None)
            betas_std = pd.Series(beta_ext[1:], index=X_cols, dtype=float)
        except Exception:
            betas_std = pd.Series(0.0, index=X_cols, dtype=float)

        scales = {"X_mean": X_mean, "X_std": X_std_safe, "y_mean": y_mean, "y_std": y_std_safe}
        return betas_std, scales

    @staticmethod
    def _apply_linear_sensitivity_scenario(
        betas,
        adjustments: dict,
        *,
        scales: Optional[Dict[str, np.ndarray]] = None,
        X_cols: Optional[List[str]] = None
    ) -> float:
        """
        Wendet (standardisierte) Betas auf ΔX an.
        Akzeptiert betas als:
        - pd.Series (Index = Feature-Namen)
        - dict {feature: beta}
        - list/ndarray (dann MUSS X_cols zur Ausrichtung mitgegeben werden)
        - (betas_std: pd.Series, scales: dict)  # direkt aus _estimate_local_betas(...)
        Wenn 'scales' + 'X_cols' gesetzt sind, wird auf Originalskala zurücktransformiert,
        sonst unstandardisiert multipliziert.
        """
        if not adjustments:
            return 0.0

        # --- Tuple direkt aus _estimate_local_betas(...) akzeptieren ---
        if isinstance(betas, tuple) and len(betas) == 2 and hasattr(betas[0], "index"):
            betas, scales = betas  # betas_std: Series, scales: dict

        # --- In Mapping umwandeln ---
        beta_map: Dict[str, float] = {}
        if hasattr(betas, "to_dict"):  # pd.Series
            beta_map = {str(k): float(v) for k, v in betas.to_dict().items()}
        elif isinstance(betas, dict):
            beta_map = {str(k): float(v) for k, v in betas.items()}
        elif isinstance(betas, (list, tuple, np.ndarray)):
            if X_cols is None:
                raise ValueError(
                    "apply_linear_sensitivity_scenario: Wenn 'betas' eine Liste/Array ist, "
                    "muss 'X_cols' zur Ausrichtung mitgegeben werden."
                )
            arr = np.asarray(betas, dtype=float)
            if len(arr) != len(X_cols):
                raise ValueError(
                    f"apply_linear_sensitivity_scenario: Länge betas ({len(arr)}) != len(X_cols) ({len(X_cols)})."
                )
            beta_map = {str(c): float(b) for c, b in zip(X_cols, arr)}
        else:
            raise TypeError(
                "apply_linear_sensitivity_scenario: 'betas' muss Series, dict, list/ndarray "
                "oder (Series, scales)-Tuple sein."
            )

        # --- Standardisierte Variante (falls möglich) ---
        if scales is not None and X_cols is not None and \
           isinstance(scales, dict) and \
           "X_std" in scales and "y_std" in scales:

            X_std = np.asarray(scales["X_std"], dtype=float)
            y_std = float(scales["y_std"]) if float(scales.get("y_std", 1.0)) != 0 else 1.0
            idx = {c: i for i, c in enumerate(X_cols)}

            z_sum = 0.0
            for k, dx in adjustments.items():
                if k in beta_map and k in idx:
                    i = idx[k]
                    xstd = float(X_std[i]) if i < len(X_std) and X_std[i] != 0 else 1.0
                    try:
                        z_sum += beta_map[k] * (float(dx) / xstd)
                    except Exception:
                        pass
            return float(z_sum * y_std)

        # --- Fallback: unstandardisierte Multiplikation ---
        delta = 0.0
        for k, dx in adjustments.items():
            b = beta_map.get(k)
            if b is None:
                continue
            try:
                delta += float(b) * float(dx)
            except Exception:
                pass
        return float(delta)

    # ---------- Model Path ----------

    def _get_model_filepath(self, cfg: "Config" = None) -> str:
        """
        Liefert einen dateibasierten, datensatzsensitiven Cache-Pfad für das Modell.
        """
        import os
        import json
        import hashlib

        cfg = cfg or getattr(self, "cfg", None)
        if cfg is None:
            raise ValueError("ScenarioAnalysis._get_model_filepath: cfg fehlt und self.cfg ist nicht gesetzt.")

        def _as_tuple(x):
            try:
                if x is None: return tuple()
                if isinstance(x, (list, tuple)): return tuple(x)
                return (x,)
            except Exception:
                return tuple()

        config_tuple = (
            getattr(cfg, "target_col", ""),
            _as_tuple(getattr(cfg, "target_candidates", [])),
            _as_tuple(getattr(cfg, "exog_fixed_list", [])),
            getattr(cfg, "agg_method_target", ""),
            _as_tuple(getattr(cfg, "agg_methods_exog", [])),
            _as_tuple(getattr(cfg, "exog_month_lags", [])),
            _as_tuple(getattr(cfg, "target_lags_q", [])),
            bool(getattr(cfg, "add_trend_features", False)),
            int(getattr(cfg, "trend_degree", 1)),
            bool(getattr(cfg, "add_seasonality", False)),
            str(getattr(cfg, "seasonality_mode", "")),
            str(getattr(cfg, "target_transform", "")),
            getattr(cfg, "forecast_horizon_quarters", None),
        )
        config_str = json.dumps(config_tuple, sort_keys=True, default=str)

        data_fp = "NA"
        try:
            excel_path = getattr(cfg, "excel_path")
            st = os.stat(excel_path)
            data_fp = f"{int(st.st_mtime)}|{int(st.st_size)}|{os.path.basename(excel_path)}"
        except Exception:
            pass

        key = f"{config_str}|{data_fp}"
        config_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
        target_name = getattr(cfg, "target_col", "target")
        filename = f"nowcast_{target_name}_{config_hash}.pkl"

        model_dir = getattr(cfg, "model_dir", "models")
        os.makedirs(model_dir, exist_ok=True)

        return os.path.join(model_dir, filename)

# =============================================================================
# BEISPIEL-NUTZUNG
# =============================================================================
if __name__ == "__main__":
    cfg = Config(
        excel_path=str((APP_ROOT / "data" / "output.xlsx").resolve()),
        sheet_name="final_dataset",
        date_col="Datum",
        target_col="Einlagen",
        agg_methods_exog=["last", "mean"],
        agg_method_target="mean",

        # exog_month_lags=[-12, -6, -3, -1],
        # target_lags_q=[1, 2, 4],

        exog_month_lags=[],
        target_lags_q=[],
        add_trend_features=True,
        trend_degree=1,
        add_seasonality=True,
        seasonality_mode="dummies",
        target_transform="none",
        target_standardize=True,
        use_cached_model=True,
        random_state=42,
        output_dir=str((SCENARIO_DIR / "outputs").resolve()),
        model_dir=str((SCENARIO_DIR / "models_scenario").resolve()),  # eigener Ordner
        min_train_quarters=24,
        gap_quarters=1,
        forecast_horizon_quarters=4,
        future_exog_strategy="mixed",
        future_exog_drift_window_q=8,
        future_exog_seasonal_period_q=4,
        exog_fixed_list=["lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"],
        # Optional:
        # target_lower_bound=0.0,
    )

    sa = ScenarioAnalysis(cfg)

    # # t0 Szenario (additive Deltas in Originaleinheiten der aggregierten Spalten)
    # scenario_t0 = {
    #     "lt_interest_rate__last": 0.5,
    #     "inflation__mean": 1.0,
    #     "unemployment__last": 0.2,
    #     "gdp__mean": -0.3,
    #     "property_prices__last": -2.0,
    # }
    # res_now = sa.nowcast_t0(scenario_adjustments=scenario_t0, persist=True)
    # print(json.dumps(res_now, indent=2, ensure_ascii=False))

    # Zukunfts-Pfade für H=4 (Skalar / Liste[L=H] / Dict{1..H})
    scenario_future = {
        # "lt_interest_rate__last": 0.4,  # gleicher Delta für alle 4Q
        # "inflation__mean": [0.6, 0.35, 0.2, 0.1],
        # "unemployment__last": {2: 0.1, 3: 0.15, 4: 0.1},
        # "property_prices__last": [-1.5, -1.1, -0.8, -0.6],
    }
    res_fc = sa.forecast(H=4, scenario_future=scenario_future, persist=True)
    print(pd.DataFrame(res_fc["table"]).to_string(index=False))

# -*- coding: utf-8 -*-
"""
Integration der forecaster_pipeline.py ins Dashboard.
KORRIGIERT: Robustere Import-Guards, klares HAS_PIPELINE-Flag, saubere Logger-Initialisierung.
"""

# -------------------------
# Standard-Imports
# -------------------------
import json
import logging
import os
import sys
import tempfile  # wichtig für NamedTemporaryFile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# -------------------------
# Logger einrichten (ohne doppelte Handler)
# -------------------------
LOGGER_NAME = "GVB_Dashboard"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

def _supports_utf8() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").upper()
    return "UTF-8" in enc or "UTF8" in enc

def _sym(text: str) -> str:
    """
    Ersetzt hübsche Unicode-Symbole durch ASCII-Fallbacks, falls das Terminal
    vermutlich kein UTF-8 rendert (Windows-CMD o.ä.).
    """
    if _supports_utf8():
        return text
    repl = {"✓": "OK", "→": "->", "—": "-", "–": "-", "…": "...", "»": ">>", "«": "<<"}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text

# Zweiter, kurz benannter Logger für den Adapter (wird in den Callbacks genutzt)
LOGGER = logging.getLogger("DashboardForecastAdapter")
if not LOGGER.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Pfade ermitteln und sys.path erweitern
# ---------------------------------------------------------------------------
# Das Verzeichnis, in dem forecast_integration.py liegt (forecaster/)
FORECASTER_DIR = Path(__file__).parent.resolve()

# Das Root-Verzeichnis (parent von forecaster/)
ROOT_DIR = FORECASTER_DIR.parent.resolve()

# Beide Verzeichnisse zum Python-Pfad hinzufügen (falls nicht bereits vorhanden)
for path_to_add in [str(FORECASTER_DIR), str(ROOT_DIR)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

LOGGER.info(f"[Pipeline] Forecaster-Dir: {FORECASTER_DIR}")
LOGGER.info(f"[Pipeline] Root-Dir: {ROOT_DIR}")
LOGGER.info(f"[Pipeline] sys.path erweitert")

# ---------------------------------------------------------------------------
# Pipeline optional importieren (robust mit Unterordner-Support)
# ---------------------------------------------------------------------------
HAS_PIPELINE: bool = False
_PIPELINE_IMPORT_ERROR: Optional[Exception] = None

# Platzhalter; werden bei erfolgreichem Import überschrieben
PipelineConfig = None            # type: ignore[assignment]
ModelArtifact = None             # type: ignore[assignment]
get_model_filepath = None        # type: ignore[assignment]
run_production_pipeline = None   # type: ignore[assignment]

# Wir versuchen mehrere Varianten in sicherer Reihenfolge
try:
    # 1) Direkter Import (jetzt mit Pfad-Fix)
    from forecaster_pipeline import (  # type: ignore
        PipelineConfig,
        run_production_pipeline,
        ModelArtifact,
        get_model_filepath,
    )
    HAS_PIPELINE = True
    LOGGER.info("[Pipeline] ✓ Import erfolgreich (direkter Import)")
except Exception as e1:
    try:
        # 2) Alternativ: Config heißt 'Config' in manchen Repos
        from forecaster_pipeline import (  # type: ignore
            Config as PipelineConfig,
            run_production_pipeline,
            ModelArtifact,
            get_model_filepath,
        )
        HAS_PIPELINE = True
        LOGGER.info("[Pipeline] ✓ Import erfolgreich (Config → PipelineConfig)")
    except Exception as e2:
        try:
            # 3) Relativ (falls als Paket installiert/ausgeführt)
            from .forecaster_pipeline import (  # type: ignore
                PipelineConfig,
                run_production_pipeline,
                ModelArtifact,
                get_model_filepath,
            )
            HAS_PIPELINE = True
            LOGGER.info("[Pipeline] ✓ Import erfolgreich (relativer Import)")
        except Exception as e3:
            try:
                # 4) Relativ mit Config-Alias
                from .forecaster_pipeline import (  # type: ignore
                    Config as PipelineConfig,
                    run_production_pipeline,
                    ModelArtifact,
                    get_model_filepath,
                )
                HAS_PIPELINE = True
                LOGGER.info("[Pipeline] ✓ Import erfolgreich (relativ, Config → PipelineConfig)")
            except Exception as e4:
                try:
                    # 5) Absoluter Import mit forecaster-Prefix
                    from forecaster.forecaster_pipeline import (  # type: ignore
                        PipelineConfig,
                        run_production_pipeline,
                        ModelArtifact,
                        get_model_filepath,
                    )
                    HAS_PIPELINE = True
                    LOGGER.info("[Pipeline] ✓ Import erfolgreich (forecaster.forecaster_pipeline)")
                except Exception as e5:
                    try:
                        # 6) Absoluter Import mit forecaster-Prefix + Config-Alias
                        from forecaster.forecaster_pipeline import (  # type: ignore
                            Config as PipelineConfig,
                            run_production_pipeline,
                            ModelArtifact,
                            get_model_filepath,
                        )
                        HAS_PIPELINE = True
                        LOGGER.info("[Pipeline] ✓ Import erfolgreich (forecaster.forecaster_pipeline, Config → PipelineConfig)")
                    except Exception as e6:
                        # Alle Versuche fehlgeschlagen → Flags setzen und Fehler protokollieren
                        HAS_PIPELINE = False
                        _PIPELINE_IMPORT_ERROR = e6 or e5 or e4 or e3 or e2 or e1
                        PipelineConfig = None            # type: ignore[assignment]
                        ModelArtifact = None             # type: ignore[assignment]
                        get_model_filepath = None        # type: ignore[assignment]
                        run_production_pipeline = None   # type: ignore[assignment]
                        
                        # Detaillierte Fehlerausgabe
                        LOGGER.error("=" * 80)
                        LOGGER.error("[Pipeline] ❌ ALLE Import-Versuche fehlgeschlagen!")
                        LOGGER.error(f"[Pipeline] Forecaster-Dir: {FORECASTER_DIR}")
                        LOGGER.error(f"[Pipeline] Root-Dir: {ROOT_DIR}")
                        LOGGER.error(f"[Pipeline] Aktuelles Verzeichnis: {Path.cwd()}")
                        LOGGER.error(f"[Pipeline] Python-Pfad (erste 5): {sys.path[:5]}")
                        LOGGER.error("=" * 80)
                        LOGGER.error("[Pipeline] Fehler-Details pro Versuch:")
                        LOGGER.error(f"  1) Direkter Import: {type(e1).__name__}: {e1}")
                        LOGGER.error(f"  2) Config-Alias: {type(e2).__name__}: {e2}")
                        LOGGER.error(f"  3) Relativer Import: {type(e3).__name__}: {e3}")
                        LOGGER.error(f"  4) Relativ + Config: {type(e4).__name__}: {e4}")
                        LOGGER.error(f"  5) forecaster.* Import: {type(e5).__name__}: {e5}")
                        LOGGER.error(f"  6) forecaster.* + Config: {type(e6).__name__}: {e6}")
                        LOGGER.error("=" * 80)
                        
                        # Prüfe ob forecaster_pipeline.py existiert
                        pipeline_file = FORECASTER_DIR / "forecaster_pipeline.py"
                        if pipeline_file.exists():
                            LOGGER.error(f"[Pipeline] ✓ Datei existiert: {pipeline_file}")
                            LOGGER.error(f"[Pipeline] Dateigröße: {pipeline_file.stat().st_size} bytes")
                            
                            # Versuche die ersten Zeilen zu lesen
                            try:
                                with open(pipeline_file, 'r', encoding='utf-8') as f:
                                    first_lines = [f.readline() for _ in range(5)]
                                LOGGER.error("[Pipeline] Erste Zeilen der Datei:")
                                for i, line in enumerate(first_lines, 1):
                                    LOGGER.error(f"  {i}: {line.rstrip()}")
                            except Exception as read_err:
                                LOGGER.error(f"[Pipeline] Konnte Datei nicht lesen: {read_err}")
                        else:
                            LOGGER.error(f"[Pipeline] ✗ Datei NICHT gefunden: {pipeline_file}")
                            LOGGER.error(f"[Pipeline] Dateien im Verzeichnis:")
                            try:
                                for item in FORECASTER_DIR.iterdir():
                                    LOGGER.error(f"  - {item.name}")
                            except Exception as list_err:
                                LOGGER.error(f"[Pipeline] Konnte Verzeichnis nicht listen: {list_err}")

# Nach dem Guard exportieren wir klar, was dieses Modul bereitstellt
__all__ = [
    "DashboardForecastAdapter",
    "HAS_PIPELINE",
    "PipelineConfig",
    "ModelArtifact",
    "get_model_filepath",
    "run_production_pipeline",
    "_PIPELINE_IMPORT_ERROR",
]





class DashboardForecastAdapter:
    """
    Adapter zwischen Dashboard und forecaster_pipeline – korrigiert & mit Debug-Logs.
    ERWEITERT: Berechnet 80% und 95% Konfidenzintervalle für Forecasts.

    Kernaufgaben:
    - Parsing/Normalisierung der GVB-Daten (Target) & exogenen Reihen
    - Re-Sampling auf Monatsanfang (MS)
    - Temporären Excel-Export (Schema, das die Pipeline erwartet)
    - Bau der Config & orchestrierter Pipeline-Run (inkl. Cache-Check)
    - Berechnung von Konfidenzintervallen (80%, 95%)
    """

    # ------------------------------------------------------------
    # Init
    # ------------------------------------------------------------
    def __init__(self, gvb_store_json: str, exog_store_json: str):
        # GVB laden
        self.gvb_data = pd.read_json(gvb_store_json, orient="split")
        self.gvb_data["date"] = pd.to_datetime(self.gvb_data["date"], errors="coerce")
        self.gvb_data = (
            self.gvb_data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        )

        # Exogene laden (robust)
        self.exog_data = self._load_exog_frame(exog_store_json)


        LOGGER.info(f"[Adapter|Exog] Frame: shape={self.exog_data.shape if hasattr(self, 'exog_data') else None}")
        if not self.exog_data.empty:
            LOGGER.info(f"[Adapter|Exog] Columns (Top): {self.exog_data.columns.tolist()[:12]}")
            if 'date' not in self.exog_data.columns:
                LOGGER.warning("[Adapter|Exog] Achtung: 'date' Spalte fehlt im Exog-Frame!")


        # für Debug/Summary
        self.pipeline_info: Dict[str, object] = {}

        LOGGER.info(_sym("✓ Pipeline-Integration geladen"))

        # --- INPUT DIAGNOSTICS ---------------------------------------------
        # Erkenne früh lange Null-Plateaus / auffällige Datenlagen
        try:
            diag = {}
            # Erwartete Struktur: Tall-Format mit Spalten 'datatype' ('bestand'/'fluss') + Werten in 'bestand'/'fluss'
            for mode, vcol in (("bestand", "bestand"), ("fluss", "fluss")):
                if "datatype" in self.gvb_data.columns and vcol in self.gvb_data.columns:
                    s = (
                        self.gvb_data.loc[self.gvb_data["datatype"] == mode, vcol]
                        .dropna()
                        .astype(float)
                    )
                    if not s.empty:
                        share_zero = float((s == 0).mean())
                        # längster Null-Run
                        run = best = 0
                        for val in (s == 0).values:
                            run = run + 1 if val else 0
                            best = max(best, run)
                        diag[f"{mode}_share_zero"] = share_zero
                        diag[f"{mode}_longest_zero_run"] = int(best)
            if diag:
                LOGGER.info(f"[DIAG|INPUT] {diag}")
            self._input_diagnostics = diag
        except Exception as _e:
            LOGGER.warning(f"[DIAG|INPUT] übersprungen: {_e}")


    # ------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------
    @staticmethod
    def _load_exog_frame(payload) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame()

        if isinstance(payload, str):
            # Vorrang: pandas 'split' JSON
            try:
                df = pd.read_json(payload, orient="split")
            except Exception:
                try:
                    df = pd.DataFrame(json.loads(payload))
                except Exception:
                    return pd.DataFrame()
        elif isinstance(payload, (list, dict)):
            df = pd.DataFrame(payload)
        else:
            return pd.DataFrame()

        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _to_ms(ts: pd.Series) -> pd.Series:
        """Trimmt Datumswerte auf Monatsanfang (MS)."""
        return ts.dt.to_period("M").dt.to_timestamp(how="start")



    @staticmethod
    def _expand_exog_variants(exog_names: list[str]) -> dict[str, set[str]]:
        variants: dict[str, set[str]] = {}
        for x in map(str, exog_names or []):
            x = x.strip()
            cand = {x}
            cand.add(f"{x}__last")
            cand.add(f"{x}__last__")
            variants[x] = cand
        return variants


    @staticmethod
    def _validate_exog_presence_in_excel(
        excel_path: str,
        selected_exog: list[str],
        *,
        sheet_name: str = "final_dataset",
        strict: bool = True,
        logger=LOGGER,
    ) -> None:
        if not selected_exog:
            logger.info("[Guard] Keine selected_exog übergeben – Validierung übersprungen.")
            return
        try:
            cols = list(pd.read_excel(excel_path, sheet_name=sheet_name, nrows=0).columns)
        except Exception as e:
            msg = f"[Guard] Konnte Excel zur Exog-Validierung nicht lesen: {excel_path} (sheet='{sheet_name}'): {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        cols_set = {str(c) for c in cols}
        variants_map = DashboardForecastAdapter._expand_exog_variants(selected_exog)

        missing: list[str] = []
        resolved: dict[str, str] = {}
        for req, cands in variants_map.items():
            match = next((c for c in cands if c in cols_set), None)
            if match is None:
                missing.append(req)
            else:
                resolved[req] = match

        if missing:
            logger.error("[Guard] Exogene Variablen fehlen im Excel-Export.")
            logger.error(f"[Guard] Erwartet (selected_exog): {selected_exog}")
            logger.error(f"[Guard] Gefunden (Excel-Spalten): {cols[:25]}{' ...' if len(cols)>25 else ''}")
            logger.error(f"[Guard] Fehlend (ohne akzeptierte Varianten): {missing}")
            if strict:
                raise RuntimeError(
                    f"Exogs fehlen im Excel-Export: {missing}. "
                    "Abbruch (strict=True). Bitte Exog-Mapping/Export korrigieren."
                )
            else:
                logger.warning("[Guard] strict=False – entferne fehlende Exogs aus cfg.selected_exog und fahre fort.")
        else:
            logger.info(f"[Guard] Exog-Validierung OK. Aufgelöste Spalten: {resolved}")



    @staticmethod
    def _validate_exog_presence_in_df(
        df_final: pd.DataFrame,
        selected_exog: list[str],
        *,
        strict: bool = True,
        logger=LOGGER,
    ) -> None:
        if not selected_exog:
            logger.info("[Guard] Keine selected_exog übergeben – Validierung übersprungen.")
            return
        cols_set = {str(c) for c in df_final.columns}
        variants_map = DashboardForecastAdapter._expand_exog_variants(selected_exog)
        missing = [req for req, cands in variants_map.items() if not any(c in cols_set for c in cands)]
        if missing:
            logger.error(f"[Guard] Exogs fehlen im DataFrame vor Excel-Export: {missing}")
            if strict:
                raise RuntimeError(
                    f"Exogs fehlen im DF/Export: {missing}. Abbruch (strict=True). "
                    "Bitte Exog-Mapping/Export korrigieren."
                )




    def _project_exog_series_monthly(
        self,
        s: pd.Series,
        h_months: int,
        *,
        period_m: int = 12,
        drift_window_m: int = 24,
        stable_std_threshold: float = 1e-6,
    ) -> np.ndarray:
        """
        Projiziert eine monatliche (MS) Exog 's' um h_months in die Zukunft.
        - Stabile Reihen (sehr geringe Varianz) → LOCF
        - Sonst: Seasonal-Naive (Monatsklasse) + linearer Drift (letztes Fenster)
        """
        s = pd.to_numeric(pd.Series(s).dropna(), errors="coerce").dropna()
        n = len(s)
        if n == 0 or h_months <= 0:
            return np.array([], dtype=float)

        # Stabile Reihe? → LOCF
        if float(s.std(ddof=1) if n > 1 else 0.0) < stable_std_threshold:
            last_val = float(s.iloc[-1])
            return np.full(h_months, last_val, dtype=float)

        # Saisonmittel pro Monat (1..12)
        # Achtung: s muss MS-indexiert sein – in prepare_* reindizieren wir vorher auf vollen MS-Index
        month_idx = s.index.month if isinstance(s.index, pd.DatetimeIndex) else pd.Series(np.arange(n) % period_m) + 1
        seas_means = s.groupby(month_idx).mean().to_dict()  # {1: µ_Jan, 2: µ_Feb, ...}

        # Drift (lineare Steigung) aus letztem Fenster
        win = min(max(drift_window_m, 6), n)
        y = s.iloc[-win:].values.astype(float)
        x = np.arange(win, dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = 0.0 if denom == 0.0 else ((x - x_mean) * (y - y_mean)).sum() / denom
        last_val = float(s.iloc[-1])

        out = np.zeros(h_months, dtype=float)
        # Wir gehen davon aus, dass Index monatlich bis zum letzten Punkt durchläuft.
        last_idx = s.index[-1] if isinstance(s.index, pd.DatetimeIndex) else None
        for t in range(1, h_months + 1):
            # Monatsklasse der Zukunft
            if isinstance(last_idx, pd.Timestamp):
                future_month = (last_idx + pd.DateOffset(months=t)).month
            else:
                # Falls kein DatetimeIndex vorhanden ist (sollte nicht passieren, aber fail-safe)
                future_month = ((n + (t - 1)) % period_m) + 1

            seas = float(seas_means.get(future_month, 0.0))
            # Additives Modell: linearer Drift + saisonaler Level
            val = (last_val + t * slope) + (seas - y_mean)  # mean-center die Saison, damit Level nicht doppelt zählt
            # Falls Historie sehr kurz → weich auf LOCF
            if n < period_m or win < 6:
                val = last_val
            out[t - 1] = val

        return out


    def _extend_exogs_to_future_ms(
        self,
        exog_monthly: pd.DataFrame,
        horizon_months: int,
    ) -> pd.DataFrame:
        """
        Hängt an exog_monthly (MS-Index) die nächsten horizon_months Monate mit
        prognostizierten Werten an. Erwartet Spalten: ['date', <exogs...>].
        """
        if horizon_months is None or int(horizon_months) <= 0:
            return exog_monthly

        df = exog_monthly.copy()
        if "date" not in df.columns or df.empty:
            return df

        # Sicherstellen: durchgehender MS-Index
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        start = df["date"].min().to_period("M").to_timestamp(how="start")
        end = df["date"].max().to_period("M").to_timestamp(how="start")
        full_idx = pd.date_range(start=start, end=end, freq="MS")

        exog_cols = [c for c in df.columns if c != "date"]
        df_full = (
            df.set_index("date")
            .reindex(full_idx)
            .sort_index()
        )
        # Lücken in Historie per ffill schließen (keine 0!)
        if exog_cols:
            df_full[exog_cols] = df_full[exog_cols].ffill()

        # Zukunftsindex aufbauen
        future_idx = pd.date_range(end + pd.offsets.MonthBegin(1), periods=int(horizon_months), freq="MS")
        fut = pd.DataFrame(index=future_idx, columns=exog_cols, dtype=float)

        # Pro Spalte projizieren
        for c in exog_cols:
            hist = df_full[c].dropna()
            if hist.empty:
                continue
            fut[c] = self._project_exog_series_monthly(hist, h_months=int(horizon_months))

        # Zusammenführen
        out = pd.concat([df_full, fut], axis=0).reset_index().rename(columns={"index": "date"})
        return out






    # ------------------------------------------------------------
    # NEU: Konfidenzintervall-Berechnung
    # ------------------------------------------------------------
    @staticmethod
    def _calculate_confidence_intervals(
        predictions: np.ndarray,
        residuals: np.ndarray,
        confidence_levels: List[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Berechnet multiple Konfidenzintervalle für Prognosen.
        
        Args:
            predictions: Array mit Punktprognosen
            residuals: Array mit Residuen aus dem Training/CV
            confidence_levels: Liste der gewünschten Konfidenzlevels (z.B. [80, 95])
        
        Returns:
            Dict mit lower/upper bounds für jedes Level
            Format: {'lower_80': array, 'upper_80': array, 'lower_95': array, ...}
        """
        if confidence_levels is None:
            confidence_levels = [80, 95]
        
        # Standardfehler aus Residuen schätzen
        std_error = np.std(residuals, ddof=1)  # Sample standard deviation
        
        # Z-Scores für verschiedene Konfidenzlevels (Normal-Verteilung)
        z_scores = {
            68: 0.9945,   # 68% CI (±1σ)
            80: 1.282,    # 80% CI
            90: 1.645,    # 90% CI
            95: 1.96,     # 95% CI
            99: 2.576     # 99% CI
        }
        
        intervals = {}
        
        for level in confidence_levels:
            z = z_scores.get(level, 1.96)  # Default: 95%
            margin = z * std_error
            
            intervals[f'lower_{level}'] = predictions - margin
            intervals[f'upper_{level}'] = predictions + margin
        
        LOGGER.info(f"[CI] Konfidenzintervalle berechnet: {confidence_levels}%, std_error={std_error:.4f}")
        
        return intervals

    @staticmethod
    def _add_confidence_intervals_to_forecast(
        forecast_df: pd.DataFrame,
        residuals: np.ndarray,
        confidence_levels: List[int] = None,
        forecast_col: str = "Forecast"
    ) -> pd.DataFrame:
        """
        Fügt Konfidenzintervalle zum Forecast-DataFrame hinzu.
        
        Args:
            forecast_df: DataFrame mit Prognosen
            residuals: Residuen aus Training/CV
            confidence_levels: Gewünschte Konfidenzlevels
            forecast_col: Name der Forecast-Spalte
        
        Returns:
            Erweiterter DataFrame mit CI-Spalten (yhat_lower_80, yhat_upper_80, etc.)
        """
        if confidence_levels is None:
            confidence_levels = [80, 95]
        
        df = forecast_df.copy()
        
        # Predictions extrahieren
        if forecast_col not in df.columns:
            LOGGER.warning(f"[CI] Spalte '{forecast_col}' nicht gefunden, verwende erste numerische Spalte")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            forecast_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if forecast_col is None:
            LOGGER.error("[CI] Keine Forecast-Spalte gefunden!")
            return df
        
        predictions = df[forecast_col].values
        
        # Konfidenzintervalle berechnen
        intervals = DashboardForecastAdapter._calculate_confidence_intervals(
            predictions=predictions,
            residuals=residuals,
            confidence_levels=confidence_levels
        )
        
        # Zum DataFrame hinzufügen (Format: yhat_lower_XX, yhat_upper_XX)
        for key, values in intervals.items():
            df[f'yhat_{key}'] = values
        
        LOGGER.info(f"[CI] Hinzugefügte Spalten: {list(intervals.keys())}")
        
        return df

    # ------------------------------------------------------------
    # Datenaufbereitung (MS-Index, kein Backfill)
    # ------------------------------------------------------------
    def prepare_pipeline_data(
        self,
        target: str,
        selected_exog: List[str],
        use_flows: bool = False,
        *,
        horizon_quarters: int = 0,  # NEU: für Exog-Projektion (MS → +3M pro Quartal)
    ) -> pd.DataFrame:
        """
        Bereitet Daten für die Pipeline vor – MS-Index, fügt exogene Variablen ein
        und projiziert exogene Variablen bis zum Prognosehorizont in die Zukunft.
        """
        data_type = "fluss" if use_flows else "bestand"
        value_col = "fluss" if use_flows else "bestand"

        LOGGER.info(f"[Adapter] use_flows={use_flows} -> data_type='{data_type}'")
        LOGGER.info(f"[Adapter] requested_exog={selected_exog}")
        LOGGER.info(f"[Adapter] exog_data.columns (Top): {self.exog_data.columns.tolist()[:12] if not self.exog_data.empty else []}")


        # 1) Zielvariable filtern/aggregieren
        if target == "gesamt":
            base = self.gvb_data[self.gvb_data["datatype"] == data_type]
            target_data = base.groupby("date", as_index=False)[value_col].sum()
        else:
            base = self.gvb_data[
                (self.gvb_data["ebene1"] == target) &
                (self.gvb_data["datatype"] == data_type)
            ]
            target_data = base.groupby("date", as_index=False)[value_col].sum()

        target_data = target_data.rename(columns={value_col: "target_value"})
        target_data["date"] = pd.to_datetime(target_data["date"], errors="coerce")
        target_data = target_data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 2) Quartals-Target → Monatsanfang (MS)
        tq = (target_data.set_index("date")["target_value"].to_period("Q").groupby(level=0).last())
        tq_start = tq.to_timestamp(how="start")

        start_ms = target_data["date"].min().to_period("M").to_timestamp(how="start")
        end_ms   = target_data["date"].max().to_period("M").to_timestamp(how="start")
        monthly_index_hist = pd.date_range(start=start_ms, end=end_ms, freq="MS")

        target_monthly = tq_start.reindex(monthly_index_hist).ffill()
        target_monthly_df = target_monthly.rename("target_value").rename_axis("date").reset_index()

        # 3) Exogene Variablen aus Store
        if selected_exog and not self.exog_data.empty:
            available_exog = [c for c in selected_exog if c in self.exog_data.columns]

            if available_exog:
                LOGGER.info(_sym(f"[Adapter] Füge {len(available_exog)} exogene Variablen hinzu: {available_exog}"))

                exog_monthly = self.exog_data[["date"] + available_exog].copy()
                exog_monthly["date"] = pd.to_datetime(exog_monthly["date"], errors="coerce")
                exog_monthly = exog_monthly.dropna(subset=["date"])

                # Monatsanfang normalisieren + pro Monat letzten Wert
                exog_monthly["date"] = self._to_ms(exog_monthly["date"])
                exog_monthly = exog_monthly.groupby("date", as_index=False).last()

                # Suffix __last (Kompatibilität zur Pipeline)
                rename_dict = {col: f"{col}__last" for col in available_exog}
                exog_monthly = exog_monthly.rename(columns=rename_dict)

                # ✅ NEU: Exogs bis in die Zukunft fortschreiben (MS)
                future_months = int(horizon_quarters or 0) * 3
                if future_months > 0:
                    exog_monthly = self._extend_exogs_to_future_ms(exog_monthly, future_months)

                LOGGER.debug(f"[Adapter] Exog-Spalten nach Projection: {exog_monthly.columns.tolist()}")
            else:
                LOGGER.warning(
                    "[Adapter] Keine der gewählten exogenen Variablen in exog_data gefunden. "
                    f"Gesucht: {selected_exog} | Verfügbar: {self.exog_data.columns.tolist()}"
                )
                exog_monthly = pd.DataFrame({"date": monthly_index_hist})
        else:
            exog_monthly = pd.DataFrame({"date": monthly_index_hist})

        # 4) Merge (Left) – Zielreihe bleibt nur historisch; Exogs enthalten ggf. Zukunft
        result = (
            pd.merge(target_monthly_df, exog_monthly, on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )

        # 5) Nur Exogs ffillen (Target nicht anfassen)
        exog_cols_only = [c for c in result.columns if c not in ["date", "target_value"]]
        if exog_cols_only:
            result[exog_cols_only] = result[exog_cols_only].ffill()

        # 6) Debug/Checks
        tgt_non_null = result["target_value"].dropna()
        tmin = float(tgt_non_null.min()) if not tgt_non_null.empty else np.nan
        tmax = float(tgt_non_null.max()) if not tgt_non_null.empty else np.nan

        LOGGER.info(_sym("\n[Adapter] Pipeline-Daten (MS-normalisiert + Exog-Forecast):"))
        LOGGER.info(f"  Input-Quartale: {tq.shape[0]}")
        LOGGER.info(f"  Output-Monate:  {len(result)}  (inkl. Exog-Zukunft: +{int(horizon_quarters)*3 if horizon_quarters else 0}M)")
        LOGGER.info(f"  Target-Range:  {tmin:.3f} - {tmax:.3f}")
        LOGGER.info(f"  Exog-Spalten:  {exog_cols_only[:10]}{'...' if len(exog_cols_only)>10 else ''}")
        if result.isna().sum().sum() > 0:
            nz = result.isna().sum()
            LOGGER.warning(f"  NaN nach Merge: {dict(nz[nz>0])}")

        if result["target_value"].isna().all():
            raise ValueError("Target-Variable enthält nur NaN-Werte.")

        self.pipeline_info = {
            "n_in_quarters": int(tq.shape[0]),
            "n_out_months": int(len(result)),
            "target_min": tmin,
            "target_max": tmax,
            "exog_cols": exog_cols_only,
        }

        return result


    # ------------------------------------------------------------
    # Excel-Export für die Pipeline
    # ------------------------------------------------------------
    def create_temp_excel(self, df: pd.DataFrame, target_col: str = "target_value") -> str:
        """
        Erstellt temporäre Excel-Datei für die Pipeline.
        WICHTIG: Exogene Spalten müssen bereits vorhanden sein.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp_path = tmp.name
        tmp.close()  # Dateihandle schließen, Path weiterverwenden

        # Target umbenennen
        rename_map = {"date": "Datum", target_col: "PH_EINLAGEN"}
        df_export = df.rename(columns=rename_map).copy()
        df_export["Datum"] = pd.to_datetime(df_export["Datum"], errors="coerce")

        LOGGER.info(_sym("\n[Adapter] Excel-Export:"))
        LOGGER.info(f"  Shape: {df_export.shape}")
        LOGGER.info(f"  Spalten: {df_export.columns.tolist()}")

        if not df_export.empty:
            dmin = df_export["Datum"].min()
            dmax = df_export["Datum"].max()
            tmin = df_export["PH_EINLAGEN"].min()
            tmax = df_export["PH_EINLAGEN"].max()
            LOGGER.info(f"  Datum-Range: {dmin} - {dmax}")
            LOGGER.info(f"  Target-Range: {tmin:.1f} - {tmax:.1f}")

            exog_cols = [c for c in df_export.columns if c not in ["Datum", "PH_EINLAGEN"]]
            if exog_cols:
                LOGGER.info(f"  Exogene Variablen ({len(exog_cols)}): {exog_cols}")
                for col in exog_cols[:3]:  # Erste 3 anzeigen
                    non_na = int(df_export[col].notna().sum())
                    LOGGER.info(f"    {col}: {non_na}/{len(df_export)} Werte")

        df_export.to_excel(tmp_path, sheet_name="final_dataset", index=False)

        # Validierung
        try:
            v = pd.read_excel(tmp_path, sheet_name="final_dataset")
            LOGGER.info(
                f"[Adapter] Excel-Validierung: {len(v)} Zeilen, Spalten: {list(v.columns)}"
            )
        except Exception as e:
            LOGGER.warning(f"[Adapter] Excel-Validierung fehlgeschlagen: {e}")

        return tmp_path

    # ------------------------------------------------------------
    # Config-Aufbau für die Pipeline
    # ------------------------------------------------------------
    def _build_config(
            self,
        *,
        excel_path: str,
        horizon: int,
        use_cached: bool,
        selected_exog: List[str]
        ) -> "PipelineConfig":
        """
        Baut die Pipeline-Konfiguration:
        - liest die Zielspalte dynamisch aus dem Excel (alles außer 'Datum')
        - bereinigt ausgewählte Exogs (entfernt interne Flags)
        - setzt Cache-Tag konsistent nach Sektor/UI-Target/Modus/Horizont
        - namespaced Model-/Output-Verzeichnisse pro Sektor
        - nutzt PipelineConfig (alias für Config) aus forecaster_pipeline
        - filtert unbekannte/inkompatible Keyword-Argumente dynamisch weg
        """

        # 0) Pipeline vorhanden?
        if not HAS_PIPELINE or PipelineConfig is None:
            raise RuntimeError(
                f"Forecast-Pipeline nicht verfügbar. Detail: {_PIPELINE_IMPORT_ERROR!r}"
            )

        # 1) Exogs bereinigen + UI-Metadaten
        exog_final = list(selected_exog or [])
        exog_clean = [x for x in exog_final if "__flows_flag__" not in str(x)]

        ui_target = (
            self.pipeline_info.get("ui_target") if isinstance(self.pipeline_info, dict) else None
        ) or "Wertpapiere"

        # Modus primär aus UI (Switch), fallback auf Heuristik aus Exogs
        ui_mode = "fluss" if (
            (isinstance(self.pipeline_info, dict) and self.pipeline_info.get("use_flows"))
            or any("__flows_flag__" in str(x) for x in exog_final)
        ) else "bestand"

        # NEU: Sektor (PH/NFK) aus UI für Cache & Pfade
        sektor = (
            (self.pipeline_info.get("sektor") if isinstance(self.pipeline_info, dict) else None)
            or "PH"
        )
        sektor_slug = str(sektor).strip().upper()  # "PH" / "NFK"

        cache_tag = f"{sektor_slug.lower()}_{str(ui_target).lower()}_{ui_mode}_h{int(horizon or 4)}"

        # 2) Zielspalte dynamisch aus Excel ziehen
        try:
            cols = list(pd.read_excel(excel_path, sheet_name="final_dataset", nrows=0).columns)
        except Exception:
            cols = ["Datum", "PH_EINLAGEN"]  # Fallback
        target_col = next((c for c in cols if str(c).lower() != "datum"), None) or "PH_EINLAGEN"

        # 3) Output-/Model-Ordner sicherstellen (NEU: pro Sektor namespacen)
        forecaster_dir = Path(__file__).parent
        output_dir = (forecaster_dir / "trained_outputs" / sektor_slug).resolve()
        model_dir = (forecaster_dir / "trained_models" / sektor_slug).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 4) Gewünschte (volle) Konfiguration als dict aufbauen
        desired_kwargs = dict(
            # Datenquelle
            excel_path=excel_path,
            sheet_name="final_dataset",
            date_col="Datum",
            target_col=str(target_col),

            # Aggregations-/Feature-Parameter
            agg_methods_exog=["last"],
            agg_method_target="mean",
            exog_month_lags=[-12, -6, -3, -1],
            target_lags_q=[1, 2, 4],
            add_trend_features=True,
            trend_degree=2,
            add_seasonality=True,
            seasonality_mode="dummies",

            # Transformation/Standardisierung
            target_transform="none",
            target_standardize=True,

            # Forecasting
            forecast_horizon=int(horizon or 4),

            # Zukunfts-Strategien für Exogs
            future_exog_strategy="mixed",
            future_exog_drift_window_q=8,
            future_exog_seasonal_period_q=4,

            # Pfade/Cache
            output_dir=str(output_dir),
            model_dir=str(model_dir),
            use_cached_model=bool(use_cached),
            random_state=42,
            cache_tag=cache_tag,

            # Exogs
            selected_exog=list(exog_clean),
        )

        # 4b) Dynamisch nur die vom tatsächlichen Config-Dataclass erlaubten Keys durchlassen
        allowed_keys = set()
        ignored_keys = []
        try:
            from dataclasses import fields as _dc_fields, is_dataclass as _is_dc
            if _is_dc(PipelineConfig):
                allowed_keys = {f.name for f in _dc_fields(PipelineConfig)}
            else:
                import inspect as _inspect
                sig = _inspect.signature(PipelineConfig)
                allowed_keys = {p.name for p in sig.parameters.values()
                                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        except Exception:
            # Minimal-Whitelist als Fallback
            allowed_keys = {
                "excel_path", "sheet_name", "date_col", "target_col",
                "agg_methods_exog", "agg_method_target", "exog_month_lags",
                "target_lags_q", "add_trend_features", "trend_degree",
                "add_seasonality", "seasonality_mode", "target_transform",
                "target_standardize", "forecast_horizon", "future_exog_strategy",
                "future_exog_drift_window_q", "future_exog_seasonal_period_q",
                "output_dir", "model_dir", "use_cached_model", "random_state",
                "cache_tag", "selected_exog"
            }

        filtered_kwargs = {}
        for k, v in desired_kwargs.items():
            if k in allowed_keys:
                filtered_kwargs[k] = v
            else:
                ignored_keys.append(k)

        if ignored_keys:
            LOGGER.warning(
                "[Adapter|Config] Ignoriere unbekannte Config-Parameter (nicht im PipelineConfig vorhanden): "
                + ", ".join(sorted(map(str, ignored_keys)))
            )

        # 5) Konfiguration instanziieren
        cfg = PipelineConfig(**filtered_kwargs)

        # 6) Für volle Kompatibilität: beide Felder setzen (falls vorhanden)
        try:
            cfg.selected_exog = list(exog_clean)
        except Exception:
            pass
        if hasattr(cfg, "exog_cols"):
            try:
                cfg.exog_cols = list(exog_clean)
            except Exception:
                pass

        # 7) Logging
        try:
            LOGGER.info(f"[Adapter] Config.target_col: {getattr(cfg, 'target_col', None)}")
            LOGGER.info(f"[Adapter] Config.selected_exog: {getattr(cfg, 'selected_exog', None)}")
            if hasattr(cfg, "exog_cols"):
                LOGGER.info(f"[Adapter] Config.exog_cols: {getattr(cfg, 'exog_cols', None)}")
            LOGGER.info(
                f"[Adapter] cache_tag={getattr(cfg, 'cache_tag', None)}, "
                f"horizon={getattr(cfg, 'forecast_horizon', None)}, "
                f"ui_mode={ui_mode}, sektor={sektor_slug}"
            )
        except Exception:
            pass

        return cfg


    # ------------------------------------------------------------
    # NEU: Residuen aus Pipeline/Modell extrahieren
    # ------------------------------------------------------------
    def _extract_residuals_from_pipeline(
        self,
        metadata: dict,
        model_path: str = None,
        reference_value: float = None,   # optional: z.B. letzte Historik oder erste Prognose
        n_synth: int = 1000              # mehr Punkte → stabilere CI-Schätzung
    ) -> np.ndarray:
        """
        Extrahiert Residuen für CI-Berechnung.
        Präferenz: echte CV-Residuen → Modell-Artefakt → (synthetisch aus RMSE; inkl. Rückskalierung).
        """

        # 0) kleine Hilfsfunktionen
        def _load_artifact(path: str):
            if not path or not ModelArtifact:
                return None
            if hasattr(ModelArtifact, "exists") and not ModelArtifact.exists(path):
                return None
            try:
                return ModelArtifact.load(path)
            except Exception as e:
                LOGGER.warning(f"[CI] Artifact konnte nicht geladen werden: {e}")
                return None

        def _finite_diff_derivative(inv_func, z0: float, eps: float = 1e-3) -> float:
            """Numerische Ableitung dy/dz ~ (f(z0+eps)-f(z0-eps))/(2*eps)."""
            try:
                y_plus  = inv_func(np.array([[z0 + eps]])).ravel()[0]
                y_minus = inv_func(np.array([[z0 - eps]])).ravel()[0]
                return float((y_plus - y_minus) / (2.0 * eps))
            except Exception:
                return np.nan

        # 1) echte Residuen aus Metadata
        if metadata and "cv_residuals" in metadata and isinstance(metadata["cv_residuals"], (list, tuple)):
            res = np.array(metadata["cv_residuals"], dtype=float)
            res = res[np.isfinite(res)]
            if res.size > 0:
                LOGGER.info(f"[CI] Residuen aus Metadata geladen: {res.size} Werte")
                return res

        # 2) echte Residuen aus Modell-Artefakt
        art = _load_artifact(model_path) if model_path else None
        if art and isinstance(getattr(art, "metadata", None), dict):
            md = art.metadata or {}
            if "cv_residuals" in md and isinstance(md["cv_residuals"], (list, tuple)):
                res = np.array(md["cv_residuals"], dtype=float)
                res = res[np.isfinite(res)]
                if res.size > 0:
                    LOGGER.info(f"[CI] Residuen aus Modell geladen: {res.size} Werte")
                    return res

        # 3) synthetische Residuen aus CV-Performance (mit Skalencheck & Rückskalierung)
        if metadata and isinstance(metadata.get("cv_performance"), dict):
            cv = metadata["cv_performance"]
            rmse_val = cv.get("cv_rmse", cv.get("rmse", None))
            if isinstance(rmse_val, (int, float)) and np.isfinite(rmse_val):
                rmse = float(rmse_val)
                scale_flag = (metadata.get("cv_metrics_scale") or "").lower() or "original"

                if scale_flag == "original":
                    # bereits in Originaleinheiten → direkt nutzen
                    residuals = np.random.normal(0.0, rmse, size=int(n_synth))
                    LOGGER.info(f"[CI] Synthetische Residuen (original scale) mit RMSE={rmse:.4f}, n={n_synth}")
                    return residuals

                # Skala ist 'transformed' → zurückskalieren
                # Versuche, den im Modell gespeicherten Transformer zu nutzen:
                tj = getattr(art, "tj", None) if art else None

                # Referenzwert y0 bestimmen (für lokale Ableitung). Reihenfolge:
                # caller→metadata.context.last_hist_value→metadata.context.first_forecast→y_train_summary.mean
                y0 = None
                if isinstance(reference_value, (int, float)) and np.isfinite(reference_value):
                    y0 = float(reference_value)
                elif isinstance(metadata.get("context"), dict):
                    ctx = metadata["context"]
                    for key in ("last_hist_value", "first_forecast"):
                        v = ctx.get(key)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            y0 = float(v); break
                if y0 is None:
                    ysum = (metadata.get("y_train_summary") or {})
                    if isinstance(ysum.get("mean"), (int, float)) and np.isfinite(ysum["mean"]):
                        y0 = float(ysum["mean"])

                # Falls kein y0 verfügbar ist, setze einen konservativen Default
                if y0 is None:
                    y0 = 0.0

                if tj is not None and hasattr(tj, "inverse_transform") and hasattr(tj, "transform"):
                    try:
                        # z0 = T(y0), Ableitung dy/dz an z0 per Finite-Difference
                        z0 = float(tj.transform(np.array([[y0]])).ravel()[0])
                        d_dz = _finite_diff_derivative(tj.inverse_transform, z0, eps=1e-3)
                        if np.isfinite(d_dz) and d_dz > 0:
                            rmse_back = rmse * d_dz
                            residuals = np.random.normal(0.0, rmse_back, size=int(n_synth))
                            LOGGER.warning(
                                f"[CI] RMSE rückskaliert via lokale Ableitung: rmse_t={rmse:.4f} → rmse_y={rmse_back:.4f} (y0={y0:.4f})"
                            )
                            return residuals
                    except Exception as e:
                        LOGGER.warning(f"[CI] Rückskalierung (Ableitung) fehlgeschlagen: {e}")

                    # Monte-Carlo-Fallback: stichprobenweise im Transform-Raum stören und zurücktransformieren
                    try:
                        z0 = float(tj.transform(np.array([[y0]])).ravel()[0])
                        eps = np.random.normal(0.0, rmse, size=int(n_synth)).reshape(-1, 1)
                        y_sim = tj.inverse_transform((z0 + eps).astype(float)).ravel()
                        # Differenz zur Referenz y0 als "residuum"
                        residuals = (y_sim - y0).astype(float)
                        residuals = residuals[np.isfinite(residuals)]
                        if residuals.size > 0:
                            LOGGER.warning(
                                f"[CI] RMSE rückskaliert via Monte-Carlo: rmse_t={rmse:.4f} → std_y≈{np.std(residuals):.4f} (y0={y0:.4f})"
                            )
                            return residuals
                    except Exception as e:
                        LOGGER.warning(f"[CI] Rückskalierung (Monte-Carlo) fehlgeschlagen: {e}")

                # Wenn kein Transformer verfügbar → grober Heuristik-Fallback
                LOGGER.warning(
                    "[CI] cv_rmse ist 'transformed', aber kein Transformer verfügbar – "
                    "nutze konservativen Default (keine Rückskalierung möglich)."
                )
                return np.random.normal(0.0, rmse, size=int(n_synth))

        # 4) letzter Fallback (sehr konservativ)
        LOGGER.error("[CI] Keine Residuen verfügbar – verwende Default-Annahme (std=0.1)")
        return np.random.normal(0.0, 0.1, size=int(n_synth))


    def _generate_backtest_results(
        self,
        model,
        tj,  # optionaler Ziel-Transformer (kann None sein)
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
        dates_train: Union[pd.Series, pd.DataFrame, pd.Index, np.ndarray],
        n_splits: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Expanding-Window Backtest (1-Schritt) mit robustem Typ-/NaN-Handling und
        Guards gegen 0-Linien. Gibt (backtest_df, residuals_df) zurück.
        """

        import numpy as np
        from typing import Union
        from sklearn.base import clone

        EPS = 1e-8

        # -------- Helper ----------
        def _as_series(obj, name=None) -> pd.Series:
            """Konvertiert DataFrame/Index/ndarray sicher in 1D Series."""
            if isinstance(obj, pd.Series):
                return obj
            if isinstance(obj, pd.DataFrame):
                if obj.shape[1] == 1:
                    return obj.iloc[:, 0]
                raise ValueError(f"[Backtest] Erwartete 1 Spalte für {name or 'Series'}, bekam {obj.shape[1]}")
            if isinstance(obj, (pd.Index, np.ndarray, list, tuple)):
                arr = np.asarray(obj)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.ravel()
                if arr.ndim != 1:
                    raise ValueError(f"[Backtest] Erwartete 1D Array für {name or 'Series'}, bekam shape={arr.shape}")
                return pd.Series(arr)
            raise ValueError(f"[Backtest] Unbekannter Typ für {name or 'Series'}: {type(obj)}")

        def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            out = df.replace([np.inf, -np.inf], np.nan)
            out = out.ffill().bfill()
            if out.isna().any().any():
                med = out.median(numeric_only=True)
                out = out.fillna(med)
            return out

        def _fold_variance_ok(y: pd.Series) -> bool:
            y = pd.to_numeric(y, errors="coerce").dropna()
            return (len(y) >= 5) and (float(np.nanstd(y)) > EPS)

        def _baseline_forecast(y_tr: pd.Series, idx_len: int) -> np.ndarray:
            y_tr = pd.to_numeric(y_tr, errors="coerce").dropna()
            if y_tr.empty:
                return np.full(idx_len, np.nan, dtype=float)
            return np.full(idx_len, float(y_tr.iloc[-1] if y_tr.std() > EPS else y_tr.mean()), dtype=float)

        def _guard_zero_predictions(yhat: np.ndarray, y_tr: pd.Series) -> np.ndarray:
            yhat = np.asarray(yhat, dtype=float)
            if yhat.size == 0:
                return yhat
            share_zero = (np.isfinite(yhat) & (np.abs(yhat) < EPS)).mean()
            if share_zero > 0.7 and (abs(float(pd.to_numeric(y_tr, errors="coerce").mean())) > EPS):
                return _baseline_forecast(y_tr, len(yhat))
            return yhat

        # -------- Inputs säubern ----------
        X = X_train.copy()
        X.index = pd.RangeIndex(len(X))  # neutraler Index

        y = _as_series(y_train, "y_train").astype(float)
        y.index = pd.RangeIndex(len(y))

        d = _as_series(dates_train, "dates_train")
        d = pd.to_datetime(d, errors="coerce")
        d.index = pd.RangeIndex(len(d))

        if not (len(X) == len(y) == len(d)):
            raise ValueError(f"[Backtest] Längen-Inkonsistenz: X={len(X)}, y={len(y)}, dates={len(d)}")

        n = len(y)
        if n < (n_splits + 4):
            n_splits = max(2, min(5, n // 4))

        # Feature-Imputing (nie fillna(0))
        X = _safe_impute(X)

        # Expanding Splits (1-Schritt)
        def expanding_splits(n_total: int, min_train: int, gap: int, horizon: int):
            for test_end in range(min_train + gap, n_total - horizon + 1):
                train_end = test_end - gap
                tr_idx = list(range(0, train_end))
                te_idx = list(range(test_end, test_end + horizon))
                yield tr_idx, te_idx

        min_train = max(12, n // (n_splits + 1))
        gap = 0
        horizon = 1

        preds = np.full(n, np.nan, dtype=float)

        # Split-Schleife
        for tr, te in expanding_splits(n, min_train, gap, horizon):
            if not tr or not te:
                continue

            X_tr = X.iloc[tr].copy()
            X_te = X.iloc[te].copy()
            y_tr = y.iloc[tr].copy()

            # Restliche NaNs mit Spaltenmedian ersetzen
            med = X_tr.median(numeric_only=True)
            X_tr = X_tr.fillna(med)
            X_te = X_te.fillna(med)

            # Degenerations-Guards
            if not _fold_variance_ok(y_tr):
                preds[te] = _baseline_forecast(y_tr, len(te))
                continue

            if (X_tr.std(numeric_only=True) < 1e-12).all():
                preds[te] = _baseline_forecast(y_tr, len(te))
                continue

            # Modell klonen
            m = clone(model)

            # Ziel-Transformation splitsicher
            use_tj = (tj is not None)
            if use_tj:
                try:
                    from copy import deepcopy
                    tj_local = deepcopy(tj)  # eigenen Fit pro Fold
                    # Wenn es ein sklearn-Transformer ist:
                    if hasattr(tj_local, "fit") and hasattr(tj_local, "transform"):
                        tj_local.fit(y_tr.values.reshape(-1, 1))
                        y_tr_t = tj_local.transform(y_tr.values.reshape(-1, 1)).ravel()
                        m.fit(X_tr.values, y_tr_t)
                        yhat_t = m.predict(X_te.values)
                        # Inversion: inverse_transform oder custom inverse()
                        if hasattr(tj_local, "inverse_transform"):
                            yhat = tj_local.inverse_transform(yhat_t.reshape(-1, 1)).ravel()
                        elif hasattr(tj_local, "inverse"):
                            yhat = tj_local.inverse(yhat_t)
                        else:
                            # kein Invers – dann auf Transform verzichten
                            m = clone(model)
                            m.fit(X_tr.values, y_tr.values)
                            yhat = m.predict(X_te.values)
                    else:
                        # Transformer ohne sklearn-API -> direkt ohne Transform
                        m.fit(X_tr.values, y_tr.values)
                        yhat = m.predict(X_te.values)
                except Exception:
                    # Fallback ohne Transform
                    m = clone(model)
                    m.fit(X_tr.values, y_tr.values)
                    yhat = m.predict(X_te.values)
            else:
                # kein Transformer
                m.fit(X_tr.values, y_tr.values)
                yhat = m.predict(X_te.values)

            # Guard gegen "alles 0"
            yhat = _guard_zero_predictions(yhat, y_tr)

            preds[te] = np.asarray(yhat, dtype=float)

        # Ergebnis-DF
        mask = ~np.isnan(preds)
        backtest_df = pd.DataFrame({
            "date": d[mask].values,
            "actual": y[mask].values,
            "predicted": preds[mask],
        })
        backtest_df["error"] = backtest_df["actual"] - backtest_df["predicted"]
        backtest_df["abs_error"] = backtest_df["error"].abs()
        backtest_df = backtest_df.sort_values("date").reset_index(drop=True)

        residuals_df = backtest_df[["date", "error", "abs_error"]].copy()

        return backtest_df, residuals_df



    # ------------------------------------------------------------
    # Hauptmethode: Pipeline laufen lassen (mit Cache-Handling & CI)
    # ------------------------------------------------------------

    def run_forecast(
        self,
        target: str,
        selected_exog: List[str],
        horizon: int,
        use_cached: bool,
        force_retrain: bool,
        use_flows: bool = False,
        confidence_levels: List[int] = None,
        preload_model_path: Optional[str] = None,
    ):
        """
        Führt Forecast durch mit Konfidenzintervallen und Backtest-Generierung.
        Enthält:
        - kurzes Warten auf Exog-Verfügbarkeit (falls Loader async)
        - robustes Auflösen/Validieren der Exogs
        - Config-Bau mit Cache-Check (Feature-Mismatch invalidiert Cache)
        - Pipeline-Run + CI-Anreicherung + Backtest
        """
        if confidence_levels is None:
            confidence_levels = [80, 95]

        LOGGER.info(_sym("\n[Adapter] ===== RUN FORECAST ====="))
        LOGGER.info(f"[Adapter] Target: {target} | Horizon: {horizon} | use_cached={use_cached} | force_retrain={force_retrain}")
        LOGGER.info(f"[Adapter] Selected exog ({len(selected_exog)}): {selected_exog}")
        LOGGER.info(f"[Adapter] use_flows={use_flows}")
        LOGGER.info(f"[Adapter] confidence_levels={confidence_levels}")

        self.pipeline_info["ui_target"] = target

        temp_excel: Optional[str] = None
        try:
            # -------- B) Exogene ggf. kurz "heranpolling", falls noch im Laden ----------
            try:
                import time
                requested = list(selected_exog or [])
                LOGGER.info(f"[Adapter] requested_exog={requested}")

                def _get_exog_df():
                    if hasattr(self, "get_exog_data") and callable(getattr(self, "get_exog_data")):
                        return self.get_exog_data()
                    return getattr(self, "exog_data", None) if hasattr(self, "exog_data") else None

                def _has_any_requested(dfx: Optional[pd.DataFrame], codes: list[str]) -> bool:
                    if not isinstance(dfx, pd.DataFrame) or dfx.empty or not codes:
                        return False
                    cs = {str(c) for c in dfx.columns}
                    for code in codes:
                        if (code in cs) or (f"{code}__last" in cs) or (f"{code}__last__" in cs):
                            return True
                    return False

                cur_exog_df = _get_exog_df()
                if isinstance(cur_exog_df, pd.DataFrame):
                    LOGGER.info(f"[Adapter|Exog] Frame: shape={cur_exog_df.shape}")
                    LOGGER.info(f"[Adapter] exog_data.columns (Top): {list(cur_exog_df.columns[:5])}")
                else:
                    LOGGER.info("[Adapter|Exog] Frame: None")

                need_exog = bool(requested)
                have_exog_now = _has_any_requested(cur_exog_df, requested)

                if need_exog and not have_exog_now:
                    # Download (best effort) anstoßen
                    if hasattr(self, "fetch_exogs") and callable(getattr(self, "fetch_exogs")):
                        try:
                            self.fetch_exogs(requested)
                        except Exception as _e_fetch:
                            LOGGER.warning(f"[Adapter|Exog] fetch_exogs() Fehler/kein Hook: {_e_fetch}")

                    # Kurzes Polling-Fenster
                    max_wait_sec = 12.0
                    poll_sec = 0.3
                    t0 = time.time()
                    while time.time() - t0 < max_wait_sec and not have_exog_now:
                        time.sleep(poll_sec)
                        cur_exog_df = _get_exog_df()
                        have_exog_now = _has_any_requested(cur_exog_df, requested)

                    if have_exog_now:
                        LOGGER.info("[Adapter|Exog] Exogs verfügbar – fahre mit Datenaufbereitung fort.")
                    else:
                        LOGGER.warning("[Adapter|Exog] Timeout: Exogs noch nicht verfügbar – fahre ohne Exogene fort.")
            except Exception as _e_wait:
                LOGGER.warning(f"[Adapter|Exog] Wait-Block übersprungen: {_e_wait}")

            # -------- 1) Daten vorbereiten (MS, kein Backfill) ----------
            prepared_df = self.prepare_pipeline_data(
                target=target,
                selected_exog=selected_exog,
                use_flows=use_flows,
                horizon_quarters=int(horizon or 0),  # ← wichtig: MS-Projektion Exogs bis in Zukunft
            )

            # 1a) Exogs im DF robust auflösen + sanft filtern
            def _resolve_exogs_in_df(df: pd.DataFrame, requested: List[str]) -> tuple[list[str], dict]:
                cols_set = {str(c) for c in df.columns}
                variants_map = self._expand_exog_variants(requested or [])
                resolved_list: list[str] = []
                resolved_map: dict[str, str] = {}
                missing: list[str] = []
                for req, cands in variants_map.items():
                    match = next((c for c in cands if c in cols_set), None)
                    if match is None:
                        missing.append(req)
                    else:
                        resolved_map[req] = match
                        resolved_list.append(match)
                return resolved_list, {"missing": missing, "mapping": resolved_map}

            resolved_exogs, diag_resolve = _resolve_exogs_in_df(prepared_df, list(selected_exog or []))
            if diag_resolve["missing"]:
                LOGGER.warning(
                    "[Guard] Einige gewünschte Exogs fehlen im DataFrame und werden herausgefiltert: "
                    + ", ".join(diag_resolve["missing"])
                )
            if resolved_exogs:
                LOGGER.info(f"[Guard] Exogs im DF aufgelöst: {diag_resolve['mapping']}")
            else:
                LOGGER.warning("[Guard] Keine Exogs im DF auflösbar – Forecast läuft ohne Exogene weiter.")

            # Für nachfolgende Schritte merken
            self.pipeline_info["exog_cols"] = list(resolved_exogs)

            # 1b) DF-Guard (non-strict: nur Hinweis/Logging)
            try:
                self._validate_exog_presence_in_df(
                    df_final=prepared_df,
                    selected_exog=list(selected_exog or []),
                    strict=False,   # wegen vorigem Filter absichtlich nicht strikt
                    logger=LOGGER,
                )
            except Exception as e_guard_df:
                LOGGER.warning(f"[Guard/DF] (non-strict) Hinweis: {e_guard_df}")

            # -------- 2) Excel erzeugen & prüfen ----------
            temp_excel = self.create_temp_excel(prepared_df)

            # 2a) Excel-Guard (strict=True) nur gegen effektiv vorhandene Exogs
            try:
                self._validate_exog_presence_in_excel(
                    excel_path=temp_excel,
                    selected_exog=list(resolved_exogs),
                    sheet_name="final_dataset",
                    strict=True,
                    logger=LOGGER,
                )
            except Exception as e_guard_xlsx:
                LOGGER.error(f"[Guard/Excel] Validierung fehlgeschlagen: {e_guard_xlsx}")
                raise

            # -------- 3) Config bauen (mit effektiv vorhandenen Exogs) ----------
            used_exog = list(self.pipeline_info.get("exog_cols") or [])
            cfg = self._build_config(
                excel_path=temp_excel,
                horizon=horizon,
                use_cached=use_cached,
                selected_exog=used_exog,
            )

            # 3a) Optional: Preload-PKL (z. B. aus Preset) übernehmen, wenn kompatibel
            try:
                if preload_model_path and os.path.exists(preload_model_path):
                    exp_path = get_model_filepath(cfg)
                    try:
                        art_pre = ModelArtifact.load(preload_model_path)
                        ok, issues = art_pre.is_compatible(cfg)
                    except Exception as _e_pre:
                        ok, issues = False, [f'Preload-Ladefehler: {_e_pre}']
                    if ok:
                        if not os.path.exists(exp_path):
                            import shutil
                            Path(exp_path).parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(preload_model_path, exp_path)
                        LOGGER.info(f"[Adapter] ✓ Preloaded Modell übernommen: {preload_model_path} → {exp_path}")
                    else:
                        LOGGER.warning(f"[Adapter] Preloaded Modell inkompatibel, retrain nötig. Gründe: {issues}")
            except Exception as _e_pl:
                LOGGER.warning(f"[Adapter] Konnte Preload-PKL nicht verwenden: {_e_pl}")

            # Kurze Config-Summary
            LOGGER.info("\n[Adapter] Config summary:")
            for k in ["forecast_horizon","agg_method_target","agg_methods_exog","exog_month_lags",
                    "target_lags_q","add_trend_features","trend_degree","add_seasonality",
                    "future_exog_strategy","target_transform","target_standardize"]:
                LOGGER.info(f"  {k}: {getattr(cfg, k, None)}")
            LOGGER.info(f"  Excel: {cfg.excel_path}")
            LOGGER.info(f"  Model dir: {cfg.model_dir}")
            LOGGER.info(f"  selected_exog/exog_cols: {getattr(cfg, 'selected_exog', [])}")

            # -------- 4) Cache-Check bzgl. Exog-Änderungen ----------
            model_path = get_model_filepath(cfg)
            if ModelArtifact and hasattr(ModelArtifact, "exists") and ModelArtifact.exists(model_path):
                try:
                    art = ModelArtifact.load(model_path)
                    old_exogs = set(art.metadata.get("exog_cols", []))
                    cur_exogs = set(getattr(cfg, "selected_exog", []))
                    if old_exogs and (old_exogs != cur_exogs):
                        LOGGER.warning("\n[Cache] Feature-Mismatch:")
                        LOGGER.warning(f"  Gecacht: {old_exogs}")
                        LOGGER.warning(f"  Aktuell: {cur_exogs}")
                        LOGGER.warning(f"[Cache] Lösche alten Cache: {model_path}")
                        try:
                            os.remove(model_path)
                        except Exception as _e_rm:
                            LOGGER.warning(f"[Cache] Konnte Cache nicht löschen: {_e_rm}")
                        force_retrain = True
                except Exception as _e_load:
                    LOGGER.warning(f"[Cache] Konnte Cache nicht prüfen: {_e_load}")

            # -------- 5) Pipeline ausführen ----------
            LOGGER.info(_sym("\n[Adapter] Starte Pipeline."))
            LOGGER.info(f"  Cache: {'verwendet' if use_cached else 'ignoriere'}")
            LOGGER.info(f"  Force-Retrain: {force_retrain}")
            forecast_df, metadata = run_production_pipeline(cfg, force_retrain)

            # -------- 6) Modellpfad + Snapshot persistieren ----------
            model_path = get_model_filepath(cfg)
            if isinstance(metadata, dict):
                metadata["model_path"] = model_path
                metadata["confidence_levels"] = confidence_levels

            snapshot_dir = Path(getattr(cfg, "output_dir", "outputs"))
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"snapshot_{Path(model_path).stem}.parquet"
            try:
                prepared_df.to_parquet(snapshot_path)
                if isinstance(metadata, dict):
                    metadata["exog_snapshot_path"] = str(snapshot_path)
            except Exception as _e_pq:
                LOGGER.warning(f"[Adapter] Snapshot konnte nicht geschrieben werden: {_e_pq}")

            # -------- 7) Konfidenzintervalle hinzufügen ----------
            try:
                LOGGER.info(_sym("\n[Adapter] Berechne Konfidenzintervalle"))
                residuals = self._extract_residuals_from_pipeline(metadata, model_path)

                if residuals is None or (isinstance(residuals, (list, np.ndarray, pd.Series)) and len(residuals) == 0):
                    # Fallback: CV-RMSE nutzen, falls vorhanden
                    cv_rmse = None
                    try:
                        cv_perf = (metadata or {}).get("cv_performance", {})
                        cv_rmse = float(cv_perf.get("cv_rmse") or cv_perf.get("rmse"))
                    except Exception:
                        pass
                    if cv_rmse and np.isfinite(cv_rmse):
                        LOGGER.warning(f"[CI] Keine Residuen gefunden – erzeuge synthetische Residuen (RMSE={cv_rmse:.4f}, n=100)")
                        rng = np.random.default_rng(42)
                        residuals = rng.normal(0.0, cv_rmse, size=100)
                    else:
                        LOGGER.warning("[CI] Keine Residuen verfügbar und kein CV-RMSE – CI werden übersprungen")
                        residuals = None

                if residuals is not None:
                    forecast_df = self._add_confidence_intervals_to_forecast(
                        forecast_df=forecast_df,
                        residuals=residuals,
                        confidence_levels=confidence_levels,
                        forecast_col="Forecast"
                    )
                    # std_error in Metadaten ablegen (hilfreich für Coverage-Diagnostik)
                    if isinstance(metadata, dict):
                        try:
                            res_arr = np.asarray(residuals, dtype=float)
                            if res_arr.size > 1:
                                se = float(np.std(res_arr, ddof=1))
                                metadata["ci_std_error"] = se
                                metadata.setdefault("diagnostics", {}).setdefault("ci", {})["std_error"] = se
                        except Exception:
                            pass

                    LOGGER.info(f"[CI] Konfidenzintervalle erfolgreich hinzugefügt")
                    LOGGER.info(f"[CI] Forecast-Spalten: {forecast_df.columns.tolist()}")

            except Exception as e_ci:
                LOGGER.error(f"[CI] Fehler bei Konfidenzintervall-Berechnung: {e_ci}")
                LOGGER.warning("[CI] Forecast wird ohne Konfidenzintervalle zurückgegeben")

            # -------- 8) Backtest erzeugen ----------
            try:
                LOGGER.info("[Adapter] Generiere Backtest-Daten.")
                if ModelArtifact and hasattr(ModelArtifact, 'exists') and ModelArtifact.exists(model_path):
                    artifact = ModelArtifact.load(model_path)
                    X_train = artifact.metadata.get('X_train')
                    y_train = artifact.metadata.get('y_train')
                    dates_train = artifact.metadata.get('dates_train')

                    if X_train is None or y_train is None or dates_train is None:
                        LOGGER.info("[Backtest] Training-Daten nicht in Artifact, extrahiere aus prepared_df")
                        from forecaster_pipeline import (aggregate_to_quarter, add_deterministic_features, build_quarterly_lags)
                        df_in = prepared_df.copy()
                        if cfg is not None:
                            rename_map = {}
                            if 'date' in df_in.columns and getattr(cfg, "date_col", "date") not in df_in.columns:
                                rename_map['date'] = cfg.date_col
                            if 'target_value' in df_in.columns and getattr(cfg, "target_col", "target") not in df_in.columns:
                                rename_map['target_value'] = cfg.target_col
                            if rename_map:
                                df_in = df_in.rename(columns=rename_map)
                        df_q = aggregate_to_quarter(df_in, cfg)
                        df_q = add_deterministic_features(df_q, cfg)
                        df_feats = build_quarterly_lags(df_q, cfg)

                        X_cols = artifact.X_cols
                        X_train = df_feats[X_cols]
                        y_train = df_feats[cfg.target_col]
                        if 'Q_end' in df_feats.columns:
                            dates_train = df_feats['Q_end']
                        else:
                            dates_train = df_feats[getattr(cfg, "date_col", "Datum")]

                    # Backtest rechnen (robust, mit Transform-Handling)
                    backtest_df, residuals_df = self._build_backtest_from_artifact(
                        artifact=artifact,
                        X_train=X_train,
                        y_train=y_train,
                        dates_train=dates_train,
                    )
                    # in Metadaten anhängen
                    if isinstance(metadata, dict):
                        metadata["backtest_results"] = backtest_df.to_dict(orient="records")
                        metadata["backtest_residuals"] = residuals_df.to_dict(orient="records")

                else:
                    LOGGER.warning("[Backtest] Kein gespeichertes Modell gefunden – Backtest entfällt.")
            except Exception as e_bt:
                LOGGER.warning(f"[Backtest] Fehler bei Backtest-Erstellung: {e_bt}")

            return forecast_df, metadata

        except Exception as e:
            LOGGER.exception(f"[Adapter] RUN_FORECAST fehlgeschlagen: {e}")
            raise
        finally:
            # Aufräumen (temp Excel)
            try:
                if temp_excel and os.path.exists(temp_excel):
                    os.remove(temp_excel)
            except Exception:
                pass

    
    # def run_forecast(
    #     self,
    #     target: str,
    #     selected_exog: List[str],
    #     horizon: int,
    #     use_cached: bool,
    #     force_retrain: bool,
    #     use_flows: bool = False,
    #     confidence_levels: List[int] = None,
    #     preload_model_path: Optional[str] = None,
    # ):
    #     """
    #     Führt Forecast durch mit Konfidenzintervallen und Backtest-Generierung.
    #     (mit Punkt A: Exog-Resolve + Guards + Polling auf Exog-Verfügbarkeit)
    #     """
    #     if confidence_levels is None:
    #         confidence_levels = [80, 95]

    #     LOGGER.info(_sym("\n[Adapter] ===== RUN FORECAST ====="))
    #     LOGGER.info(f"[Adapter] Target: {target} | Horizon: {horizon} | use_cached={use_cached} | force_retrain={force_retrain}")
    #     LOGGER.info(f"[Adapter] Selected exog ({len(selected_exog)}): {selected_exog}")
    #     LOGGER.info(f"[Adapter] use_flows={use_flows}")
    #     LOGGER.info(f"[Adapter] confidence_levels={confidence_levels}")

    #     self.pipeline_info["ui_target"] = target

    #     temp_excel: Optional[str] = None
    #     try:
    #         # --- B) Warten bis Exogs verfügbar sind (falls Loader asynchron arbeitet) --------
    #         # Hooks (optional):
    #         #   - self.fetch_exogs(codes: list[str]) -> None   : stößt Download an
    #         #   - self.get_exog_data() -> pd.DataFrame | None  : liefert aktuellen Exog-Frame
    #         #   - self.exog_data                                : DataFrame im Adapter
    #         try:
    #             import time
    #             requested = list(selected_exog or [])
    #             LOGGER.info(f"[Adapter] requested_exog={requested}")

    #             def _get_exog_df():
    #                 if hasattr(self, "get_exog_data") and callable(getattr(self, "get_exog_data")):
    #                     return self.get_exog_data()
    #                 return getattr(self, "exog_data", None) if hasattr(self, "exog_data") else None

    #             def _has_any_requested(dfx: Optional[pd.DataFrame], codes: list[str]) -> bool:
    #                 if not isinstance(dfx, pd.DataFrame) or dfx.empty or not codes:
    #                     return False
    #                 cs = {str(c) for c in dfx.columns}
    #                 for code in codes:
    #                     if (code in cs) or (f"{code}__last" in cs) or (f"{code}__last__" in cs):
    #                         return True
    #                 return False

    #             cur_exog_df = _get_exog_df()
    #             if isinstance(cur_exog_df, pd.DataFrame):
    #                 LOGGER.info(f"[Adapter|Exog] Frame: shape={cur_exog_df.shape}")
    #                 LOGGER.info(f"[Adapter] exog_data.columns (Top): {list(cur_exog_df.columns[:5])}")
    #             else:
    #                 LOGGER.info("[Adapter|Exog] Frame: None")

    #             need_exog = bool(requested)
    #             have_exog_now = _has_any_requested(cur_exog_df, requested)

    #             if need_exog and not have_exog_now:
    #                 # Download (best effort) anstoßen
    #                 if hasattr(self, "fetch_exogs") and callable(getattr(self, "fetch_exogs")):
    #                     try:
    #                         self.fetch_exogs(requested)
    #                     except Exception as _e_fetch:
    #                         LOGGER.warning(f"[Adapter|Exog] fetch_exogs() Fehler/kein Hook: {_e_fetch}")

    #                 # Kurzes Polling-Fenster
    #                 max_wait_sec = 12.0
    #                 poll_sec = 0.3
    #                 t0 = time.time()
    #                 while time.time() - t0 < max_wait_sec and not have_exog_now:
    #                     time.sleep(poll_sec)
    #                     cur_exog_df = _get_exog_df()
    #                     have_exog_now = _has_any_requested(cur_exog_df, requested)

    #                 if have_exog_now:
    #                     LOGGER.info("[Adapter|Exog] Exogs verfügbar – fahre mit Datenaufbereitung fort.")
    #                 else:
    #                     LOGGER.warning("[Adapter|Exog] Timeout: Exogs noch nicht verfügbar – fahre ohne Exogene fort.")
    #         except Exception as _e_wait:
    #             LOGGER.warning(f"[Adapter|Exog] Wait-Block übersprungen: {_e_wait}")

    #         # 1) Daten vorbereiten (MS, kein Backfill) – erst NACH dem Wait-Block!
    #         prepared_df = self.prepare_pipeline_data(
    #             target=target,
    #             selected_exog=selected_exog,
    #             use_flows=use_flows,
    #             horizon_quarters=int(horizon or 0),
    #         )

    #         # ---- 1a) Resolve & Filter der Exogs im DataFrame (sanft, nicht abbrechen) ----
    #         def _resolve_exogs_in_df(df: pd.DataFrame, requested: List[str]) -> tuple[list[str], dict]:
    #             cols_set = {str(c) for c in df.columns}
    #             variants_map = self._expand_exog_variants(requested or [])
    #             resolved_list: list[str] = []
    #             resolved_map: dict[str, str] = {}
    #             missing: list[str] = []
    #             for req, cands in variants_map.items():
    #                 match = next((c for c in cands if c in cols_set), None)
    #                 if match is None:
    #                     missing.append(req)
    #                 else:
    #                     resolved_map[req] = match
    #                     resolved_list.append(match)
    #             return resolved_list, {"missing": missing, "mapping": resolved_map}

    #         resolved_exogs, diag_resolve = _resolve_exogs_in_df(prepared_df, list(selected_exog or []))
    #         if diag_resolve["missing"]:
    #             LOGGER.warning(
    #                 "[Guard] Einige gewünschte Exogs fehlen im DataFrame und werden herausgefiltert: "
    #                 + ", ".join(diag_resolve["missing"])
    #             )
    #         if resolved_exogs:
    #             LOGGER.info(f"[Guard] Exogs im DF aufgelöst: {diag_resolve['mapping']}")
    #         else:
    #             LOGGER.warning("[Guard] Keine Exogs im DF auflösbar – Forecast läuft ohne Exogene weiter.")

    #         # Für die nachfolgenden Schritte merken
    #         self.pipeline_info["exog_cols"] = list(resolved_exogs)

    #         # 1b) DF-Guard (non-strict: nur Hinweis/Logging, kein Abbruch)
    #         try:
    #             self._validate_exog_presence_in_df(
    #                 df_final=prepared_df,
    #                 selected_exog=list(selected_exog or []),
    #                 strict=False,   # wegen vorigem Filter absichtlich nicht strikt
    #                 logger=LOGGER,
    #             )
    #         except Exception as e_guard_df:
    #             LOGGER.warning(f"[Guard/DF] (non-strict) Hinweis: {e_guard_df}")

    #         # 2) Excel erzeugen
    #         temp_excel = self.create_temp_excel(prepared_df)

    #         # 2a) Excel-Guard gegen die **gefilterten** Exogs (strict=True)
    #         try:
    #             self._validate_exog_presence_in_excel(
    #                 excel_path=temp_excel,
    #                 selected_exog=list(resolved_exogs),  # nur die wirklich vorhandenen
    #                 sheet_name="final_dataset",
    #                 strict=True,
    #                 logger=LOGGER,
    #             )
    #         except Exception as e_guard_xlsx:
    #             LOGGER.error(f"[Guard/Excel] Validierung fehlgeschlagen: {e_guard_xlsx}")
    #             raise

    #         # 3) Config bauen – mit den **effektiv** vorhandenen Exogs
    #         used_exog = list(self.pipeline_info.get("exog_cols") or [])
    #         cfg = self._build_config(
    #             excel_path=temp_excel,
    #             horizon=horizon,
    #             use_cached=use_cached,
    #             selected_exog=used_exog,
    #         )

    #         # 3a) Optional: Vorgegebenes Modell aus Preset nutzen (PKL-Reuse)
    #         try:
    #             if preload_model_path and os.path.exists(preload_model_path):
    #                 exp_path = get_model_filepath(cfg)
    #                 try:
    #                     art_pre = ModelArtifact.load(preload_model_path)
    #                     ok, issues = art_pre.is_compatible(cfg)
    #                 except Exception as _e_pre:
    #                     ok, issues = False, [f'Preload-Ladefehler: {_e_pre}']
    #                 if ok:
    #                     if not os.path.exists(exp_path):
    #                         import shutil
    #                         Path(exp_path).parent.mkdir(parents=True, exist_ok=True)
    #                         shutil.copy2(preload_model_path, exp_path)
    #                     LOGGER.info(f"[Adapter] ✓ Preloaded Modell übernommen: {preload_model_path} → {exp_path}")
    #                 else:
    #                     LOGGER.warning(f"[Adapter] Preloaded Modell inkompatibel, retrain nötig. Gründe: {issues}")
    #         except Exception as _e_pl:
    #             LOGGER.warning(f"[Adapter] Konnte Preload-PKL nicht verwenden: {_e_pl}")

    #         # Kurze Config-Zusammenfassung
    #         LOGGER.info("\n[Adapter] Config summary:")
    #         dbg_keys = [
    #             "forecast_horizon", "agg_method_target", "agg_methods_exog", "exog_month_lags",
    #             "target_lags_q", "add_trend_features", "trend_degree", "add_seasonality",
    #             "future_exog_strategy", "target_transform", "target_standardize",
    #         ]
    #         for k in dbg_keys:
    #             LOGGER.info(f"  {k}: {getattr(cfg, k, None)}")
    #         LOGGER.info(f"  Excel: {cfg.excel_path}")
    #         LOGGER.info(f"  Model dir: {cfg.model_dir}")
    #         LOGGER.info(f"  selected_exog/exog_cols: {getattr(cfg, 'selected_exog', [])}")

    #         # 4) Cache-Check bzgl. Exog-Änderungen
    #         model_path = get_model_filepath(cfg)
    #         if ModelArtifact and hasattr(ModelArtifact, "exists") and ModelArtifact.exists(model_path):
    #             try:
    #                 art = ModelArtifact.load(model_path)
    #                 old_exogs = set(art.metadata.get("exog_cols", []))
    #                 cur_exogs = set(getattr(cfg, "selected_exog", []))
    #                 if old_exogs and (old_exogs != cur_exogs):
    #                     LOGGER.warning("\n[Cache] Feature-Mismatch:")
    #                     LOGGER.warning(f"  Gecacht: {old_exogs}")
    #                     LOGGER.warning(f"  Aktuell: {cur_exogs}")
    #                     LOGGER.warning(f"[Cache] Lösche alten Cache: {model_path}")
    #                     try:
    #                         os.remove(model_path)
    #                     except Exception as _e_rm:
    #                         LOGGER.warning(f"[Cache] Konnte Cache nicht löschen: {_e_rm}")
    #                     force_retrain = True
    #             except Exception as _e_load:
    #                 LOGGER.warning(f"[Cache] Konnte Cache nicht prüfen: {_e_load}")

    #         # 5) Pipeline ausführen
    #         LOGGER.info(_sym("\n[Adapter] Starte Pipeline."))
    #         LOGGER.info(f"  Cache: {'verwendet' if use_cached else 'ignoriere'}")
    #         LOGGER.info(f"  Force-Retrain: {force_retrain}")
    #         forecast_df, metadata = run_production_pipeline(cfg, force_retrain)

    #         # 6) Modellpfad + Snapshot persistieren
    #         model_path = get_model_filepath(cfg)
    #         if isinstance(metadata, dict):
    #             metadata["model_path"] = model_path
    #             metadata["confidence_levels"] = confidence_levels

    #         snapshot_dir = Path(getattr(cfg, "output_dir", "outputs"))
    #         snapshot_dir.mkdir(parents=True, exist_ok=True)
    #         snapshot_path = snapshot_dir / f"snapshot_{Path(model_path).stem}.parquet"
    #         try:
    #             prepared_df.to_parquet(snapshot_path)
    #             if isinstance(metadata, dict):
    #                 metadata["exog_snapshot_path"] = str(snapshot_path)
    #         except Exception as _e_pq:
    #             LOGGER.warning(f"[Adapter] Snapshot konnte nicht geschrieben werden: {_e_pq}")

    #         # 7) Residuen extrahieren & CIs hinzufügen
    #         try:
    #             LOGGER.info(_sym("\n[Adapter] Berechne Konfidenzintervalle"))
    #             residuals = self._extract_residuals_from_pipeline(metadata, model_path)

    #             if residuals is None or (isinstance(residuals, (list, np.ndarray, pd.Series)) and len(residuals) == 0):
    #                 cv_rmse = None
    #                 try:
    #                     cv_perf = (metadata or {}).get("cv_performance", {})
    #                     cv_rmse = float(cv_perf.get("cv_rmse") or cv_perf.get("rmse"))
    #                 except Exception:
    #                     pass
    #                 if cv_rmse and np.isfinite(cv_rmse):
    #                     LOGGER.warning(f"[CI] Keine Residuen gefunden – erzeuge synthetische Residuen (RMSE={cv_rmse:.4f}, n=100)")
    #                     rng = np.random.default_rng(42)
    #                     residuals = rng.normal(0.0, cv_rmse, size=100)
    #                 else:
    #                     LOGGER.warning("[CI] Keine Residuen verfügbar und kein CV-RMSE – CI werden übersprungen")
    #                     residuals = None

    #             if residuals is not None:
    #                 forecast_df = self._add_confidence_intervals_to_forecast(
    #                     forecast_df=forecast_df,
    #                     residuals=residuals,
    #                     confidence_levels=confidence_levels,
    #                     forecast_col="Forecast"
    #                 )
    #                 if isinstance(metadata, dict):
    #                     try:
    #                         res_arr = np.asarray(residuals, dtype=float)
    #                         if res_arr.size > 1:
    #                             se = float(np.std(res_arr, ddof=1))
    #                             metadata["ci_std_error"] = se
    #                             metadata.setdefault("diagnostics", {}).setdefault("ci", {})["std_error"] = se
    #                     except Exception:
    #                         pass

    #                 LOGGER.info(f"[CI] Konfidenzintervalle erfolgreich hinzugefügt")
    #                 LOGGER.info(f"[CI] Forecast-Spalten: {forecast_df.columns.tolist()}")

    #         except Exception as e_ci:
    #             LOGGER.error(f"[CI] Fehler bei Konfidenzintervall-Berechnung: {e_ci}")
    #             LOGGER.warning("[CI] Forecast wird ohne Konfidenzintervalle zurückgegeben")

    #         # --- OUTPUT DIAGNOSTICS: Forecast-Flatline -------------------------
    #         try:
    #             yhat = None
    #             for c in ["Forecast", "yhat", "y_pred"]:
    #                 if c in forecast_df.columns:
    #                     yhat = forecast_df[c].astype(float)
    #                     break
    #             if yhat is not None and len(yhat) >= 2:
    #                 std_yhat = float(yhat.std(ddof=1))
    #                 mean_yhat = float(yhat.mean())
    #                 flat = std_yhat < 1e-8
    #                 (metadata.setdefault("diagnostics", {})
    #                         .setdefault("forecast", {})
    #                         .update({"std": std_yhat, "mean": mean_yhat, "is_flatline": flat}))
    #                 if flat:
    #                     LOGGER.warning("[DIAG|FORECAST] Prognose ist (nahezu) konstant – mögliche Ursachen: "
    #                                 "degenerates Target/Features, starke Regularisierung oder fehlerhafte Transformation")
    #         except Exception as _e:
    #             LOGGER.warning(f"[DIAG|FORECAST] übersprungen: {_e}")

    #         # 8) Backtest – unverändert
    #         try:
    #             LOGGER.info("[Adapter] Generiere Backtest-Daten...")
    #             if ModelArtifact and hasattr(ModelArtifact, 'exists') and ModelArtifact.exists(model_path):
    #                 artifact = ModelArtifact.load(model_path)
    #                 X_train = artifact.metadata.get('X_train')
    #                 y_train = artifact.metadata.get('y_train')
    #                 dates_train = artifact.metadata.get('dates_train')

    #                 if X_train is None or y_train is None or dates_train is None:
    #                     LOGGER.info("[Backtest] Training-Daten nicht in Artifact, extrahiere aus prepared_df")
    #                     from forecaster_pipeline import (aggregate_to_quarter, add_deterministic_features, build_quarterly_lags)
    #                     df_in = prepared_df.copy()
    #                     if cfg is not None:
    #                         rename_map = {}
    #                         if 'date' in df_in.columns and getattr(cfg, "date_col", "date") not in df_in.columns:
    #                             rename_map['date'] = cfg.date_col
    #                         if 'target_value' in df_in.columns and getattr(cfg, "target_col", "target") not in df_in.columns:
    #                             rename_map['target_value'] = cfg.target_col
    #                         if rename_map:
    #                             df_in = df_in.rename(columns=rename_map)
    #                     df_q = aggregate_to_quarter(df_in, cfg)
    #                     df_q = add_deterministic_features(df_q, cfg)
    #                     df_feats = build_quarterly_lags(df_q, cfg)

    #                     X_cols = artifact.X_cols
    #                     X_train = df_feats[X_cols]
    #                     y_train = df_feats[cfg.target_col]
    #                     if 'Q_end' in df_feats.columns:
    #                         dates_train = df_feats['Q_end']
    #                     elif 'Q' in df_feats.columns:
    #                         dates_train = df_feats['Q']
    #                     else:
    #                         dates_train = df_feats.index

    #                 try:
    #                     yv = pd.Series(y_train).astype(float)
    #                     diag_train = {
    #                         "y_std": float(yv.std(ddof=1)) if len(yv) > 1 else 0.0,
    #                         "y_mean": float(yv.mean()),
    #                         "y_share_zero": float((yv == 0).mean())
    #                     }
    #                     if diag_train["y_std"] < 1e-8 and len(yv) >= 8:
    #                         LOGGER.warning("[DIAG|TRAIN] Target ist (nahezu) konstant – Flatline-Risiko hoch")
    #                     (metadata.setdefault("diagnostics", {})
    #                             .setdefault("train", {})
    #                             .update(diag_train))
    #                 except Exception as _e:
    #                     LOGGER.warning(f"[DIAG|TRAIN] übersprungen: {_e}")

    #                 if X_train is not None and y_train is not None and dates_train is not None:
    #                     backtest_results, backtest_residuals = self._generate_backtest_results(
    #                         model=artifact.model,
    #                         tj=getattr(artifact, "tj", None),
    #                         X_train=X_train,
    #                         y_train=y_train,
    #                         dates_train=dates_train,
    #                         n_splits=5
    #                     )

    #                     if isinstance(metadata, dict) and isinstance(backtest_results, pd.DataFrame):
    #                         metadata['backtest_results'] = backtest_results.to_dict(orient='records')
    #                         if isinstance(backtest_residuals, pd.DataFrame):
    #                             metadata['backtest_residuals'] = backtest_residuals.to_dict(orient='records')

    #                     try:
    #                         artifact.metadata['backtest_results'] = backtest_results
    #                         artifact.metadata['backtest_residuals'] = backtest_residuals
    #                         artifact.save(model_path)
    #                     except Exception as _e_save:
    #                         LOGGER.warning(f"[Adapter] Konnte Backtest im Artifact nicht speichern: {_e_save}")

    #                     LOGGER.info("[Adapter] ✓ Backtest-Daten erfolgreich generiert und gespeichert")
    #                     LOGGER.info(f"[Backtest] {len(backtest_results)} Vorhersagen, {len(backtest_residuals)} Residuen")

    #                     try:
    #                         bt = backtest_results.copy()
    #                         share_zero_pred = float((bt['predicted'].astype(float) == 0).mean()) if 'predicted' in bt.columns else 0.0
    #                         flat_pred = False
    #                         if 'predicted' in bt.columns and len(bt) >= 2:
    #                             flat_pred = bool(bt['predicted'].astype(float).std(ddof=1) < 1e-8)
    #                         (metadata.setdefault("diagnostics", {})
    #                                 .setdefault("backtest", {})
    #                                 .update({
    #                                     "share_zero_pred": share_zero_pred,
    #                                     "flat_pred": flat_pred,
    #                                     "n_points": int(len(bt))
    #                                 }))
    #                         if share_zero_pred > 0.3 or flat_pred:
    #                             LOGGER.warning(f"[DIAG|BACKTEST] Auffällig: {share_zero_pred:.0%} der historischen "
    #                                         f"Vorhersagen sind exakt 0 oder nahezu konstant")
    #                     except Exception as _e:
    #                         LOGGER.warning(f"[DIAG|BACKTEST] übersprungen: {_e}")
    #                 else:
    #                     LOGGER.warning("[Adapter] Training-Daten nicht verfügbar für Backtest")
    #             else:
    #                 LOGGER.warning("[Adapter] Model-Artifact nicht gefunden für Backtest")
    #         except Exception as e_bt:
    #             LOGGER.warning(f"[Adapter] Backtest-Generierung fehlgeschlagen: {e_bt}")

    #         # 9) Ergebnis-Log
    #         LOGGER.info(_sym("\n[Adapter] Pipeline erfolgreich:"))
    #         if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
    #             try:
    #                 LOGGER.info(f"  Prognose: {forecast_df['Forecast'].values}")
    #                 for level in confidence_levels:
    #                     if f'yhat_lower_{level}' in forecast_df.columns:
    #                         lower = forecast_df[f'yhat_lower_{level}'].values
    #                         upper = forecast_df[f'yhat_upper_{level}'].values
    #                         LOGGER.info(f"  {level}% CI: [{lower[0]:.2f}, {upper[0]:.2f}] ... [{lower[-1]:.2f}, {upper[-1]:.2f}]")
    #             except Exception:
    #                 pass

    #         if isinstance(metadata, dict):
    #             best = metadata.get("best_params", {})
    #             cv = metadata.get("cv_performance", {})
    #             LOGGER.info(f"  → Beste Parameter: {best}")
    #             if cv:
    #                 LOGGER.info(f"  → CV-RMSE: {cv.get('cv_rmse', cv.get('rmse', 'n/a'))}")
    #             mc = metadata.get("model_complexity", {})
    #             if mc:
    #                 LOGGER.info(f"  → {mc.get('n_features', 'n/a')} Features")
    #             if 'backtest_results' in metadata:
    #                 bt_len = len(metadata['backtest_results']) if isinstance(metadata['backtest_results'], list) else 0
    #                 LOGGER.info(f"  → Backtest: {bt_len} historische Vorhersagen verfügbar")

    #         return forecast_df, metadata

    #     except Exception as e:
    #         LOGGER.exception(f"[Adapter] FEHLER: {e}")
    #         raise

    #     finally:
    #         if temp_excel:
    #             try:
    #                 Path(temp_excel).unlink(missing_ok=True)
    #             except Exception:
    #                 try:
    #                     os.remove(temp_excel)
    #                 except Exception:
    #                     pass


    # ------------------------------------------------------------
    # NEU: CI-Coverage berechnen (für Qualitäts-Monitoring)
    # ------------------------------------------------------------
    @staticmethod
    def calculate_ci_coverage(
        actual_values: np.ndarray,
        predictions: np.ndarray,
        lower_bounds: Dict[int, np.ndarray],
        upper_bounds: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """
        Berechnet die tatsächliche Coverage der Konfidenzintervalle.
        
        Args:
            actual_values: Array mit tatsächlichen Werten
            predictions: Array mit Punktprognosen
            lower_bounds: Dict mit unteren Grenzen pro Level {80: array, 95: array}
            upper_bounds: Dict mit oberen Grenzen pro Level {80: array, 95: array}
        
        Returns:
            Dict mit Coverage-Prozenten pro Level {80: 78.5, 95: 94.2}
        """
        coverage = {}
        
        for level in lower_bounds.keys():
            if level not in upper_bounds:
                continue
            
            lower = lower_bounds[level]
            upper = upper_bounds[level]
            
            # Prüfe wie viele Werte im Intervall liegen
            within_ci = (actual_values >= lower) & (actual_values <= upper)
            coverage_pct = np.mean(within_ci) * 100
            
            coverage[level] = float(coverage_pct)
            
            LOGGER.info(f"[CI Coverage] {level}% CI → tatsächliche Coverage: {coverage_pct:.1f}%")
        
        return coverage

    # ------------------------------------------------------------
    # Models auflisten (für UI)
    # ------------------------------------------------------------
    @staticmethod
    def get_available_models() -> List[str]:
        """Listet verfügbare Modelle (für UI-Dialog)."""
        try:
            from forecaster_pipeline import list_saved_models  # type: ignore
            return list_saved_models()
        except Exception:
            return []
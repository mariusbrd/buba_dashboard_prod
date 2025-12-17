# forecast_integration.py
# Aufgabe: Adapter zwischen Dashboard und Kernpipeline. Nimmt Dashboard Daten an, bereitet sie für die Pipeline auf und formt das Ergebnis für die Anzeige.

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
from typing import Dict, List, Optional, Tuple, Union, Sequence
import re
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

# Kompatibel zu app.py: App-Root des Projekts
try:
    APP_ROOT: Path = ROOT_DIR
except Exception:
    APP_ROOT = Path.cwd()

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
PipelineConfig = None  # type: ignore[assignment]
ModelArtifact = None  # type: ignore[assignment]
get_model_filepath = None  # type: ignore[assignment]
run_production_pipeline = None  # type: ignore[assignment]

# Wir versuchen mehrere Varianten in sicherer Reihenfolge
try:
    # 1) Direkter Import (jetzt mit Pfad-Fix)
    from src.forecaster.forecaster_pipeline import (  # type: ignore
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
        from src.forecaster.forecaster_pipeline import (  # type: ignore
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
                    from src.backend.forecaster.forecaster_pipeline import (  # type: ignore
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
                        from src.backend.forecaster.forecaster_pipeline import (  # type: ignore
                            Config as PipelineConfig,
                            run_production_pipeline,
                            ModelArtifact,
                            get_model_filepath,
                        )

                        HAS_PIPELINE = True
                        LOGGER.info("[Pipeline] ✓ Import erfolgreich (src.frontend.forecaster.forecaster_pipeline, Config → PipelineConfig)")
                    except Exception as e6:
                        # Alle Versuche fehlgeschlagen → Flags setzen und Fehler protokollieren
                        HAS_PIPELINE = False
                        _PIPELINE_IMPORT_ERROR = e6 or e5 or e4 or e3 or e2 or e1
                        PipelineConfig = None  # type: ignore[assignment]
                        ModelArtifact = None  # type: ignore[assignment]
                        get_model_filepath = None  # type: ignore[assignment]
                        run_production_pipeline = None  # type: ignore[assignment]

                        # Detaillierte Fehlerausgabe
                        LOGGER.error("=" * 80)
                        LOGGER.error("[Pipeline] ❌ ALLE Import-Versuche fehlgeschlagen!")
                        LOGGER.error(f"[Pipeline] Forecaster-Dir: {FORECASTER_DIR}")
                        LOGGER.error(f"[Pipeline] Root-Dir (APP_ROOT): {APP_ROOT}")
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
                                with open(pipeline_file, "r", encoding="utf-8") as f:
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
    - SCHRITT 6: Dashboard-freundliches Ergebnis bauen
    """

    # ------------------------------------------------------------
    # Init
    # ------------------------------------------------------------

    def __init__(
        self,
        gvb_store_json: str,
        exog_store_json: str,
        custom_final_dataset: Optional[Union[dict, str]] = None,
    ):
        # GVB laden
        self.gvb_data = pd.read_json(gvb_store_json, orient="split")
        self.gvb_data["date"] = pd.to_datetime(self.gvb_data["date"], errors="coerce")
        self.gvb_data = (
            self.gvb_data
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Exogene laden (robust)
        self.exog_data = self._load_exog_frame(exog_store_json)

        # Fallback Cache
        if (
            self.exog_data is None
            or self.exog_data.empty
            or list(self.exog_data.columns) == ["date"]
            or self.exog_data.shape[1] <= 1
        ):
            cache_df = self._load_exog_from_default_cache()
            if not cache_df.empty:
                self.exog_data = cache_df
                LOGGER.info(
                    "[Adapter|Exog] Cache-Fallback beim Init verwendet "
                    "(loader/financial_cache/output.xlsx)."
                )

        LOGGER.info(
            f"[Adapter|Exog] Frame: shape="
            f"{self.exog_data.shape if hasattr(self, 'exog_data') else None}"
        )
        if not self.exog_data.empty:
            LOGGER.info(
                f"[Adapter|Exog] Columns (Top): "
                f"{self.exog_data.columns.tolist()[:12]}"
            )
            if "date" not in self.exog_data.columns:
                LOGGER.warning(
                    "[Adapter|Exog] Achtung: 'date' Spalte fehlt im Exog-Frame!"
                )

        # Optional: benutzerdefiniertes PIPELINE_PREP aus Upload
        self.custom_prepared_df: Optional[pd.DataFrame] = self._load_custom_prepared_df(
            custom_final_dataset
        )
        if self.custom_prepared_df is not None:
            LOGGER.info(
                _sym(
                    f"[Adapter] Custom PIPELINE_PREP aktiv – "
                    f"shape={self.custom_prepared_df.shape}"
                )
            )

        # für Debug/Summary
        self.pipeline_info: Dict[str, object] = {}

        LOGGER.info(_sym("✓ Pipeline-Integration geladen"))

        # --- INPUT DIAGNOSTICS ---------------------------------------------
        try:
            diag = {}
            for mode, vcol in (("bestand", "bestand"), ("fluss", "fluss")):
                if "datatype" in self.gvb_data.columns and vcol in self.gvb_data.columns:
                    s = (
                        self.gvb_data.loc[self.gvb_data["datatype"] == mode, vcol]
                        .dropna()
                        .astype(float)
                    )
                    if not s.empty:
                        share_zero = float((s == 0).mean())
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




    # ------------------------------------------------------------------
    # Namens-Normalisierung für Exogs
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_exog_key(raw: str) -> str:
        """
        Normiert einen Exog-Namen so, dass verschiedene Schreibweisen
        (Punkte, Unterstriche, Merge-Suffixe, __last, Lags etc.)
        auf denselben Schlüssel gemappt werden.
        """
        if raw is None:
            return ""
        name = str(raw)

        # 1) bekannte Suffixe für Feature-Engineering entfernen
        #    z.B. "__last", "__last__lag-1Q", "__l", "__lag-1Q", etc.
        name = re.sub(r"(__last.*$|__lag.*$|__l$)", "", name)

        # 2) Merge-Suffixe aus pd.merge entfernen: "_x", "_y"
        name = re.sub(r"_[xy]$", "", name)

        # 3) alles, was kein Buchstabe/Ziffer ist, durch "_" ersetzen
        name = re.sub(r"[^A-Za-z0-9]+", "_", name)

        # 4) Mehrfach-Underscores reduzieren und trimmen
        name = re.sub(r"_+", "_", name).strip("_")

        # 5) für den Vergleich in Kleinbuchstaben
        return name.lower()

    @staticmethod
    def _normalize_exog_name_for_match(name: str) -> str:
        """
        Bringt UI-Codes (mit Punkten) und Spaltennamen (mit Unterstrichen und Suffixen)
        auf eine gemeinsame Vergleichsform.

        Beispiele:
        - "IRS.M.DE.L.L40.CI.0000.EUR.N.Z"             -> "IRS_M_DE_L_L40_CI_0000_EUR_N_Z"
        - "IRS_M_DE_L_L40_CI_0000_EUR_N_Z__last"       -> "IRS_M_DE_L_L40_CI_0000_EUR_N_Z"
        - "QSA.Q.N.DE.W0.S1M.S1.N.A.LE.F._Z._Z.XDC..." -> "QSA_Q_N_DE_W0_S1M_S1_N_A_LE_F__Z__Z_XDC__T_S_V_N__T"
        """
        if not isinstance(name, str):
            name = str(name)

        s = name.upper()

        # Punkte und sonstige Sonderzeichen vereinheitlichen
        # – entspricht im Prinzip deiner YAML-Definition (re.sub r"[^A-Za-z0-9_]+", "_")
        s = s.replace(".", "_")
        s = re.sub(r"[^A-Z0-9_]+", "_", s)

        # Mehrfache Unterstriche zusammenziehen
        s = re.sub(r"_+", "_", s)

        # führende/abschließende Unterstriche weg
        s = s.strip("_")

        # typisches Suffix "__LAST" (und Varianten wie "__L", "__LAG-1Q") für Vergleich ignorieren
        # alles ab dem ersten "__LAST" oder "__LAG" etc abschneiden
        s = re.sub(r"(__LAST.*)$", "", s)
        s = re.sub(r"(__LAG.*)$", "", s)
        # falls du weitere Suffixmuster verwendest, hier ergänzen

        return s

    # ------------------------------------------------------------
    # Orchestrierungs-Schritte (neu)
    # ------------------------------------------------------------
    def _step_1_prepare_from_dashboard(
            self,
            *,
            target: str,
            selected_exog: list[str],
            horizon: int,
            use_flows: bool,
            sektor: Optional[str],
        ) -> dict:
            """
            Schritt 1: Alles aus dem Dashboard einlesen, Exogs bereitstellen, Daten für die Pipeline vorbereiten,
            Zielspalte bestimmen und Cache-Tag erzeugen.

            NEU / robust (non-blocking):
            - es wird ein run-spezifischer Ordner erzeugt: loader/runs/<cache_tag>/<timestamp>/
            - dort erwarten wir den Loader-Output (output.xlsx)
            - wenn er (jetzt) nicht da ist, schauen wir nur einmal auf den globalen Output
            - es wird eine Übergabe-Datei loader/active_run.json geschrieben (Handshake Adapter → Loader)
            - es wird zurückgegeben, welche Quelle tatsächlich benutzt werden konnte (loader_source)
            - KEIN Warten / KEINE Timeouts mehr auf Loader-Dateien
            """
            from pathlib import Path
            from datetime import datetime, timezone
            import json
            import pandas as pd
            import re

            ui_target = str(target)
            ui_mode = "fluss" if use_flows else "bestand"
            ui_sektor = sektor or "PH"

            LOGGER.info(_sym("\n[Adapter] ===== STEP 1: Dashboard-Daten vorbereiten ====="))
            LOGGER.info("[Step1] Target=%s, Horizon=%s, Mode=%s, Sektor=%s", ui_target, horizon, ui_mode, ui_sektor)
            LOGGER.info("[Step1] requested_exog=%s", selected_exog)

            # ---------------------------------------------------------------------
            # 1) Exogene beschaffen – jetzt einmalig, ohne Polling
            # ---------------------------------------------------------------------
            try:
                requested = list(selected_exog or [])

                def _get_exog_df():
                    if hasattr(self, "get_exog_data") and callable(getattr(self, "get_exog_data")):
                        return self.get_exog_data()
                    if hasattr(self, "exog_data") and isinstance(self.exog_data, pd.DataFrame):
                        return self.exog_data
                    return None

                cur_exog_df = _get_exog_df()
                need_exog = bool(requested)

                if need_exog and isinstance(cur_exog_df, pd.DataFrame) and not cur_exog_df.empty:
                    LOGGER.info("[Step1|Exog] Initial exog_data.columns (Top): %s", list(cur_exog_df.columns)[:10])

                have_all_now = self._has_all_requested_exogs(cur_exog_df, requested)

                if need_exog and not have_all_now:
                    # optionaler Hook, um Exogs sofort anzustoßen
                    if hasattr(self, "fetch_exogs") and callable(getattr(self, "fetch_exogs")):
                        try:
                            LOGGER.info("[Step1|Exog] fetch_exogs() wird aufgerufen für: %s", requested)
                            self.fetch_exogs(requested)
                        except Exception as _e_fetch:
                            LOGGER.warning("[Step1|Exog] fetch_exogs() Fehler/kein Hook: %s", _e_fetch)

                    # einmal versuchen, den Standard-Cache zu laden
                    cache_df = self._load_exog_from_default_cache()
                    if cache_df is not None and not cache_df.empty:
                        LOGGER.info(
                            "[Step1|Exog] Default-Cache geladen: shape=%s, columns(top)=%s",
                            cache_df.shape,
                            list(cache_df.columns)[:10],
                        )

                        # Union von Dashboard-Exogs und Cache-Exogs auf 'date'
                        if isinstance(cur_exog_df, pd.DataFrame) and not cur_exog_df.empty:
                            try:
                                cur = cur_exog_df.set_index("date")
                                cache = cache_df.set_index("date")
                            except KeyError:
                                LOGGER.warning(
                                    "[Step1|Exog] Cache oder aktuelle Exogs ohne 'date'-Spalte – "
                                    "verwende bestehende exog_data unverändert."
                                )
                                merged = cur_exog_df
                            else:
                                overlap = [c for c in cache.columns if c in cur.columns]
                                if overlap:
                                    LOGGER.info(
                                        "[Step1|Exog] Cache enthält bereits vorhandene Spalten – "
                                        "Dashboard-Daten haben Vorrang: %s",
                                        overlap,
                                    )
                                    cache = cache.drop(columns=overlap)

                                merged = cur.join(cache, how="outer").reset_index()

                                LOGGER.info(
                                    "[Step1|Exog] Exog-Union angewendet: vorher=%d Spalten, cache=%d Spalten, nachher=%d Spalten",
                                    cur_exog_df.shape[1],
                                    cache_df.shape[1],
                                    merged.shape[1],
                                )
                            self.exog_data = merged
                            cur_exog_df = merged
                        else:
                            # Wir hatten vorher nichts – dann darf der Cache komplett übernehmen
                            self.exog_data = cache_df
                            cur_exog_df = cache_df
                    else:
                        cur_exog_df = _get_exog_df()

                    have_all_now = self._has_all_requested_exogs(cur_exog_df, requested)

                    if have_all_now:
                        LOGGER.info("[Step1|Exog] Alle angeforderten Exogs verfügbar – fahre fort.")
                    else:
                        available = set(cur_exog_df.columns) if isinstance(cur_exog_df, pd.DataFrame) else set()
                        missing = []
                        for code in requested:
                            variants = self._exog_variants_for_name(code)
                            if not any(v in available for v in variants):
                                missing.append(code)
                        LOGGER.warning(
                            "[Step1|Exog] Einige Exogs sind aktuell nicht verfügbar – fahre mit vorhandenen fort: %s",
                            ", ".join(missing) if missing else "<unbekannt>",
                        )
            except Exception as _e_wait:
                LOGGER.warning("[Step1|Exog] Exog-Beschaffung (non-blocking) übersprungen: %s", _e_wait)

            # ---------------------------------------------------------------------
            # 2) Daten für Pipeline vorbereiten
            # ---------------------------------------------------------------------
            LOGGER.info(
                "[Step1] Starte prepare_pipeline_data: target=%s, requested_exogs=%s, use_flows=%s, horizon_quarters=%s",
                ui_target,
                list(selected_exog or []),
                use_flows,
                int(horizon or 0),
            )
            prepared_df = self.prepare_pipeline_data(
                target=ui_target,
                selected_exog=selected_exog,
                use_flows=use_flows,
                horizon_quarters=int(horizon or 0),
            )
            LOGGER.info(
                "[Step1] Daten für Pipeline vorbereitet: shape=%s, Spalten=%s",
                prepared_df.shape,
                prepared_df.columns.tolist(),
            )
            LOGGER.debug("[Step1] Vorschau auf vorbereiteten DF:\n%s", prepared_df.head(3))

            # ---------------------------------------------------------------------
            # 3) Exogs im DataFrame auflösen, damit wir wissen, welche wirklich da sind
            #    → robustes Matching: UI-Name (mit Punkten) vs. DF-Spalten (mit Unterstrichen + Suffixen)
            # ---------------------------------------------------------------------
            def _normalize_exog_name_for_match(name: str) -> str:
                """
                Bringt UI-Codes (mit Punkten) und Spaltennamen (mit Unterstrichen/Suffixen)
                auf eine gemeinsame Vergleichsform.
                Beispiele:
                  - "IRS.M.DE.L.L40.CI.0000.EUR.N.Z"
                    -> "IRS_M_DE_L_L40_CI_0000_EUR_N_Z"
                  - "IRS_M_DE_L_L40_CI_0000_EUR_N_Z__last"
                    -> "IRS_M_DE_L_L40_CI_0000_EUR_N_Z"
                  - "QSA.Q.N.DE.W0.S1M.S1.N.A.LE.F._Z._Z.XDC._T.S.V.N._T"
                    -> "QSA_Q_N_DE_W0_S1M_S1_N_A_LE_F__Z__Z_XDC__T_S_V_N__T"
                """
                if not isinstance(name, str):
                    name = str(name)

                s = name.upper()
                s = s.replace(".", "_")
                s = re.sub(r"[^A-Z0-9_]+", "_", s)
                s = re.sub(r"_+", "_", s)
                s = s.strip("_")

                # Suffixe wie __LAST / __LAG... kappen, damit UI-Code und Feature-Name matchen
                s = re.sub(r"(__LAST.*)$", "", s)
                s = re.sub(r"(__LAG.*)$", "", s)

                return s

            requested_exogs = list(selected_exog or [])
            mapping: dict[str, str] = {}
            missing: list[str] = []

            if not prepared_df.empty and requested_exogs:
                # DF-Spalten in kanonische Form bringen
                col_norm_map: dict[str, str] = {}
                for col in prepared_df.columns:
                    norm_col = _normalize_exog_name_for_match(col)
                    if norm_col:
                        # falls mehrfach vorhanden, erste Spalte behalten
                        col_norm_map.setdefault(norm_col, col)

                # requested UI-Namen auf DF-Spalten mappen
                for raw_name in requested_exogs:
                    norm_req = _normalize_exog_name_for_match(raw_name)
                    if not norm_req:
                        missing.append(raw_name)
                        continue

                    if norm_req in col_norm_map:
                        mapping[raw_name] = col_norm_map[norm_req]
                    else:
                        missing.append(raw_name)

            resolved_exogs = list(mapping.values())

            diag_resolve = {
                "mapping": mapping,
                "missing": missing,
            }

            LOGGER.info("[Step1|ExogResolve] Gewünschte Exogs (UI): %s", requested_exogs)
            LOGGER.info("[Step1|ExogResolve] Mapping requested→actual: %s", diag_resolve.get("mapping"))

            if missing:
                LOGGER.warning(
                    "[Step1|ExogResolve] %d gewünschte Exogs NICHT im Prepared-DF gefunden und werden ignoriert: %s",
                    len(missing),
                    ", ".join(missing),
                )
                LOGGER.debug("[Step1|ExogResolve] Prepared-DF-Spalten: %s", prepared_df.columns.tolist())

            if resolved_exogs:
                LOGGER.info(
                    "[Step1|ExogResolve] %d Exogs werden effektiv an die Pipeline übergeben: %s",
                    len(resolved_exogs),
                    resolved_exogs,
                )
            else:
                LOGGER.warning(
                    "[Step1|ExogResolve] Keine Exogs im DF auflösbar – Forecast läuft ohne Exogene weiter."
                )

            # ---------------------------------------------------------------------
            # 4) DataFrame auf Exog-Vollständigkeit prüfen (nicht strikt)
            # ---------------------------------------------------------------------
            LOGGER.info(
                "[Step1|Guard/DF] Starte Exog-Validierung: requested_exogs=%s, resolved_exogs=%s",
                requested_exogs,
                list(resolved_exogs or []),
            )
            try:
                self._validate_exog_presence_in_df(
                    df_final=prepared_df,
                    selected_exog=requested_exogs,  # bewusst UI-Namen prüfen
                    strict=False,
                    logger=LOGGER,
                )
            except Exception as e_guard_df:
                LOGGER.warning("[Step1|Guard/DF] (non-strict) Hinweis aus Exog-Validierung: %s", e_guard_df)

            # ---------------------------------------------------------------------
            # 5) Zielspalte aus UI Namen ableiten
            # ---------------------------------------------------------------------
            excel_target_col_from_ui = self._ui_target_to_excel_col(ui_target, sektor=ui_sektor)
            LOGGER.info(
                "[Step1] Excel-Target aus UI abgeleitet: ui_target=%s, sektor=%s → excel_col=%s",
                ui_target,
                ui_sektor,
                excel_target_col_from_ui,
            )

            # ---------------------------------------------------------------------
            # 6) Cache-Tag bestimmen – ab hier kennen wir den run-spezifischen Kontext
            # ---------------------------------------------------------------------
            run_cache_tag = self._make_cache_tag(
                sektor=ui_sektor,
                excel_target_col=excel_target_col_from_ui,
                ui_mode=ui_mode,
                horizon=horizon,
                selected_exog=list(resolved_exogs or []),
            )
            LOGGER.info(
                "[Step1] Generierter run_cache_tag=%s (resolved_exogs=%s, horizon=%s, ui_mode=%s)",
                run_cache_tag,
                list(resolved_exogs or []),
                horizon,
                ui_mode,
            )

            # ---------------------------------------------------------------------
            # 7) run-spezifischen Ordner + erwarteten Loader-Output bestimmen
            #    + Übergabe-Datei für den Loader schreiben
            # ---------------------------------------------------------------------
            base_dir = Path(__file__).resolve().parent.parent
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            run_loader_dir = base_dir / "loader" / "runs" / run_cache_tag / ts
            run_loader_dir.mkdir(parents=True, exist_ok=True)
            expected_run_output = run_loader_dir / "output.xlsx"

            LOGGER.info("[Step1] Run-Verzeichnis angelegt/gefunden: %s", run_loader_dir)
            LOGGER.info("[Step1] Erwarteter Loader-Output für diesen Run: %s", expected_run_output)

            # Übergabe-Datei schreiben, damit der Loader denselben Pfad nimmt
            try:
                handover_path = base_dir / "loader" / "active_run.json"
                handover_payload = {
                    "run_cache_tag": run_cache_tag,
                    "run_loader_dir": str(run_loader_dir),
                    "expected_output": str(expected_run_output),
                    # wichtig: die vom Dashboard gewünschten Codes, nicht nur die im DF aufgelösten
                    "requested_exog": list(selected_exog or []),
                    "ui_target": ui_target,
                    "ui_mode": ui_mode,
                    "ui_sektor": ui_sektor,
                    "horizon": int(horizon or 0),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                handover_path.parent.mkdir(parents=True, exist_ok=True)
                handover_path.write_text(
                    json.dumps(handover_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                LOGGER.info("[Step1] Übergabe-Datei für Loader geschrieben: %s", handover_path)
                LOGGER.debug(
                    "[Step1] Inhalt Übergabe-Datei (active_run.json): %s",
                    json.dumps(handover_payload, ensure_ascii=False),
                )
            except Exception as _e_handover:
                LOGGER.warning("[Step1] Übergabe-Datei konnte nicht geschrieben werden: %s", _e_handover)

            # ---------------------------------------------------------------------
            # 8) Loader-Quelle nur feststellen (non-blocking)
            # ---------------------------------------------------------------------
            global_loader_xlsx = base_dir / "loader" / "financial_cache" / "output.xlsx"
            if expected_run_output.exists():
                loader_source = "run"
                LOGGER.info(
                    "[Step1] Loader-Output bereits run-spezifisch vorhanden – wird in Step 2 bevorzugt."
                )
            elif global_loader_xlsx.exists():
                loader_source = "global"
                LOGGER.info(
                    "[Step1] Kein run-spezifischer Loader-Output, aber globaler Loader-Output vorhanden."
                )
            else:
                loader_source = "none"
                LOGGER.error(
                    "[Step1] Weder run-spezifischer noch globaler Loader-Output sind aktuell vorhanden – "
                    "Step 2 wird eigenes Excel erzeugen."
                )
            LOGGER.debug(
                "[Step1] Loader-Check: expected_run_output=%s exists=%s, global_loader_xlsx=%s exists=%s",
                expected_run_output,
                expected_run_output.exists(),
                global_loader_xlsx,
                global_loader_xlsx.exists(),
            )

            # ---------------------------------------------------------------------
            # 9) pipeline_info aktualisieren (nur Info)
            # ---------------------------------------------------------------------
            if hasattr(self, "pipeline_info") and isinstance(self.pipeline_info, dict):
                pi = self.pipeline_info or {}
                pi["ui_target"] = ui_target
                pi["ui_mode"] = ui_mode
                pi["sektor"] = ui_sektor
                pi["exog_cols"] = list(resolved_exogs or [])
                pi["run_cache_tag"] = run_cache_tag
                pi["run_loader_dir"] = str(run_loader_dir)
                pi["expected_run_output"] = str(expected_run_output)
                pi["loader_source"] = loader_source
                self.pipeline_info = pi
                LOGGER.debug("[Step1] pipeline_info aktualisiert: %s", pi)

            # ---------------------------------------------------------------------
            # 10) Kontext zurückgeben
            # ---------------------------------------------------------------------
            return {
                "ui_target": ui_target,
                "ui_mode": ui_mode,
                "ui_sektor": ui_sektor,
                "horizon": horizon,
                "selected_exog": list(selected_exog or []),
                "prepared_df": prepared_df,
                "resolved_exogs": list(resolved_exogs or []),
                "excel_target_col_from_ui": excel_target_col_from_ui,
                "run_cache_tag": run_cache_tag,
                "run_loader_dir": str(run_loader_dir),
                "expected_run_output": str(expected_run_output),
                "loader_source": loader_source,
            }


    def _step_2_build_temp_excel_and_config(
        self,
        ctx: dict,
        *,
        use_cached: bool,
        force_retrain: bool,
        preload_model_path: Optional[str],
    ) -> dict:
        """
        Schritt 2: Excel erzeugen oder wiederverwenden, Exogs in Excel prüfen,
        Pipeline-Config bauen, Preload prüfen, Cache auf Mismatch prüfen.

        Anpassung zu Punkt 2 + 3:
        - Wir bevorzugen den lauf-spezifischen Ordner aus Step 1 (ctx["run_loader_dir"])
        für das Wiederverwenden und Schreiben der Run-Excel.
        - Erst wenn dort nichts passt, schauen wir in den allgemeinen Ordner
        loader/runs/<run_cache_tag>/.
        - Neue Excel + Meta werden – wenn möglich – direkt im lauf-spezifischen Ordner abgelegt.
        - Wir schreiben in die Meta auch, welche Loader-Quelle Step 1 tatsächlich nutzen konnte
        (ctx["loader_source"]) und welchen Loader-Output wir erwartet haben
        (ctx["expected_run_output"]).
        """
        from pathlib import Path
        from datetime import datetime
        import shutil
        import json
        import os
        import pandas as pd  # wir brauchen das hier zum Spalten-Check

        LOGGER.info(_sym("\n[Adapter] ===== STEP 2: Excel und Config ====="))

        prepared_df: pd.DataFrame = ctx["prepared_df"]
        resolved_exogs: list[str] = ctx["resolved_exogs"]
        ui_target: str = ctx["ui_target"]
        ui_mode: str = ctx["ui_mode"]
        ui_sektor: str = ctx["ui_sektor"]
        horizon: int = ctx["horizon"]
        excel_target_col_from_ui: str = ctx["excel_target_col_from_ui"]
        run_cache_tag: str = ctx["run_cache_tag"]

        # neu aus Step 1
        loader_source: str = ctx.get("loader_source", "unknown")
        expected_run_output: str = ctx.get("expected_run_output", "")

        base_dir = Path(__file__).resolve().parent.parent

        # neu: vom Step 1 mitgegeben – das ist der wirklich lauf-spezifische Ordner
        run_loader_dir = Path(ctx.get("run_loader_dir", base_dir / "loader" / "runs" / run_cache_tag))
        # alter Sammel-Ordner für alle Läufe dieses cache_tags
        runs_dir = base_dir / "loader" / "runs" / run_cache_tag

        temp_excel_path: Path
        excel_target_col: str = excel_target_col_from_ui
        reused_excel_path: Optional[Path] = None

        # kleine Hilfsfunktion für Spalten-Check
        def _excel_cols_for(path: Path) -> tuple[list[str], Optional[str]]:
            try:
                cols = list(pd.read_excel(path, sheet_name="final_dataset", nrows=0).columns)
                non_dt = [c for c in cols if c.lower() != "datum"]
                excel_target_detected = non_dt[0] if non_dt else None
                return cols, excel_target_detected
            except Exception as _e_cols:
                LOGGER.warning(f"[Step2|Reuse] Konnte Excel {path} nicht inspizieren: {_e_cols}")
                return [], None

        # ------------------------------------------------------------------
        # 2a) vorhandene Excel zuerst im lauf-spezifischen Ordner wiederverwenden
        # ------------------------------------------------------------------
        candidate_dirs: list[Path] = []
        # 1. Priorität: timestamp-Ordner aus Step 1
        if run_loader_dir.exists():
            candidate_dirs.append(run_loader_dir)
        # 2. Priorität: allgemeiner Ordner für dieses cache_tag (bisheriges Verhalten)
        if runs_dir.exists() and runs_dir not in candidate_dirs:
            candidate_dirs.append(runs_dir)

        LOGGER.info(
            f"[Step2|Reuse] Suche Run-Excels in {len(candidate_dirs)} Ordner(n): "
            + ", ".join(str(d) for d in candidate_dirs)
        )

        for search_dir in candidate_dirs:
            candidates = sorted(
                search_dir.glob("*_final_dataset.xlsx"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                LOGGER.info(f"[Step2|Reuse] Keine *_final_dataset.xlsx unter {search_dir}")
                continue

            LOGGER.info(
                f"[Step2|Reuse] Ordner {search_dir} – {len(candidates)} Kandidat(en) zum Wiederverwenden gefunden"
            )

            for cand in candidates:
                meta = None

                # 1) versuche timestamp-spezifische Meta
                ts_meta = cand.with_name(cand.stem.replace("_final_dataset", "") + "_run_meta.json")
                # 2) fallback: generische Meta im selben Ordner
                generic_meta = cand.parent / "run_meta.json"

                for mp in (ts_meta, generic_meta):
                    if mp.exists():
                        try:
                            with mp.open("r", encoding="utf-8") as fh:
                                meta = json.load(fh)
                            LOGGER.info(f"[Step2|Reuse] Meta-Datei zu {cand.name} gefunden: {mp.name}")
                            break
                        except Exception as _e_meta:
                            LOGGER.warning(f"[Step2|Reuse] Konnte Meta-Datei {mp} nicht lesen: {_e_meta}")

                selected_exogs_current = list(resolved_exogs or [])

                if meta:
                    # strenger Vergleich gegen die Meta
                    meta_ok = (
                        meta.get("cache_tag") == run_cache_tag
                        and meta.get("sector") == ui_sektor
                        and meta.get("ui_mode") == ui_mode
                        and int(meta.get("forecast_horizon", horizon)) == int(horizon)
                        and set(meta.get("selected_exog") or []) == set(selected_exogs_current)
                    )

                    if not meta_ok:
                        LOGGER.info(
                            f"[Step2|Reuse] Kandidat {cand.name} in {search_dir} verworfen – Meta passt nicht."
                        )
                        continue

                    meta_target = meta.get("target_col") or meta.get("excel_target_col")
                    if meta_target and meta_target == excel_target_col_from_ui:
                        reused_excel_path = cand
                        excel_target_col = meta_target
                        LOGGER.info(f"[Step2] ✓ Früheren Run wiederverwendet (Meta-Match): {cand}")
                        break
                    else:
                        # zur Sicherheit noch die tatsächlichen Spalten prüfen
                        cols, detected_target = _excel_cols_for(cand)
                        if detected_target and detected_target == excel_target_col_from_ui:
                            reused_excel_path = cand
                            excel_target_col = detected_target
                            LOGGER.info(f"[Step2] ✓ Früheren Run wiederverwendet (Excel-Match): {cand}")
                            break
                        else:
                            LOGGER.info(
                                f"[Step2|Reuse] Kandidat {cand.name} verworfen – Target in Excel/Meta stimmt nicht."
                            )
                            continue
                else:
                    # kein Meta → nur Excel-Struktur vergleichen
                    cols, detected_target = _excel_cols_for(cand)
                    if not cols or not detected_target:
                        continue

                    if detected_target != excel_target_col_from_ui:
                        LOGGER.info(
                            f"[Step2|Reuse] Kandidat {cand.name} verworfen – Target {detected_target}≠{excel_target_col_from_ui}"
                        )
                        continue

                    excel_exogs = set([c for c in cols if c not in ("Datum", "datum") and c != detected_target])
                    if excel_exogs == set(selected_exogs_current):
                        reused_excel_path = cand
                        excel_target_col = detected_target
                        LOGGER.info(f"[Step2] ✓ Früheren Run wiederverwendet (Excel-only-Match): {cand}")
                        break
                    else:
                        LOGGER.info(
                            f"[Step2|Reuse] Kandidat {cand.name} verworfen – Exogs im Excel ≠ aktuelle Exogs."
                        )

            if reused_excel_path:
                # wir haben bereits was passendes gefunden – andere Ordner brauchen wir nicht mehr ansehen
                break

        # ------------------------------------------------------------------
        # 2b) ggf. neues Excel erzeugen
        # ------------------------------------------------------------------
        if reused_excel_path:
            temp_excel_path = reused_excel_path
        else:
            LOGGER.info("[Step2] Keine passende Run-Excel gefunden – erzeuge neue Run-Excel.")

            # wenn wir einen lauf-spezifischen Ordner haben, legen wir dort ab
            if run_loader_dir.exists():
                run_loader_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                temp_excel_path = run_loader_dir / f"{ts}_final_dataset.xlsx"

                # Excel wie in create_temp_excel schreiben
                rename_map = {"date": "Datum", "target_value": excel_target_col_from_ui}
                df_export = prepared_df.rename(columns=rename_map).copy()
                df_export["Datum"] = pd.to_datetime(df_export["Datum"], errors="coerce")

                LOGGER.info(_sym("\n[Step2] Excel-Export (run-spezifisch):"))
                LOGGER.info(f"  Ziel: {temp_excel_path}")
                LOGGER.info(f"  Shape: {df_export.shape}")
                LOGGER.info(f"  Spalten: {df_export.columns.tolist()}")

                df_export.to_excel(temp_excel_path, sheet_name="final_dataset", index=False)
                excel_target_col = excel_target_col_from_ui

                # Validierung lesen wir kurz ein
                try:
                    _ = pd.read_excel(temp_excel_path, sheet_name="final_dataset", nrows=1)
                except Exception as _e_val:
                    LOGGER.warning(f"[Step2] Konnte frisch erzeugte Excel nicht validieren: {_e_val}")
            else:
                # Fallback: altes Verhalten
                excel_ret = self.create_temp_excel(
                    prepared_df,
                    target_col="target_value",
                    excel_target_col=excel_target_col_from_ui,
                    cache_tag=run_cache_tag,
                )
                if isinstance(excel_ret, tuple):
                    temp_excel_path, excel_target_col = excel_ret
                else:
                    temp_excel_path = Path(excel_ret)
                    try:
                        cols = list(pd.read_excel(temp_excel_path, sheet_name="final_dataset", nrows=0).columns)
                        excel_target_col = next((c for c in cols if c.lower() != "datum"), excel_target_col_from_ui)
                    except Exception:
                        excel_target_col = excel_target_col_from_ui

            # passende Meta-Datei neben der neu erzeugten Excel ablegen
            try:
                meta_out = {
                    "cache_tag": run_cache_tag,
                    "sector": ui_sektor,
                    "ui_mode": ui_mode,
                    "target_col": excel_target_col,
                    "forecast_horizon": horizon,
                    "selected_exog": list(resolved_exogs or []),
                    "ui_target": ui_target,
                    # neu: Info aus Step 1 mitgeben
                    "loader_source": loader_source,
                    "expected_run_output": expected_run_output,
                    "run_loader_dir": str(run_loader_dir),
                }
                meta_name = temp_excel_path.with_name(
                    temp_excel_path.stem.replace("_final_dataset", "") + "_run_meta.json"
                )
                with meta_name.open("w", encoding="utf-8") as fh:
                    json.dump(meta_out, fh, ensure_ascii=False, indent=2)
                LOGGER.info(f"[Step2] Meta-Datei für Run geschrieben: {meta_name}")
            except Exception as _e_meta_write:
                LOGGER.warning(f"[Step2] Konnte Meta-Datei nicht schreiben: {_e_meta_write}")

        # ------------------------------------------------------------------
        # 2c) Exog-Validierung in Excel (jetzt strikt, weil Run-Excel)
        # ------------------------------------------------------------------
        self._validate_exog_presence_in_excel(
            excel_path=str(temp_excel_path),
            selected_exog=list(resolved_exogs or []),
            sheet_name="final_dataset",
            strict=True,
            logger=LOGGER,
        )

        # ------------------------------------------------------------------
        # 2d) Config bauen
        # ------------------------------------------------------------------
        try:
            cfg = self._build_config(
                excel_path=str(temp_excel_path),
                horizon=horizon,
                use_cached=use_cached,
                selected_exog=list(resolved_exogs or []),
                ui_target=ui_target,
                ui_mode=ui_mode,
                sektor=ui_sektor,
                excel_target_col=excel_target_col,
                cache_tag=run_cache_tag,
            )
        except TypeError:
            # Backward-compat
            cfg = self._build_config(
                excel_path=str(temp_excel_path),
                horizon=horizon,
                use_cached=use_cached,
                selected_exog=list(resolved_exogs or []),
                ui_target=ui_target,
                ui_mode=ui_mode,
                sektor=ui_sektor,
                cache_tag=run_cache_tag,
            )

        # ------------------------------------------------------------------
        # 2e) Preload prüfen (Modell-PKL) – jetzt mit strengerem Matching
        # ------------------------------------------------------------------
        accepted_preload = False
        if preload_model_path and os.path.exists(preload_model_path):
            try:
                art_pre = ModelArtifact.load(preload_model_path)
                meta_pre = getattr(art_pre, "metadata", {}) or {}
                current_exogs = list(getattr(cfg, "selected_exog", []) or [])

                is_ok = self._preload_meta_compatible(
                    meta_pre,
                    excel_target_col=excel_target_col,
                    selected_exog=current_exogs,
                    run_cache_tag=run_cache_tag,
                    ui_mode=ui_mode,
                    ui_sektor=ui_sektor,
                    horizon=horizon,
                )

                if not is_ok:
                    LOGGER.warning(
                        "[Step2|Preload] Preload-Metadaten passen nicht zum aktuellen UI-Setup – Preload wird ignoriert."
                    )
                else:
                    exp_path = get_model_filepath(cfg)
                    if not os.path.exists(exp_path):
                        Path(exp_path).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(preload_model_path, exp_path)

                    # Metadata im Zielmodell aktualisieren
                    try:
                        art_fix = ModelArtifact.load(exp_path)
                        m = getattr(art_fix, "metadata", {}) or {}
                        m["exog_cols"] = current_exogs
                        m["target_col"] = excel_target_col
                        m.setdefault("cache_tag", run_cache_tag)
                        m.setdefault("ui_mode", ui_mode)
                        m.setdefault("sector", ui_sektor)
                        m.setdefault("forecast_horizon", horizon)
                        art_fix.metadata = m
                        if hasattr(art_fix, "save"):
                            art_fix.save(exp_path)
                    except Exception as _e_fix:
                        LOGGER.warning(f"[Step2|Preload] Konnte Ziel-Metadata nicht aktualisieren: {_e_fix}")

                    LOGGER.info(f"[Step2] ✓ Preloaded Modell übernommen: {preload_model_path} → {exp_path}")
                    accepted_preload = True
            except Exception as _e_pl:
                LOGGER.warning(f"[Step2|Preload] Konnte Preload-PKL nicht verwenden: {_e_pl}")

        # ------------------------------------------------------------------
        # 2f) Cache auf Mismatch prüfen – mit aktuellem Setup abgleichen
        # ------------------------------------------------------------------
        model_path = get_model_filepath(cfg)
        current_exogs = list(getattr(cfg, "selected_exog", []) or [])
        if (not accepted_preload) and ModelArtifact and hasattr(ModelArtifact, "exists") and ModelArtifact.exists(
            model_path
        ):
            try:
                art = ModelArtifact.load(model_path)
                meta_art = getattr(art, "metadata", {}) or {}

                is_ok = self._preload_meta_compatible(
                    meta_art,
                    excel_target_col=excel_target_col,
                    selected_exog=current_exogs,
                    run_cache_tag=run_cache_tag,
                    ui_mode=ui_mode,
                    ui_sektor=ui_sektor,
                    horizon=horizon,
                )

                if not is_ok:
                    LOGGER.warning("\n[Step2|Cache] Modell im Cache passt nicht zum aktuellen Setup – wird gelöscht.")
                    LOGGER.warning(f"[Step2|Cache] Gefundene Meta: {meta_art}")
                    try:
                        os.remove(model_path)
                    except Exception as _e_rm:
                        LOGGER.warning(f"[Step2|Cache] Konnte Cache nicht löschen: {_e_rm}")
                    force_retrain = True
            except Exception as _e_load:
                LOGGER.warning(f"[Step2|Cache] Konnte Cache nicht prüfen: {_e_load}")

        # ------------------------------------------------------------------
        # 2g) Config Summary
        # ------------------------------------------------------------------
        LOGGER.info("\n[Step2] Config summary:")
        for k in [
            "forecast_horizon",
            "agg_method_target",
            "agg_methods_exog",
            "exog_month_lags",
            "target_lags_q",
            "add_trend_features",
            "trend_degree",
            "add_seasonality",
            "future_exog_strategy",
            "target_transform",
            "target_standardize",
        ]:
            LOGGER.info(f"  {k}: {getattr(cfg, k, None)}")
        LOGGER.info(f"  Excel: {cfg.excel_path}")
        LOGGER.info(f"  Model dir: {cfg.model_dir}")
        LOGGER.info(f"  selected_exog/exog_cols: {getattr(cfg, 'selected_exog', [])}")
        LOGGER.info(f"  Excel target col: {excel_target_col}")
        LOGGER.info(f"  Loader-Quelle aus Step 1: {loader_source}")

        return {
            **ctx,
            "temp_excel_path": str(temp_excel_path),
            "excel_target_col": excel_target_col,
            "cfg": cfg,
            "force_retrain": force_retrain,
            "model_path": model_path,
        }

    def _step_3_run_pipeline(self, ctx: dict) -> dict:
        """
        Schritt 3: Pipeline wirklich ausführen, Metadata ergänzen, Snapshot schreiben.
        """
        LOGGER.info(_sym("\n[Adapter] ===== STEP 3: Pipeline ausführen ====="))

        cfg = ctx["cfg"]
        force_retrain = ctx["force_retrain"]
        prepared_df: pd.DataFrame = ctx["prepared_df"]

        LOGGER.info(_sym("\n[Adapter] Starte Pipeline."))
        LOGGER.info(f"  Cache: {'verwendet' if getattr(cfg, 'use_cached_model', False) else 'ignoriere'}")
        LOGGER.info(f"  Force-Retrain: {force_retrain}")
        forecast_df, metadata = run_production_pipeline(cfg, force_retrain)

        # Modellpfad ermitteln
        model_path = get_model_filepath(cfg)
        if isinstance(metadata, dict):
            metadata["model_path"] = model_path

        # Snapshot schreiben – jetzt gekapselt
        snapshot_path = self._write_artifact(
            kind="snapshot_parquet",
            df=prepared_df,
            base_dir=getattr(cfg, "output_dir", "outputs"),
            filename=f"snapshot_{Path(model_path).stem}.parquet",
        )
        if isinstance(metadata, dict) and snapshot_path:
            metadata["exog_snapshot_path"] = str(snapshot_path)

        return {
            **ctx,
            "forecast_df": forecast_df,
            "metadata": metadata,
            "model_path": model_path,
        }

    def _step_4_enrich_result_for_dashboard(
        self,
        ctx: dict,
        *,
        confidence_levels: Optional[list[int]],
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Schritt 4: Konfidenzintervalle berechnen, Backtest anhängen, final_dataset für Dash persistieren.

        Härten (Punkt 8):
        - robustes Finden der Forecast-Spalte (Fallback-Kandidaten)
        - wenn keine Residuen und kein cv_rmse → CI wird sauber übersprungen und im Metadata vermerkt
        - Backtest nur, wenn Modell/Trainingsdaten wirklich vorhanden
        - Persistenz-Teil holt sich fehlende Imports selbst
        """
        from datetime import datetime
        from pathlib import Path
        import os
        import tempfile
        import numpy as np
        import pandas as pd  # damit wir hier sicher sind

        LOGGER.info(_sym("\n[Adapter] ===== STEP 4: Ergebnis anreichern ====="))

        forecast_df: pd.DataFrame = ctx["forecast_df"]
        metadata: dict = ctx.get("metadata") or {}
        model_path: str = ctx["model_path"]
        cfg = ctx["cfg"]
        temp_excel_path: str = ctx.get("temp_excel_path", "")
        prepared_df: pd.DataFrame = ctx["prepared_df"]

        if confidence_levels is None:
            confidence_levels = [80, 95]

        # wir legen uns einen diagnostics-Knoten an, damit das Dashboard Ursachen sieht
        if not isinstance(metadata, dict):
            metadata = {}
        diagnostics_ci = metadata.setdefault("diagnostics", {}).setdefault("ci", {})

        # ------------------------------------------------------------------
        # 4a) Konfidenzintervalle
        # ------------------------------------------------------------------
        if forecast_df is None or forecast_df.empty:
            LOGGER.warning("[CI] Forecast-DataFrame ist leer – CI wird übersprungen.")
            diagnostics_ci["applied"] = False
            diagnostics_ci["reason"] = "empty_forecast"
        else:
            try:
                LOGGER.info(_sym("\n[Adapter] Berechne Konfidenzintervalle"))
                residuals = self._extract_residuals_from_pipeline(metadata, model_path)

                # wenn nichts brauchbares zurückkam → versuchen wir cv_rmse
                if residuals is None or (hasattr(residuals, "__len__") and len(residuals) == 0):
                    cv_rmse = None
                    try:
                        cv_perf = (metadata or {}).get("cv_performance", {})
                        cv_rmse = float(cv_perf.get("cv_rmse") or cv_perf.get("rmse"))
                    except Exception:
                        cv_rmse = None

                    if cv_rmse and np.isfinite(cv_rmse):
                        LOGGER.warning(
                            f"[CI] Keine Residuen gefunden – erzeuge synthetische Residuen (RMSE={cv_rmse:.4f}, n=100)"
                        )
                        rng = np.random.default_rng(42)
                        residuals = rng.normal(0.0, cv_rmse, size=100)
                        diagnostics_ci["residual_source"] = "synthetic_from_cv_rmse"
                        diagnostics_ci["cv_rmse"] = float(cv_rmse)
                    else:
                        LOGGER.warning("[CI] Keine Residuen verfügbar und kein CV-RMSE – CI werden übersprungen")
                        diagnostics_ci["applied"] = False
                        diagnostics_ci["reason"] = "no_residuals_and_no_cv_rmse"
                        residuals = None

                if residuals is not None:
                    # robustes Finden der Forecast-Spalte
                    forecast_col = "Forecast"
                    if forecast_col not in forecast_df.columns:
                        # alternative Kandidaten
                        candidates = ["forecast", "yhat", "y_pred", "prediction"]
                        found = next((c for c in candidates if c in forecast_df.columns), None)
                        if found is None:
                            # letzte Chance: erste numerische Spalte nehmen
                            num_cols = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
                            found = num_cols[0] if num_cols else None
                        forecast_col = found or "Forecast"

                    if forecast_col not in forecast_df.columns:
                        LOGGER.warning(
                            "[CI] Keine passende Forecast-Spalte gefunden – CI wird nicht angefügt (gesucht wurde z.B. 'Forecast')."
                        )
                        diagnostics_ci["applied"] = False
                        diagnostics_ci["reason"] = "forecast_column_not_found"
                    else:
                        forecast_df = self._add_confidence_intervals_to_forecast(
                            forecast_df=forecast_df,
                            residuals=residuals,
                            confidence_levels=confidence_levels,
                            forecast_col=forecast_col,
                        )
                        # std_error für UI ablegen
                        try:
                            res_arr = np.asarray(residuals, dtype=float)
                            if res_arr.size > 1:
                                se = float(np.std(res_arr, ddof=1))
                                diagnostics_ci["std_error"] = se
                        except Exception:
                            pass

                        diagnostics_ci["applied"] = True
                        diagnostics_ci["used_forecast_col"] = forecast_col
                        diagnostics_ci["confidence_levels"] = confidence_levels
                        LOGGER.info("[CI] Konfidenzintervalle erfolgreich hinzugefügt")
                        LOGGER.info(f"[CI] Forecast-Spalten: {forecast_df.columns.tolist()}")
            except Exception as e_ci:
                LOGGER.error(f"[CI] Fehler bei Konfidenzintervall-Berechnung: {e_ci}")
                LOGGER.warning("[CI] Forecast wird ohne Konfidenzintervalle zurückgegeben")
                diagnostics_ci["applied"] = False
                diagnostics_ci["reason"] = f"exception: {e_ci!r}"

        # ------------------------------------------------------------------
        # 4b) Backtest
        # ------------------------------------------------------------------
        try:
            LOGGER.info("[Adapter] Generiere Backtest-Daten.")
            can_do_backtest = (
                ModelArtifact
                and hasattr(ModelArtifact, "exists")
                and isinstance(model_path, str)
                and model_path
                and ModelArtifact.exists(model_path)
            )
            if can_do_backtest:
                artifact = ModelArtifact.load(model_path)
                X_train = artifact.metadata.get("X_train")
                y_train = artifact.metadata.get("y_train")
                dates_train = artifact.metadata.get("dates_train")

                if X_train is None or y_train is None or dates_train is None:
                    LOGGER.info("[Backtest] Training-Daten nicht in Artifact, extrahiere aus prepared_df")
                    try:
                        from src.forecaster.forecaster_pipeline import (  # type: ignore
                            aggregate_to_quarter,
                            add_deterministic_features,
                            build_quarterly_lags,
                        )

                        df_in = prepared_df.copy()
                        if cfg is not None:
                            rename_map = {}
                            if "date" in df_in.columns and getattr(cfg, "date_col", "date") not in df_in.columns:
                                rename_map["date"] = cfg.date_col
                            if "target_value" in df_in.columns and getattr(cfg, "target_col", "target") not in df_in.columns:
                                rename_map["target_value"] = cfg.target_col
                            if rename_map:
                                df_in = df_in.rename(columns=rename_map)

                        df_q = aggregate_to_quarter(df_in, cfg)
                        df_q = add_deterministic_features(df_q, cfg)
                        df_feats = build_quarterly_lags(df_q, cfg)

                        X_cols = getattr(artifact, "X_cols", None)
                        if X_cols is None:
                            raise ValueError("Artifact enthält keine X_cols – Backtest nicht möglich.")
                        X_train = df_feats.reindex(columns=X_cols)
                        y_train = df_feats[cfg.target_col]
                        if "Q_end" in df_feats.columns:
                            dates_train = df_feats["Q_end"]
                        else:
                            dates_train = df_feats[getattr(cfg, "date_col", "Datum")]
                    except Exception as e_bt_prep:
                        LOGGER.warning(f"[Backtest] Vorbereitung aus prepared_df fehlgeschlagen: {e_bt_prep}")
                        X_train = None
                        y_train = None
                        dates_train = None

                if X_train is not None and y_train is not None and dates_train is not None:
                    backtest_df, residuals_df = self._build_backtest_from_artifact(
                        artifact=artifact,
                        X_train=X_train,
                        y_train=y_train,
                        dates_train=dates_train,
                    )
                    if isinstance(metadata, dict):
                        metadata["backtest_results"] = backtest_df.to_dict(orient="records")
                        metadata["backtest_residuals"] = residuals_df.to_dict(orient="records")
                else:
                    LOGGER.warning("[Backtest] Trainingsdaten konnten nicht rekonstruiert werden – Backtest entfällt.")
            else:
                LOGGER.warning("[Backtest] Kein gespeichertes Modell gefunden – Backtest entfällt.")
        except Exception as e_bt:
            LOGGER.warning(f"[Backtest] Fehler bei Backtest-Erstellung: {e_bt}")

        # ------------------------------------------------------------------
        # 4c) final_dataset persistent machen für Dash – jetzt gekapselt
        # ------------------------------------------------------------------
        try:
            base_outdir = Path(getattr(cfg, "output_dir", tempfile.gettempdir()))
            dash_dir = base_outdir / "dash_forecasts"
            dash_dir.mkdir(parents=True, exist_ok=True)

            dash_export = {
                "output_dir": str(base_outdir),
                "model_dir": str(getattr(cfg, "model_dir", "")),
                "model_path": str(model_path),
                "production_forecast_csv": str(base_outdir / "production_forecast.csv"),
                "production_metadata_json": str(base_outdir / "production_forecast_metadata.json"),
                "future_design_csv": str(base_outdir / "future_design_debug.csv"),
                "pipeline_config": cfg.to_dict() if hasattr(cfg, "to_dict") else getattr(cfg, "__dict__", {}),
            }

            if temp_excel_path and os.path.exists(temp_excel_path):
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                persisted = self._write_artifact(
                    kind="dash_final_dataset",
                    src_path=temp_excel_path,
                    base_dir=dash_dir,
                    filename=f"final_dataset_{stamp}.xlsx",
                )
                if persisted:
                    dash_export["final_dataset_path"] = str(persisted)

            if isinstance(metadata, dict):
                metadata["dash_export"] = dash_export
        except Exception as e_persist:
            LOGGER.warning(f"[Adapter] Konnte final_dataset nicht persistent speichern: {e_persist}")

        return forecast_df, metadata


    # ------------------------------------------------------------
    # run_forecast: Orchestrator
    # ------------------------------------------------------------
    def run_forecast(
        self,
        target: str,
        selected_exog: list[str],
        horizon: int,
        use_cached: bool,
        force_retrain: bool,
        use_flows: bool = False,
        confidence_levels: list[int] | None = None,
        preload_model_path: str | None = None,
        sektor: str | None = None,
    ):
        """
        Führt einen Forecast über die Produktions-Pipeline aus.
        Jetzt in 4 klaren Schritten organisiert + SCHRITT 6 (Dashboard-Payload).

        Anpassung zu Punkt 6:
        - Im finally-Block löschen wir nur echte temporäre Dateien (z.B. aus dem OS-Tempdir)
        oder Dateien, die nicht im run-spezifischen loader-Verzeichnis liegen.
        - Run-spezifische Dateien wie
            loader/runs/<cache_tag>/<timestamp>/_final_dataset.xlsx
        bleiben erhalten, damit Loader / spätere Runs sie wiederverwenden können.
        """
        import os
        from pathlib import Path
        import tempfile

        temp_excel_path: str | None = None

        try:
            # Schritt 1
            ctx1 = self._step_1_prepare_from_dashboard(
                target=target,
                selected_exog=selected_exog,
                horizon=horizon,
                use_flows=use_flows,
                sektor=sektor,
            )

            # Schritt 2
            ctx2 = self._step_2_build_temp_excel_and_config(
                ctx1,
                use_cached=use_cached,
                force_retrain=force_retrain,
                preload_model_path=preload_model_path,
            )
            temp_excel_path = ctx2.get("temp_excel_path")

            # Schritt 3
            ctx3 = self._step_3_run_pipeline(ctx2)

            # Schritt 4
            forecast_df, metadata = self._step_4_enrich_result_for_dashboard(
                ctx3,
                confidence_levels=confidence_levels,
            )

            # Schritt 6 (Dashboard-Payload)
            try:
                dash_result = self._step_6_build_dashboard_result(forecast_df, metadata)
                if isinstance(metadata, dict):
                    metadata["dashboard_result"] = dash_result
            except Exception as _e_step6:
                LOGGER.warning(f"[Step6] Dashboard-Payload konnte nicht gebaut werden: {_e_step6}")

            return forecast_df, metadata

        except Exception as e:
            LOGGER.exception(f"[Adapter] RUN_FORECAST fehlgeschlagen: {e}")
            raise
        finally:
            # nur temporäre Dateien aufräumen – run-spezifische Artefakte behalten
            try:
                if temp_excel_path:
                    p = Path(temp_excel_path)
                    # Projektbasis
                    base_dir = Path(__file__).resolve().parent.parent

                    # Flags, ob die Datei in einem der "dauerhaften" Verzeichnisse liegt
                    is_in_loader_runs = ("loader" in p.parts and "runs" in p.parts)
                    is_in_loader_financial = ("loader" in p.parts and "financial_cache" in p.parts)

                    # OS-Tempdir erkennen
                    tmpdir = Path(tempfile.gettempdir())
                    is_in_os_tmp = tmpdir in p.parents

                    # Wenn es eine echte Temp-Datei ist ODER nicht im loader/* run-Kontext liegt → löschen
                    if (not is_in_loader_runs) and (not is_in_loader_financial):
                        if p.exists():
                            try:
                                os.remove(p)
                                LOGGER.info(f"[run_forecast|cleanup] temporäre Excel entfernt: {p}")
                            except Exception as _e_rm:
                                LOGGER.warning(f"[run_forecast|cleanup] Konnte temporäre Datei nicht löschen: {_e_rm}")
                    else:
                        # bewusst liegen lassen
                        LOGGER.info(f"[run_forecast|cleanup] Excel wird behalten (run-/loader-Kontext): {p}")
            except Exception as _e_final:
                LOGGER.warning(f"[run_forecast|cleanup] Aufräumen übersprungen: {_e_final}")

    # ------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------
    @staticmethod
    def _load_exog_frame(payload) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame()

        if isinstance(payload, str):
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
    def _load_exog_from_default_cache(base_dir: "Path | None" = None) -> pd.DataFrame:
        """
        Versucht, den von deinem Loader geschriebenen Standard-Cache
        (loader/financial_cache/output.xlsx) zu laden.
        """
        try:
            from pathlib import Path

            if base_dir is None:
                base_dir = Path(__file__).resolve().parent.parent
            cache_file = base_dir / "loader" / "financial_cache" / "output.xlsx"
            if not cache_file.exists():
                return pd.DataFrame()
            df_loader = pd.read_excel(cache_file)
            if "Datum" in df_loader.columns and "date" not in df_loader.columns:
                df_loader = df_loader.rename(columns={"Datum": "date"})
            if "date" in df_loader.columns:
                df_loader["date"] = pd.to_datetime(df_loader["date"], errors="coerce")
                df_loader = df_loader.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df_loader
        except Exception as _e_cache:
            LOGGER.warning(f"[Adapter|Exog] Cache-Fallback konnte nicht geladen werden: {_e_cache}")
            return pd.DataFrame()

    @staticmethod
    def _load_custom_prepared_df(
        payload: Optional[Union[dict, str]]
    ) -> Optional[pd.DataFrame]:
        """
        Lädt ein benutzerdefiniertes PIPELINE_PREP-DataFrame aus dem
        custom-final-dataset-store.

        Erwartet ein Dict der Form:
            {
                "filename": "...",
                "uploaded_at": "...",
                "sheet_name": "...",
                "json": "<df.to_json(orient='split')>"
            }
        oder direkt einen JSON-String im Split-Format.
        """
        if payload is None:
            return None

        try:
            obj: dict
            if isinstance(payload, str):
                # könnte bereits ein JSON-String sein
                try:
                    obj = json.loads(payload)
                except Exception:
                    # oder direkt df.to_json(...)
                    df = pd.read_json(payload, orient="split")
                    return df
            elif isinstance(payload, dict):
                obj = payload
            else:
                LOGGER.warning(
                    "[Adapter] custom_final_dataset hat unerwarteten Typ: %r", type(payload)
                )
                return None

            df_json = obj.get("json")
            if not df_json:
                LOGGER.warning(
                    "[Adapter] custom_final_dataset ohne 'json'-Feld – wird ignoriert."
                )
                return None

            df = pd.read_json(df_json, orient="split")

            if df.empty:
                LOGGER.warning("[Adapter] custom_final_dataset ist leer.")
                return None

            # Datumsspalte normalisieren
            if "date" not in df.columns:
                for cand in ("Datum", "DATE", "Date", "ds", "time", "Time"):
                    if cand in df.columns:
                        df = df.rename(columns={cand: "date"})
                        break

            if "date" not in df.columns:
                LOGGER.warning(
                    "[Adapter] custom_final_dataset hat keine 'date'/'Datum' Spalte – "
                    "wird ignoriert."
                )
                return None

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            return df

        except Exception as e:
            LOGGER.warning(
                "[Adapter] Konnte custom_final_dataset nicht laden/parsen: %s", e
            )
            return None

    @staticmethod
    def _make_cache_tag(
        sektor: str | None,
        excel_target_col: str,
        ui_mode: str,
        horizon: int,
        selected_exog: list[str] | None,
    ) -> str:
        import hashlib

        sektor_slug = (sektor or "PH").strip().upper()
        target_slug = (excel_target_col or "").strip().lower().replace(" ", "_")
        mode = (ui_mode or "bestand").strip().lower()
        if selected_exog:
            exog_sorted = sorted(str(x) for x in selected_exog)
            exog_part = "§".join(exog_sorted)
        else:
            exog_part = "noexog"

        sig_input = "|".join(
            [
                sektor_slug.lower(),
                target_slug,
                mode,
                f"h{int(horizon or 4)}",
                exog_part,
            ]
        )
        cache_sig = hashlib.md5(sig_input.encode("utf-8")).hexdigest()[:8]
        return f"{sektor_slug.lower()}_{target_slug}_{mode}_h{int(horizon or 4)}_{cache_sig}"

    @staticmethod
    def _ui_target_to_excel_col(ui_target: str, sektor: str | None = "PH") -> str:
        sektor_prefix = (sektor or "PH").strip().upper()
        name = (ui_target or "").strip().lower()

        base_map = {
            "einlagen": "EINLAGEN",
            "wertpapiere": "WERTPAPIERE",
            "versicherungen": "VERSICHERUNGEN",
            "kredite": "KREDITE",
            "gesamt": "GESAMT",
            "gvb": "GVB",
        }

        if name in base_map:
            col = base_map[name]
        else:
            col = name.replace(" ", "_").upper() if name else "EINLAGEN"

        if col == "GVB":
            return col

        if col.startswith(sektor_prefix + "_"):
            return col

        return f"{sektor_prefix}_{col}"

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
    def _exog_variants_for_name(name: str) -> set[str]:
        """
        Erzeugt Varianten eines Exog-Namens zum Matching.
        
        WICHTIG: Verwendet nun die normalisierte Form für besseres Matching.
        """
        name = str(name).strip()
        
        # Normalisierte Form (Punkte -> Unterstriche)
        norm_name = DashboardForecastAdapter._normalize_exog_name_for_match(name)
        
        # Erstelle Varianten sowohl mit Original als auch mit normalisierter Form
        variants = {
            name,                    # Original (z.B. mit Punkten)
            norm_name,               # Normalisiert (mit Unterstrichen)
            f"{norm_name}__last",    # Mit __last Suffix
            f"{norm_name}__last__",  # Mit __last__ Suffix
        }
        
        # Füge auch Varianten mit dem Original-Namen hinzu (falls unterschiedlich)
        if name != norm_name:
            variants.add(f"{name}__last")
            variants.add(f"{name}__last__")
        
        return variants

    @staticmethod
    def _has_all_requested_exogs(
        self,
        df: pd.DataFrame,
        requested: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Prüft, ob alle requested-Exogs in df prinzipiell verfügbar wären.
        Die Signatur erlaubt sowohl Aufrufe mit df als auch mit (df, requested),
        damit der bisherige Code keinen TypeError mehr auslöst.

        Wird nur als "non-blocking" Guard genutzt – im Zweifel lieber True
        zurückgeben, statt die Pipeline zu blockieren.
        
        KORREKTUR: Nutzt nun _resolve_exogs_in_df korrekt.
        """
        if requested is None:
            # falls der Aufrufer keine Liste übergibt, einfach "alles gut" sagen
            return True

        if df is None or df.empty:
            return False

        # WICHTIG: _resolve_exogs_in_df ist eine Klassenmethode, kein self-Parameter nötig
        mapping = DashboardForecastAdapter._resolve_exogs_in_df(requested, df)
        
        # "alle gefunden" wenn mapping alle requested enthält
        return len(mapping) == len([r for r in requested if r is not None])


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

    @staticmethod
    def _get_file_stats(path: str):
        import os

        try:
            st = os.stat(path)
            return True, float(st.st_mtime), int(st.st_size)
        except FileNotFoundError:
            return False, 0.0, 0
        except Exception:
            return False, 0.0, 0

    @staticmethod
    def _wait_for_loader_output(
        excel_path: str,
        previous_mtime: float,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        require_stable: bool = True,
        logger=LOGGER,
    ) -> bool:
        """
        Wartet darauf, dass der Loader eine Excel-Datei fertig geschrieben hat.

        Verbesserungen (Punkt 7):
        - besseres Logging (wir sehen, ob die Datei nie da war / schon da war)
        - toleranter bei Fällen, in denen das FS den mtime nicht ändert
        - 'stable'-Check bleibt, aber wir akzeptieren auch den Fall:
            * Datei existiert
            * Größe > 0
            * und wir haben sie mindestens 2x gleich groß gesehen
        """
        import time

        start = time.time()
        last_size = None
        stable_count = 0
        first_seen = None

        logger.info(
            "[Adapter|Wait] Warte auf Loader-Output: %s (prev_mtime=%.3f, timeout=%.1fs, require_stable=%s)",
            excel_path,
            previous_mtime,
            timeout,
            require_stable,
        )

        while True:
            exists, mtime, size = DashboardForecastAdapter._get_file_stats(excel_path)

            if exists:
                if first_seen is None:
                    first_seen = time.time()
                    logger.info(
                        "[Adapter|Wait] Loader-Output existiert jetzt: %s (mtime=%.3f, size=%s)",
                        excel_path,
                        mtime,
                        size,
                    )

                # Fall A: klassischer Fall – Datei ist neuer als vorher
                is_newer_than_before = mtime > previous_mtime

                # Fall B: wir hatten keinen vorherigen mtime (0.0) → jede existierende Datei ist jetzt relevant
                is_relevant_without_mtime = (previous_mtime == 0.0)

                if not require_stable and (is_newer_than_before or is_relevant_without_mtime):
                    logger.info("[Adapter|Wait] Loader-Output erkannt (%s).", excel_path)
                    return True

                # require_stable = True → wir wollen mindestens 2 gleiche Größen hintereinander sehen
                if last_size is None:
                    # erster Messpunkt
                    last_size = size
                else:
                    if size == last_size and size > 0:
                        stable_count += 1
                    else:
                        # Größe hat sich geändert → wieder von vorn zählen
                        stable_count = 0
                        last_size = size

                # wir akzeptieren Stabilität nach 2 gleichen Beobachtungen
                if (is_newer_than_before or is_relevant_without_mtime) and stable_count >= 1:
                    logger.info(
                        "[Adapter|Wait] Loader-Output stabil (%s, size=%s, mtime=%.3f).",
                        excel_path,
                        size,
                        mtime,
                    )
                    return True

            # Timeout prüfen
            if time.time() - start > timeout:
                logger.warning(
                    "[Adapter|Wait] Timeout nach %.1fs beim Warten auf Loader-Output (%s). "
                    "Fahre mit aktuellem Stand fort.",
                    timeout,
                    excel_path,
                )
                return False

            time.sleep(poll_interval)


    def _project_exog_series_monthly(
        self,
        s: pd.Series,
        h_months: int,
        *,
        period_m: int = 12,
        drift_window_m: int = 24,
        stable_std_threshold: float = 1e-6,
    ) -> np.ndarray:
        s = pd.to_numeric(pd.Series(s).dropna(), errors="coerce").dropna()
        n = len(s)
        if n == 0 or h_months <= 0:
            return np.array([], dtype=float)

        if float(s.std(ddof=1) if n > 1 else 0.0) < stable_std_threshold:
            last_val = float(s.iloc[-1])
            return np.full(h_months, last_val, dtype=float)

        month_idx = s.index.month if isinstance(s.index, pd.DatetimeIndex) else pd.Series(np.arange(n) % period_m) + 1
        seas_means = s.groupby(month_idx).mean().to_dict()

        win = min(max(drift_window_m, 6), n)
        y = s.iloc[-win:].values.astype(float)
        x = np.arange(win, dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = 0.0 if denom == 0.0 else ((x - x_mean) * (y - y_mean)).sum() / denom
        last_val = float(s.iloc[-1])

        out = np.zeros(h_months, dtype=float)
        last_idx = s.index[-1] if isinstance(s.index, pd.DatetimeIndex) else None
        for t in range(1, h_months + 1):
            if isinstance(last_idx, pd.Timestamp):
                future_month = (last_idx + pd.DateOffset(months=t)).month
            else:
                future_month = ((n + (t - 1)) % period_m) + 1

            seas = float(seas_means.get(future_month, 0.0))
            val = (last_val + t * slope) + (seas - y_mean)
            if n < period_m or win < 6:
                val = last_val
            out[t - 1] = val

        return out

    def _extend_exogs_to_future_ms(self, exog_monthly: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
        if horizon_months is None or int(horizon_months) <= 0:
            return exog_monthly

        df = exog_monthly.copy()
        if "date" not in df.columns or df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        start = df["date"].min().to_period("M").to_timestamp(how="start")
        end = df["date"].max().to_period("M").to_timestamp(how="start")
        full_idx = pd.date_range(start=start, end=end, freq="MS")

        exog_cols = [c for c in df.columns if c != "date"]
        df_full = df.set_index("date").reindex(full_idx).sort_index()
        if exog_cols:
            df_full[exog_cols] = df_full[exog_cols].ffill()

        future_idx = pd.date_range(end + pd.offsets.MonthBegin(1), periods=int(horizon_months), freq="MS")
        fut = pd.DataFrame(index=future_idx, columns=exog_cols, dtype=float)

        for c in exog_cols:
            hist = df_full[c].dropna()
            if hist.empty:
                continue
            fut[c] = self._project_exog_series_monthly(hist, h_months=int(horizon_months))

        out = pd.concat([df_full, fut], axis=0).reset_index().rename(columns={"index": "date"})
        return out

    @staticmethod
    def _calculate_confidence_intervals(
        predictions: np.ndarray,
        residuals: np.ndarray,
        confidence_levels: List[int] = None,
    ) -> Dict[str, np.ndarray]:
        if confidence_levels is None:
            confidence_levels = [80, 95]

        std_error = np.std(residuals, ddof=1)
        z_scores = {
            68: 0.9945,
            80: 1.282,
            90: 1.645,
            95: 1.96,
            99: 2.576,
        }

        intervals = {}

        for level in confidence_levels:
            z = z_scores.get(level, 1.96)
            margin = z * std_error

            intervals[f"lower_{level}"] = predictions - margin
            intervals[f"upper_{level}"] = predictions + margin

        LOGGER.info(f"[CI] Konfidenzintervalle berechnet: {confidence_levels}%, std_error={std_error:.4f}")

        return intervals

    @staticmethod
    def _add_confidence_intervals_to_forecast(
        forecast_df: pd.DataFrame,
        residuals: np.ndarray,
        confidence_levels: List[int] = None,
        forecast_col: str = "Forecast",
    ) -> pd.DataFrame:
        if confidence_levels is None:
            confidence_levels = [80, 95]

        df = forecast_df.copy()

        if forecast_col not in df.columns:
            LOGGER.warning(f"[CI] Spalte '{forecast_col}' nicht gefunden, verwende erste numerische Spalte")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            forecast_col = numeric_cols[0] if len(numeric_cols) > 0 else None

        if forecast_col is None:
            LOGGER.error("[CI] Keine Forecast-Spalte gefunden!")
            return df

        predictions = df[forecast_col].values

        intervals = DashboardForecastAdapter._calculate_confidence_intervals(
            predictions=predictions,
            residuals=residuals,
            confidence_levels=confidence_levels,
        )

        for key, values in intervals.items():
            df[f"yhat_{key}"] = values

        LOGGER.info(f"[CI] Hinzugefügte Spalten: {list(intervals.keys())}")

        return df

    @staticmethod
    def _cap_seasonal_quarter_lows(
        tq: pd.Series,
        *,
        quarter: int = 3,
        back_years: int = 4,
        factor: float = 1.0,
        logger=LOGGER,
    ) -> pd.Series:
        if tq is None or tq.empty:
            return tq

        tq = tq.sort_index()
        out = tq.copy()

        for per in tq.index:
            if getattr(per, "quarter", None) != quarter:
                continue

            prev_periods = [per - 4 * i for i in range(1, back_years + 1)]
            prev_vals = [tq.get(p, np.nan) for p in prev_periods]
            prev_vals = pd.Series(prev_vals, dtype="float64").dropna()
            if prev_vals.empty:
                continue

            seasonal_median = float(prev_vals.median())
            cur_val = float(out.loc[per])

            threshold = seasonal_median * factor
            if cur_val < threshold:
                logger.info(
                    f"[Smooth] Deckel {per} (Q{quarter}): {cur_val:.2f} -> {threshold:.2f} "
                    f"(median {back_years}y Q{quarter} = {seasonal_median:.2f})"
                )
                out.loc[per] = threshold

        return out
    
    
    def prepare_pipeline_data(
        self,
        target: str,
        selected_exog: List[str],
        use_flows: bool = False,
        *,
        horizon_quarters: int = 0,
    ) -> pd.DataFrame:
        """
        KORREKTUR: Verwendet nun _resolve_exogs_in_df für robustes Exog-Matching.
        Optional: nutzt ein benutzerdefiniertes PIPELINE_PREP DataFrame,
        falls über custom-final-dataset-store geladen.
        """
        data_type = "fluss" if use_flows else "bestand"
        value_col = "fluss" if use_flows else "bestand"

        LOGGER.info(f"[Adapter] use_flows={use_flows} -> data_type='{data_type}'")
        LOGGER.info(f"[Adapter] requested_exog={selected_exog}")
        LOGGER.info(
            f"[Adapter] exog_data.columns (Top): "
            f"{self.exog_data.columns.tolist()[:12] if not self.exog_data.empty else []}"
        )

        # ------------------------------------------------------------------
        # 0) Spezialfall: benutzerdefiniertes PIPELINE_PREP aus Upload
        # ------------------------------------------------------------------
        custom_df = getattr(self, "custom_prepared_df", None)
        if custom_df is not None and isinstance(custom_df, pd.DataFrame) and not custom_df.empty:
            df = custom_df.copy()

            # Datumsspalte normalisieren
            if "date" not in df.columns:
                for cand in ("Datum", "Date", "DATE", "ds", "time", "Time"):
                    if cand in df.columns:
                        df = df.rename(columns={cand: "date"})
                        break

            if "date" not in df.columns:
                raise ValueError(
                    "[Adapter] Custom-Dataset hat keine 'date' oder 'Datum' Spalte. "
                    "Bitte das unveränderte PIPELINE_PREP-Sheet als Grundlage verwenden."
                )

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            # Zielspalte identifizieren, falls noch nicht 'target_value'
            if "target_value" not in df.columns:
                non_date = [c for c in df.columns if c != "date"]
                numeric_candidates = [
                    c
                    for c in non_date
                    if pd.to_numeric(df[c], errors="coerce").notna().any()
                ]
                if numeric_candidates:
                    df = df.rename(columns={numeric_candidates[0]: "target_value"})

            if "target_value" not in df.columns:
                raise ValueError(
                    "[Adapter] Custom-Dataset enthält keine Zielspalte 'target_value' "
                    "und keine numerische Spalte, die automatisch zugeordnet werden konnte."
                )

            exog_cols_only = [c for c in df.columns if c not in ("date", "target_value")]

            tgt_non_null = df["target_value"].dropna()
            tmin = float(tgt_non_null.min()) if not tgt_non_null.empty else np.nan
            tmax = float(tgt_non_null.max()) if not tgt_non_null.empty else np.nan

            LOGGER.info(
                _sym(
                    f"[Adapter] Verwende benutzerdefiniertes PIPELINE_PREP (Upload). "
                    f"rows={len(df)}, exog_cols={exog_cols_only[:10]}{'...' if len(exog_cols_only) > 10 else ''}"
                )
            )

            pi = getattr(self, "pipeline_info", {}) or {}
            pi.setdefault("ui_target", target)
            pi.update(
                {
                    "custom_prepared": True,
                    "prepared_rows": int(len(df)),
                    "prepared_exog_cols": exog_cols_only,
                    "target_min": tmin,
                    "target_max": tmax,
                    "data_type": data_type,
                    "use_flows": use_flows,
                }
            )
            self.pipeline_info = pi

            return df

        # ------------------------------------------------------------------
        # 1) Standardpfad: Aggregation aus GVB + Exog Store
        # ------------------------------------------------------------------
        if target == "gesamt":
            base = self.gvb_data[self.gvb_data["datatype"] == data_type]
            target_data = base.groupby("date", as_index=False)[value_col].sum()
        else:
            base = self.gvb_data
            base = base[(base["ebene1"] == target) & (base["datatype"] == data_type)]
            target_data = base.groupby("date", as_index=False)[value_col].sum()

        target_data = target_data.rename(columns={value_col: "target_value"})
        target_data["date"] = pd.to_datetime(target_data["date"], errors="coerce")
        target_data = (
            target_data.dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        tq = (
            target_data.set_index("date")["target_value"]
            .to_period("Q")
            .groupby(level=0)
            .last()
        )

        tq = self._cap_seasonal_quarter_lows(
            tq,
            quarter=3,
            back_years=4,
            factor=1.0,
            logger=LOGGER,
        )

        tq_start = tq.to_timestamp(how="start")

        start_ms = target_data["date"].min().to_period("M").to_timestamp(how="start")
        end_ms = target_data["date"].max().to_period("M").to_timestamp(how="start")
        monthly_index_hist = pd.date_range(start=start_ms, end=end_ms, freq="MS")

        target_monthly = tq_start.reindex(monthly_index_hist).ffill()
        target_monthly_df = (
            target_monthly.rename("target_value")
            .rename_axis("date")
            .reset_index()
        )

        # ====================================================================
        # KORRIGIERTER TEIL: Exog-Resolution mit _resolve_exogs_in_df
        # ====================================================================
        resolved_exog_map: dict[str, str] = {}
        if selected_exog and not self.exog_data.empty:
            if "date" not in self.exog_data.columns:
                LOGGER.warning(
                    "[Adapter] Exog-Frame hat keine 'date'-Spalte – Exogs werden übersprungen."
                )
                exog_monthly = pd.DataFrame({"date": monthly_index_hist})
            else:
                # Verwende die zentrale _resolve_exogs_in_df Funktion
                resolved_exog_map = self._resolve_exogs_in_df(
                    selected_exog,
                    self.exog_data,
                )

                missing_exogs = [
                    req for req in selected_exog if req not in resolved_exog_map
                ]

                if missing_exogs:
                    LOGGER.warning(
                        "[Adapter] Einige gewünschte exogene Variablen wurden im "
                        "exog_data-Frame nicht gefunden (auch nicht als __last): "
                        f"{missing_exogs}"
                    )

                if resolved_exog_map:
                    # Verwende die aufgelösten (tatsächlichen) Spaltennamen
                    available_exog = list(dict.fromkeys(resolved_exog_map.values()))
                    LOGGER.info(
                        _sym(
                            f"[Adapter] Füge {len(available_exog)} exogene Variablen "
                            f"hinzu (aufgelöst): {available_exog}"
                        )
                    )

                    exog_monthly = self.exog_data[["date"] + available_exog].copy()
                    exog_monthly["date"] = pd.to_datetime(
                        exog_monthly["date"],
                        errors="coerce",
                    )
                    exog_monthly = exog_monthly.dropna(subset=["date"])

                    exog_monthly["date"] = self._to_ms(exog_monthly["date"])
                    exog_monthly = (
                        exog_monthly.groupby("date", as_index=False).last()
                    )

                    # Stelle sicher, dass alle Spalten ein __last Suffix haben
                    rename_dict = {}
                    for col in available_exog:
                        if col.endswith("__last") or col.endswith("__last__"):
                            rename_dict[col] = col
                        else:
                            rename_dict[col] = f"{col}__last"

                    exog_monthly = exog_monthly.rename(columns=rename_dict)

                    future_months = int(horizon_quarters or 0) * 3
                    if future_months > 0:
                        exog_monthly = self._extend_exogs_to_future_ms(
                            exog_monthly,
                            future_months,
                        )

                    LOGGER.debug(
                        f"[Adapter] Exog-Spalten nach Projection: "
                        f"{exog_monthly.columns.tolist()}"
                    )
                else:
                    exog_monthly = pd.DataFrame({"date": monthly_index_hist})
        else:
            exog_monthly = pd.DataFrame({"date": monthly_index_hist})

        result = (
            pd.merge(target_monthly_df, exog_monthly, on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )

        exog_cols_only = [c for c in result.columns if c not in ["date", "target_value"]]
        if exog_cols_only:
            result[exog_cols_only] = result[exog_cols_only].ffill()

        tgt_non_null = result["target_value"].dropna()
        tmin = float(tgt_non_null.min()) if not tgt_non_null.empty else np.nan
        tmax = float(tgt_non_null.max()) if not tgt_non_null.empty else np.nan

        LOGGER.info(_sym("\n[Adapter] Pipeline-Daten (MS-normalisiert + Exog-Forecast):"))
        LOGGER.info(f"  Input-Quartale: {tq.shape[0]}")
        LOGGER.info(
            f"  Output-Monate:  {len(result)}  "
            f"(inkl. Exog-Zukunft: +{int(horizon_quarters) * 3 if horizon_quarters else 0}M)"
        )
        LOGGER.info(f"  Target-Range:  {tmin:.3f} - {tmax:.3f}")
        LOGGER.info(
            f"  Exog-Spalten:  {exog_cols_only[:10]}"
            f"{'...' if len(exog_cols_only) > 10 else ''}"
        )
        if result.isna().sum().sum() > 0:
            nz = result.isna().sum()
            LOGGER.warning(f"  NaN nach Merge: {dict(nz[nz > 0])}")

        if result["target_value"].isna().all():
            raise ValueError("Target-Variable enthält nur NaN-Werte.")

        pi = getattr(self, "pipeline_info", {}) or {}
        pi.setdefault("ui_target", target)
        pi.update(
            {
                "n_in_quarters": int(tq.shape[0]),
                "n_out_months": int(len(result)),
                "target_min": tmin,
                "target_max": tmax,
                "exog_cols": exog_cols_only,
                "data_type": data_type,
                "use_flows": use_flows,
                "exog_resolved_from_prepare": resolved_exog_map,
            }
        )
        self.pipeline_info = pi

        return result

    def create_temp_excel(
        self,
        df: pd.DataFrame,
        target_col: str = "target_value",
        excel_target_col: str = "PH_EINLAGEN",
        cache_tag: str | None = None,
    ) -> tuple[str, str]:
        from datetime import datetime
        from pathlib import Path
        import tempfile
        import json

        # Basisverzeichnis des Projekts
        base_dir = Path(__file__).resolve().parent.parent

        # 1) Versuche: lauf-spezifischen Ordner aus dem Handshake nehmen
        run_dir: Path | None = None
        if cache_tag:
            active_path = base_dir / "loader" / "active_run.json"
            if active_path.exists():
                try:
                    active = json.loads(active_path.read_text(encoding="utf-8"))
                    if active.get("run_cache_tag") == cache_tag and active.get("run_loader_dir"):
                        rd = Path(active["run_loader_dir"])
                        rd.mkdir(parents=True, exist_ok=True)
                        run_dir = rd
                        LOGGER.info(
                            f"[create_temp_excel] benutze run-spezifischen Ordner aus active_run.json: {run_dir}"
                        )
                except Exception as _e_active:
                    LOGGER.warning(f"[create_temp_excel] active_run.json konnte nicht gelesen werden: {_e_active}")

        # 2) Wenn kein lauf-spezifischer Ordner vorhanden, aber cache_tag gegeben:
        #    verwende wie bisher loader/runs/<cache_tag>/
        if run_dir is None and cache_tag:
            run_dir = base_dir / "loader" / "runs" / cache_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[create_temp_excel] benutze runs-Ordner für cache_tag={cache_tag}: {run_dir}")

        # 3) Ziel-Dateipfad bestimmen
        if run_dir is not None:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            tmp_path = str(run_dir / f"{ts}_final_dataset.xlsx")
        else:
            # echter System-Temp-Fallback
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp_path = tmp.name
            tmp.close()
            LOGGER.info(f"[create_temp_excel] Fallback auf echtes Tempfile: {tmp_path}")

        # ------------------------------------------------------------------
        # Excel-Inhalt aufbereiten
        # ------------------------------------------------------------------
        rename_map = {"date": "Datum", target_col: excel_target_col}
        df_export = df.rename(columns=rename_map).copy()
        df_export["Datum"] = pd.to_datetime(df_export["Datum"], errors="coerce")

        LOGGER.info(_sym("\n[Adapter] Excel-Export:"))
        LOGGER.info(f"  Ziel: {tmp_path}")
        LOGGER.info(f"  Shape: {df_export.shape}")
        LOGGER.info(f"  Spalten: {df_export.columns.tolist()}")

        if not df_export.empty:
            dmin = df_export["Datum"].min()
            dmax = df_export["Datum"].max()
            tmin = df_export[excel_target_col].min()
            tmax = df_export[excel_target_col].max()
            LOGGER.info(f"  Datum-Range: {dmin} - {dmax}")
            LOGGER.info(f"  Target-Range ({excel_target_col}): {tmin:.1f} - {tmax:.1f}")

            exog_cols = [c for c in df_export.columns if c not in ["Datum", excel_target_col]]
            if exog_cols:
                LOGGER.info(f"  Exogene Variablen ({len(exog_cols)}): {exog_cols}")
                for col in exog_cols[:3]:
                    non_na = int(df_export[col].notna().sum())
                    LOGGER.info(f"    {col}: {non_na}/{len(df_export)} Werte")

        # schreiben
        df_export.to_excel(tmp_path, sheet_name="final_dataset", index=False)

        # kurze Validierung
        try:
            v = pd.read_excel(tmp_path, sheet_name="final_dataset")
            LOGGER.info(f"[Adapter] Excel-Validierung: {len(v)} Zeilen, Spalten: {list(v.columns)}")
        except Exception as e:
            LOGGER.warning(f"[Adapter] Excel-Validierung fehlgeschlagen: {e}")

        return tmp_path, excel_target_col

    def _build_config(
        self,
        *,
        excel_path: str,
        horizon: int,
        use_cached: bool,
        selected_exog: List[str],
        ui_target: str = None,
        ui_mode: str = None,
        sektor: str = None,
        excel_target_col: str = None,
        cache_tag: str | None = None,
    ) -> "PipelineConfig":
        import hashlib
        from pathlib import Path
        import pandas as pd

        if not HAS_PIPELINE or PipelineConfig is None:
            raise RuntimeError(f"Forecast-Pipeline nicht verfügbar. Detail: {_PIPELINE_IMPORT_ERROR!r}")

        exog_final = list(selected_exog or [])
        exog_clean = [x for x in exog_final if "__flows_flag__" not in str(x)]

        if excel_target_col:
            excel_target_effective = str(excel_target_col)
        else:
            try:
                cols = list(pd.read_excel(excel_path, sheet_name="final_dataset", nrows=0).columns)
            except Exception:
                cols = ["Datum", "PH_EINLAGEN"]
                LOGGER.warning(
                    "[Adapter|Config] Konnte Excel nicht lesen (%s) – verwende Fallback-Spalten.",
                    excel_path,
                )
            excel_target_effective = next((c for c in cols if str(c).lower() != "datum"), None) or "PH_EINLAGEN"

        if ui_target:
            ui_norm = str(ui_target).strip().lower()
            excel_norm = str(excel_target_effective).strip().lower()
            if ui_norm != excel_norm:
                LOGGER.warning(
                    "[Adapter|Config|MISMATCH] UI-Target='%s' ≠ Excel-Target='%s'. "
                    "Es wird das Excel-Target verwendet.",
                    ui_target,
                    excel_target_effective,
                )

        if ui_mode:
            ui_mode_effective = ui_mode
        else:
            if any("__flows_flag__" in str(x) for x in exog_final):
                ui_mode_effective = "fluss"
            else:
                ui_mode_effective = "bestand"

        sektor_effective = sektor or "PH"
        sektor_slug = str(sektor_effective).strip().upper()

        if cache_tag:
            cache_tag_effective = cache_tag
        else:
            cache_tag_effective = self._make_cache_tag(
                sektor=sektor_effective,
                excel_target_col=excel_target_effective,
                ui_mode=ui_mode_effective,
                horizon=horizon,
                selected_exog=exog_clean,
            )

        forecaster_dir = Path(__file__).parent
        output_dir = (forecaster_dir / "trained_outputs" / sektor_slug).resolve()
        model_dir = (forecaster_dir / "trained_models" / sektor_slug).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        desired_kwargs = dict(
            excel_path=excel_path,
            sheet_name="final_dataset",
            date_col="Datum",
            target_col=str(excel_target_effective),
            agg_methods_exog=["last"],
            agg_method_target="mean",
            exog_month_lags=[-12, -6, -3, -1],
            target_lags_q=[1, 2, 4],
            add_trend_features=True,
            trend_degree=2,
            add_seasonality=True,
            seasonality_mode="dummies",
            target_transform="none",
            target_standardize=True,
            forecast_horizon=int(horizon or 4),
            future_exog_strategy="mixed",
            future_exog_drift_window_q=8,
            future_exog_seasonal_period_q=4,
            output_dir=str(output_dir),
            model_dir=str(model_dir),
            use_cached_model=bool(use_cached),
            random_state=42,
            cache_tag=cache_tag_effective,
            selected_exog=list(exog_clean),
        )

        allowed_keys = set()
        ignored_keys = []
        try:
            from dataclasses import fields as _dc_fields, is_dataclass as _is_dc

            if _is_dc(PipelineConfig):
                allowed_keys = {f.name for f in _dc_fields(PipelineConfig)}
            else:
                import inspect as _inspect

                sig = _inspect.signature(PipelineConfig)
                allowed_keys = {
                    p.name
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                }
        except Exception:
            allowed_keys = {
                "excel_path",
                "sheet_name",
                "date_col",
                "target_col",
                "agg_methods_exog",
                "agg_method_target",
                "exog_month_lags",
                "target_lags_q",
                "add_trend_features",
                "trend_degree",
                "add_seasonality",
                "seasonality_mode",
                "target_transform",
                "target_standardize",
                "forecast_horizon",
                "future_exog_strategy",
                "future_exog_drift_window_q",
                "future_exog_seasonal_period_q",
                "output_dir",
                "model_dir",
                "use_cached_model",
                "random_state",
                "cache_tag",
                "selected_exog",
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

        cfg = PipelineConfig(**filtered_kwargs)

        try:
            cfg.selected_exog = list(exog_clean)
        except Exception:
            pass
        if hasattr(cfg, "exog_cols"):
            try:
                cfg.exog_cols = list(exog_clean)
            except Exception:
                pass

        try:
            LOGGER.info(f"[Adapter] Config.target_col (aus Excel/Caller): {excel_target_effective}")
            if ui_target:
                LOGGER.info(f"[Adapter] UI-target (nur Info): {ui_target}")
            LOGGER.info(f"[Adapter] Config.selected_exog: {getattr(cfg, 'selected_exog', None)}")
            if hasattr(cfg, "exog_cols"):
                LOGGER.info(f"[Adapter] Config.exog_cols: {getattr(cfg, 'exog_cols', None)}")
            LOGGER.info(
                f"[Adapter] cache_tag={getattr(cfg, 'cache_tag', None)}, "
                f"horizon={getattr(cfg, 'forecast_horizon', None)}, "
                f"ui_mode={ui_mode}, sektor={sektor}"
            )
        except Exception:
            pass

        return cfg

    def _extract_residuals_from_pipeline(
        self,
        metadata: dict,
        model_path: str = None,
        reference_value: float = None,
        n_synth: int = 1000,
    ) -> np.ndarray:
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
            try:
                y_plus = inv_func(np.array([[z0 + eps]])).ravel()[0]
                y_minus = inv_func(np.array([[z0 - eps]])).ravel()[0]
                return float((y_plus - y_minus) / (2.0 * eps))
            except Exception:
                return np.nan

        if metadata and "cv_residuals" in metadata and isinstance(metadata["cv_residuals"], (list, tuple)):
            res = np.array(metadata["cv_residuals"], dtype=float)
            res = res[np.isfinite(res)]
            if res.size > 0:
                LOGGER.info(f"[CI] Residuen aus Metadata geladen: {res.size} Werte")
                return res

        art = _load_artifact(model_path) if model_path else None
        if art and isinstance(getattr(art, "metadata", None), dict):
            md = art.metadata or {}
            if "cv_residuals" in md and isinstance(md["cv_residuals"], (list, tuple)):
                res = np.array(md["cv_residuals"], dtype=float)
                res = res[np.isfinite(res)]
                if res.size > 0:
                    LOGGER.info(f"[CI] Residuen aus Modell geladen: {res.size} Werte")
                    return res

        if metadata and isinstance(metadata.get("cv_performance"), dict):
            cv = metadata["cv_performance"]
            rmse_val = cv.get("cv_rmse", cv.get("rmse", None))
            if isinstance(rmse_val, (int, float)) and np.isfinite(rmse_val):
                rmse = float(rmse_val)
                scale_flag = (metadata.get("cv_metrics_scale") or "").lower() or "original"

                if scale_flag == "original":
                    residuals = np.random.normal(0.0, rmse, size=int(n_synth))
                    LOGGER.info(f"[CI] Synthetische Residuen (original scale) mit RMSE={rmse:.4f}, n={n_synth}")
                    return residuals

                tj = getattr(art, "tj", None) if art else None

                y0 = None
                if isinstance(reference_value, (int, float)) and np.isfinite(reference_value):
                    y0 = float(reference_value)
                elif isinstance(metadata.get("context"), dict):
                    ctx = metadata["context"]
                    for key in ("last_hist_value", "first_forecast"):
                        v = ctx.get(key)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            y0 = float(v)
                            break
                if y0 is None:
                    ysum = metadata.get("y_train_summary") or {}
                    if isinstance(ysum.get("mean"), (int, float)) and np.isfinite(ysum["mean"]):
                        y0 = float(ysum["mean"])
                if y0 is None:
                    y0 = 0.0

                if tj is not None and hasattr(tj, "inverse_transform") and hasattr(tj, "transform"):
                    try:
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

                    try:
                        z0 = float(tj.transform(np.array([[y0]])).ravel()[0])
                        eps = np.random.normal(0.0, rmse, size=int(n_synth)).reshape(-1, 1)
                        y_sim = tj.inverse_transform((z0 + eps).astype(float)).ravel()
                        residuals = (y_sim - y0).astype(float)
                        residuals = residuals[np.isfinite(residuals)]
                        if residuals.size > 0:
                            LOGGER.warning(
                                f"[CI] RMSE rückskaliert via Monte-Carlo: rmse_t={rmse:.4f} → std_y≈{np.std(residuals):.4f} (y0={y0:.4f})"
                            )
                            return residuals
                    except Exception as e:
                        LOGGER.warning(f"[CI] Rückskalierung (Monte-Carlo) fehlgeschlagen: {e}")

                LOGGER.warning(
                    "[CI] cv_rmse ist 'transformed', aber kein Transformer verfügbar – "
                    "nutze konservativen Default (keine Rückskalierung möglich)."
                )
                return np.random.normal(0.0, rmse, size=int(n_synth))

        LOGGER.error("[CI] Keine Residuen verfügbar – verwende Default-Annahme (std=0.1)")
        return np.random.normal(0.0, 0.1, size=int(n_synth))

    def _generate_backtest_results(
        self,
        model,
        tj,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
        dates_train: Union[pd.Series, pd.DataFrame, pd.Index, np.ndarray],
        n_splits: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        import numpy as np
        from sklearn.base import clone

        EPS = 1e-8

        def _as_series(obj, name=None) -> pd.Series:
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

        X = X_train.copy()
        X.index = pd.RangeIndex(len(X))

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

        X = _safe_impute(X)

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

        for tr, te in expanding_splits(n, min_train, gap, horizon):
            if not tr or not te:
                continue

            X_tr = X.iloc[tr].copy()
            X_te = X.iloc[te].copy()
            y_tr = y.iloc[tr].copy()

            med = X_tr.median(numeric_only=True)
            X_tr = X_tr.fillna(med)
            X_te = X_te.fillna(med)

            if not _fold_variance_ok(y_tr):
                preds[te] = _baseline_forecast(y_tr, len(te))
                continue

            if (X_tr.std(numeric_only=True) < 1e-12).all():
                preds[te] = _baseline_forecast(y_tr, len(te))
                continue

            m = clone(model)

            use_tj = tj is not None
            if use_tj:
                try:
                    from copy import deepcopy

                    tj_local = deepcopy(tj)
                    if hasattr(tj_local, "fit") and hasattr(tj_local, "transform"):
                        tj_local.fit(y_tr.values.reshape(-1, 1))
                        y_tr_t = tj_local.transform(y_tr.values.reshape(-1, 1)).ravel()
                        m.fit(X_tr.values, y_tr_t)
                        yhat_t = m.predict(X_te.values)
                        if hasattr(tj_local, "inverse_transform"):
                            yhat = tj_local.inverse_transform(yhat_t.reshape(-1, 1)).ravel()
                        elif hasattr(tj_local, "inverse"):
                            yhat = tj_local.inverse(yhat_t)
                        else:
                            m = clone(model)
                            m.fit(X_tr.values, y_tr.values)
                            yhat = m.predict(X_te.values)
                    else:
                        m.fit(X_tr.values, y_tr.values)
                        yhat = m.predict(X_te.values)
                except Exception:
                    m = clone(model)
                    m.fit(X_tr.values, y_tr.values)
                    yhat = m.predict(X_te.values)
            else:
                m.fit(X_tr.values, y_tr.values)
                yhat = m.predict(X_te.values)

            yhat = _guard_zero_predictions(yhat, y_tr)

            preds[te] = np.asarray(yhat, dtype=float)

        mask = ~np.isnan(preds)
        backtest_df = pd.DataFrame(
            {
                "date": d[mask].values,
                "actual": y[mask].values,
                "predicted": preds[mask],
            }
        )
        backtest_df["error"] = backtest_df["actual"] - backtest_df["predicted"]
        backtest_df["abs_error"] = backtest_df["error"].abs()
        backtest_df = backtest_df.sort_values("date").reset_index(drop=True)

        residuals_df = backtest_df[["date", "error", "abs_error"]].copy()

        return backtest_df, residuals_df

    def _build_backtest_from_artifact(
        self,
        artifact,
        X_train=None,
        y_train=None,
        dates_train=None,
        n_splits: int = 5,
    ):
        import pandas as pd

        if artifact is None or not hasattr(artifact, "model") or artifact.model is None:
            raise ValueError("[Backtest] Kein gültiges ModelArtifact vorhanden")

        model = artifact.model
        tj = getattr(artifact, "tj", None)

        def _to_df(x):
            if x is None:
                return None
            if isinstance(x, pd.DataFrame):
                return x
            try:
                return pd.DataFrame(x)
            except Exception:
                return None

        def _to_series(x):
            if x is None:
                return None
            if isinstance(x, pd.Series):
                return x
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                return x.iloc[:, 0]
            try:
                return pd.Series(x)
            except Exception:
                return None

        X_df = _to_df(X_train)
        y_ser = _to_series(y_train)
        d_ser = _to_series(dates_train)

        if X_df is None or y_ser is None or d_ser is None:
            LOGGER.warning("[Backtest] Trainingsdaten unvollständig – gebe leere Frames zurück")
            return pd.DataFrame(), pd.DataFrame()

        backtest_df, residuals_df = self._generate_backtest_results(
            model=model,
            tj=tj,
            X_train=X_df,
            y_train=y_ser,
            dates_train=d_ser,
            n_splits=n_splits,
        )
        LOGGER.info(f"[Backtest] {len(backtest_df)} Punkte erzeugt")
        return backtest_df, residuals_df

    # ------------------------------------------------------------
    # neue kleine Hilfen aus run_forecast herausgezogen
    # ------------------------------------------------------------
    @staticmethod
    def _exog_sets_compatible(old_exogs: set[str] | None, cur_exogs: set[str] | None) -> bool:
        old_exogs = set(old_exogs or [])
        cur_exogs = set(cur_exogs or [])
        if not old_exogs and not cur_exogs:
            return True
        if not old_exogs and cur_exogs:
            return False
        if old_exogs and not cur_exogs:
            return False
        return old_exogs == cur_exogs
    

    def _preload_meta_compatible(
        self,
        meta: dict,
        *,
        excel_target_col: str,
        selected_exog: list[str],
        run_cache_tag: str | None = None,
        ui_mode: str | None = None,
        ui_sektor: str | None = None,
        horizon: int | None = None,
    ) -> bool:
        """
        Prüft, ob die Metadaten eines Preload-/Cache-Artifacts zu dem passen,
        was das Dashboard gerade haben will.
        """
        if meta is None:
            return False

        # 1) Target
        m_target = meta.get("target_col") or meta.get("target")
        if m_target and m_target != excel_target_col:
            return False

        # 2) Exogs
        old_exogs = set(meta.get("exog_cols") or [])
        cur_exogs = set(selected_exog or [])
        if old_exogs or cur_exogs:
            if old_exogs != cur_exogs:
                return False

        # 3) cache_tag (falls im Modell drin – ist nicht garantiert)
        if run_cache_tag and meta.get("cache_tag"):
            if meta.get("cache_tag") != run_cache_tag:
                return False

        # 4) optionale Infos
        if ui_mode and meta.get("ui_mode") and meta.get("ui_mode") != ui_mode:
            return False
        if ui_sektor and meta.get("sector") and meta.get("sector") != ui_sektor:
            return False
        if horizon is not None and meta.get("forecast_horizon") is not None:
            try:
                if int(meta.get("forecast_horizon")) != int(horizon):
                    return False
            except Exception:
                pass

        return True







    @classmethod
    def _resolve_exogs_in_df(
        cls,
        requested_exogs: Sequence[str],
        df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Versucht, die in requested_exogs (UI/ECB-Namen) angeforderten Exogs auf
        tatsächliche Spaltennamen im df zu mappen.

        KORREKTUR: Verwendet nun konsequent _normalize_exog_name_for_match für
        robustes Matching zwischen Punkt- und Unterstrich-Notation.

        Rückgabe:
            {requested_name: actual_column_name}
        """
        if not requested_exogs or df is None or df.empty:
            return {}

        # Map: normalisierter Name → tatsächliche df-Spalte
        col_map: Dict[str, str] = {}
        for col in df.columns:
            norm = cls._normalize_exog_name_for_match(col)
            if norm:  # leere Strings ignorieren
                # Wenn mehrere Spalten denselben Normnamen haben, nehmen wir die erste
                col_map.setdefault(norm, col)

        mapping: Dict[str, str] = {}

        for raw_name in requested_exogs:
            norm_req = cls._normalize_exog_name_for_match(raw_name)
            if not norm_req:
                continue

            # Direkter Treffer: gleicher Normalname
            if norm_req in col_map:
                mapping[raw_name] = col_map[norm_req]
                LOGGER.debug(f"[ExogResolve] '{raw_name}' -> '{col_map[norm_req]}' (norm: '{norm_req}')")
                continue
            
            # Fallback: Suche nach Spalten, die mit dem normalisierten Namen beginnen
            # (für Fälle mit zusätzlichen Suffixen wie __last, __l, etc.)
            for col_norm, actual_col in col_map.items():
                if col_norm.startswith(norm_req):
                    mapping[raw_name] = actual_col
                    LOGGER.debug(f"[ExogResolve] '{raw_name}' -> '{actual_col}' (prefix-match: '{norm_req}')")
                    break

        # Logging für nicht gefundene Exogs
        missing = [req for req in requested_exogs if req not in mapping]
        if missing:
            LOGGER.debug(f"[ExogResolve] Nicht gefunden: {missing}")
            LOGGER.debug(f"[ExogResolve] Verfügbare Spalten (normalisiert): {sorted(col_map.keys())[:10]}...")

        return mapping


    def _try_reuse_previous_run_excel(
        self,
        base_dir: Path,
        cache_tag: str,
        excel_target_col: str,
        requested_exogs: list[str],
    ) -> Optional[str]:
        import shutil

        runs_dir = base_dir / "loader" / "runs" / cache_tag
        if not runs_dir.exists():
            LOGGER.info("[Adapter|Reuse] Kein runs-Ordner für cache_tag=%s gefunden.", cache_tag)
            return None

        candidates = sorted(
            runs_dir.glob("*_final_dataset.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            LOGGER.info("[Adapter|Reuse] Keine *_final_dataset.xlsx unter %s", runs_dir)
            return None

        for xlsx in candidates:
            try:
                df_head = pd.read_excel(xlsx, sheet_name="final_dataset", nrows=0)
            except Exception as exc:
                LOGGER.warning("[Adapter|Reuse] Konnte %s nicht lesen: %s", xlsx, exc)
                continue

            cols = {str(c) for c in df_head.columns}
            if excel_target_col not in cols:
                continue

            all_ok = True
            for code in (requested_exogs or []):
                variants = self._exog_variants_for_name(code)
                if not any(v in cols for v in variants):
                    all_ok = False
                    break

            if not all_ok:
                continue

            LOGGER.info("[Adapter|Reuse] Verwende früheren Run: %s", xlsx)

            try:
                fin_dir = base_dir / "loader" / "financial_cache"
                fin_dir.mkdir(parents=True, exist_ok=True)
                fin_out = fin_dir / "output.xlsx"
                shutil.copy2(xlsx, fin_out)
                LOGGER.info("[Adapter|Reuse] Nach financial_cache gespiegelt: %s", fin_out)
            except Exception as exc:
                LOGGER.warning("[Adapter|Reuse] Konnte Datei nicht in financial_cache spiegeln: %s", exc)

            return str(xlsx)

        LOGGER.info("[Adapter|Reuse] Kein passender früherer Run gefunden.")
        return None

    # ------------------------------------------------------------
    # NEU: Datei-Schreib-Layer (Schritt 5)
    # ------------------------------------------------------------
    def _write_artifact(
        self,
        *,
        kind: str,
        df: Optional[pd.DataFrame] = None,
        src_path: Optional[str] = None,
        base_dir: Union[str, Path, None] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Zentraler Writer für Artefakte.
        kind:
          - "snapshot_parquet": df -> <base_dir>/<filename>.parquet
          - "dash_final_dataset": src_path -> <base_dir>/<filename>.xlsx (copy)
        """
        from pathlib import Path
        import shutil

        if base_dir is None:
            base_dir = tempfile.gettempdir()
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # generischer Name, falls nichts angegeben wurde
            filename = f"{kind}.bin"

        target_path = base_dir / filename

        try:
            if kind == "snapshot_parquet":
                if df is None:
                    LOGGER.warning("[Writer] snapshot_parquet ohne DataFrame aufgerufen – übersprungen.")
                    return None
                df.to_parquet(target_path)
                LOGGER.info(f"[Writer] Snapshot geschrieben: {target_path}")
                return str(target_path)

            elif kind == "dash_final_dataset":
                if not src_path or not os.path.exists(src_path):
                    LOGGER.warning(f"[Writer] dash_final_dataset: Quelle existiert nicht ({src_path}) – übersprungen.")
                    return None
                shutil.copy2(src_path, target_path)
                LOGGER.info(f"[Writer] Dash-Excel persistiert: {target_path}")
                return str(target_path)

            else:
                LOGGER.warning(f"[Writer] Unbekannter kind='{kind}' – nichts geschrieben.")
                return None
        except Exception as e:
            LOGGER.warning(f"[Writer] Schreiben fehlgeschlagen ({kind}): {e}")
            return None

    # ------------------------------------------------------------
    # NEU: Dashboard-Payload bauen (Schritt 6)
    # ------------------------------------------------------------
    def _step_6_build_dashboard_result(self, forecast_df: pd.DataFrame, metadata: dict) -> dict:
        """
        Schritt 6: Alles, was Dashboard typischerweise braucht, in ein einziges Paket legen.
        - Forecast als Records
        - Spaltennamen
        - Pipeline-/Adapter-Infos
        - Verknüpfung zu Artefakten aus Schritt 4
        """
        LOGGER.info(_sym("[Adapter] ===== STEP 6: Dashboard-Payload bauen ====="))

        if forecast_df is None or forecast_df.empty:
            LOGGER.warning("[Step6] Forecast-DataFrame ist leer – Dashboard-Payload wird minimal.")
            forecast_records = []
            columns = []
        else:
            forecast_records = forecast_df.to_dict(orient="records")
            columns = list(forecast_df.columns)

        pipeline_info = getattr(self, "pipeline_info", {}) or {}
        dash_export = (metadata or {}).get("dash_export") if isinstance(metadata, dict) else None
        model_path = (metadata or {}).get("model_path") if isinstance(metadata, dict) else None

        # kleine Summary
        forecast_summary = {
            "n_rows": len(forecast_records),
            "n_columns": len(columns),
            "has_intervals": any(col.startswith("yhat_") for col in columns),
        }

        dashboard_result = {
            "forecast": forecast_records,
            "columns": columns,
            "forecast_summary": forecast_summary,
            "pipeline_info": pipeline_info,
            "dash_export": dash_export,
            "model_path": model_path,
        }

        LOGGER.info(
            f"[Step6] Dashboard-Payload: rows={forecast_summary['n_rows']}, "
            f"cols={forecast_summary['n_columns']}, has_intervals={forecast_summary['has_intervals']}"
        )

        return dashboard_result

    # ------------------------------------------------------------
    # NEU: CI-Coverage berechnen (für Qualitäts-Monitoring)
    # ------------------------------------------------------------
    @staticmethod
    def calculate_ci_coverage(
        actual_values: np.ndarray,
        predictions: np.ndarray,
        lower_bounds: Dict[int, np.ndarray],
        upper_bounds: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        coverage = {}

        for level in lower_bounds.keys():
            if level not in upper_bounds:
                continue

            lower = lower_bounds[level]
            upper = upper_bounds[level]

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
            from src.forecaster.forecaster_pipeline import list_saved_models  # type: ignore

            return list_saved_models()
        except Exception:
            return []



    # ------------------------------------------------------------
    # Reload model
    # ------------------------------------------------------------
    
    @staticmethod
    def _find_artifact_in_run_dir(run_dir: str | Path, meta: dict | None = None) -> str | None:
        p = Path(run_dir) if run_dir else None
        if not p or not p.exists():
            return None
        # 1) Pfad aus Meta respektieren
        if isinstance(meta, dict):
            for key in ("model_path", "artifact_path"):
                cand = meta.get(key)
                if cand and Path(cand).exists():
                    return str(Path(cand).resolve())
        # 2) Sonst jüngstes *.pkl im Ordner nehmen
        pkls = list(p.glob("*.pkl"))
        if pkls:
            pkls.sort(key=lambda q: q.stat().st_mtime, reverse=True)
            return str(pkls[0].resolve())
        return None

    # ---------- NEU: Excel-Target -> UI-Target remappen ----------
    @staticmethod
    def _excel_col_to_ui_target(excel_col: str, sektor: str | None = "PH") -> str:
        """
        Macht aus 'PH_WERTPAPIERE' wieder 'Wertpapiere' (UI-Wert aus deinem Dropdown).
        """
        if not excel_col:
            return "Wertpapiere"  # Fallback
        s = str(excel_col).upper().strip()
        # Sektor-Präfix weg
        if sektor:
            pref = str(sektor).upper().strip() + "_"
            if s.startswith(pref):
                s = s[len(pref):]
        # Fixe Map zurück auf UI-Werte
        rev = {
            "EINLAGEN": "Einlagen",
            "WERTPAPIERE": "Wertpapiere",
            "VERSICHERUNGEN": "Versicherungen",
            "KREDITE": "Kredite",
            "GESAMT": "gesamt",
            "GVB": "gesamt",  # falls du GVB als Gesamt verwendest
        }
        if s in rev:
            return rev[s]
        # grober Fallback: underscores raus und case normalisieren
        return s.replace("_", " ").title()

    # ---------- NEU: Run-Payload -> Args für run_forecast ----------
    def build_args_from_selected_run(self, selected_run: dict, *, ui_defaults: dict | None = None) -> dict:
        """
        selected_run: Payload aus dem Modal/Store (enthält id/path/meta/meta_raw etc.).
        ui_defaults:  optionale Fallbacks aus der aktuellen UI (target/horizon/sektor/flows).
        Gibt kwargs zurück, die du 1:1 an run_forecast(...) übergeben kannst.
        """
        defaults = ui_defaults or {}
        meta = (
            selected_run.get("meta_raw")
            or selected_run.get("meta")
            or selected_run.get("meta_dict")
            or {}
        )

        sektor = meta.get("sector") or defaults.get("sektor") or "PH"
        excel_target = meta.get("target_col") or meta.get("excel_target_col")
        ui_target = self._excel_col_to_ui_target(excel_target, sektor=sektor) if excel_target else (defaults.get("target") or "Wertpapiere")

        horizon = int(meta.get("forecast_horizon") or defaults.get("horizon") or 4)
        exogs = list(meta.get("selected_exog") or [])
        use_flows = str(meta.get("ui_mode", defaults.get("ui_mode", "bestand"))).lower() == "fluss"

        run_dir = selected_run.get("path") or selected_run.get("run_loader_dir") or ""
        preload = self._find_artifact_in_run_dir(run_dir, meta)

        return dict(
            target=ui_target,
            selected_exog=exogs,
            horizon=horizon,
            use_cached=True,           # beim Replay sinnvoll
            force_retrain=False,       # wir versuchen zuerst das Artefakt/Caching
            use_flows=use_flows,
            preload_model_path=preload,  # kann None sein; dein Step 2 kann damit umgehen
            sektor=sektor,
        )

    

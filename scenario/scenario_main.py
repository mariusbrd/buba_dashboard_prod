# === Path Helpers: immer unter scenario/data arbeiten ===
from pathlib import Path

# Ordner dieses Moduls (…/scenario)
SCENARIO_DIR = Path(__file__).resolve().parent

# Projekt-Root (ein Ordner über scenario), analog zu den anderen Modulen
try:
    APP_ROOT: Path = SCENARIO_DIR.parent
except Exception:
    APP_ROOT = Path.cwd()

# Datenverzeichnis für Scenario-Module
SCENARIO_DATA_DIR = SCENARIO_DIR / "data"
SCENARIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

def scenario_data_path(*parts: str) -> Path:
    return SCENARIO_DATA_DIR.joinpath(*parts)

# unsere drei "offiziellen" Dateien im Unterordner
DEFAULT_OUTPUT_XLSX    = scenario_data_path("output.xlsx")
DEFAULT_ANALYSIS_XLSX  = scenario_data_path("analysis_data.xlsx")
DEFAULT_OVERRIDES_XLSX = scenario_data_path("scenario_overrides.xlsx")



# Robust gegen zirkuläre Importe: versuche Foundation-Helper aus app.py zu importieren,
# sonst fallback-only.
# Robust gegen zirkuläre Importe: versuche Foundation-Helper aus app.py zu importieren,
# sonst fallback-only.
try:
    from app import get_quarter_end_date  # nur diese eine Helper-Funktion
except Exception:
    def get_quarter_end_date(year: int, quarter: int):
        import pandas as pd
        QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
        QUARTER_END_DAY   = {3: 31, 6: 30, 9: 30, 12: 31}
        m_end = QUARTER_END_MONTH[int(quarter)]
        d_end = QUARTER_END_DAY[m_end]
        return pd.Timestamp(int(year), m_end, d_end)

# ➜ NEU: parse_german_number hier verfügbar machen (Import + Fallback)
try:
    from app import parse_german_number  # robuster Parser aus der Foundation
except Exception:
    import re, math
    _NUM_RE = re.compile(r'^[\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?$')
    def parse_german_number(value):
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return None if (isinstance(value, float) and math.isnan(value)) else float(value)
        s = str(value).strip()
        if s == "": return None
        has_percent = "%" in s
        s = re.sub(r"[^0-9,\.\+\-eE%]", "", s)
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
        s = s.replace("%", "").strip()
        if not _NUM_RE.match(s):
            s2 = re.sub(r"[^0-9\.\+\-eE]", "", s)
            if not _NUM_RE.match(s2): return None
            s = s2
        try:
            num = float(s)
        except Exception:
            return None
        return num/100.0 if has_percent else num

# ➜ NEU: DataFrame-Store-Loader (Import + Fallback)
try:
    from app import load_dataframe_from_store  # bevorzugt Foundation-Implementierung
except Exception:
    import json
    import pandas as pd

    def load_dataframe_from_store(
        payload,
        fallback=None,
        *,
        date_col=None,
        coerce_dates: bool = True,
        sort_by_date: bool = True,
    ) -> pd.DataFrame:
        """Minimaler Fallback: JSON-String, {'columns','data'}, dict -> einzeilig, Liste von dicts."""
        def _empty():
            return fallback.copy() if isinstance(fallback, pd.DataFrame) else pd.DataFrame()

        if payload is None:
            return _empty()

        # 1) JSON-String (orient='split' oder generisch)
        if isinstance(payload, str):
            try:
                return pd.read_json(payload, orient="split")
            except Exception:
                try:
                    obj = json.loads(payload)
                except Exception:
                    return _empty()
                payload = obj  # weiter unten normal behandeln

        # 2) Dict mit DataTable-Shape
        if isinstance(payload, dict):
            if {"columns", "data"} <= set(payload.keys()):
                try:
                    cols = payload.get("columns", [])
                    cols = [c["id"] if isinstance(c, dict) and "id" in c else c for c in cols]
                    df = pd.DataFrame(payload.get("data", []), columns=cols if cols else None)
                except Exception:
                    return _empty()
            else:
                try:
                    df = pd.DataFrame([payload]) if payload else _empty()
                except Exception:
                    return _empty()

        # 3) Liste (typisch: Liste von Dicts)
        elif isinstance(payload, list):
            try:
                df = pd.DataFrame(payload)
            except Exception:
                return _empty()
        else:
            return _empty()

        # optional Datumsspalte normalisieren
        if date_col and coerce_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            if sort_by_date:
                df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        return df



# --- Callback-Proxy: sammelt @app.callback(...) Deklarationen für spätere Registrierung
import dash

class _CallbackProxy:
    """Sammelt @app.callback(...) Deklarationen; wird später auf die echte Dash-App gemappt."""
    def __init__(self):
        self._registrations = []  # [(args, kwargs, fn), ...]

    def callback(self, *args, **kwargs):
        def _decorator(func):
            self._registrations.append((args, kwargs, func))
            return func
        return _decorator

# WICHTIG: 'app' ist hier absichtlich KEINE echte Dash-App, sondern der Proxy!
app = _CallbackProxy()

# --- Safe logging shim (verhindert NameError für Log/logger) ---
import logging
logger = logging.getLogger("GVB_Dashboard")

class _SafeLog:
    """Fallback-Logger mit sauberer Level-Zuordnung.

    - debug:    Detail-Logs und Tabellen (scenario_table)
    - info:     wichtige High-Level-Ereignisse (scenario)
    - warning:  potentielle Probleme
    - error:    echte Fehler
    """
    def debug(self, msg): logger.debug(msg)
    def info(self, msg): logger.info(msg)
    def warn(self, msg): logger.warning(msg)
    def warning(self, msg): logger.warning(msg)
    def error(self, msg): logger.error(msg)
    def scenario(self, msg): logger.info(msg)
    def scenario_table(self, msg): logger.debug(msg)

# Falls 'Log' bereits von app.py übergeben/gesetzt wird, nutzen; sonst fallback:
Log = globals().get("Log", _SafeLog())
# =================== Szenario-Analyse: Funktionen & Callbacks ===================

#from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc



from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
import traceback


# Pfade & erwartete Spalten
CANON_EXOG = ["lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"]

# hier NICHT nochmal /scenario/data anhängen – oben existiert SCENARIO_DATA_DIR schon
DEFAULT_ANALYSIS_PATH = SCENARIO_DATA_DIR / "analysis_data.xlsx"

from pathlib import Path
from typing import Optional, Dict, Any, List


def ensure_analysis_data_on_startup(
    base_candidates: Optional[List[Path]] = None,
    analysis_path: Optional[Path] = None,
) -> Path:
    """
    Stellt sicher, dass scenario/data/analysis_data.xlsx bereits zum App-Start existiert.

    Strategie:
      1) 'output.xlsx' finden (Kandidaten siehe unten).
      2) Falls gefunden: analysis_data.xlsx (nur Historie) via _create_analysis_data_from_scenarios() bauen.
      3) Falls nicht gefunden: Minimal-Workbook mit gültiger Struktur schreiben.
    """
    import pandas as pd
    import numpy as np

    app_dir = Path(__file__).parent             # …/scenario
    scenario_dir = app_dir / "data"             # ✓ richtig: …/scenario/data
    scenario_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_path or (scenario_dir / "analysis_data.xlsx")

    # ----- YAML-Output-Pfad auslesen (optional) -----
    def _discover_output_from_yaml() -> Optional[Path]:
        try:
            import yaml
        except Exception:
            return None
        for yml in [
            app_dir / "config.yaml",
            app_dir / "config.yml",
            app_dir.parent / "scenario" / "config.yaml",  # falls jemand in …/<root>/scenario/config.yaml speichert
            app_dir.parent / "scenario" / "config.yml",
            Path("scenario/config.yaml"),
            Path("scenario/config.yml"),
        ]:
            try:
                if yml.exists():
                    y = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
                    out = y.get("output_path")
                    if out:
                        p = Path(out)
                        if not p.is_absolute():
                            p = yml.parent / p
                        return p
            except Exception as e:
                logger.warning(f"[Startup] WARN: YAML lesen fehlgeschlagen ({yml}): {e}")
        return None

    # ----- Kandidaten für output.xlsx -----
    yaml_output = _discover_output_from_yaml()
    default_candidates = [
        app_dir / "data" / "transformed_output.xlsx",
        app_dir / "data" / "output.xlsx",
        app_dir / "output.xlsx",
        app_dir.parent / "data" / "output.xlsx",  # optional: eine Ebene höher
        Path("/mnt/data/output.xlsx"),
    ]
    candidates: List[Path] = []
    if base_candidates:
        candidates.extend(base_candidates)
    if yaml_output:
        candidates.append(yaml_output)
    candidates.extend(default_candidates)

    base_excel = next((Path(p) for p in candidates if p and Path(p).exists()), None)

    # Entscheiden, ob rebuild nötig ist
    need_rebuild = False
    if base_excel and base_excel.exists():
        if not analysis_path.exists():
            need_rebuild = True
        else:
            try:
                need_rebuild = base_excel.stat().st_mtime > analysis_path.stat().st_mtime
            except Exception:
                need_rebuild = True
    else:
        if not analysis_path.exists():
            need_rebuild = True

    # Fall A: Basisdatei vorhanden → (re)build über _create_analysis_data_from_scenarios
    if base_excel and base_excel.exists():
        try:
            created = _create_analysis_data_from_scenarios(
                base_excel_path=base_excel,
                manual_vals={},      # keine UI-Overrides beim Startup
                quarter_labels=[],   # nur Historie
                output_path=analysis_path,
            )
            logger.info(f"[Startup] ✓ analysis_data.xlsx erstellt/aktualisiert: {created.resolve()}")
            return created
        except Exception as e:
            logger.warning(f"[Startup] WARN: Erstellung aus {base_excel.name} fehlgeschlagen: {e}")
            # Fallback unten greift

    # Fall B: kein output.xlsx → Minimaldatei (gültige Struktur, keine Werte)
    if need_rebuild:
        cols = [
            "Datum",
            "Einlagen", "Wertpapiere", "Versicherungen", "Kredite", "Gesamt GVB",
            "lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation",
        ]
        df_empty = pd.DataFrame(columns=cols)
        df_empty["Datum"] = pd.to_datetime(df_empty["Datum"])
        df_empty.to_excel(analysis_path, sheet_name="final_dataset", index=False)
        logger.info(f"[Startup] ✓ Minimal- analysis_data.xlsx erzeugt: {analysis_path.resolve()}")
    else:
        logger.debug(f"[Startup] ✓ analysis_data.xlsx ist bereits aktuell: {analysis_path.resolve()}")

    return analysis_path









# --- Analyzer robust importieren ----------------------------------------------
try:
    from scenario_analyzer import Config, ScenarioAnalysis
except Exception:
    import importlib.util, sys
    _base = Path(__file__).parent
    _cands = [
        _base / "scenario" / "scenario_analyzer.py",
        _base / "scenario_analyzer.py",
        Path("/mnt/data/scenario_analyzer.py"),
    ]
    _loaded = False
    for _p in _cands:
        if _p.exists():
            spec = importlib.util.spec_from_file_location("scenario_analyzer", str(_p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            Config, ScenarioAnalysis = mod.Config, mod.ScenarioAnalysis
            _loaded = True
            break
    if not _loaded:
        raise





# ========== CALLBACK 1: Initial-Ansicht (Historie + Tabelle) ==========
@app.callback(
    Output("scenario-comparison-chart", "figure"),
    [
        Input("url", "pathname"),
        Input("scenario-target-dropdown", "value"),
        Input("reset-exog-overrides-btn", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def show_initial_historical_view(pathname, target_ui, reset_clicks):
    """
    Zeigt nur die historischen Daten ohne Forecast.
    Wird beim Seitenwechsel oder Target-Änderung aktualisiert.
    """
    if pathname not in {"/scenario", "/scenario-analysis"}:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Target mapping
        target_map = {
            "gesamt": "Gesamt GVB",
            "Einlagen": "Einlagen",
            "Wertpapiere": "Wertpapiere",
            "Versicherungen": "Versicherungen",
            "Kredite": "Kredite",
        }
        target_col = target_map.get(target_ui, "Gesamt GVB")
        
        # Analysis-Excel laden
        analysis_excel = _find_analysis_or_scenario_excel()
        if not analysis_excel.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {analysis_excel}")
        
        df = pd.read_excel(analysis_excel, sheet_name="final_dataset", engine="openpyxl")
        df["Datum"] = pd.to_datetime(df["Datum"])
        df = df.sort_values("Datum").reset_index(drop=True)
        
        # Nur Zeilen mit vorhandenen Target-Werten (keine NaN)
        df_hist = df[df[target_col].notna()].copy()
        
        if df_hist.empty:
            raise ValueError(f"Keine historischen Daten für {target_col}")
        
        # Einfacher Plot: nur Historie
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["Datum"],
            y=df_hist[target_col],
            mode="lines",
            name="Historische Daten",
            line=dict(width=2, color="#636EFA")
        ))
        
        fig.update_layout(
            title=f"Historischer Verlauf: {target_col}",
            template="plotly_white",
            xaxis_title=None,
            yaxis_title=f"{target_col} (Mrd. EUR)",
            hovermode="x unified",
            height=560,
            annotations=[{
                'text': 'Klicken Sie auf "Analyse starten" für Nowcasting-Prognose',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.95,
                'showarrow': False,
                'font': {'size': 12, 'color': '#6c757d'},
                'bgcolor': 'rgba(255,255,255,0.8)',
                'borderpad': 10
            }]
        )
        
        return fig
        
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="[ScenarioInit] Fehler beim Laden der historischen Daten",
            template="plotly_white",
            height=560,
            annotations=[{
                'text': f'Fehler: {str(e)}',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5,
                'showarrow': False,
                'font': {'color': 'red'}
            }]
        )
        return empty_fig


def _validate_scenario_inputs(manual_vals: Dict[str, List[Optional[float]]], 
                              H: int = 4) -> Tuple[bool, List[str]]:
    """
    Validiert User-Eingaben auf Plausibilität.
    
    Returns:
        (is_valid, warnings) - True wenn mindestens eine gültige Eingabe vorhanden
    """
    warnings = []
    total_values = 0
    
    for var_key, vals in manual_vals.items():
        non_none = [v for v in vals if v is not None]
        total_values += len(non_none)
        
        if not non_none:
            continue
        
        # Plausibilitätschecks
        if var_key == "lt_interest_rate":
            if any(v < -5 or v > 20 for v in non_none):
                warnings.append(f"Zinssatz außerhalb plausibler Range (-5% bis 20%)")
        
        elif var_key == "inflation":
            if any(v < -10 or v > 30 for v in non_none):
                warnings.append(f"Inflation außerhalb plausibler Range (-10% bis 30%)")
        
        elif var_key == "unemployment":
            if any(v < 0 or v > 30 for v in non_none):
                warnings.append(f"Arbeitslosenquote außerhalb plausibler Range (0% bis 30%)")
        
        elif var_key == "gdp":
            if any(abs(v) > 1000 for v in non_none):
                warnings.append(f"BIP-Werte erscheinen unplausibel (|Wert| > 1000)")
        
        elif var_key == "property_prices":
            if any(abs(v) > 500 for v in non_none):
                warnings.append(f"Immobilienpreis-Index außerhalb plausibler Range")
    
    if total_values == 0:
        warnings.append("KRITISCH: Keine gültigen User-Eingaben gefunden!")
        return False, warnings
    
    return True, warnings




@app.callback(
    [
        Output("scenario-comparison-chart", "figure", allow_duplicate=True),
        Output("scenario-kpi-cards", "children"),
        Output("driver-analysis-chart", "figure"),
    ],
    Input("run-scenario-analysis-btn", "n_clicks"),
    [
        State("scenario-target-dropdown", "value"),
        State("exog-override-table", "columns"),
        State("exog-override-table", "data"),
        State("scenario-preset-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def run_scenario_forecast_with_analyzer(n_clicks, target_ui, tbl_cols, tbl_rows, scenario_preset):
    """
    Vollständige Decision-Tree-Analyse mit robustem Δ-Fallback und Treiber-Analyse.

    Neu (Option A):
    - Für JEDE konkrete Szenario-Kombination (Target + Overrides + Quartale) wird ein
      stabiler Fingerabdruck berechnet (MD5, gekürzt).
    - Daraus entsteht eine szenario-spezifische Datei:
        scenario/data/analysis_data_<hash>.xlsx
    - Existiert diese Datei schon, wird sie wiederverwendet → Modell-Cache greift → identische Ergebnisse.
    - Existiert sie noch nicht, wird sie aus einer sauberen Basis neu gebaut.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    try:
        from datetime import datetime
        import json, hashlib, math

        Log.debug("=" * 80)
        Log.info("[ScenarioForecast] START - Nowcasting mit Decision Tree")
        Log.scenario_table(f"[ScenarioForecast] Button-Klick #{n_clicks}")

        # ---------- 0) Preset normalisieren ----------
        preset_norm = (scenario_preset or "").strip().lower()
        baseline_preset = preset_norm in {"baseline", "basis"}

        # ---------- 1) User-Overrides parsen ----------
        H = 4  # Forecast-Horizont in Quartalen
        manual_vals, quarter_labels = _parse_exog_override_table(tbl_cols, tbl_rows, H=H)
        Log.scenario_table("[ScenarioForecast] Geparste User-Eingaben:")
        for k, vals in manual_vals.items():
            non_none = [v for v in vals if v is not None]
            if non_none:
                Log.scenario_table(f"  {k}: {non_none}")
        Log.scenario_table(f"  Quarter-Labels (UI): {quarter_labels}")

        # ---------- 2) Basis-Datei finden ----------
        base_excel = _find_analysis_or_scenario_excel()
        Log.scenario_table(f"[ScenarioForecast] Basis-Excel (Finder): {base_excel}")
        if not base_excel.exists():
            raise FileNotFoundError(
                "Basis-Datei nicht gefunden!\n"
                f"Erwartet: {base_excel}\n\n"
                "Bitte zuerst Szenario-Daten laden (Downloader) oder Script ausführen."
            )
        Log.scenario_table(f"[ScenarioForecast] Basis gefunden: {base_excel.stat().st_size/1024:.1f} KB")

        # ---------- 2a) möglichst 'saubere' Basis ableiten ----------
        # Wenn der Finder uns bereits eine analysis_data... zurückgibt, versuchen wir,
        # stattdessen die ursprüngliche output.xlsx im selben Ordner zu nehmen.
        clean_base_excel = base_excel
        if base_excel.name.startswith("analysis_data"):
            candidate = base_excel.parent / "output.xlsx"
            if candidate.exists():
                clean_base_excel = candidate
                Log.scenario_table("[ScenarioForecast] Überschreibe Basis mit output.xlsx (saubere Basis).")
            else:
                Log.scenario_table("[ScenarioForecast] Keine output.xlsx gefunden – nutze analysis_data als Basis.")

        # ---------- 2b) Szenario-Fingerprint bauen ----------
        import json, hashlib, math  # lokal importiert, damit der Block für sich funktioniert

        def _clean_val(x):
            if x is None:
                return None
            if isinstance(x, float):
                if math.isnan(x):
                    return None
                return float(x)
            return x

        payload = {
            "target": target_ui or "",
            "quarters": quarter_labels or [],
            "manual_vals": {k: [_clean_val(v) for v in vals] for k, vals in manual_vals.items()},
        }
        payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        scenario_hash = hashlib.md5(payload_str.encode("utf-8")).hexdigest()[:10]
        Log.scenario_table(f"[ScenarioForecast] Szenario-Fingerprint: {scenario_hash}")

        # ---------- 3) Szenario-spezifische analysis_data erstellen/verwenden ----------
        analysis_dir = SCENARIO_DATA_DIR
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_excel = analysis_dir / f"analysis_data_{scenario_hash}.xlsx"

        if analysis_excel.exists():
            # exakt dieses Szenario gab es schon → wiederverwenden
            created_path = analysis_excel
            Log.scenario_table("[ScenarioForecast] Verwende bestehende Szenario-Datei (kein Rebuild).")
        else:
            # neues Szenario → aus sauberer Basis aufbauen
            Log.scenario_table(f"[ScenarioForecast] Erstelle neue Szenario-Datei: {analysis_excel.name}")
            Log.scenario_table(f"  Basis: {clean_base_excel.name}")
            Log.scenario_table(f"  Quartale (UI): {quarter_labels}")
            Log.scenario_table(f"  Variablen: {list(manual_vals.keys())}")

            created_path = _create_analysis_data_from_scenarios(
                base_excel_path=clean_base_excel,
                manual_vals={},  # <-- leer lassen
                quarter_labels=quarter_labels,
                output_path=analysis_excel,
            )

        if not created_path.exists():
            raise RuntimeError("Szenario-Datei wurde nicht erstellt / gefunden.")
        Log.scenario_table(
            f"[ScenarioForecast] ✓ Datei bereit: {created_path.name} ({created_path.stat().st_size/1024:.1f} KB)"
        )

        # schnelle Validierung
        _ = pd.read_excel(created_path, sheet_name="final_dataset", nrows=5)
        Log.scenario_table("[ScenarioForecast] ✓ Validierung: final_dataset lesbar")


        # ---------- 4) Datei laden & Target ----------
        Log.scenario_table(f"[ScenarioForecast] Lade {created_path.name}…")
        df_full = pd.read_excel(created_path, sheet_name="final_dataset")
        df_full["Datum"] = pd.to_datetime(df_full["Datum"])
        df_full = df_full.sort_values("Datum").reset_index(drop=True)
        Log.scenario_table(f"[ScenarioForecast] Geladen: {len(df_full)} Zeilen")
        Log.scenario_table(f"  Zeitraum: {df_full['Datum'].min().date()} - {df_full['Datum'].max().date()}")
        Log.scenario_table(f"  Spalten: {list(df_full.columns)}")

        required_cols = ["Datum", "lt_interest_rate", "property_prices", "gdp", "unemployment", "inflation"]
        missing = [c for c in required_cols if c not in df_full.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten in Szenario-Datei: {missing}")

        gvb_cols = ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite", "Gesamt GVB"]
        Log.scenario_table(f"  GVB-Komponenten: {[c for c in gvb_cols if c in df_full.columns]}")

        target_map = {
            "gesamt": "Gesamt GVB",
            "Einlagen": "Einlagen",
            "Wertpapiere": "Wertpapiere",
            "Versicherungen": "Versicherungen",
            "Kredite": "Kredite",
        }
        target_col = target_map.get(target_ui, "Gesamt GVB")
        Log.scenario_table(f"[ScenarioForecast] Target: {target_col}")
        if target_col not in df_full.columns:
            raise ValueError(f"Target '{target_col}' nicht in Szenario-Datei gefunden!")

        # ---------- 5) Analyzer konfigurieren ----------
        cfg = Config(
            excel_path=str(created_path),
            sheet_name="final_dataset",
            date_col="Datum",
            target_col=target_col,
            agg_methods_exog=["last"],
            agg_method_target="last",
            exog_month_lags=[-12, -6, -3, -1],
            target_lags_q=[1, 2, 4],
            add_trend_features=False,
            trend_degree=1,
            add_seasonality=False,
            seasonality_mode="dummies",
            forecast_horizon_quarters=H,
            use_cached_model=True,                 # bleibt AN, damit wir pro Szenario wiederverwenden können
            model_dir="scenario/models_scenario",
            output_dir="scenario/outputs",
            future_exog_strategy="mixed",
            future_exog_drift_window_q=8,
            future_exog_seasonal_period_q=4,
        )
        sa = ScenarioAnalysis(cfg)
        _ = sa.load_and_prepare()

        # ---------- 6) Szenario-Beginn (letztes gültiges Target) ----------
        last_target_ts = pd.to_datetime(df_full.loc[df_full[target_col].notna(), "Datum"].max())
        if pd.isna(last_target_ts):
            raise ValueError("Keine historischen Target-Werte gefunden (alle NaN).")
        last_hist_qend = (last_target_ts + pd.offsets.QuarterEnd(0))
        first_scenario_qend = last_hist_qend + pd.offsets.QuarterEnd(1)
        Log.scenario_table(
            f"[ScenarioForecast] Szenario-Beginn (letztes gültiges Target-Q_end={last_hist_qend.date()}): "
            f"{first_scenario_qend.date()}"
        )

        # Historie/Szenario-Teilung fürs Logging
        df_hist = df_full[df_full["Datum"] < first_scenario_qend].copy()
        df_scn = df_full[df_full["Datum"] >= first_scenario_qend].copy()
        Log.scenario_table(f"[ScenarioForecast] Split @ {first_scenario_qend.date()}")
        Log.scenario_table(f"  Historie: {len(df_hist)} rows (bis {df_hist['Datum'].max().date() if len(df_hist) else 'n/a'})")
        Log.scenario_table(f"  Szenario: {len(df_scn)} rows")
        if len(df_hist) < 20:
            raise ValueError(f"Zu wenig Historie: {len(df_hist)} Zeilen (min. 20)")

        # ---------- 6b) Analyzer-Frames defensiv trimmen ----------
        if hasattr(sa, "_df_q") and isinstance(sa._df_q, pd.DataFrame) and "Q_end" in sa._df_q.columns:
            mask_q = sa._df_q["Q_end"] <= last_hist_qend
            if mask_q.any():
                sa._df_q = sa._df_q[mask_q].copy()
                Log.scenario_table(f"[ScenarioForecast] Nach Trim df_q: {len(sa._df_q)} rows")
            else:
                Log.scenario_table("[ScenarioForecast] Kein df_q vor Szenario-Grenze → lasse df_q ungetrimmt.")

        if hasattr(sa, "_df_det") and isinstance(sa._df_det, pd.DataFrame) and "Q_end" in sa._df_det.columns:
            mask_det = sa._df_det["Q_end"] <= last_hist_qend
            if mask_det.any():
                sa._df_det = sa._df_det[mask_det].copy()
                Log.scenario_table(f"[ScenarioForecast] Nach Trim df_det: {len(sa._df_det)} rows")
            else:
                Log.scenario_table("[ScenarioForecast] Kein df_det vor Szenario-Grenze → lasse df_det ungetrimmt.")

        if hasattr(sa, "_df_feats") and isinstance(sa._df_feats, pd.DataFrame) and "Q_end" in sa._df_feats.columns:
            mask_feats = sa._df_feats["Q_end"] <= last_hist_qend
            hist_feats = sa._df_feats[mask_feats].copy()
            if len(hist_feats) >= 12:
                sa._df_feats = hist_feats
                Log.scenario_table(f"[ScenarioForecast] Nach Trim df_feats: {len(sa._df_feats)} rows")
            else:
                Log.scenario_table(
                    f"[ScenarioForecast] Trim df_feats hätte {len(hist_feats)} rows ergeben (<12) – verwende ungetrimmte Features."
                )
        else:
            Log.warn("[ScenarioForecast] WARNUNG: Analyzer-Features ohne Q_end – überspringe Trim.")

        # ---------- 7) Baseline Forecast ----------
        Log.scenario_table("[ScenarioForecast] Baseline Forecast…")
        try:
            base_res = sa.forecast(H=H, scenario_future=None, persist=True)
        except TypeError:
            base_res = sa.forecast(H=H, scenario_future=None, persist=False)
        base_df = pd.DataFrame(base_res.get("table", []))
        if base_df.empty:
            raise ValueError("Baseline Forecast ist leer")
        Log.scenario_table(f"[ScenarioForecast] Baseline: {base_df['Forecast_Baseline'].values}")

        # ---------- 7b) FRÜHER EXIT bei 'baseline' ----------
        if baseline_preset:
            Log.scenario_table("[ScenarioForecast] preset='baseline' → übernehme Baseline 1:1; skippe Deltas/LeafLinear/Contribs.")

            def _quarter_ends_from(start_qend: pd.Timestamp, H: int) -> list:
                return [(start_qend + pd.offsets.QuarterEnd(i)).to_pydatetime() for i in range(1, H + 1)]

            y_hist_clean = sa._df_det[[sa.cfg.target_col, "Q_end"]].copy()
            y_hist_clean = y_hist_clean[
                y_hist_clean[sa.cfg.target_col].notna() & (y_hist_clean["Q_end"] <= last_hist_qend)
            ].sort_values("Q_end")
            if y_hist_clean.empty:
                raise ValueError("Keine gültigen historischen Target-Werte gefunden")

            hist_x = pd.to_datetime(y_hist_clean["Q_end"]).dt.to_pydatetime().tolist()
            hist_y = y_hist_clean[sa.cfg.target_col].astype(float).tolist()

            fut_dates_expected = _quarter_ends_from(last_hist_qend, H)
            base_y = base_df["Forecast_Baseline"].tolist()[:H]
            scn_y = base_y[:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, mode="lines", name="Historie",
                line=dict(width=2, color="#636EFA"),
                hovertemplate="<b>Historie</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
            ))
            if len(fut_dates_expected) == len(base_y):
                fig.add_trace(go.Scatter(
                    x=fut_dates_expected, y=base_y, mode="lines", name="Forecast (Baseline)",
                    line=dict(dash="dot", width=2),
                    hovertemplate="<b>Baseline</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=fut_dates_expected, y=scn_y, mode="lines", name="Forecast (Szenario)",
                    line=dict(dash="dot", width=3),
                    hovertemplate="<b>Szenario</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
                ))
            fig.update_layout(
                template="plotly_white",
                title=f"Forecast – {target_col}",
                xaxis_title="Datum",
                yaxis_title=f"{target_col} (Mrd. EUR)",
                legend_title="Serie",
                hovermode="x unified",
                height=560,
            )
            if hist_x:
                cutoff = pd.to_datetime(max(hist_x)).to_pydatetime()
                fig.add_shape(
                    type="line", xref="x", yref="paper",
                    x0=cutoff, x1=cutoff, y0=0, y1=1,
                    line=dict(width=1, dash="dash"), opacity=0.5,
                )

            baseline_end = float(base_y[-1]) if base_y else float("nan")
            scenario_end = float(scn_y[-1]) if scn_y else float("nan")
            abs_impact = 0.0
            rel_impact = 0.0
            risk_label, risk_cls = "Niedrig", "text-success"

            kpi_cards = html.Div([
                dbc.Card(dbc.CardBody([
                    html.H4(f"{abs_impact:+.2f}", className="text-primary"),
                    html.P("Abweichung", className="mb-0"),
                    html.Small("Mrd. EUR", className="text-muted")
                ]), className="text-center mb-2"),
                dbc.Card(dbc.CardBody([
                    html.H4(f"{rel_impact:+.2f}%", className="text-info"),
                    html.P("Relativ", className="mb-0"),
                    html.Small("vs. Baseline", className="text-muted")
                ]), className="text-center mb-2"),
                dbc.Card(dbc.CardBody([
                    html.H4(risk_label, className=risk_cls),
                    html.P("Risiko", className="mb-0")
                ]), className="text-center mb-2"),
                dbc.Card(dbc.CardBody([
                    html.H6("Modell", className="fw-bold mb-2"),
                    html.Small(f"R²: {sa.metadata.get('cv_performance', {}).get('cv_r2', 'n/a')}", className="d-block"),
                    html.Small("Direkte Modellreaktion", className="text-muted d-block")
                ]), className="text-center"),
            ])

            driver_fig = go.Figure()
            driver_fig.update_layout(
                title="Treiber-Analyse",
                template="plotly_white",
                height=300,
                annotations=[{
                    'text': 'Baseline-Szenario: keine Abweichung zur Baseline',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 12, 'color': '#6c757d'}
                }]
            )

            Log.scenario_table("[ScenarioForecast] LineChartSync: Szenario ≡ Baseline (strict).")
            Log.info("[ScenarioForecast] SUCCESS (Baseline-Shortcut)")
            return fig, kpi_cards, driver_fig

        # ---------- 8) Szenario-Deltas vorbereiten ----------
        Log.scenario_table("[ScenarioForecast] Berechne Szenario-Deltas…")
        scenario_future, fut_q = _build_future_deltas_for_analyzer(
            sa, manual_vals, H=H, force_hist_qend=last_hist_qend
        )
        Log.scenario_table(f"[ScenarioForecast] Deltas: {len(scenario_future)} Spalten")
        sf_cols = set(scenario_future.keys())
        x_cols = set(sa.X_cols or [])
        inter = sorted(sf_cols & x_cols)
        if not inter:
            Log.warn("[ScenarioForecast] WARNUNG: Keine Übereinstimmung zwischen scenario_future und X_cols!")
        else:
            Log.scenario_table(f"[ScenarioForecast] scenario_future ∩ X_cols = {inter}")

        # ---------- 9) Δ-Signalstärke prüfen ----------
        def _max_abs_in_scenario_future(sf_dict, H):
            max_abs = 0.0
            for v in (sf_dict or {}).values():
                if isinstance(v, dict):
                    for h in range(1, H + 1):
                        val = v.get(h, None)
                        if val is not None and np.isfinite(val):
                            max_abs = max(max_abs, abs(float(val)))
                elif isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                    for z in list(v)[:H]:
                        if z is not None and np.isfinite(z):
                            max_abs = max(max_abs, abs(float(z)))
                elif isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                    max_abs = max(max_abs, abs(float(v)))
            return max_abs

        max_abs_delta = _max_abs_in_scenario_future(scenario_future, H)
        if (scenario_future is None) or (max_abs_delta < 1e-6):
            Log.scenario_table(f"[ScenarioDeltas] |Δ|_max={max_abs_delta:.2e} < 1e-6 → ignoriere Deltas; Szenario = Baseline.")
            scn_df = base_df.copy()
            scn_res = base_res
        else:
            Log.scenario_table("[ScenarioForecast] Szenario Forecast…")
            try:
                scn_res = sa.forecast(H=H, scenario_future=(scenario_future or None), persist=True)
            except TypeError:
                scn_res = sa.forecast(H=H, scenario_future=(scenario_future or None), persist=False)
            scn_df = pd.DataFrame(scn_res.get("table", []))
            if scn_df.empty:
                raise ValueError("Szenario Forecast ist leer")

        Log.scenario_table(f"[ScenarioForecast] Szenario: {scn_df['Forecast_Scenario'].values}")

        diff = scn_df["Forecast_Scenario"].values - base_df["Forecast_Baseline"].values
        Log.scenario_table(f"[ScenarioForecast] Differenzen: {diff}")

        # ---------- 11) Sensitivity-Fallback ----------
        fallback_used = False
        if (np.nanmax(np.abs(diff)) < 1e-6):
            Log.warn("[ScenarioForecast] WARNUNG: Forecasts identisch! (prüfe Feature-Namen/Scenario-Future)")
            Log.scenario_table("[ScenarioForecast] Wende linearen Sensitivitäts-Fallback (z-Score) an…")

            sa_ref = (locals().get("analyzer")
                      or locals().get("sa")
                      or globals().get("analyzer")
                      or globals().get("sa"))
            if sa_ref is None:
                raise RuntimeError("ScenarioAnalysis-Instanz nicht gefunden (erwartet z.B. 'analyzer' oder 'sa').")
            sa = sa_ref

            X = sa._df_feats.loc[:, sa.X_cols].copy()

            def _resolve_target_series(sa):
                name_candidates = []
                for attr in ("y_col", "target_col", "target", "y_name"):
                    if hasattr(sa, attr):
                        val = getattr(sa, attr)
                        if isinstance(val, str) and val:
                            name_candidates.append(val)
                if hasattr(sa, "cfg"):
                    for key in ("target_col", "target", "y_col"):
                        v = getattr(sa.cfg, key, None)
                        if isinstance(v, str) and v:
                            name_candidates.append(v)
                name_candidates += ["Gesamt GVB", "target", "y"]
                for nm in name_candidates:
                    for df in [getattr(sa, "_df_feats", None),
                               getattr(sa, "_df_all", None),
                               getattr(sa, "df", None)]:
                        if isinstance(df, pd.DataFrame) and (nm in df.columns):
                            s = df[nm]
                            if pd.api.types.is_numeric_dtype(s):
                                return nm, s
                if isinstance(sa._df_feats, pd.DataFrame):
                    rest = [c for c in sa._df_feats.columns if c not in sa.X_cols]
                    rest = [c for c in rest if pd.api.types.is_numeric_dtype(sa._df_feats[c]) and not c.endswith("_id")]
                    if len(rest) == 1:
                        nm = rest[0]
                        return nm, sa._df_feats[nm]
                return None, None

            y_name, y_series = _resolve_target_series(sa)
            if y_series is None:
                Log.warn("[SensitivityFallback] WARNUNG: Target-Spalte nicht eindeutig – verwende Backup-Skalierung.")
                y_std = 1.0
            else:
                y_std_raw = np.nanstd(y_series.values)
                y_std = float(y_std_raw if np.isfinite(y_std_raw) and (y_std_raw > 0) else 1.0)

            X_std = X.std(ddof=0).replace(0, np.nan).fillna(1.0)

            try:
                beta_std = ScenarioAnalysis._estimate_local_betas(
                    sa._df_feats, sa.X_cols, sa.cfg,
                    window_q=min(24, max(12, len(sa._df_feats) // 3)),
                    standardize=True
                )
            except Exception:
                Xz = (X - X.mean()) / X_std
                yz = (y_series - y_series.mean()) / (y_std if y_std > 0 else 1.0) if y_series is not None else pd.Series(0.0, index=X.index)
                lam = 1e-3
                XtX = Xz.T @ Xz
                beta_std = pd.Series(
                    np.linalg.solve(XtX.values + lam * np.eye(XtX.shape[0]), Xz.T.values @ yz.values),
                    index=sa.X_cols
                )

            new_scn = []
            for h in range(1, H + 1):
                adj_h_raw = {}
                for k, v in (scenario_future or {}).items():
                    val = None
                    if isinstance(v, dict):
                        val = v.get(h, None)
                    elif isinstance(v, (list, tuple, np.ndarray, pd.Series)) and len(v) >= h:
                        val = v[h - 1]
                    elif isinstance(v, (int, float, np.floating)):
                        val = float(v)
                    if val is None:
                        continue
                    try:
                        adj_h_raw[k] = float(val)
                    except Exception:
                        pass

                adj_h_raw = {k: v for k, v in adj_h_raw.items() if k in sa.X_cols}
                if not adj_h_raw:
                    base_h = float(base_df["Forecast_Baseline"].iloc[h - 1])
                    new_scn.append(base_h)
                    continue

                adj_h_z = {}
                for k, v in adj_h_raw.items():
                    sigma = float(X_std.get(k, 1.0))
                    if sigma <= 0 or not np.isfinite(sigma):
                        sigma = 1.0
                    adj_h_z[k] = float(np.clip(v / sigma, -5.0, 5.0))

                b = beta_std.reindex(adj_h_z.keys()).fillna(0.0).values
                x = np.array([adj_h_z[k] for k in adj_h_z.keys()], dtype=float)
                delta_y_z = float(np.dot(b, x))
                delta_y_z = float(np.clip(delta_y_z, -3.0, 3.0))
                delta_y = delta_y_z * y_std

                base_h = float(base_df["Forecast_Baseline"].iloc[h - 1])
                new_scn.append(base_h + delta_y)

                parts = ", ".join([f"{k}:{adj_h_raw[k]:.4g} (z={adj_h_z[k]:.3g})" for k in adj_h_z.keys()])
                Log.scenario_table(f"[SensitivityFallback] h={h} | Δy_z={delta_y_z:.3f} → Δy={delta_y:.3f} | {parts}")

            scn_df["Forecast_Scenario"] = new_scn
            diff = scn_df["Forecast_Scenario"].values - base_df["Forecast_Baseline"].values
            Log.scenario_table(f"[ScenarioForecast] Fallback-Delta (stabilisiert): {np.round(diff, 3)}")
            fallback_used = True

        # ---------- 12) Visualisierung ----------
        def _quarter_ends_from(start_qend: pd.Timestamp, H: int) -> list:
            return [(start_qend + pd.offsets.QuarterEnd(i)).to_pydatetime() for i in range(1, H + 1)]

        y_hist_clean = sa._df_det[[sa.cfg.target_col, "Q_end"]].copy()
        y_hist_clean = y_hist_clean[
            y_hist_clean[sa.cfg.target_col].notna() & (y_hist_clean["Q_end"] <= last_hist_qend)
        ].sort_values("Q_end")
        if y_hist_clean.empty:
            raise ValueError("Keine gültigen historischen Target-Werte gefunden")

        hist_x = pd.to_datetime(y_hist_clean["Q_end"]).dt.to_pydatetime().tolist()
        hist_y = y_hist_clean[sa.cfg.target_col].astype(float).tolist()
        Log.scenario_table(f"[Chart] Historie bis {last_hist_qend.date()}, {len(hist_x)} Punkte")

        fut_dates_expected = _quarter_ends_from(last_hist_qend, H)
        base_y = base_df["Forecast_Baseline"].tolist()[:H]
        scn_y = scn_df["Forecast_Scenario"].tolist()[:H]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y, mode="lines", name="Historie",
            line=dict(width=2, color="#636EFA"),
            hovertemplate="<b>Historie</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
        ))
        if len(fut_dates_expected) == len(base_y):
            fig.add_trace(go.Scatter(
                x=fut_dates_expected, y=base_y, mode="lines", name="Forecast (Baseline)",
                line=dict(dash="dot", width=2),
                hovertemplate="<b>Baseline</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
            ))
        if len(fut_dates_expected) == len(scn_y):
            fig.add_trace(go.Scatter(
                x=fut_dates_expected, y=scn_y, mode="lines", name="Forecast (Szenario)",
                line=dict(dash="dot", width=3),
                hovertemplate="<b>Szenario</b><br>%{x|%Y-%m-%d}<br>%{y:.2f} Mrd. EUR<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_white",
            title=f"Forecast – {target_col}",
            xaxis_title="Datum",
            yaxis_title=f"{target_col} (Mrd. EUR)",
            legend_title="Serie",
            hovermode="x unified",
            height=560,
        )
        if hist_x:
            cutoff = pd.to_datetime(max(hist_x)).to_pydatetime()
            fig.add_shape(
                type="line",
                xref="x", yref="paper",
                x0=cutoff, x1=cutoff,
                y0=0, y1=1,
                line=dict(width=1, dash="dash"),
                opacity=0.5,
            )

        # ---------- 13) KPI-Cards ----------
        baseline_end = float(base_y[-1]) if base_y else float("nan")
        scenario_end = float(scn_y[-1]) if scn_y else float("nan")
        if not np.isfinite(baseline_end) or not np.isfinite(scenario_end):
            abs_impact, rel_impact = 0.0, 0.0
        else:
            abs_impact = scenario_end - baseline_end
            rel_impact = (abs_impact / baseline_end * 100.0) if baseline_end else 0.0

        if abs(rel_impact) >= 5:
            risk_label, risk_cls = "Hoch", "text-danger"
        elif abs(rel_impact) >= 2:
            risk_label, risk_cls = "Mittel", "text-warning"
        else:
            risk_label, risk_cls = "Niedrig", "text-success"

        kpi_cards = html.Div([
            dbc.Card(dbc.CardBody([
                html.H4(f"{abs_impact:+.2f}", className="text-primary"),
                html.P("Abweichung", className="mb-0"),
                html.Small("Mrd. EUR", className="text-muted")
            ]), className="text-center mb-2"),
            dbc.Card(dbc.CardBody([
                html.H4(f"{rel_impact:+.2f}%", className="text-info"),
                html.P("Relativ", className="mb-0"),
                html.Small("vs. Baseline", className="text-muted")
            ]), className="text-center mb-2"),
            dbc.Card(dbc.CardBody([
                html.H4(risk_label, className=risk_cls),
                html.P("Risiko", className="mb-0")
            ]), className="text-center mb-2"),
            dbc.Card(dbc.CardBody([
                html.H6("Modell", className="fw-bold mb-2"),
                html.Small(f"R²: {sa.metadata.get('cv_performance', {}).get('cv_r2', 'n/a')}", className="d-block"),
                html.Small("Fallback aktiv" if fallback_used else "Direkte Modellreaktion", className="text-muted d-block")
            ]), className="text-center"),
        ])

        # ---------- 14) Treiber-Analyse ----------
        var_labels = {
            "lt_interest_rate": "Zinssatz (10Y)",
            "property_prices": "Immobilienpreise",
            "gdp": "BIP",
            "unemployment": "Arbeitslosenquote",
            "inflation": "Inflation",
        }

        contrib_rows = []

        contrib_dict = None
        if isinstance(scn_res, dict):
            for k, v in scn_res.items():
                if isinstance(k, str) and "contrib" in k.lower() and isinstance(v, dict) and v:
                    contrib_dict = v
                    break
            if contrib_dict is None:
                for k in ("leaf_linear", "driver_contribs", "parts", "x_contribs"):
                    v = scn_res.get(k)
                    if isinstance(v, dict) and v:
                        contrib_dict = v
                        break

        if (not fallback_used) and contrib_dict:
            for raw_key, series_or_val in contrib_dict.items():
                for vk, nice in var_labels.items():
                    if raw_key.startswith(vk):
                        try:
                            if isinstance(series_or_val, (list, tuple, np.ndarray, pd.Series)):
                                c_val = float(series_or_val[-1])
                            else:
                                c_val = float(series_or_val)
                            contrib_rows.append({"Variable": nice, "Beitrag": c_val})
                        except Exception:
                            pass

        if (not contrib_rows) and (max_abs_delta >= 1e-6):
            Log.scenario_table("[ScenarioForecast] Berechne isolierte Beiträge (Fallback)…")
            for var_key, nice in var_labels.items():
                single = {}
                for suffix in ("__last", "__mean"):
                    col = f"{var_key}{suffix}"
                    if col in scenario_future:
                        single[col] = scenario_future[col]
                if not single:
                    continue
                try:
                    try:
                        single_res = sa.forecast(H=H, scenario_future=single, persist=True)
                    except TypeError:
                        single_res = sa.forecast(H=H, scenario_future=single, persist=False)
                    single_df = pd.DataFrame(single_res.get("table", []))
                    if single_df.empty:
                        continue
                    c = float(single_df["Forecast_Scenario"].iloc[-1] - single_df["Forecast_Baseline"].iloc[-1])
                    contrib_rows.append({"Variable": nice, "Beitrag": c})
                    Log.scenario_table(f"  {var_key}: Beitrag = {c:+.3f}")
                except Exception as e:
                    Log.scenario_table(f"  {var_key}: Fehler bei Berechnung: {e}")

        if contrib_rows:
            cdf = pd.DataFrame(contrib_rows).groupby("Variable", as_index=False)["Beitrag"].sum()
            cdf = cdf.sort_values("Beitrag", key=lambda s: s.abs(), ascending=True)
            driver_fig = go.Figure()
            driver_fig.add_trace(go.Bar(
                x=cdf['Beitrag'],
                y=cdf['Variable'],
                orientation='h',
                marker=dict(color=['#dc3545' if x < 0 else '#28a745' for x in cdf['Beitrag']]),
                text=[f"{x:+.2f}" for x in cdf['Beitrag']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Beitrag: %{x:+.2f} Mrd. EUR<extra></extra>'
            ))
            driver_fig.update_layout(
                title="Isolierte Beiträge je Variable (Mrd. EUR)",
                template="plotly_white",
                xaxis_title="Beitrag zur Abweichung",
                yaxis_title=None,
                height=300,
                showlegend=False,
                margin=dict(l=150, r=50, t=50, b=50)
            )
            driver_fig.add_shape(
                type="line", xref="x", yref="y",
                x0=min(0, float(cdf['Beitrag'].min() or 0)), x1=max(0, float(cdf['Beitrag'].max() or 0)),
                y0=0, y1=0,
                line=dict(color="gray", dash="dash", width=1)
            )
            Log.scenario_table(f"[ScenarioForecast] Treiber-Chart erstellt mit {len(cdf)} Variablen")
        else:
            driver_fig = go.Figure()
            driver_fig.update_layout(
                title="Treiber-Analyse",
                template="plotly_white",
                height=300,
                annotations=[{
                    'text': 'Keine signifikanten Beiträge (oder Δ<1e-6)',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 12, 'color': '#6c757d'}
                }]
            )
            Log.scenario_table("[ScenarioForecast] Keine Treiber-Beiträge verfügbar")

        Log.info("[ScenarioForecast] SUCCESS")
        return fig, kpi_cards, driver_fig

    except Exception as e:
        return _create_error_response(e, "Nowcasting-Analyse")




# ==============================
# Szenario-Presets (Q+1..Q+4)
# ==============================
# Regeln:
# - "add_pp": additive Änderungen in Prozentpunkten.
# - "mul_pct": prozentuale Änderung in % (multiplikativ auf das Niveau).
# Du kannst die Werte jederzeit feinjustieren.

SCENARIO_PRESETS = {
    "baseline": {
        "label": "Baseline (Bundesdurchschnitt): Unverändert",
        "rules": {
            "lt_interest_rate": {"add_pp": [0.00, 0.00, 0.00, 0.00]},
            "property_prices": {"mul_pct": [0.0, 0.0, 0.0, 0.0]},
            "gdp": {"mul_pct": [0.0, 0.0, 0.0, 0.0]},
            "unemployment": {"add_pp": [0.00, 0.00, 0.00, 0.00]},
            "inflation": {"add_pp": [0.00, 0.00, 0.00, 0.00]},
        },
    },
    "rate_shock": {
        "label": "Zinsschock",
        "rules": {
            "lt_interest_rate": {"add_pp": [1.50, 1.20, 0.80, 0.50]},
            "property_prices": {"mul_pct": [-2.0, -3.0, -2.0, -1.0]},
            "gdp": {"mul_pct": [-0.4, -0.6, -0.5, -0.3]},
            "unemployment": {"add_pp": [0.20, 0.30, 0.30, 0.20]},
            "inflation": {"add_pp": [-0.20, -0.40, -0.50, -0.30]},
        },
    },
    "high_inflation": {
        "label": "Hohe Inflation",
        "rules": {
            "lt_interest_rate": {"add_pp": [1.00, 0.75, 0.50, 0.25]},
            "property_prices": {"mul_pct": [-1.0, -1.0, 0.0, 0.0]},
            "gdp": {"mul_pct": [1.0, 1.2, 1.0, 0.8]},
            "unemployment": {"add_pp": [0.10, 0.20, 0.20, 0.10]},
            "inflation": {"add_pp": [2.00, 1.50, 1.00, 0.50]},
        },
    },
    "property_down": {
        "label": "Immobilienpreisrückgang",
        "rules": {
            "lt_interest_rate": {"add_pp": [-0.25, -0.25, 0.00, 0.00]},
            "property_prices": {"mul_pct": [-5.0, -3.0, -2.0, -1.0]},
            "gdp": {"mul_pct": [-0.2, -0.3, -0.2, -0.1]},
            "unemployment": {"add_pp": [0.10, 0.20, 0.20, 0.10]},
            "inflation": {"add_pp": [-0.10, -0.20, -0.20, -0.10]},
        },
    },
    "recession": {
        "label": "Rezession",
        "rules": {
            "lt_interest_rate": {"add_pp": [-1.00, -0.75, -0.50, -0.25]},
            "property_prices": {"mul_pct": [-3.0, -4.0, -2.0, -1.0]},
            "gdp": {"mul_pct": [-1.0, -0.8, -0.3, 0.0]},
            "unemployment": {"add_pp": [0.50, 0.70, 0.60, 0.40]},
            "inflation": {"add_pp": [-0.50, -0.70, -0.60, -0.40]},
        },
    },
    "boom": {
        "label": "Wirtschaftsboom",
        "rules": {
            "lt_interest_rate": {"add_pp": [0.75, 0.50, 0.25, 0.00]},
            "property_prices": {"mul_pct": [2.0, 3.0, 2.0, 1.0]},
            "gdp": {"mul_pct": [1.0, 1.2, 1.0, 0.8]},
            "unemployment": {"add_pp": [-0.20, -0.30, -0.30, -0.20]},
            "inflation": {"add_pp": [0.30, 0.40, 0.30, 0.20]},
        },
    },
}

# erlaubt auch Auswahl per Label-Text im Dropdown
SCENARIO_VALUE_ALIASES = {
    SCENARIO_PRESETS["baseline"]["label"].lower(): "baseline",
    SCENARIO_PRESETS["rate_shock"]["label"].lower(): "rate_shock",
    SCENARIO_PRESETS["high_inflation"]["label"].lower(): "high_inflation",
    SCENARIO_PRESETS["property_down"]["label"].lower(): "property_down",
    SCENARIO_PRESETS["recession"]["label"].lower(): "recession",
    SCENARIO_PRESETS["boom"]["label"].lower(): "boom",
}


@app.callback(
    Output("exog-override-table", "data", allow_duplicate=True),
    Output("exog-baseline-store", "data", allow_duplicate=True),  # NEW: quartalsgenaue Baseline-Referenz für Styles
    Output("exog-override-table", "style_data_conditional", allow_duplicate=True),
    [
        Input("exog-override-table", "columns"),        # triggert beim ersten Render
        Input("apply-exog-overrides-btn", "n_clicks"),
        Input("scenario-preset-dropdown", "value"),
    ],
    [
        State("exog-override-table", "data"),
    ],
    # wegen allow_duplicate=True:
    prevent_initial_call="initial_duplicate",
)
def apply_scenario_to_table_on_click(columns, n_clicks, selected_scenario, current_rows):
    """
    Logik:
    - Initial-Load (Trigger: exog-override-table.columns): schreibe IMMER die aus
      Drift+Saisonalität extrapolierte Baseline in die Tabelle und aktualisiere
      exog-baseline-store mit quartalsgenauen Referenzen (per Variable).
    - 'baseline' (per Button/Dropdown): dito (Baseline + Store updaten).
    - andere Presets: wende Regeln je Quartal relativ zur Baseline an (keine Kaskade)
      und schreibe dieselbe Baseline als Referenz in den Store (für die Styles).
    Styles: werden NICHT hier gesetzt (dash.no_update), denn dafür gibt es deinen separaten Style-Callback.
    """
    import dash
    import numpy as np
    import time
    from datetime import datetime, timezone
    from dash import callback_context as ctx

    MAX_LOG_ROWS = 8

    # ---------- kleine Helfer fürs Log ----------
    def _fmt_series(vals):
        def _f(v):
            if v is None:
                return "None"
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)
        return "[" + ", ".join(_f(v) for v in (vals or [])) + "]"

    def _fmt_rule(rule):
        if not rule:
            return "None"
        parts = []
        if "add_pp" in rule:  parts.append(f"add_pp={rule['add_pp']}")
        if "mul_pct" in rule: parts.append(f"mul_pct={rule['mul_pct']}")
        if isinstance(rule.get("per_q"), (list, tuple)): parts.append(f"per_q={list(rule['per_q'])}")
        return "{" + ", ".join(parts) + "}"

    # Mapping Anzeigename ↔ Key
    label_to_key = dict(EXOG_VAR_MAP)
    key_to_label = {v: k for k, v in label_to_key.items()}

    def _compute_model_baseline_q1_q4():
        """
        Erzeugt die Baseline-Fortschreibung (Q+1..Q+4) für alle exogenen Features,
        konsistent zur Scenario-/Delta-Logik.

        Ablauf:
        - wir nehmen nur Dateien aus scenario/data
        - wir prüfen, dass es wirklich ein Sheet 'final_dataset' gibt
        - wir bauen eine Config mit dem richtigen Feld 'forecast_horizon_quarters'
        - wir trimmen die Historie auf das letzte Quartal mit gültigem Target
        - wir wählen Exog-Spalten bevorzugt als __last, sonst __mean
        - wir rufen die Impute-/Forecast-Funktion mit H=4 auf
        - wir geben ein DataFrame mit Q+1..Q+4 zurück
        """
        import logging
        import pandas as pd

        logger = logging.getLogger("GVB_Dashboard")

        # 1) passende Datei unter scenario/data finden
        base_path = _find_analysis_or_scenario_excel()

        # 2) sicherstellen, dass das Sheet wirklich da ist – sonst Alternativen testen
        checked_paths = [base_path]

        def _sheet_exists(path: Path, sheet_name: str = "final_dataset") -> bool:
            try:
                xl = pd.ExcelFile(path)
                return sheet_name in xl.sheet_names
            except Exception:
                return False

        if not _sheet_exists(base_path, "final_dataset"):
            # Fallbacks testen (z.B. output.xlsx, analysis_data.xlsx)
            for alt in (DEFAULT_OUTPUT_XLSX, DEFAULT_ANALYSIS_XLSX):
                if alt.exists() and alt not in checked_paths and _sheet_exists(alt, "final_dataset"):
                    base_path = alt
                    break
            else:
                # nichts gefunden → sauber abbrechen
                try:
                    Log.scenario_table(
                        "[ScenarioApply] ERROR: Worksheet 'final_dataset' not found in any scenario/data/*.xlsx"
                    )
                except Exception:
                    logger.error(
                        "[ScenarioApply] ERROR: Worksheet 'final_dataset' not found in any scenario/data/*.xlsx"
                    )
                raise dash.exceptions.PreventUpdate

        # 3) Config AUFBAUEN – WICHTIG: forecast_horizon_quarters, nicht forecast_horizon
        cfg = Config(
            excel_path=str(base_path),
            sheet_name="final_dataset",
            date_col="Datum",
            target_col="Gesamt GVB",         # wenn dein Sheet anders heißt, hier anpassen
            agg_methods_exog=["last", "mean"],
            agg_method_target="last",
            forecast_horizon_quarters=4,     # <<< das ist der korrekte Parametername
            future_exog_strategy="mixed",
            future_exog_drift_window_q=8,
            future_exog_seasonal_period_q=4,
        )

        # 4) Analyzer anwerfen
        sa = ScenarioAnalysis(cfg)
        _ = sa.load_and_prepare()

        # ein bisschen Feintuning draufhängen (liest der Analyzer per getattr)
        try:
            setattr(sa.cfg, "future_exog_resid_drift_shrink", 0.35)
            setattr(sa.cfg, "future_exog_level_clip_pct", 0.05)
        except Exception:
            pass

        # 5) Historie auf letztes gültiges Target trimmen
        df_det = sa._df_det
        if cfg.target_col not in df_det.columns:
            try:
                Log.error(f"[ScenarioApply] ERROR: Spalte '{cfg.target_col}' fehlt.")
            except Exception:
                logger.error(f"[ScenarioApply] ERROR: Spalte '{cfg.target_col}' fehlt.")
            return None

        mask_valid = df_det[cfg.target_col].notna()
        if not mask_valid.any():
            try:
                Log.warn("[ScenarioApply] WARN: Kein gültiges Target im Datensatz – Baseline bleibt unverändert")
            except Exception:
                logger.warning("[ScenarioApply] WARN: Kein gültiges Target im Datensatz – Baseline bleibt unverändert")
            return None

        # Q-Anker bestimmen
        if "Q_end" in df_det.columns and df_det.loc[mask_valid, "Q_end"].notna().any():
            anchor_qend = df_det.loc[mask_valid, "Q_end"].max()
            df_hist = df_det[df_det["Q_end"] <= anchor_qend].copy()
        else:
            anchor_q = df_det.loc[mask_valid, "Q"].max()
            df_hist = df_det[df_det["Q"] <= anchor_q].copy()

        if df_hist.empty:
            try:
                Log.warn("[ScenarioApply] WARN: Getrimmte Historie ist leer.")
            except Exception:
                logger.warning("[ScenarioApply] WARN: Getrimmte Historie ist leer.")
            return None

        # 6) Exogene Spalten finden – bevorzugt __last, dann __mean
        ignore = {"Q", "Q_end", cfg.target_col}
        candidates = [
            c for c in df_hist.columns
            if c not in ignore
            and "__" in c
            and not c.startswith(("DET_", "SEAS_", "TARGET__lag"))
            and (c.endswith("__last") or c.endswith("__mean"))
        ]

        by_base = {}
        for c in candidates:
            base = c.split("__", 1)[0]
            by_base.setdefault(base, set()).add(c)

        exog_cols = []
        for base, cols in by_base.items():
            if f"{base}__last" in cols:
                exog_cols.append(f"{base}__last")
            elif f"{base}__mean" in cols:
                exog_cols.append(f"{base}__mean")

        if not exog_cols:
            # dann wenigstens die Q-Metadaten für Q+1..Q+4 zurückgeben
            try:
                Log.warn("[ScenarioApply] WARN: Keine passenden Exog-Spalten gefunden – Baseline leer.")
            except Exception:
                logger.warning("[ScenarioApply] WARN: Keine passenden Exog-Spalten gefunden – Baseline leer.")
            fut = sa._future_quarters(df_hist["Q"].iloc[-1], cfg.forecast_horizon_quarters)
            return fut

        # 7) Fortschreiben (das ist deine ARIMA/Drift/Saison-Logik im Analyzer)
        window_q = getattr(cfg, "future_exog_drift_window_q", 8)
        seas_p = getattr(cfg, "future_exog_seasonal_period_q", 4)
        H = getattr(cfg, "forecast_horizon_quarters", 4)

        fut_exog = sa._impute_future_exog_quarterly_fixed(
            df_hist,
            exog_cols,
            H,                               # <<< das fehlte vorher
            strategy=cfg.future_exog_strategy,
            window_q=window_q,
            seas_p=seas_p,
        )

        # 8) Logging
        try:
            anchor_str = (
                df_hist["Q_end"].dropna().iloc[-1]
                if "Q_end" in df_hist.columns and df_hist["Q_end"].notna().any()
                else df_hist["Q"].iloc[-1]
            )
            try:
                Log.scenario_table(
                    f"[ScenarioTable] [ScenarioApply] Baseline (Drift+Saison) aus getrimmter Historie bis {anchor_str} extrapoliert"
                )
            except Exception:
                logger.debug(
                    f"[ScenarioTable] [ScenarioApply] Baseline (Drift+Saison) aus getrimmter Historie bis {anchor_str} extrapoliert"
                )
        except Exception:
            pass

        return fut_exog








    # ---------- Trigger/Initial-Load ----------
    triggered = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
    is_initial_from_columns = (triggered == "exog-override-table")

    try:
        # Quartalsspalten (Q+1..Q+4)
        q_cols = [c["id"] for c in (columns or []) if c.get("id") != "Variable"]
        if len(q_cols) < 4:
            Log.scenario_table(f"[ScenarioApply] Not enough quarter columns: {q_cols}")
            raise dash.exceptions.PreventUpdate

        def _build_baseline_store(fut_exog):
            """
            Erzeugt den quartalsgenauen Baseline-Store:
              - by_key_q:     var_key -> [q1..q4]
              - by_display_q: Anzeige-Label -> [q1..q4]
              - q_cols:       die verwendeten Quartalsspalten
            """
            by_key_q = {}
            for key in key_to_label.keys():
                col_last = f"{key}__last"
                if fut_exog is not None and col_last in getattr(fut_exog, "columns", []):
                    arr = fut_exog[col_last].tolist()[:4]
                    arr = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in arr]
                else:
                    arr = [None, None, None, None]
                by_key_q[key] = arr

            by_display_q = { key_to_label[k]: v for k, v in by_key_q.items() if k in key_to_label }

            return {
                "by_key_q": by_key_q,
                "by_display_q": by_display_q,
                "q_cols": list(q_cols),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        # ---------- Initial-Load: Baseline direkt schreiben + Store setzen ----------
        if is_initial_from_columns:
            fut_exog = _compute_model_baseline_q1_q4()

            # Input-Snapshot
            in_log = []
            for i, row in enumerate(current_rows or []):
                if i >= MAX_LOG_ROWS: break
                in_log.append(f"  • {row.get('Variable')}: in={_fmt_series([row.get(c) for c in q_cols])}")
            more_in = max(0, len((current_rows or [])) - MAX_LOG_ROWS)
            Log.scenario_table(
                "[ScenarioApply] Initial-Load via columns: Input-Snapshot (vor Anwendung):\n"
                + ("\n".join(in_log) if in_log else "  (leer)")
                + (f"\n  … und {more_in} weitere" if more_in else "")
            )

            # Tabelle mit Baseline befüllen
            new_rows, out_log = [], []
            for i, row in enumerate(current_rows or []):
                disp_label = row.get("Variable")
                var_key = label_to_key.get(disp_label)
                col_last = f"{var_key}__last" if var_key else None

                if fut_exog is not None and col_last and col_last in fut_exog.columns:
                    vals = fut_exog[col_last].tolist()[:4]
                    vals = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in vals]
                else:
                    vals = [row.get(c) for c in q_cols]  # Fallback

                nr = {"Variable": disp_label}
                for qi, col_id in enumerate(q_cols):
                    nr[col_id] = vals[qi] if qi < len(vals) else None
                new_rows.append(nr)

                if i < MAX_LOG_ROWS:
                    out_log.append(f"  • {disp_label}: baseline={_fmt_series(vals)}")

            more = max(0, len((current_rows or [])) - MAX_LOG_ROWS)
            Log.scenario_table(
                "[ScenarioApply] Initial-Load: Baseline in Tabelle geschrieben (Drift+Saison):\n"
                + ("\n".join(out_log) if out_log else "")
                + (f"\n  … und {more} weitere" if more else "")
            )

            # Quartalsgenauen Baseline-Store setzen
            baseline_store = _build_baseline_store(fut_exog)

            # Styles dem Style-Callback überlassen
            return new_rows, baseline_store, dash.no_update

        # ---------- Ab hier: echte User-Trigger (Button/Dropdown) ----------
        if triggered not in {"apply-exog-overrides-btn", "scenario-preset-dropdown"}:
            raise dash.exceptions.PreventUpdate

        scen_key = _normalize_scenario_value(selected_scenario)
        if not scen_key or scen_key not in SCENARIO_PRESETS:
            Log.scenario_table(f"[ScenarioApply] Unknown scenario: {selected_scenario!r} -> no change")
            raise dash.exceptions.PreventUpdate

        # Entprellung
        global _SCEN_APPLY_LAST_SIG, _SCEN_APPLY_LAST_TS
        if "_SCEN_APPLY_LAST_SIG" not in globals():
            _SCEN_APPLY_LAST_SIG = None
        if "_SCEN_APPLY_LAST_TS" not in globals():
            _SCEN_APPLY_LAST_TS = 0.0

        var_order = tuple((row.get("Variable") for row in (current_rows or [])))
        curr_sig = (triggered, scen_key, var_order)
        now = time.monotonic()
        if _SCEN_APPLY_LAST_SIG == curr_sig and (now - _SCEN_APPLY_LAST_TS) < 0.30:
            raise dash.exceptions.PreventUpdate
        _SCEN_APPLY_LAST_SIG = curr_sig
        _SCEN_APPLY_LAST_TS = now

        # Input-Snapshot
        in_log = []
        for i, row in enumerate(current_rows or []):
            if i >= MAX_LOG_ROWS: break
            in_log.append(f"  • {row.get('Variable')}: in={_fmt_series([row.get(c) for c in q_cols])}")
        more_in = max(0, len((current_rows or [])) - MAX_LOG_ROWS)
        Log.scenario_table(
            "[ScenarioApply] Input-Snapshot (vor Anwendung):\n"
            + ("\n".join(in_log) if in_log else "  (leer)")
            + (f"\n  … und {more_in} weitere" if more_in else "")
        )

        # Baseline (Drift+Saison) holen – dient als Referenz und als Basis für Presets
        fut_exog = _compute_model_baseline_q1_q4()
        baseline_store = _build_baseline_store(fut_exog)

        # ===== BASELINE: immer Baseline schreiben =====
        if scen_key == "baseline":
            new_rows, out_log = [], []
            for i, row in enumerate(current_rows or []):
                disp_label = row.get("Variable")
                var_key = label_to_key.get(disp_label)
                col_last = f"{var_key}__last" if var_key else None

                if fut_exog is not None and col_last and col_last in fut_exog.columns:
                    vals = fut_exog[col_last].tolist()[:4]
                    vals = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in vals]
                else:
                    vals = [row.get(c) for c in q_cols]

                nr = {"Variable": disp_label}
                for qi, col_id in enumerate(q_cols):
                    nr[col_id] = vals[qi] if qi < len(vals) else None
                new_rows.append(nr)

                if i < MAX_LOG_ROWS:
                    out_log.append(f"  • {disp_label}: baseline={_fmt_series(vals)}")

            more = max(0, len((current_rows or [])) - MAX_LOG_ROWS)
            Log.scenario_table(
                "[ScenarioApply] Baseline in Tabelle geschrieben (Drift+Saison):\n"
                + ("\n".join(out_log) if out_log else "")
                + (f"\n  … und {more} weitere" if more else "")
            )
            # Styles dem Style-Callback überlassen (der hat nun quartalsgenaue Referenzen)
            return new_rows, baseline_store, dash.no_update

        # ===== PRESETS: relativ zur Baseline =====
        rules = SCENARIO_PRESETS[scen_key]["rules"]

        new_rows, out_log = [], []
        for i, row in enumerate(current_rows or []):
            disp_label = row.get("Variable")
            var_key = label_to_key.get(disp_label)
            rule = rules.get(var_key, None)

            col_last = f"{var_key}__last" if var_key else None
            if fut_exog is not None and col_last and col_last in fut_exog.columns:
                base_series = fut_exog[col_last].tolist()[:4]
            else:
                base_series = [row.get(c) for c in q_cols]  # defensiver Fallback

            # je Quartal auf Baseline anwenden
            vals = []
            for qi in range(4):
                bq = base_series[qi]
                if bq is None or (isinstance(bq, float) and np.isnan(bq)):
                    vals.append(None)
                else:
                    bq = float(bq)
                    vals.append(bq if rule is None else _scenario_apply_to_value(bq, rule, qi))

            nr = {"Variable": disp_label}
            for qi, col_id in enumerate(q_cols):
                nr[col_id] = vals[qi]
            new_rows.append(nr)

            if i < MAX_LOG_ROWS:
                out_log.append(f"  • {disp_label}: base={_fmt_series(base_series)} | rule={_fmt_rule(rule)} | out={_fmt_series(vals)}")

        more = max(0, len((current_rows or [])) - MAX_LOG_ROWS)
        Log.scenario_table(
            f"[ScenarioApply] Applied scenario={scen_key} relativ zur Baseline (Drift+Saison):"
            + ("\n" + "\n".join(out_log) if out_log else "")
            + (f"\n  … und {more} weitere" if more else "")
        )
        # Styles dem Style-Callback überlassen
        return new_rows, baseline_store, dash.no_update

    except dash.exceptions.PreventUpdate:
        raise
    except Exception as e:
        try:
            Log.error(f"[ScenarioApply] ERROR: {e}")
        except Exception:
            logger.error(f"[ScenarioApply] ERROR: {e}")

        # Styles & Store unverändert lassen
        return current_rows, dash.no_update, dash.no_update










def _scenario_apply_to_value(base, rule, quarter_index: int) -> float | None:
    """
    base: Basiswert (float oder None)
    rule: dict wie {"add_pp":[...]} oder {"mul_pct":[...]}
    quarter_index: 0..3
    """
    if base is None:
        return None
    try:
        if "add_pp" in rule:
            delta = float(rule["add_pp"][quarter_index])
            return round(float(base) + delta, 2)
        if "mul_pct" in rule:
            pct = float(rule["mul_pct"][quarter_index]) / 100.0
            return round(float(base) * (1.0 + pct), 2)
    except Exception:
        pass
    return round(float(base), 2)

def _normalize_scenario_value(v: str | None) -> str | None:
    """
    Erlaubt sowohl direkte Keys ('rate_shock') als auch Label-Strings,
    sofern SCENARIO_VALUE_ALIASES gepflegt ist.
    """
    if not v:
        return None
    s = str(v).strip().lower()
    return SCENARIO_VALUE_ALIASES.get(s, s)




# --- Konstanten (UI-Label <-> interne Keys) -----------------------------------
EXOG_VAR_MAP: Dict[str, str] = {
    "Zinssatz (10Y)": "lt_interest_rate",
    "Immobilienpreise": "property_prices",
    "BIP": "gdp",
    "Arbeitslosenquote": "unemployment",
    "Inflation": "inflation",
}
EXOG_DISPLAY_ORDER: List[str] = [
    "Zinssatz (10Y)", "Immobilienpreise", "BIP", "Arbeitslosenquote", "Inflation"
]

# --- Datei-Finder & Loader -----------------------------------------------------
def _find_scenario_excel() -> Path:
    """Sucht output.xlsx (Basis für analysis_data.xlsx) – relativ zum Ordner dieses Moduls."""
    base = Path(__file__).parent                 # …/scenario
    candidates = [
        base / "data" / "output.xlsx",           # ✓ richtig: …/scenario/data/output.xlsx
        base / "data" / "transformed_output.xlsx",
        base.parent / "data" / "output.xlsx",    # optional: …/<project>/data/output.xlsx
        base / "output.xlsx",                    # fallback: …/scenario/output.xlsx
        Path("/mnt/data/output.xlsx"),           # CI/Dev
    ]
    for p in candidates:
        if p.exists():
            return p
    # Rückgabe des ersten Kandidaten (für eindeutige Fehlermeldung downstream)
    return candidates[0]


def _load_scenario_final_dataset() -> pd.DataFrame:
    base = Path(__file__).parent                 # …/scenario
    primary  = base / "data" / "output.xlsx"     # ✓ richtig
    fallback = base / "output.xlsx"              # fallback
    xlsx_path = primary if primary.exists() else fallback
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Scenario-Datei nicht gefunden: {primary} (Fallback: {fallback})"
        )

    df = pd.read_excel(xlsx_path, sheet_name="final_dataset", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    for cand in ["Datum", "datum", "Date", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("In 'final_dataset' wurde keine Datumsspalte gefunden.")

    df = df.rename(columns={date_col: "Datum"}).copy()
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
    df = df.dropna(subset=["Datum"])

    # sanfte Numerik-Normalisierung für alle Werte-Spalten
    value_cols = [c for c in df.columns if c != "Datum"]
    for c in value_cols:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s = s.astype(str)
            s = s.replace({"": None, "nan": None, "NaN": None, "None": None, "-": None, "—": None, "–": None})
            s = s.str.replace(r"\s", "", regex=True)
            s = s.str.replace(".", "", regex=False)  # Tausenderpunkte entfernen
            s = s.str.replace(",", ".", regex=False) # deutsches Komma → Punkt
            df[c] = pd.to_numeric(s, errors="coerce")
    return df.sort_values("Datum").reset_index(drop=True)


# --- Quartals-Utilities --------------------------------------------------------


def _next_four_quarters(base: Optional[pd.Timestamp] = None) -> List[Tuple[str, pd.Timestamp]]:
    """
    Gibt die nächsten 4 Quartale als (Label 'YYYY-Qn', Quartalsende-Timestamp) zurück,
    beginnend mit dem *nächsten* Quartal nach 'base' (oder heute).
    """
    if base is None:
        base = pd.Timestamp.now(tz="Europe/Berlin").normalize()
    y, q = int(base.year), int(((base.month - 1) // 3) + 1)
    out: List[Tuple[str, pd.Timestamp]] = []
    for k in range(1, 5):
        total = (y * 4 + (q - 1)) + k
        ny, nq = divmod(total, 4)
        nq = nq + 1
        label = f"{ny}-Q{nq}"
        ts = get_quarter_end_date(ny, nq)
        out.append((label, ts))
    return out


def _parse_exog_override_table(columns, rows, H: int = 4) -> Tuple[Dict[str, List[Optional[float]]], List[str]]:
    """
    Liest die DataTable (Q+1..Q+H) in ein Dict var_key -> [q1..qH] (Floats/None).
    Robustes Parsing für deutsche/englische Zahlformate, bevorzugt parse_german_number aus app.py.
    """
    quarter_labels = [c["id"] for c in (columns or []) if c.get("id") != "Variable"][:H]
    Log.scenario_table(f"[ParseTable] Enter | H={H} | quarter_labels={quarter_labels}")

    manual_vals: Dict[str, List[Optional[float]]] = {}

    # Parser-Strategie: bevorzugt app.parse_german_number, sonst robuster Fallback
    try:
        from app import parse_german_number as _parse_num  # type: ignore
    except Exception:
        def _parse_num(x) -> Optional[float]:
            """Robuster Zahlen-Parser (DE/EN), inkl. Prozent und Tausendertrennzeichen."""
            if x is None:
                return None
            s = str(x).strip()
            if s == "":
                return None

            # Normalisierung (geschützte Leerzeichen, alternative Minuszeichen etc.)
            s = s.replace("\u00A0", " ").replace(" ", "")
            s = s.replace("−", "-").replace("–", "-")  # verschiedene Minus-Darstellungen

            # Prozent-Erkennung
            has_pct = "%" in s

            # Nur zulässige Zeichen beibehalten (Zahlen, +-.,,e/E,%)
            import re
            s = re.sub(r"[^0-9,\.\+\-eE%]", "", s)

            # Wenn sowohl Punkt als auch Komma vorhanden sind: letztes Vorkommen ist Dezimaltrennzeichen
            try:
                if "," in s and "." in s:
                    if s.rfind(",") > s.rfind("."):
                        # Deutsch: . = Tausender, , = Dezimal
                        s = s.replace(".", "").replace(",", ".")
                    else:
                        # Englisch: , = Tausender, . = Dezimal
                        s = s.replace(",", "")
                elif "," in s and "." not in s:
                    # Deutsch: , = Dezimal
                    s = s.replace(",", ".")
                else:
                    # Englisch: . = Dezimal oder nackte Zahl
                    pass

                # Prozentzeichen entfernen vor float()
                s = s.replace("%", "")
                if s in {"+", "-", ".", "+.", "-."}:
                    return None  # kein valider numerischer Inhalt

                val = float(s)
                return val / 100.0 if has_pct else val
            except Exception:
                return None

    # Zeilen durchlaufen
    for i, row in enumerate(rows or []):
        disp_label = row.get("Variable")
        if not disp_label:
            Log.scenario_table(f"[ParseTable] Row#{i}: keine 'Variable' -> skip")
            continue

        var_key = EXOG_VAR_MAP.get(disp_label)
        if not var_key:
            Log.scenario_table(f"[ParseTable] Row#{i}: unbekanntes Label '{disp_label}' -> skip")
            continue

        vals: List[Optional[float]] = []
        for lab in quarter_labels:
            raw = row.get(lab, None)
            parsed = _parse_num(raw)
            vals.append(parsed)

        # Auf H auffüllen, falls zu kurz
        if len(vals) < H:
            vals += [None] * (H - len(vals))

        manual_vals[var_key] = vals[:H]

        # Logging: nur nicht-None Werte
        non_none = [v for v in vals[:H] if v is not None]
        if non_none:
            Log.scenario_table(f"[ParseTable] Row#{i} '{disp_label}' -> {var_key}: {[round(v, 3) for v in non_none]}")
        else:
            Log.scenario_table(f"[ParseTable] Row#{i} '{disp_label}' -> {var_key}: [alle None]")

    Log.scenario_table(f"[ParseTable] Parsed {len(manual_vals)} variables")
    return manual_vals, quarter_labels

def _debug_forecast_comparison(base_df: pd.DataFrame, scn_df: pd.DataFrame, label: str = ""):
    """Debug-Utility zum Vergleich zweier Forecast-DataFrames"""
    if label:
        logger.debug(f"\n{'='*60}\n{label}\n{'='*60}")
    logger.debug("\n[DEBUG] Baseline Forecast:")
    logger.debug(base_df[["Quarter", "Forecast_Baseline"]].to_string(index=False))
    logger.debug("\n[DEBUG] Scenario Forecast:")
    logger.debug(scn_df[["Quarter", "Forecast_Scenario"]].to_string(index=False))
    logger.debug("\n[DEBUG] Differenzen (Scenario - Baseline):")
    diff = scn_df["Forecast_Scenario"].values - base_df["Forecast_Baseline"].values
    for i, (q, d) in enumerate(zip(base_df["Quarter"], diff)):
        logger.debug(f"  {q}: {d:+.4f}")
    logger.debug(f"\n[DEBUG] Max Abs Diff: {abs(diff).max():.4f}")
    logger.debug(f"[DEBUG] Mean Abs Diff: {abs(diff).mean():.4f}\n")
def _build_future_deltas_for_analyzer(
    sa: "ScenarioAnalysis",
    manual_vals: Dict[str, List[Optional[float]]],
    H: int = 4,
    force_hist_qend: Optional[pd.Timestamp] = None,   # <<-- NEU: gemeinsamer Zeitanker
) -> Tuple[Dict[str, Dict[int, float]], pd.DataFrame]:
    """
    Berechnet Deltas zwischen User-Eingaben und extrapolierter Baseline (Drift+Saisonalität).

    Wichtige Punkte:
    - Verwendet NUR historische Daten (keine UI/Store-Daten) für die Baseline-Extrapolation.
    - Falls 'force_hist_qend' gesetzt ist, wird die Historie *hart* bis inkl. dieses Q_end
      begrenzt. Dadurch nutzen Forecast und Delta-Builder exakt denselben Zeitanker.

    Args:
        sa: ScenarioAnalysis-Instanz (nach load_and_prepare/train_or_load)
        manual_vals: Dict[var_key -> Liste[Optional[float]]] mit UI-Werten (Q+1..Q+H)
        H: Anzahl Forecast-Quartale
        force_hist_qend: Optionaler gemeinsamer Zeitanker (inklusive). Wenn None, wird
                         automatisch das max. Q_end aus sa._df_det verwendet.

    Returns:
        scenario_future: Dict[column_name -> Dict[h -> delta]]
        fut_q_df: DataFrame mit den zukünftigen Quartalsenddaten (Q+1..Q+H)
    """
    Log.scenario_table(f"[ScenarioDeltas] Enter | H={H} | manual keys={list(manual_vals.keys())}")

    # --- 1) Historische Quartals-Deterministik kopieren ----------------------
    if getattr(sa, "_df_det", None) is None or sa._df_det.empty:
        Log.error("[ScenarioDeltas] ERROR: sa._df_det ist leer oder nicht gesetzt")
        return {}, pd.DataFrame({"Q_end": []})

    df_hist_det_full = sa._df_det.copy()

    # --- 2) Gemeinsamen Zeitanker bestimmen & Historie trimmen ---------------
    if force_hist_qend is not None:
        last_hist_qend = pd.to_datetime(force_hist_qend)
        Log.scenario_table(f"[ScenarioDeltas] Forced historischer Anker: {last_hist_qend.date()}")
        df_hist_det = df_hist_det_full[df_hist_det_full["Q_end"] <= last_hist_qend].copy()
    else:
        last_hist_qend = pd.to_datetime(df_hist_det_full["Q_end"].max())
        Log.scenario_table(f"[ScenarioDeltas] Letztes historisches Quartal (auto): "
                           f"{pd.Period(last_hist_qend, freq='Q')}")
        df_hist_det = df_hist_det_full

    if df_hist_det.empty:
        Log.error("[ScenarioDeltas] ERROR: getrimmte Historie ist leer")
        return {}, pd.DataFrame({"Q_end": []})

    # --- 3) Exogene Spalten identifizieren -----------------------------------
    ignore = {"Q", "Q_end", sa.cfg.target_col}
    exog_cols = [
        c for c in df_hist_det.columns
        if c not in ignore and "__" in c and not c.startswith(("DET_", "SEAS_", "TARGET__lag"))
    ]
    if not exog_cols:
        Log.warn("[ScenarioDeltas] WARN: Keine exogenen Spalten gefunden")
        return {}, pd.DataFrame({"Q_end": []})

    Log.scenario_table(
        f"[ScenarioDeltas] Exog columns: {exog_cols[:10]}{' ...' if len(exog_cols) > 10 else ''}"
    )

    # --- 4) Baseline-Exogenen für Q+1..Q+H imputen (Drift + Saison) ----------
    try:
        fut_exog_baseline = sa._impute_future_exog_quarterly_fixed(
            df_hist_det=df_hist_det,                     # <<-- wichtig: getrimmt auf den Anker!
            exog_cols=exog_cols,
            H=H,
            strategy=getattr(sa.cfg, "future_exog_strategy", "mixed"),
            window_q=getattr(sa.cfg, "future_exog_drift_window_q", 8),
            seas_p=getattr(sa.cfg, "future_exog_seasonal_period_q", 4),
        )
    except TypeError:
        # Falls ältere Signatur ohne Keywords: auf Positional zurückfallen
        fut_exog_baseline = sa._impute_future_exog_quarterly_fixed(
            df_hist_det, exog_cols, H
        )
    except Exception as e:
        Log.error(f"[ScenarioDeltas] ERROR in imputation: {e}")
        import traceback
        traceback.print_exc()
        return {}, pd.DataFrame({"Q_end": []})

    Log.scenario_table(f"[ScenarioDeltas] Baseline extrapoliert: shape={fut_exog_baseline.shape}")
    try:
        Log.scenario_table("[ScenarioDeltas] Baseline samples:")
        for col in exog_cols[:5]:
            if col in fut_exog_baseline.columns:
                vals = fut_exog_baseline[col].values[:H]
                # defensiv runden (numpy/None-handling)
                vals_fmt = []
                for v in vals:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        vals_fmt.append(None)
                    else:
                        try:
                            vals_fmt.append(round(float(v), 3))
                        except Exception:
                            vals_fmt.append(v)
                Log.scenario_table(f"  {col}: {vals_fmt}")
    except Exception:
        pass

    # --- 5) Deltas vs. Baseline bilden ---------------------------------------
    scenario_future: Dict[str, Dict[int, float]] = {}

    # Wir bevorzugen __last, unterstützen aber __mean falls vorhanden
    for var_key, user_vals in manual_vals.items():
        if not user_vals:
            continue

        for suffix in ("__last", "__mean"):
            col = f"{var_key}{suffix}"
            if col not in fut_exog_baseline.columns:
                Log.scenario_table(f"[ScenarioDeltas] Spalte '{col}' nicht in Baseline gefunden")
                continue

            deltas: Dict[int, float] = {}
            for h in range(1, H + 1):
                user_val = user_vals[h - 1] if (h - 1) < len(user_vals) else None
                if user_val is None:
                    continue
                try:
                    base_val = fut_exog_baseline[col].iloc[h - 1]
                    if base_val is None or (isinstance(base_val, float) and np.isnan(base_val)):
                        continue
                    base_val = float(base_val)
                    uval = float(user_val)
                    delta = uval - base_val
                    if abs(delta) > 1e-3:  # Schwellwert
                        deltas[h] = delta
                        Log.scenario_table(
                            f"[ScenarioDeltas] {col} Q+{h}: user={uval:.4f} | base={base_val:.4f} | Δ={delta:+.4f}"
                        )
                    else:
                        Log.scenario_table(
                            f"[ScenarioDeltas] {col} Q+{h}: Delta zu klein ({delta:.6f}), ignoriert"
                        )
                except Exception as e:
                    Log.error(f"[ScenarioDeltas] ERROR calculating delta for {col} Q+{h}: {e}")

            if deltas:
                scenario_future[col] = deltas
                Log.scenario_table(f"[ScenarioDeltas] Collected {len(deltas)} deltas for {col}")

    total_deltas = sum(len(d) for d in scenario_future.values())
    Log.scenario_table(f"[ScenarioDeltas] Final: {len(scenario_future)} columns, {total_deltas} total deltas")
    if not scenario_future:
        Log.warn("[ScenarioDeltas] WARNING: Keine Deltas berechnet - Baseline = Scenario!")

    # --- 6) Zukunfts-Quartale (Q_end) explizit aufbauen ----------------------
    fut_q = pd.DataFrame({
        "Q_end": pd.date_range(
            start=last_hist_qend + pd.offsets.QuarterEnd(1),
            periods=H,
            freq="Q"
        )
    })

    return scenario_future, fut_q


# --- UI: 5×4 Textinputs (Default mit letztem Wert) ----------------------------
@app.callback(
    Output("exog-quarter-inputs", "children"),
    [Input("url", "pathname"),
     Input("reset-exog-overrides-btn", "n_clicks")],
    State("exog-data-store", "data"),
    prevent_initial_call=False
)
def build_exog_quarter_inputs(pathname, reset_clicks, exog_store):
    """Baut die 5×4 Eingabezeilen mit Basiswerten (letzte Ist-Werte) und Quartalslabels auf."""
    if pathname not in {"/scenario", "/scenario-analysis"}:
        raise dash.exceptions.PreventUpdate

    exog_df = pd.DataFrame(exog_store) if isinstance(exog_store, list) else pd.DataFrame(exog_store or [])
    if exog_df.empty or "date" not in exog_df.columns:
        exog_df = pd.DataFrame({"date": []})
    exog_df["date"] = pd.to_datetime(exog_df["date"], errors="coerce")
    exog_df = exog_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for var_key in EXOG_VAR_MAP.values():
        if var_key not in exog_df.columns:
            exog_df[var_key] = np.nan

    last_vals: Dict[str, Optional[float]] = {}
    for disp_label in EXOG_DISPLAY_ORDER:
        var_key = EXOG_VAR_MAP[disp_label]
        s = exog_df[var_key]
        lv = s.dropna().iloc[-1] if not s.dropna().empty else None
        last_vals[var_key] = None if lv is None else float(lv)

    quarter_labels = [lab for lab, _ in _next_four_quarters()]

    rows: List[html.Div] = []
    header = dbc.Row(
        [dbc.Col(html.Strong("Variable"), width=3)] +
        [dbc.Col(html.Strong(lab), width=2) for lab in quarter_labels],
        className="mb-1"
    )
    rows.append(header)

    for disp_label in EXOG_DISPLAY_ORDER:
        var_key = EXOG_VAR_MAP[disp_label]
        base_val = last_vals.get(var_key, None)
        inputs = []
        for lab in quarter_labels:
            inputs.append(
                dbc.Col(
                    dcc.Input(
                        id={"type": "exog-q-input", "var": var_key, "q": lab},
                        type="text",  # Komma/DE-Format zulassen
                        value=(None if base_val is None else f"{base_val:g}"),
                        placeholder="—",
                        debounce=True,
                        style={"width": "100%"}
                    ),
                    width=2
                )
            )
        row = dbc.Row(
            [dbc.Col(html.Div([html.Span(disp_label)]), width=3)] + inputs,
            className="g-2 mb-1"
        )
        rows.append(row)

    rows.append(
        html.Small(
            "Hinweis: Felder leer lassen, wenn keine Überschreibung gewünscht ist. "
            "Eingaben werden numerisch interpretiert (Dezimalpunkt oder Komma).",
            className="text-muted d-block mt-1"
        )
    )
    return rows

# --- DataTable initialisieren (unter der Szenario-Konfiguration) --------------
# =========================================================
# Callback: Eingabetabelle initialisieren + Prefill (2 Dezimalstellen)
#   - nutzt exog-data-store (falls vorhanden)
#   - Fallback: liest output.xlsx (Sheet 'final_dataset')
#   - NEU: Alias-Mapping von Store-Spalten -> erwartete EXOG_VAR_MAP-Werte
# =========================================================
def _generate_monthly_dates_for_quarter(quarter_end: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Generiert 3 Monatsdaten für ein Quartal (Monatsanfänge):
    - Monat 1, Monat 2, Monat 3 (Quartalsende-Monat)
    
    Beispiel: Q1 2025 (Ende 2025-03-31) -> [2025-01-01, 2025-02-01, 2025-03-01]
    """
    year = quarter_end.year
    end_month = quarter_end.month
    
    # Quartal bestimmen
    if end_month == 3:  # Q1
        months = [1, 2, 3]
    elif end_month == 6:  # Q2
        months = [4, 5, 6]
    elif end_month == 9:  # Q3
        months = [7, 8, 9]
    else:  # Q4 (12)
        months = [10, 11, 12]
    
    return [pd.Timestamp(year, m, 1) for m in months]


def _extract_last_non_null(df: pd.DataFrame, wanted_key: str, aliases_map: dict) -> Tuple[Optional[float], Optional[pd.Timestamp], str]:
    """
    Extrahiert den letzten nicht-NaN Wert für eine Variable aus einem DataFrame.
    
    Returns:
        (value, date, source_column) oder (None, None, "") wenn nicht gefunden
    """
    if df.empty or "date" not in df.columns and "Datum" not in df.columns:
        return None, None, ""
    
    # Datumsspalte finden
    date_col = "Datum" if "Datum" in df.columns else "date"
    
    def _to_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")
    
    # 1. Direkte Spalte
    if wanted_key in df.columns:
        s = _to_numeric_series(df[wanted_key])
        nn = s.dropna()
        if not nn.empty:
            idx = nn.index[-1]
            return round(float(nn.iloc[-1]), 2), pd.to_datetime(df.loc[idx, date_col]), wanted_key
    
    # 2. Aliase durchsuchen
    sources = [src for src, tgt in aliases_map.items() if tgt == wanted_key and src in df.columns]
    if sources:
        comb = None
        for src in sources:
            cand = _to_numeric_series(df[src])
            comb = cand if comb is None else comb.combine_first(cand)
        nn = comb.dropna()
        if not nn.empty:
            idx = nn.index[-1]
            return round(float(nn.iloc[-1]), 2), pd.to_datetime(df.loc[idx, date_col]), sources[-1]
    
    return None, None, ""

from pathlib import Path
import pandas as pd
from dash import Output, Input
import dash

# falls du das schon ganz oben im file hast, diesen Import-Block einfach weglassen
try:
    # im selben Paket wie scenario_main.py
    from .scenario_dataloader import get_excel_engine  # type: ignore
except Exception:
    # Fallback: pandas autodetect
    def get_excel_engine():
        return None


@app.callback(
    [
        Output("exog-override-table", "columns"),
        Output("exog-override-table", "data"),
        Output("exog-baseline-store", "data"),
        Output("exog-override-table", "style_data_conditional"),
        Output("exog-override-table", "tooltip_data"),
    ],
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_exog_override_table_quarterly(pathname):
    """
    Initialisiert die Exogenentabelle auf der Szenario-Seite.
    Wichtig:
    - es werden NUR Dateien aus ./scenario/data/ genommen
    - wenn nichts gefunden wird, brechen wir sauber ab
    - 'final_dataset' wird bevorzugt, aber wir fallen weich auf leere DFs
    """
    if pathname not in {"/scenario", "/scenario-analysis"}:
        raise dash.exceptions.PreventUpdate

    # 1) Anzeige-Mapping laden (wenn vorhanden)
    try:
        display_order: list[str] = EXOG_DISPLAY_ORDER  # z. B. ["Zinssatz ...", ...]
        key_map: dict[str, str] = EXOG_VAR_MAP        # z. B. {"Zinssatz ...": "lt_interest_rate", ...}
    except Exception:
        # sehr defensiver Fallback
        key_map = {
            "Zinssatz (10Y)": "lt_interest_rate",
            "Immobilienpreise": "property_prices",
            "BIP": "gdp",
            "Arbeitslosenquote": "unemployment",
            "Inflation": "inflation",
        }
        display_order = [
            "Zinssatz (10Y)",
            "Immobilienpreise",
            "BIP",
            "Arbeitslosenquote",
            "Inflation",
        ]

    # 2) Szenario-Datei im *richtigen* Ordner suchen
    base = Path(__file__).parent          # .../scenario
    data_dir = base / "data"              # .../scenario/data
    candidates = [
        data_dir / "output.xlsx",         # die „echte“ Datei aus dem Downloader
        data_dir / "transformed_output.xlsx",
    ]

    base_excel = None
    for p in candidates:
        if p.exists():
            base_excel = p
            break

    if base_excel is None:
        # nichts gefunden -> nicht crashen, sondern still stehen lassen
        try:
            Log.warn("Keine Szenario-Datei in ./scenario/data/ gefunden – Tabelle wird nicht initialisiert.")
        except Exception:
            pass
        raise dash.exceptions.PreventUpdate

    # 3) Excel öffnen (mit Engine, falls konfiguriert)
    eng = None
    try:
        eng = get_excel_engine()
    except Exception:
        eng = None

    xl = pd.ExcelFile(base_excel, engine=eng) if eng else pd.ExcelFile(base_excel)

    # kleine Lesefunktion
    def _read_sheet(name: str) -> pd.DataFrame:
        if name not in xl.sheet_names:
            return pd.DataFrame()
        df = pd.read_excel(xl, sheet_name=name, engine="openpyxl")
        # Datumsspalte normalisieren
        dcol = None
        for cand in ("Datum", "datum", "Date", "date"):
            if cand in df.columns:
                dcol = cand
                break
        if dcol is None:
            return pd.DataFrame()
        df = df.rename(columns={dcol: "Datum"}).copy()
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)
        return df

    df_final = _read_sheet("final_dataset")
    df_ecb   = _read_sheet("raw_ecb")
    df_buba  = _read_sheet("raw_buba")

    # 4) Alias-Lookup für die 5 Kernexogenen
    EXOG_ALIASES: dict[str, list[str]] = {
        "lt_interest_rate": [
            "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
            "irs.m.de.l.l40.ci.0000.eur.n.z",
            "irs_m_de_l_l40_ci_0000_eur_n_z",
            "zinssatz_10y",
            "hauptrefinanzierungssatz",
            "lt_interest_rate",
        ],
        "property_prices": [
            "RESR.Q.DE._T.N._TR.TVAL.10.TB.N.IX",
            "resr.q.de._t.n._tr.tval.10.tb.n.ix",
            "resr_q_de__t_n__tr_tval_10_tb_n_ix",
            "immobilienpreise",
            "property_prices",
        ],
        "gdp": [
            "MNA.Q.N.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
            "mna.q.n.de.w2.s1.s1_b_b1gq._z._z._z.eur.v.n",
            "mna_q_n_de_w2_s1_s1_b_b1gq__z__z__z_eur_v_n",
            "bruttoinlandsprodukt",
            "gdp",
        ],
        "unemployment": [
            "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
            "lfsi.m.de.s.unehrt.total0.15_74.t",
            "lfsi_m_de_s_unehrt_total0_15_74_t",
            "arbeitslosenquote",
            "unemployment",
        ],
        "inflation": [
            "ICP.M.DE.N.000000.4.ANR",
            "icp.m.de.n.000000.4.anr",
            "icp_m_de_n_000000_4_anr",
            "inflation_rate",
            "inflation",
        ],
    }

    def _first_matching_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
        if df is None or df.empty:
            return None
        lower = {str(c).strip().lower(): c for c in df.columns}
        for a in aliases:
            k = str(a).strip().lower()
            if k in lower:
                return lower[k]
        return None

    # 5) nächsten 4 Quartale bestimmen
    if not df_final.empty:
        last_date = pd.to_datetime(df_final["Datum"].max())
    else:
        last_date = pd.Timestamp.today()
    last_q = pd.Period(last_date, freq="Q")
    quarter_labels = [f"{(last_q + i).year}-Q{(last_q + i).quarter}" for i in range(1, 5)]

    # 6) Spalten für DataTable
    try:
        from dash.dash_table.Format import Format, Scheme  # type: ignore
        number_fmt = Format(precision=2, scheme=Scheme.fixed)
    except Exception:
        number_fmt = None

    columns = [{"name": "Variable", "id": "Variable", "editable": False}]
    for lab in quarter_labels:
        col_def = {"name": lab, "id": lab, "type": "numeric", "editable": True}
        if number_fmt is not None:
            col_def["format"] = number_fmt
        columns.append(col_def)

    # 7) letzte Ist-Werte aus den drei Sheets zusammensuchen
    latest_vals: dict[str, float | None] = {}
    for disp in display_order:
        key = key_map.get(disp)
        val = None

        # 1) final_dataset direkt
        if key and not df_final.empty and key in df_final.columns:
            s = pd.to_numeric(df_final[key], errors="coerce").dropna()
            if not s.empty:
                val = float(s.iloc[-1])

        # 2) ECB-Sheet
        if val is None and not df_ecb.empty and key in EXOG_ALIASES:
            col = _first_matching_col(df_ecb, EXOG_ALIASES[key])
            if col:
                s = pd.to_numeric(df_ecb[col], errors="coerce").dropna()
                if not s.empty:
                    val = float(s.iloc[-1])

        # 3) BuBa-Sheet
        if val is None and not df_buba.empty and key in EXOG_ALIASES:
            col = _first_matching_col(df_buba, EXOG_ALIASES[key])
            if col:
                s = pd.to_numeric(df_buba[col], errors="coerce").dropna()
                if not s.empty:
                    val = float(s.iloc[-1])

        latest_vals[key] = val

    # 8) Tabellenzeilen aufbauen
    rows: list[dict[str, float | str | None]] = []
    for disp in display_order:
        key = key_map.get(disp)
        base_val = latest_vals.get(key)
        r = {"Variable": disp}
        for lab in quarter_labels:
            r[lab] = base_val
        rows.append(r)

    # 9) Baseline-Store für späteren Style-Callback
    baseline_store = {
        "by_display": {disp: latest_vals.get(key_map.get(disp)) for disp in display_order},
        "by_key": latest_vals,
    }

    # 10) Styles & Tooltips berechnen (Helfer ist schon in deinem File)
    styles, tooltips = _compute_exog_table_styles_and_tooltips(
        columns,
        rows,
        baseline_store,
        exog_var_map=key_map,
    )

    return columns, rows, baseline_store, styles, tooltips



@app.callback(
    Output("hc-prewarm-toast", "is_open", allow_duplicate=True),
    Output("hc-prewarm-toast", "header", allow_duplicate=True),
    Input("retrain-scenario-model-btn", "n_clicks"),  # Neuer Button
    prevent_initial_call=True
)
def force_retrain_scenario_model(n_clicks):
    """Löscht Modell-Cache und erzwingt Neutraining"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    try:
        import shutil
        from pathlib import Path
        
        models_dir = Path("scenario/models_scenario")
        if models_dir.exists():
            # Alle .pkl Dateien löschen
            deleted = 0
            for pkl_file in models_dir.glob("*.pkl"):
                pkl_file.unlink()
                deleted += 1
            
            return True, f"✓ {deleted} Modelle gelöscht – nächster Forecast trainiert neu"
        else:
            return True, "⚠ Modell-Verzeichnis nicht gefunden"
            
    except Exception as e:
        return True, f"✗ Fehler: {str(e)}"
    

@app.callback(
    Output("exog-data-store", "data", allow_duplicate=True),
    Output("apply-exog-overrides-note", "children", allow_duplicate=True),
    Input("apply-exog-overrides-btn", "n_clicks"),
    State({"type": "exog-q-input", "var": ALL, "q": ALL}, "value"),
    State({"type": "exog-q-input", "var": ALL, "q": ALL}, "id"),
    State("exog-override-table", "columns"),
    State("exog-override-table", "data"),
    State("exog-data-store", "data"),
    prevent_initial_call=True
)
def apply_exog_overrides_combined(n_clicks, pm_values, pm_ids, tbl_columns, tbl_rows, exog_store):
    """Übernimmt Werte aus den 5×4-Textfeldern in den exog-data-store und setzt eine Bestätigungsnachricht."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate  # type: ignore[name-defined]

    # --- Quartalslabel "YYYY-Qn" → Quartalsende (Foundation-Helper nutzen) ---
    _q_label_re = re.compile(r"^\s*(\d{4})-Q([1-4])\s*$")  # type: ignore[name-defined]

    def _label_to_qend_ts(label: str):
        m = _q_label_re.match(str(label))
        if not m:
            return None
        year = int(m.group(1))
        quarter = int(m.group(2))
        return get_quarter_end_date(year, quarter)  # Foundation

    # Alle Quartalsspalten aus der Tabelle erkennen
    quarter_labels = [
        c["id"]
        for c in (tbl_columns or [])
        if isinstance(c, dict) and c.get("id") and c.get("id") != "Variable"
    ]
    label_to_ts = {lab: _label_to_qend_ts(lab) for lab in quarter_labels}
    label_to_ts = {k: v for k, v in label_to_ts.items() if v is not None}

    if not label_to_ts:
        return dash.no_update, "Konnte Quartals-Labels nicht interpretieren (erwarte 'YYYY-Qn')."  # type: ignore[name-defined]

    overrides = []

    # 1) Pattern-Matching Inputs (kleine Einzel-Eingabefelder oberhalb der Tabelle)
    if pm_values and pm_ids:
        for val, idd in zip(pm_values, pm_ids):
            if not isinstance(idd, dict):
                continue
            var_key = idd.get("var")
            q_label = idd.get("q")
            ts = label_to_ts.get(q_label)
            if not var_key or ts is None:
                continue
            v = parse_german_number(val)  # Foundation
            if v is None:
                continue
            overrides.append((ts, var_key, float(v)))

    # 2) DataTable-Zellen auslesen (Anzeige-Name → technischer Key via EXOG_VAR_MAP)
    if isinstance(tbl_rows, list) and quarter_labels:
        for row in tbl_rows:
            if not isinstance(row, dict):
                continue
            disp_label = row.get("Variable")
            var_key = EXOG_VAR_MAP.get(disp_label)  # Foundation-Konstante
            if not var_key:
                continue
            for lab in quarter_labels:
                ts = label_to_ts.get(lab)
                if ts is None:
                    continue
                v = parse_german_number(row.get(lab))  # Foundation
                if v is None:
                    continue
                overrides.append((ts, var_key, float(v)))

    if not overrides:
        return dash.no_update, "Keine gültigen Eingaben gefunden. Nichts geändert."  # type: ignore[name-defined]

    # Dedupe: Falls derselbe (ts, var_key) mehrfach vorkommt → letzter Eintrag gewinnt
    # (z.B. wenn sowohl PM-Input als auch Tabelle befüllt wurde)
    ov_df = pd.DataFrame(overrides, columns=["date", "var_key", "value"])  # type: ignore[name-defined]
    ov_df = ov_df.sort_index()  # Reihenfolge beibehalten
    ov_df = ov_df.dropna(subset=["date", "var_key"])
    ov_df = ov_df.groupby(["date", "var_key"], as_index=False).tail(1).reset_index(drop=True)

    # ---------- Store robust laden (Foundation-Helper) ----------
    exog_df = load_dataframe_from_store(exog_store, fallback=pd.DataFrame({"date": []}))  # type: ignore[name-defined]
    if "date" not in exog_df.columns:
        exog_df["date"] = pd.to_datetime([])  # leer initialisieren
    exog_df["date"] = pd.to_datetime(exog_df["date"], errors="coerce")
    exog_df = exog_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # fehlende Spalten für alle bekannten EXOG-Variablen anlegen
    for vk in EXOG_VAR_MAP.values():
        if vk not in exog_df.columns:
            exog_df[vk] = np.nan  # type: ignore[name-defined]

    # fehlende Zeitpunkte (Quartalsenden) ergänzen
    needed_ts = ov_df["date"].unique().tolist()
    have_dates = set(exog_df["date"])
    missing = [ts for ts in needed_ts if ts not in have_dates]
    if missing:
        add_df = pd.DataFrame({"date": missing})  # type: ignore[name-defined]
        for vk in EXOG_VAR_MAP.values():
            add_df[vk] = np.nan
        exog_df = pd.concat([exog_df, add_df], ignore_index=True)
        exog_df = exog_df.sort_values("date").reset_index(drop=True)

    # Werte setzen
    for _, row in ov_df.iterrows():
        exog_df.loc[exog_df["date"] == row["date"], row["var_key"]] = row["value"]

    exog_df = exog_df.sort_values("date").reset_index(drop=True)

    # kleine, hilfreiche Rückmeldung
    note = f"Übernommen: {len(ov_df)} Wert(e) auf Quartalsenden."

    return exog_df.to_dict(orient="records"), note

from datetime import datetime
from dash import no_update, callback_context
from dash.exceptions import PreventUpdate

@app.callback(
    Output("model-list-modal", "is_open"),
    Output("model-list-body", "children"),
    Input("show-models-btn", "n_clicks"),
    Input("close-models-list", "n_clicks"),
    State("model-list-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_model_list(open_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "show-models-btn":
        try:
            rows = []
            # Liste aller .pkl-Modelle, neueste zuerst
            for p in sorted(MODELS_DIR.glob("*.pkl"),
                            key=lambda x: x.stat().st_mtime, reverse=True):
                stat = p.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                size_mb = stat.st_size / (1024 * 1024)
                rows.append(
                    html.Tr([
                        html.Td(p.name),
                        html.Td(f"{size_mb:.2f} MB"),
                        html.Td(mtime),
                    ])
                )

            if not rows:
                body = html.Div("Keine gespeicherten Modelle gefunden.", className="text-muted")
            else:
                body = dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Datei"),
                            html.Th("Größe"),
                            html.Th("Geändert"),
                        ])),
                        html.Tbody(rows),
                    ],
                    bordered=True, striped=True, hover=True, responsive=True, className="mb-0"
                )
        except Exception as e:
            body = html.Div(f"Fehler beim Auflisten: {e}", className="text-danger")

        return True, body

    # Schließen
    return False, no_update


@app.callback(
    [
        Output("exog-override-table", "data", allow_duplicate=True),
        Output("exog-baseline-store", "data", allow_duplicate=True),
        Output("exog-override-table", "style_data_conditional", allow_duplicate=True),
        Output("exog-override-table", "tooltip_data", allow_duplicate=True),
    ],
    Input("reset-exog-overrides-btn", "n_clicks"),
    State("exog-override-table", "columns"),
    prevent_initial_call=True,
)
def reset_exog_overrides(n_clicks, columns):
    """
    Setzt die Tabelle strikt auf die Baseline aus der output.xlsx zurück
    und aktualisiert Styles/Tooltips sowie den Baseline-Store.
    """
    import dash
    import pandas as pd
    import numpy as np
    from pathlib import Path

    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Mapping
    try:
        display_order: List[str] = EXOG_DISPLAY_ORDER
        key_map: Dict[str, str] = EXOG_VAR_MAP
    except Exception:
        key_map = {
            "Zinssatz (10Y)": "lt_interest_rate",
            "Immobilienpreise": "property_prices",
            "BIP": "gdp",
            "Arbeitslosenquote": "unemployment",
            "Inflation": "inflation",
        }
        display_order = ["Zinssatz (10Y)", "Immobilienpreise", "BIP", "Arbeitslosenquote", "Inflation"]

    # Quartalsspalten
    q_cols = [c.get("id") for c in (columns or []) if isinstance(c, dict) and c.get("id") != "Variable"]
    if not q_cols:
        raise dash.exceptions.PreventUpdate

    # output.xlsx suchen
    def _find_output_excel() -> Path:
        base = Path(__file__).parent
        for p in [
            base / "scenario" / "data" / "output.xlsx",
            base / "scenario" / "data" / "transformed_output.xlsx",
            base / "scenario" / "output.xlsx",
            base / "data" / "output.xlsx",
            base / "output.xlsx",
            Path("/mnt/data/output.xlsx"),
        ]:
            if p.exists():
                return p
        return (Path(__file__).parent / "scenario" / "data" / "output.xlsx")

    xlsx_path = _find_output_excel()
    if not xlsx_path.exists():
        raise dash.exceptions.PreventUpdate

    xl = pd.ExcelFile(xlsx_path)

    def _read_sheet(name: str) -> pd.DataFrame:
        if name not in xl.sheet_names:
            return pd.DataFrame()
        df = pd.read_excel(xl, sheet_name=name, engine="openpyxl")
        dcol = None
        for cand in ["Datum", "datum", "Date", "date"]:
            if cand in df.columns:
                dcol = cand
                break
        if dcol is None:
            return pd.DataFrame()
        df = df.rename(columns={dcol: "Datum"}).copy()
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)
        return df

    df_final = _read_sheet("final_dataset")
    df_ecb   = _read_sheet("raw_ecb")
    df_buba  = _read_sheet("raw_buba")

    EXOG_ALIASES = {
        "lt_interest_rate": [
            "IRS.M.DE.L.L40.CI.0000.EUR.N.Z", "irs.m.de.l.l40.ci.0000.eur.n.z",
            "irs_m_de_l_l40_ci_0000_eur_n_z", "zinssatz_10y", "hauptrefinanzierungssatz",
            "lt_interest_rate",
        ],
        "property_prices": [
            "RESR.Q.DE._T.N._TR.TVAL.10.TB.N.IX", "resr.q.de._t.n._tr.tval.10.tb.n.ix",
            "resr_q_de__t_n__tr_tval_10_tb_n_ix", "immobilienpreise", "property_prices",
        ],
        "gdp": [
            "MNA.Q.N.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N", "mna.q.n.de.w2.s1.s1_b_b1gq._z._z._z.eur.v.n",
            "mna_q_n_de_w2_s1_s1_b_b1gq__z__z__z_eur_v_n", "bruttoinlandsprodukt", "gdp",
        ],
        "unemployment": [
            "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T", "lfsi.m.de.s.unehrt.total0.15_74.t",
            "lfsi_m_de_s_unehrt_total0_15_74_t", "arbeitslosenquote", "unemployment",
        ],
        "inflation": [
            "ICP.M.DE.N.000000.4.ANR", "icp.m.de.n.000000.4.anr",
            "icp_m_de_n_000000_4_anr", "inflation_rate", "inflation",
        ],
    }

    def _first_matching_col(df: pd.DataFrame, aliases) -> Optional[str]:
        if df is None or df.empty:
            return None
        lower = {str(c).strip().lower(): c for c in df.columns}
        for a in aliases:
            k = str(a).strip().lower()
            if k in lower:
                return lower[k]
        return None

    def _to_num(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    # Baseline je Key (final_dataset > raw_ecb > raw_buba)
    latest_vals: Dict[str, Optional[float]] = {}
    for disp in display_order:
        key = key_map.get(disp)
        val = None
        if key and (key in df_final.columns):
            s = _to_num(df_final[key]).dropna()
            if not s.empty:
                val = float(s.iloc[-1])
        if val is None and not df_ecb.empty:
            col = _first_matching_col(df_ecb, EXOG_ALIASES[key])
            if col is not None:
                s = _to_num(df_ecb[col]).dropna()
                if not s.empty:
                    val = float(s.iloc[-1])
        if val is None and not df_buba.empty:
            col = _first_matching_col(df_buba, EXOG_ALIASES[key])
            if col is not None:
                s = _to_num(df_buba[col]).dropna()
                if not s.empty:
                    val = float(s.iloc[-1])
        latest_vals[key] = val

    # Datenzeilen neu aufbauen
    rows = []
    for disp in display_order:
        key = key_map.get(disp)
        base_val = latest_vals.get(key, None)
        r = {"Variable": disp}
        for qc in q_cols:
            r[qc] = base_val
        rows.append(r)

    baseline_store = {
        "by_display": {disp: latest_vals.get(key_map.get(disp)) for disp in display_order},
        "by_key": latest_vals
    }

    styles, tooltips = _compute_exog_table_styles_and_tooltips(columns, rows, baseline_store, exog_var_map=key_map)
    return rows, baseline_store, styles, tooltips



@app.callback(
    [
        Output("exog-override-table", "style_data_conditional", allow_duplicate=True),
        Output("exog-override-table", "tooltip_data", allow_duplicate=True),
    ],
    [
        Input("exog-override-table", "data"),
        Input("exog-override-table", "columns"),
        Input("exog-baseline-store", "data"),
    ],
    # allow_duplicate=True verlangt prevent_initial_call True oder "initial_duplicate"
    prevent_initial_call=True,
)
def update_exog_table_visuals_live(rows, columns, baseline_store):
    """
    Rechnet Styles/Tooltips neu, wenn der User in der Tabelle Werte ändert,
    Spalten wechseln (z. B. andere Quartale) oder die Baseline aktualisiert wurde.

    Erwartet in baseline_store (Option B):
      {
        "by_display_q": {"<Display>": [q1,q2,q3,q4], ...},
        "q_cols":       ["Q+1","Q+2","Q+3","Q+4"],
        ...
      }
    Fallback: unterstützt weiterhin alte Struktur mit "by_display" (Skalar-Baseline).
    """
    import dash
    from dash import exceptions as dash_exc

    # Defensive Guards: ohne Spalten oder Zeilen keine Styles/Tooltips setzen
    if not columns or not isinstance(columns, list):
        raise dash_exc.PreventUpdate
    if not rows or not isinstance(rows, list):
        # Keine Daten → keine Hervorhebung / keine Tooltips
        return [], []

    # Wenn der Store fehlt, ebenfalls neutral bleiben
    if baseline_store is None or not isinstance(baseline_store, dict):
        return [], []

    try:
        styles, tooltips = _compute_exog_table_styles_and_tooltips(columns, rows, baseline_store)
        return styles, tooltips
    except Exception as e:
        # Im Fehlerfall lieber neutral bleiben, statt falsche Färbungen zu zeigen
        Log.error(f"[ScenarioStyles] ERROR in update_exog_table_visuals_live: {e}")
        return [], []



from pathlib import Path
import pandas as pd
import dash
from dash.exceptions import PreventUpdate

def _find_scenario_excel_only_data_dir() -> Path | None:
    """
    Sucht ausschließlich unterhalb von scenario/data nach einer Excel-Datei,
    die ein Sheet 'final_dataset' enthält.
    Root-Dateien oder Dateien in scenario/ selbst werden bewusst ignoriert.
    """
    base = Path(__file__).parent          # .../scenario
    data_dir = base / "data"              # .../scenario/data

    # unsere beiden legitimen Kandidaten in genau dieser Reihenfolge
    candidates = [
        data_dir / "analysis_data.xlsx",
        data_dir / "output.xlsx",
    ]

    for f in candidates:
        if not f.exists():
            continue
        try:
            xl = pd.ExcelFile(f)
        except Exception:
            # kaputt → nächster
            continue
        if "final_dataset" in xl.sheet_names:
            return f

    # nichts Passendes gefunden
    return None



def _create_error_response(error: Exception, context: str = "") -> Tuple[go.Figure, html.Div, go.Figure]:
    """Erstellt einheitliche Error-Response für Scenario-Callback"""
    import traceback
    
    error_msg = str(error)
    traceback_str = traceback.format_exc()
    
    # Log full traceback
    Log.error(f"[ERROR] {context}")
    Log.error(f"[ERROR] Message: {error_msg}")
    Log.error(f"[ERROR] Traceback:\n{traceback_str}")
    
    # Empty figure mit Fehlertext
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title=f"Fehler: {context or 'Nowcasting-Analyse'}",
        template="plotly_white",
        height=560,
        annotations=[{
            'text': f'<b>Fehler:</b><br>{error_msg}<br><br>Details siehe Konsole',
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': 0.5,
            'showarrow': False,
            'font': {'color': 'red', 'size': 12},
            'align': 'center'
        }]
    )
    
    # Detaillierte Alert-Box
    kpis = dbc.Alert([
        html.H5("Fehler bei der Analyse", className="alert-heading"),
        html.P(error_msg, className="mb-3"),
        html.Hr(),
        html.P("Mögliche Ursachen:", className="mb-2 fw-bold"),
        html.Ul([
            html.Li("Zu wenig historische Daten (min. 20 Quartale benötigt)"),
            html.Li("Ungültige oder fehlende Eingaben in der Tabelle"),
            html.Li("Fehlende Spalten in analysis_data.xlsx"),
            html.Li("Inkompatible Modell-Konfiguration"),
        ], className="mb-3"),
        html.P("Bitte prüfen Sie:", className="mb-2 fw-bold"),
        html.Ul([
            html.Li("Console-Logs für Details (F12 → Console)"),
            html.Li("Eingabe-Tabelle auf korrekte Werte"),
            html.Li("Verfügbarkeit von analysis_data.xlsx"),
        ]),
        html.Hr(),
        html.Small(f"Technical Details: {context}", className="text-muted")
    ], color="danger")
    
    return empty_fig, kpis, go.Figure()


@app.callback(
    Output("scenario-preset-dropdown", "value"),
    Input("reset-exog-overrides-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_scenario_preset_value(n_clicks):
    import dash
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return "baseline"

from pathlib import Path
from typing import Optional, Dict, Any, List

from pathlib import Path
from typing import Dict, List, Optional, Any

def _create_analysis_data_from_scenarios(
    base_excel_path: Path,
    manual_vals: Dict[str, List[Optional[float]]],
    quarter_labels: List[str],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Baut eine analysis_data.xlsx auf Basis der vorhandenen Szenario-Excel (output.xlsx).
    Ziel:
    - Sheet 'final_dataset' muss die 5 kanonischen Exogenen enthalten:
        lt_interest_rate, property_prices, gdp, unemployment, inflation
    - Falls sie in der Quell-Excel fehlen, werden sie aus raw_ecb / raw_buba
      geholt oder notfalls leer angelegt und mit dem letzten Wert gefüllt.
    - Die per UI übergebenen Szenario-Quartale (quarter_labels) werden als
      Monate an das Ende angehängt und dort mit manual_vals überschrieben.
    """

    import pandas as pd
    import numpy as np

    # --------------------------------------------------
    # kleiner Logging-Helper (failsafe)
    # --------------------------------------------------
    def _log(msg: str):
        try:
            Log.scenario(f"[CreateAnalysis] {msg}")  # type: ignore
        except Exception:
            try:
                Log.scenario_table(f"[CreateAnalysis] {msg}")  # type: ignore
            except Exception:
                pass

    _log(f"Enter | base={base_excel_path}")
    _log(f"Quarter labels: {quarter_labels}")

    # --------------------------------------------------
    # 0) Zielpfad: NUR ./scenario/data, NICHT ./scenario/scenario/data
    # --------------------------------------------------
    if output_path is None:
        data_dir = Path(__file__).parent / "data"   # <- richtig: .../scenario/data
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "analysis_data.xlsx"

    # --------------------------------------------------
    # 1) Basisdatei öffnen & final_dataset laden
    # --------------------------------------------------
    xl = pd.ExcelFile(base_excel_path)
    if "final_dataset" not in xl.sheet_names:
        raise FileNotFoundError(
            f"Basisdatei {base_excel_path} enthält kein Sheet 'final_dataset'."
        )

    df_base = pd.read_excel(xl, sheet_name="final_dataset")
    # Datumsspalte normalisieren
    date_col = None
    for cand in ("Datum", "datum", "Date", "date"):
        if cand in df_base.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("Keine Datumsspalte im Sheet 'final_dataset' gefunden.")
    df_base = df_base.rename(columns={date_col: "Datum"}).copy()
    df_base["Datum"] = pd.to_datetime(df_base["Datum"], errors="coerce")
    df_base = df_base.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)

    # störende Hilfsspalten loswerden
    if "gvb_imputed" in df_base.columns:
        df_base = df_base.drop(columns=["gvb_imputed"])

    # --------------------------------------------------
    # 2) evtl. vorhandene Roh-Sheets einlesen
    # --------------------------------------------------
    def _read_sheet(name: str) -> pd.DataFrame:
        if name not in xl.sheet_names:
            return pd.DataFrame()
        df = pd.read_excel(xl, sheet_name=name, engine="openpyxl")
        dcol = None
        for cand in ("Datum", "datum", "Date", "date"):
            if cand in df.columns:
                dcol = cand
                break
        if dcol is None:
            return pd.DataFrame()
        df = df.rename(columns={dcol: "Datum"})
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"]).sort_values("Datum")
        return df

    df_ecb = _read_sheet("raw_ecb")
    df_buba = _read_sheet("raw_buba")

    # --------------------------------------------------
    # 3) Kanonische Exog-Definition + Aliase
    # --------------------------------------------------
    CANON_EXOG: Dict[str, List[str]] = {
        "lt_interest_rate": [
            "lt_interest_rate",
            "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
            "irs.m.de.l.l40.ci.0000.eur.n.z",
            "zinssatz_10y",
            "hauptrefinanzierungssatz",
        ],
        "property_prices": [
            "property_prices",
            "RESR.Q.DE._T.N._TR.TVAL.10.TB.N.IX",
            "immobilienpreise",
        ],
        "gdp": [
            "gdp",
            "MNA.Q.N.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
            "bruttoinlandsprodukt",
        ],
        "unemployment": [
            "unemployment",
            "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
            "arbeitslosenquote",
        ],
        "inflation": [
            "inflation",
            "ICP.M.DE.N.000000.4.ANR",
            "inflation_rate",
        ],
    }

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        if df is None or df.empty:
            return None
        norm_map = {_norm(c): c for c in df.columns}
        # exakte Normalform
        for a in aliases:
            na = _norm(a)
            if na in norm_map:
                return norm_map[na]
        # weicher Fallback
        for a in aliases:
            na = _norm(a)
            if len(na) < 5:
                continue
            for nc, orig in norm_map.items():
                if na in nc or nc in na:
                    return orig
        return None

    # --------------------------------------------------
    # 4) Exogene aus ECB/BuBa/final_dataset zusammensetzen
    # --------------------------------------------------
    exog_frames = []
    for key, aliases in CANON_EXOG.items():
        series = None
        src = None

        # 1) ECB
        if not df_ecb.empty:
            col = _find_col(df_ecb, aliases)
            if col:
                tmp = df_ecb[["Datum", col]].copy()
                tmp.rename(columns={col: key}, inplace=True)
                series = tmp
                src = f"raw_ecb::{col}"

        # 2) BuBa
        if series is None and not df_buba.empty:
            col = _find_col(df_buba, aliases)
            if col:
                tmp = df_buba[["Datum", col]].copy()
                tmp.rename(columns={col: key}, inplace=True)
                series = tmp
                src = f"raw_buba::{col}"

        # 3) final_dataset
        if series is None and key in df_base.columns:
            tmp = df_base[["Datum", key]].copy()
            series = tmp
            src = "final_dataset"

        # 4) gar nichts gefunden → leere Serie über Basis-Zeiten
        if series is None:
            series = pd.DataFrame({"Datum": df_base["Datum"], key: np.nan})
            src = "generated_na"

        _log(f"exog map: {key} <- {src}")
        exog_frames.append(series)

    # alle Exogs in einen DF bringen
    exog_df = None
    for f in exog_frames:
        if exog_df is None:
            exog_df = f.copy()
        else:
            exog_df = exog_df.merge(f, on="Datum", how="outer")

    exog_df = exog_df.sort_values("Datum").reset_index(drop=True)

    # --------------------------------------------------
    # 5) Merge Basis + Exogs
    # --------------------------------------------------
    df_merged = df_base.merge(exog_df, on="Datum", how="left")

    # fehlende Exog-Spalten sicherstellen
    for key in CANON_EXOG.keys():
        if key not in df_merged.columns:
            df_merged[key] = np.nan

    # --------------------------------------------------
    # 6) Wenn keine Szenario-Quartale → einfach schreiben
    # --------------------------------------------------
    if not quarter_labels:
        df_merged.to_excel(output_path, sheet_name="final_dataset", index=False)
        _log(f"Written (history only): {output_path}")
        return output_path

    # --------------------------------------------------
    # 7) Szenario-Quartale in Monatsdaten umwandeln
    # --------------------------------------------------
    # Helper, der ein Quartalslabel "2026-Q1" in Quartalsende-Timestamp wandelt.
    def _label_to_qend(label: str) -> Optional[pd.Timestamp]:
        try:
            year_str, q_str = label.split("-Q")
            year = int(year_str)
            q = int(q_str)
            return pd.Period(f"{year}Q{q}", freq="Q").to_timestamp(how="end")
        except Exception:
            return None

    def _monthly_dates_in_quarter(qend: pd.Timestamp) -> List[pd.Timestamp]:
        # letztes Quartal: qend = 2026-03-31 → wir wollen Jan, Feb, Mär
        start = (qend - pd.offsets.QuarterEnd(0)).to_period("Q").to_timestamp(how="start")
        return list(pd.date_range(start, qend, freq="M"))

    # abhängige/spätere Spalten, die wir nicht füllen wollen
    dep_vars = ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite", "Gesamt GVB"]
    dep_cols = [c for c in dep_vars if c in df_merged.columns]

    # manuellen Dict cleanen
    manual_vals = {str(k): (v or []) for k, v in (manual_vals or {}).items()}

    new_rows: List[Dict[str, Any]] = []

    for qi, qlabel in enumerate(quarter_labels):
        qend = _label_to_qend(qlabel)
        if qend is None:
            continue
        for mdate in _monthly_dates_in_quarter(qend):
            row: Dict[str, Any] = {"Datum": mdate}

            # 7a) Exogene: UI-Wert oder letzter Historienwert
            for key in CANON_EXOG.keys():
                # UI könnte mit kanonischen keys gekommen sein
                ui_vals = manual_vals.get(key)
                if ui_vals and qi < len(ui_vals) and ui_vals[qi] is not None:
                    row[key] = float(ui_vals[qi])
                    continue

                # ansonsten letzten Wert aus Historie nehmen
                last = pd.to_numeric(df_merged[key], errors="coerce").dropna()
                row[key] = float(last.iloc[-1]) if not last.empty else np.nan

            # 7b) abhängige Spalten auf NaN lassen (werden später berechnet)
            for c in dep_cols:
                row[c] = np.nan

            # 7c) alle übrigen Spalten konservativ auf letzten Wert setzen
            for c in df_merged.columns:
                if c in row or c == "Datum":
                    continue
                last = pd.to_numeric(df_merged[c], errors="coerce").dropna()
                row[c] = float(last.iloc[-1]) if not last.empty else np.nan

            new_rows.append(row)

    df_scen = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=df_merged.columns)

    # Spalten angleichen und zusammenführen
    all_cols = list(dict.fromkeys(list(df_merged.columns) + list(df_scen.columns)))
    df_merged = df_merged.reindex(columns=all_cols)
    df_scen   = df_scen.reindex(columns=all_cols)
    df_out    = pd.concat([df_merged, df_scen], ignore_index=True)

    # --------------------------------------------------
    # 8) Schreiben
    # --------------------------------------------------
    df_out.to_excel(output_path, sheet_name="final_dataset", index=False)
    _log(f"Written (history + scenario): {output_path}")
    return output_path










def _trim_to_last_complete_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trimmt DataFrame auf den letzten Zeitpunkt, wo alle relevanten Spalten 
    (Exogene + GVB-Komponenten) nicht-NaN sind.
    
    Args:
        df: DataFrame mit 'Datum'-Spalte
        
    Returns:
        Getrimmter DataFrame
    """
    if df.empty or "Datum" not in df.columns:
        return df
    
    df = df.sort_values("Datum").reset_index(drop=True)
    
    # Relevante Spalten identifizieren
    exog_cols = [c for c in EXOG_VAR_MAP.values() if c in df.columns]
    gvb_components = ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite"]
    gvb_cols = [c for c in gvb_components if c in df.columns]
    
    # Falls Gesamt GVB vorhanden, auch berücksichtigen
    if "Gesamt GVB" in df.columns:
        gvb_cols.append("Gesamt GVB")
    
    check_cols = exog_cols + gvb_cols
    
    if not check_cols:
        Log.warn("[Trim] WARN: Keine relevanten Spalten zum Trimmen gefunden")
        return df
    
    Log.scenario_table(f"[Trim] Checking completeness for columns: {check_cols}")
    
    # Von hinten nach vorne: Finde letzte vollständige Zeile
    last_complete_idx = None
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        if all(pd.notna(row[col]) for col in check_cols):
            last_complete_idx = idx
            break
    
    if last_complete_idx is None:
        Log.warn("[Trim] WARN: Keine vollständige Zeile gefunden - behalte alle Daten")
        return df
    
    cutoff_date = df.loc[last_complete_idx, "Datum"]
    Log.scenario_table(f"[Trim] Last complete row at index={last_complete_idx}, date={cutoff_date}")
    
    # Auf diese Zeile (inkl.) kürzen
    df_trimmed = df.iloc[:last_complete_idx + 1].copy()
    
    removed = len(df) - len(df_trimmed)
    if removed > 0:
        Log.scenario_table(f"[Trim] Removed {removed} incomplete rows from end")
    
    return df_trimmed



from typing import Optional
import re
import pandas as pd

# Einmal global (vermeidet jedes Mal Neu-Kompilieren)
_Q_LABEL_RE = re.compile(r"^\s*(\d{4})\s*-?\s*[Qq]\s*([1-4])\s*$")

def _label_to_ts_qend(lab: str) -> Optional[pd.Timestamp]:
    """
    Konvertiert Quarter-Label (z.B. '2024-Q3', '2024Q3', '2024 q3') in das Quartalsende.
    Nutzt Foundation-Helper get_quarter_end_date(year, quarter).
    """
    s = "" if lab is None else str(lab).strip()
    m = _Q_LABEL_RE.match(s)
    if not m:
        return None
    year = int(m.group(1))
    quarter = int(m.group(2))
    try:
        return get_quarter_end_date(year, quarter)  # Foundation-Helper statt _quarter_end
    except Exception:
        return None












# --- Datenset beim Seitenaufruf in Store laden (falls benötigt) ----------------
@app.callback(
    Output("scenario-data-store", "data"),
    Input("url", "pathname"),
    prevent_initial_call=False
)
def load_scenario_data_on_nav(pathname):
    if pathname not in {"/scenario", "/scenario-analysis"}:
        raise dash.exceptions.PreventUpdate
    try:
        df = _load_scenario_final_dataset()
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"⚠️ Szenario-Daten konnten nicht geladen werden: {e}")
        return None





def _find_analysis_excel() -> Path | None:
    """
    Bevorzugt scenario/data/analysis_data.xlsx.
    Optional: nimm die jüngste *.xlsx in scenario/data, falls der Standardname fehlt.
    """
    if DEFAULT_ANALYSIS_XLSX.exists():
        return DEFAULT_ANALYSIS_XLSX

    # Fallback: jüngste Excel in scenario/data
    cands = sorted(SCENARIO_DATA_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


from pathlib import Path
import pandas as pd


def _find_analysis_or_scenario_excel() -> Path:
    """
    Sucht NUR im Unterordner scenario/data – aber prüft,
    ob die gefundene Datei wirklich für die Szenario-Analyse taugt.

    Reihenfolge:
    1) analysis_data.xlsx → aber nur, wenn sie ein Sheet 'final_dataset'
       UND die 5 Exog-Spalten hat
    2) output.xlsx → falls analysis_data.xlsx fehlt oder unvollständig ist,
       wird daraus direkt eine neue analysis_data.xlsx gebaut
    3) scenario_overrides.xlsx → nur als Notlösung
    4) sonst: jüngste *.xlsx in scenario/data
    """
    import pandas as pd  # lokal, damit die Funktion auch alleine funktioniert

    # Hilfsprüfer ---------------------------------------------------------
    def _is_valid_analysis(path: Path) -> bool:
        """
        prüft, ob path ein 'final_dataset' + alle 5 Exog-Spalten hat.
        """
        if not path.exists():
            return False
        try:
            xl = pd.ExcelFile(path)
            if "final_dataset" not in xl.sheet_names:
                return False
            df = pd.read_excel(xl, sheet_name="final_dataset", nrows=5, engine="openpyxl")
        except Exception:
            return False

        needed = [
            "Datum",
            "lt_interest_rate",
            "property_prices",
            "gdp",
            "unemployment",
            "inflation",
        ]
        return all(col in df.columns for col in needed)

    # 1) zuerst: gibt es schon eine gültige analysis_data.xlsx?
    if DEFAULT_ANALYSIS_XLSX.exists() and _is_valid_analysis(DEFAULT_ANALYSIS_XLSX):
        return DEFAULT_ANALYSIS_XLSX

    # 2) analysis fehlt oder ist unvollständig → können wir aus output.xlsx bauen?
    if DEFAULT_OUTPUT_XLSX.exists():
        # verwendet deine Hilfsfunktion zum Erzeugen
        _create_analysis_data_from_scenarios(
            base_excel_path=DEFAULT_OUTPUT_XLSX,
            manual_vals={},        # keine Overrides beim Start
            quarter_labels=[],     # keine extra Quartale beim Start
            output_path=DEFAULT_ANALYSIS_XLSX,
        )
        return DEFAULT_ANALYSIS_XLSX

    # 3) erst dann: scenario_overrides.xlsx (hat oft KEIN final_dataset,
    #    aber wir geben sie wenigstens zurück)
    if DEFAULT_OVERRIDES_XLSX.exists():
        return DEFAULT_OVERRIDES_XLSX

    # 4) letzter Fallback: irgendeine xlsx aus scenario/data
    cands = sorted(
        SCENARIO_DATA_DIR.glob("*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]

    # wenn gar nichts da ist → klarer Fehler
    raise FileNotFoundError("Keine Szenario-/Analyse-Datei in scenario/data gefunden.")


def _save_uploaded_excel(content_bytes: bytes, filename: str) -> Path:
    target = scenario_data_path(filename)
    target.write_bytes(content_bytes)
    return target


# -------------------------------------------------------------
# Helper: Style & Tooltip aus Baseline berechnen (vollständig)
# -------------------------------------------------------------

def _compute_exog_table_styles_and_tooltips(table_columns, table_rows, baseline_store, exog_var_map=None):
    """
    Erzeugt style_data_conditional + tooltip_data für die exog-override-table.

    - table_columns: DataTable columns (List[dict])
    - table_rows:    DataTable data (List[dict]) – jede Zeile: {"Variable": <Display>, "Q+1": 123.4, ...}
    - baseline_store:
        * Neu (Option B):
            {
              "by_display_q": {"Zinssatz (10Y)": [q1,q2,q3,q4], ...},
              "q_cols":       ["Q+1","Q+2","Q+3","Q+4"],
              ... (weitere Metadaten)
            }
        * Alt (Fallback):
            {
              "by_display": {"Zinssatz (10Y)": <Skalar>, ...}
            }
    - exog_var_map: Mapping Display -> canonical key (optional) (nicht benötigt für Styling)

    Rückgabe: (style_data_conditional: List[dict], tooltip_data: List[dict])
    """
    import math
    import numpy as np

    # Farb-Palette (dezent)
    COLORS = {
        "green_strong": "#C8E6C9",   # kräftigeres Grün
        "green_light":  "#E8F5E9",   # leichtes Grün
        "red_strong":   "#FFCDD2",   # kräftigeres Rot
        "red_light":    "#FFEBEE",   # leichtes Rot
        "neutral":      "transparent"
    }

    # Spalten-IDs der Quartale (alles außer 'Variable')
    quarter_cols = [c.get("id") for c in (table_columns or []) if isinstance(c, dict) and c.get("id") != "Variable"]

    # --- Baseline aus dem Store laden ---
    by_display_q = {}
    q_cols_store = None
    base_by_display_scalar = {}

    if isinstance(baseline_store, dict):
        # Neue, quartalsgenaue Struktur
        by_display_q = baseline_store.get("by_display_q") or {}
        q_cols_store = baseline_store.get("q_cols") or None
        # Rückwärtskompatibel: ggf. alte Skalar-Struktur
        base_by_display_scalar = baseline_store.get("by_display", {}) or {}

    # Mapping Spalte -> Quartalsindex bestimmen
    # Priorität: Reihenfolge aus dem Store (q_cols) falls vorhanden, sonst aktuelle Tabellen-Reihenfolge.
    if q_cols_store and all(col in quarter_cols for col in q_cols_store):
        col_to_qidx = {col: idx for idx, col in enumerate(q_cols_store)}
    else:
        col_to_qidx = {col: idx for idx, col in enumerate(quarter_cols)}

    styles = []
    tooltips = []

    # Hilfsfunktion: Delta → Farbe
    def _color_for_delta(base, val):
        try:
            if base is None or (isinstance(base, float) and math.isnan(base)) or val is None or (isinstance(val, float) and math.isnan(val)):
                return COLORS["neutral"]
            if float(base) == 0.0:
                # Bei Baseline 0 nur absolute Abweichung leicht einfärben
                ad = abs(float(val) - float(base))
                if ad == 0:
                    return COLORS["neutral"]
                return COLORS["green_light"] if (val - base) > 0 else COLORS["red_light"]
            pct = (float(val) - float(base)) / abs(float(base)) * 100.0
            # Schwellenwerte
            if pct >= 10.0:
                return COLORS["green_strong"]
            if 5.0 <= pct < 10.0:
                return COLORS["green_light"]
            if -10.0 <= pct <= -5.0:
                return COLORS["red_light"]
            if pct < -10.0:
                return COLORS["red_strong"]
            return COLORS["neutral"]
        except Exception:
            return COLORS["neutral"]

    # Hilfsfunktionen: Formatierung
    def _fmt_num(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "—"
            return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    def _fmt_pct(b, v):
        try:
            if b is None or v is None or (isinstance(b, float) and math.isnan(b)) or (isinstance(v, float) and math.isnan(v)):
                return "—"
            if float(b) == 0.0:
                return "—"
            p = (float(v) - float(b)) / abs(float(b)) * 100.0
            sign = "+" if p >= 0 else ""
            return f"{sign}{p:.1f}%"
        except Exception:
            return "—"

    def _tooltip_text(base, val):
        base_s = _fmt_num(base)
        val_s  = _fmt_num(val)
        if base is None or (isinstance(base, float) and math.isnan(base)) or val is None or (isinstance(val, float) and math.isnan(val)):
            delta_abs = "—"
        else:
            try:
                delta = float(val) - float(base)
                sign = "+" if delta >= 0 else ""
                delta_abs = f"{sign}{delta:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                delta_abs = "—"
        delta_pct = _fmt_pct(base, val)
        return f"Baseline: {base_s}\nWert: {val_s}\nΔ: {delta_abs} ({delta_pct})"

    # Tooltips als Liste pro Zeile: dict(column_id -> text)
    for r_idx, row in enumerate(table_rows or []):
        disp = row.get("Variable")

        # quartalsgenaue Baseline-Serie (Prio) oder Skalar (Fallback)
        base_series = by_display_q.get(disp)
        base_scalar = base_by_display_scalar.get(disp)

        # Zeilen-Tooltip-Header
        # (zeigt – wenn vorhanden – die 4 Baseline-Werte; sonst den Skalar)
        if isinstance(base_series, (list, tuple)) and len(base_series) >= 4:
            base_header = "[" + ", ".join(_fmt_num(b) for b in base_series[:4]) + "]"
        else:
            base_header = _fmt_num(base_scalar)

        tip_row = {"Variable": f"Baseline {disp}: {base_header}"}

        for qc in quarter_cols:
            val = row.get(qc)

            # Baseline pro Zelle wählen:
            base = None
            if isinstance(base_series, (list, tuple)) and len(base_series) >= 4:
                qi = col_to_qidx.get(qc, None)
                if qi is not None and 0 <= qi < len(base_series):
                    base = base_series[qi]
            else:
                # Fallback: alter Skalar (führt zu Einfärbung, wenn sich Werte unterscheiden)
                base = base_scalar

            # Style-Regel pro Zelle
            styles.append({
                "if": {"row_index": r_idx, "column_id": qc},
                "backgroundColor": _color_for_delta(base, val)
            })
            # Tooltip-Text pro Zelle
            tip_row[qc] = _tooltip_text(base, val)

        tooltips.append(tip_row)

    return styles, tooltips

# =================== Ende Szenario-Analyse: Funktionen & Callbacks =============





def collect_scenario_callbacks():
    """Gibt die im Proxy gesammelten Callback-Deklarationen zurück."""
    return list(getattr(app, "_registrations", []))


def register_scenario_callbacks(real_app: "dash.Dash", Log):
    """
    Registriert alle in diesem Modul deklarierten Szenario-Callbacks auf der echten Dash-App.
    Aufruf: in app.py NACH der App-Erzeugung.
    """
    for args, kwargs, fn in collect_scenario_callbacks():
        real_app.callback(*args, **kwargs)(fn)

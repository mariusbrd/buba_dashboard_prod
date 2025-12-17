# --- Zusätzliche Standard-Imports ---
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import callback_context  # optional, falls du lieber callback_context statt ctx nutzt


# --- Pfad-Logik: Overview-Ordner + App-Root ---
OVERVIEW_DIR = Path(__file__).resolve().parent  # .../overview

try:
    APP_ROOT: Path = OVERVIEW_DIR.parent        # Projektroot (dort liegt typischerweise app.py)
except Exception:
    APP_ROOT = Path.cwd()


# --- Safe logging shim: vermeidet NameError für Log/logger ---
import logging
logger = logging.getLogger("GVB_Dashboard")

class _SafeLog:
    def info(self, msg): logger.info(msg)
    def warn(self, msg): logger.warning(msg)
    def error(self, msg): logger.error(msg)
    def scenario(self, msg): logger.info(msg)
    def scenario_table(self, msg): logger.info(msg)

# Platzhalter (werden durch app.py via register_overview_callbacks injiziert)
_DataManager = None
_StoreSource = None
_DiskSource  = None
Log = globals().get("Log", _SafeLog())

# (Optional) Projekt-Helper — wenn in deiner Umgebung verfügbar
try:
    from foundation import year_tickvals_biennial, window_from_slider, map_sektor
    from foundation.colors import get_category_color, get_hierarchical_color, GVB_COLOR_SEQUENCE
    from data_manager import DataManager, StoreSource, DiskSource
except Exception:
    pass



# --- (Optional) Projekt-Helper importieren; bei Dev-Umgebungen defensiv ---
try:
    # Passe diese Pfade an deine tatsächliche Struktur an
    # Passe diese Pfade an deine tatsächliche Struktur an
    from foundation import year_tickvals_biennial, window_from_slider, map_sektor
    from foundation.colors import get_category_color, get_hierarchical_color, GVB_COLOR_SEQUENCE
    from data_manager import DataManager, StoreSource, DiskSource
    from main.app import format_axis_quarters # <-- NEU
except Exception:
    # In manchen Setups kommen diese aus 'app' oder anderen Paketen; notfalls später via try/except in den Funktionen
    pass


# --- Callback-Proxy: sammelt @app.callback(...) Deklarationen für spätere Registrierung
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table






def format_axis_quarters(fig, date_iterable):
    """
    Formatiert die X-Achse als KATEGORISCH (String-basiert), um "Qx YYYY"
    als Header im Hover zu erzwingen, während die visuelle Sortierung erhalten bleibt.
    Ticks weiterhin nur alle 5 Jahre (z.B. Q1 2020).
    """
    try:
        if date_iterable is None or len(date_iterable) == 0:
            return

        dt_index = pd.to_datetime(list(date_iterable))
        if dt_index.empty:
            return

        min_date = dt_index.min()
        max_date = dt_index.max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return

        # 1. Master-Timeline erstellen (Alle Quartale im Bereich)
        # Damit Plotly die richtige Reihenfolge kennt (Nicht alphabetisch Q1 2021 < Q4 2020)
        try:
            start_q = pd.Timestamp(min_date).to_period('Q').start_time
            end_q = pd.Timestamp(max_date).to_period('Q').end_time
        except Exception:
            return
            
        full_qs = pd.date_range(start=start_q, end=end_q, freq='QS')
        if len(full_qs) == 0:
            return

        # Mapping: Timestamp -> "Qx YYYY"
        # UND: Erstellen der geordneten Kategorie-Liste
        def to_q_str(d):
            return f"Q{d.quarter} {d.year}"
            
        category_order = [to_q_str(d) for d in full_qs]
        
        # 2. Alle Traces auf String-Werte mappen
        for trace in fig.data:
            if getattr(trace, 'x', None) is None:
                continue
            try:
                # Altdaten (Datetimes) zu Strings konvertieren
                ts_series = pd.to_datetime(pd.Series(trace.x), errors='coerce')
                # NaT ignorieren/leeren
                new_x = ts_series.apply(lambda x: to_q_str(x) if pd.notna(x) else None).tolist()
                trace.x = new_x
                
                # Hovertemplate bereinigen (Header macht jetzt den Job)
                trace.hovertemplate = "%{fullData.name}: %{y}<extra></extra>"
            except Exception:
                pass

        # 3. Ticks berechnen (5 Jahres Abstand)
        # Wir suchen die Strings "Q1 2020", "Q1 2025" etc.
        min_year = min_date.year
        max_year = max_date.year
        tick_vals = []
        
        years = range(min_year, max_year + 1)
        span = max_year - min_year
        
        target_years = []
        if span >= 5:
            target_years = [y for y in years if y % 5 == 0]
        else:
            target_years = list(years)
            
        # Wir setzen den Tick genau auf das String-Label "Q1 YYYY"
        for y in target_years:
            val = f"Q1 {y}"
            if val in category_order:
                tick_vals.append(val)
        
        # 4. Layout Update
        fig.update_xaxes(
            type='category',
            categoryorder='array',
            categoryarray=category_order,
            
            tickmode='array',
            tickvals=tick_vals, # Die String-Literale sind jetzt die Koordinaten
            tickangle=0
        )
            
    except Exception:
        pass







class _CallbackProxy:
    """Sammelt @app.callback(...) Deklarationen; wird später auf die echte Dash-App gemappt."""
    def __init__(self):
        self._registrations = []  # [(args, kwargs, fn), ...]

    def callback(self, *args, **kwargs):
        def _decorator(func):
            self._registrations.append((args, kwargs, func))
            return func
        return _decorator

# Wichtig: 'app' hier ist absichtlich KEINE echte Dash-App, sondern der Proxy!
app = _CallbackProxy()







@app.callback(
    [Output('chart-type-area', 'active'),
     Output('chart-type-bar', 'active')],
    [Input('chart-type-area', 'n_clicks'),
     Input('chart-type-bar', 'n_clicks')]
)
def update_chart_type_buttons(area_clicks, bar_clicks):
    """Aktualisiert den aktiven Status der Chart-Type Buttons"""
    # Nutze das bereits importierte 'ctx' aus dash
    if not ctx.triggered:
        return True, False  # Default: Area aktiv
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return (button_id == 'chart-type-area', button_id == 'chart-type-bar')

# ==============================================================================
# DATEN-INITIALISIERUNG (Thread-sicher via Store)
# ==============================================================================

# ==============================================================================
# HELPER: Synthetische Fallback-Daten für initialize_data_stores
# ==============================================================================

def _create_synthetic_gvb_data() -> dict:
    """
    Erstellt minimale synthetische GVB-Daten als JSON-kompatibles Dict.
    Wird als Fallback verwendet, wenn gvb_output.xlsx nicht geladen werden kann.
    
    Returns:
        Dict mit {"data": [...]} für StoreSource
    """
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="Q")
    
    categories = ["Einlagen", "Wertpapiere", "Versicherungen", "Kredite"]
    subcategories = {
        "Einlagen": ["Sichteinlagen", "Termineinlagen", "Spareinlagen"],
        "Wertpapiere": ["Aktien", "Investmentfonds", "Schuldverschreibungen"],
        "Versicherungen": ["Versicherungen", "Alterssicherung"],
        "Kredite": ["Kredite", "Sonstige Verbindlichkeiten"],
    }
    
    rng = np.random.default_rng(42)
    rows = []
    
    for dt in dates:
        for cat in categories:
            for sub in subcategories[cat]:
                rows.append({
                    "date": dt.isoformat(),
                    "ebene1": cat,
                    "ebene2": sub,
                    "ebene3": sub,
                    "bestand": float(rng.uniform(50, 500)),
                    "fluss": float(rng.uniform(-20, 20)),
                    "datatype": "bestand",
                    "sektor": "PH"
                })
    
    return {"data": rows}


def _create_synthetic_exog_data() -> dict:
    """
    Erstellt minimale synthetische Exog-Daten als JSON-kompatibles Dict.
    Wird als Fallback verwendet, wenn echte Exog-Daten nicht verfügbar sind.
    
    Returns:
        Dict mit {"data": [...]} für StoreSource
    """
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="Q")
    rng = np.random.default_rng(42)
    
    rows = []
    for dt in dates:
        rows.append({
            "date": dt.isoformat(),
            "zinssatz_10y": float(rng.uniform(0.5, 3.0)),
            "inflation_rate": float(rng.uniform(1.0, 4.0)),
            "arbeitslosenquote": float(rng.uniform(4.0, 8.0)),
            "verfuegbares_einkommen": float(100 + rng.uniform(-10, 20)),
            "immobilienpreise": float(100 + rng.uniform(-15, 25)),
            "hauptrefinanzierungssatz": float(rng.uniform(0.0, 2.5)),
            "bruttoinlandsprodukt": float(100 + rng.uniform(-5, 10)),
            "sparquote": float(rng.uniform(8.0, 15.0)),
        })
    
    return {"data": rows}

@app.callback(
    [
        Output('gvb-data-store', 'data'),
        Output('exog-data-store', 'data', allow_duplicate=True),
        Output('data-metadata-store', 'data')
    ],
    Input('url', 'pathname'),
    State('gvb-data-store', 'data'),
    prevent_initial_call='initial_duplicate'  # <- genau so
)
def initialize_data_stores(pathname, existing_data):
    """
    Lädt Daten einmalig pro User-Session in Client-seitige Stores.

    Strategie:
    1) Versuche gvb_output.xlsx von Disk zu laden (DiskSource)
    2) Falls fehlgeschlagen: Fallback auf synthetische Daten (StoreSource)

    Returns:
        (gvb_json, exog_json, metadata)
    """
    # Lokale (defensive) Imports, falls Kopfbereich die Symbole nicht bereitstellt
    import dash
    from pathlib import Path
    from datetime import datetime
    import pandas as pd

    # Bereits geladen? → nichts tun
    if existing_data is not None:
        return dash.no_update, dash.no_update, dash.no_update

    # --------- Logging (sicher, ohne NameError auf Log) ---------
    try:
        Log.info("[Init] session=start | action=load_data")
    except Exception:
        logger.info("[Init] session=start | action=load_data")

    # --------- Abhängigkeiten (injected oder lazy import) ---------
    DM = globals().get("_DataManager") or globals().get("DataManager")
    SS = globals().get("_StoreSource") or globals().get("StoreSource")
    DS = globals().get("_DiskSource")  or globals().get("DiskSource")

    if DM is None or SS is None or DS is None:
        # letzter Versuch: ohne app.py importieren (vermeidet Kreislauf, falls separat vorhanden)
        try:
            from data_manager import DataManager as _DM, StoreSource as _SS, DiskSource as _DS
            DM, SS, DS = _DM, _SS, _DS
        except Exception:
            try:
                from loader.data_manager import DataManager as _DM, StoreSource as _SS, DiskSource as _DS
                DM, SS, DS = _DM, _SS, _DS
            except Exception as e:
                logger.error(f"KPI-Fehler (DataManager init): {e}")
                # Leeren Payload zurückgeben, damit Seite weiter rendert
                empty_gvb = pd.DataFrame(columns=["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss"])
                empty_exog = pd.DataFrame(columns=["date"])
                meta = {
                    'min_date': None,
                    'max_date': None,
                    'n_records': 0,
                    'used_fallback': True,
                    'error': f"DataManager unavailable: {e}",
                    'available_categories': [],
                    'loaded_at': datetime.now().isoformat()
                }
                return empty_gvb.to_json(orient='split'), empty_exog.to_json(orient='split'), meta

    # --------- Daten laden ---------
    used_fallback = False
    try:
        # Kandidaten für gvb_output.xlsx + Diagnose-Logging (App-Root basiert)
        try:
            logger.info(f"[Init] APP_ROOT={APP_ROOT.resolve()}")
            logger.info(f"[Init] OVERVIEW_DIR={OVERVIEW_DIR.resolve()}")
            logger.info(f"[Init] CWD={Path.cwd().resolve()}")
        except Exception:
            pass

        gvb_candidates = [
            # bevorzugte Pfade relativ zur App-Root-Logik
            APP_ROOT / "gvb_output.xlsx",
            APP_ROOT / "loader" / "gvb_output.xlsx",
            # zusätzlich: relativ zum overview-Ordner
            OVERVIEW_DIR / "gvb_output.xlsx",
            OVERVIEW_DIR / "loader" / "gvb_output.xlsx",
            # letzter Fallback: aktuelles Arbeitsverzeichnis
            Path.cwd() / "gvb_output.xlsx",
            Path.cwd() / "loader" / "gvb_output.xlsx",
        ]

        logger.info("[Init] Suche gvb_output.xlsx an:")
        for cand in gvb_candidates:
            try:
                logger.info(f"  • {cand.resolve()} | exists={cand.exists()}")
            except Exception as _e:
                logger.warning(f"  • {cand} | resolve-error: {_e}")

        gvb_path = next((p for p in gvb_candidates if p.exists()), None)
        if gvb_path is None:
            raise FileNotFoundError(
                f"gvb_output.xlsx nicht gefunden. Durchsucht: {[str(p) for p in gvb_candidates]}"
            )

        st = gvb_path.stat()
        logger.info(
            "[Init] found_data_source | path=%s | size=%.1f KB",
            str(gvb_path.resolve()),
            st.st_size / 1024.0,
        )

        dm = DM(DS(
            gvb_path=gvb_path,
            exog_path=None,   # Exog ggf. separat/anders geladen; hier nicht erforderlich
            gvb_sheet=None,
            exog_sheet=None
        ))
        logger.info(f"[Init] real_data_loaded | gvb_records= {len(getattr(dm, 'gvb_data', pd.DataFrame()))} GVB-Records")

    except Exception as e:
        # Fallback: synthetische Daten
        logger.error(f"[Init] real_data_load_error | err= {e}")
        logger.warning("[Init] using_synthetic_fallback=true")
        try:
            dm = DM(SS(
                gvb_payload=_create_synthetic_gvb_data(),
                exog_payload=_create_synthetic_exog_data()
            ))
            used_fallback = True
            logger.info(f"[Init] fallback_synthetic_generated | gvb_records= {len(getattr(dm, 'gvb_data', pd.DataFrame()))} Records")
        except Exception as e2:
            logger.error(f"❌ KRITISCH: Auch synthetische Daten fehlgeschlagen: {e2}")
            # Totalausfall → leere Ergebnisse zurückgeben
            empty_gvb = pd.DataFrame(columns=["date", "ebene1", "ebene2", "ebene3", "bestand", "fluss"])
            empty_exog = pd.DataFrame(columns=["date"])
            meta = {
                'min_date': None,
                'max_date': None,
                'n_records': 0,
                'used_fallback': True,
                'error': str(e2),
                'available_categories': [],
                'loaded_at': datetime.now().isoformat()
            }
            return empty_gvb.to_json(orient='split'), empty_exog.to_json(orient='split'), meta

    # --------- Serialisierung ---------
    try:
        gvb_df = getattr(dm, "gvb_data", pd.DataFrame(columns=["date"]))
        exog_df = getattr(dm, "exog_data", pd.DataFrame(columns=["date"]))
        # Defensive Typkonvertierung für Datum
        if "date" in gvb_df.columns:
            gvb_df = gvb_df.copy()
            gvb_df["date"] = pd.to_datetime(gvb_df["date"], errors="coerce")
        if "date" in exog_df.columns:
            exog_df = exog_df.copy()
            exog_df["date"] = pd.to_datetime(exog_df["date"], errors="coerce")

        gvb_json = gvb_df.to_json(date_format='iso', orient='split')
        exog_json = exog_df.to_json(date_format='iso', orient='split')
    except Exception as e:
        logger.error(f"[Init] json_serialization_error | err= {e}")
        return dash.no_update, dash.no_update, dash.no_update

    # --------- Metadaten ---------
    try:
        min_date = str(gvb_df['date'].min()) if 'date' in gvb_df.columns and not gvb_df.empty else None
        max_date = str(gvb_df['date'].max()) if 'date' in gvb_df.columns and not gvb_df.empty else None
        n_records = int(len(gvb_df))
        categories = gvb_df['ebene1'].dropna().unique().tolist() if 'ebene1' in gvb_df.columns else []

        metadata = {
            'min_date': min_date,
            'max_date': max_date,
            'n_records': n_records,
            'used_fallback': used_fallback,
            'available_categories': categories,
            'loaded_at': datetime.now().isoformat()
        }

        logger.info(
            f"✅ Daten in Store geladen: {metadata['n_records']} Records, "
            f"Zeitraum {metadata['min_date']} - {metadata['max_date']}"
            + (" (Fallback: synthetisch)" if used_fallback else "")
        )
    except Exception as e:
        logger.error(f"[Init] metadata_error | err= {e}")
        metadata = {
            'min_date': None,
            'max_date': None,
            'n_records': int(len(getattr(dm, "gvb_data", []))),
            'used_fallback': used_fallback,
            'available_categories': [],
            'loaded_at': datetime.now().isoformat(),
            'error': str(e)
        }

    return gvb_json, exog_json, metadata


# ==============================================================================
# DYNAMIC SLIDER UPDATE
# ==============================================================================

@app.callback(
    [
        Output('zeitraum-slider', 'min'),
        Output('zeitraum-slider', 'max'),
        Output('zeitraum-slider', 'value'),
        Output('zeitraum-slider', 'marks')
    ],
    Input('data-metadata-store', 'data'),
    prevent_initial_call=False
)
def update_zeitraum_slider(metadata):
    """
    Updates the time range slider based on loaded data metadata.
    Dynamically sets min, max, value, and marks based on actual data range.
    """
    import math
    from datetime import datetime
    import pandas as pd
    
    # Default values if metadata is not available
    default_min = 2000
    default_max = 2026
    default_value = [2020, 2026]
    default_marks = {year: str(year) for year in range(2000, 2030, 5)}
    
    if not metadata or metadata.get('min_date') is None or metadata.get('max_date') is None:
        return default_min, default_max, default_value, default_marks
    
    try:
        # Parse min/max dates
        min_date = pd.to_datetime(metadata['min_date'])
        max_date = pd.to_datetime(metadata['max_date'])
        
        # Convert to decimal year for quarter precision (2024.0, 2024.25, 2024.5, 2024.75)
        def date_to_quarter_decimal(dt):
            # (Month - 1) // 3 derives quarter index (0, 1, 2, 3)
            return dt.year + ((dt.month - 1) // 3) * 0.25

        slider_min = math.floor(date_to_quarter_decimal(min_date))
        
        # Calculate precise max (e.g. 2025.25 for Q2)
        precise_max = date_to_quarter_decimal(max_date)
        
        # Slider Scale: always full years for clean look
        slider_max = math.ceil(precise_max)
        
        # If precise_max is strictly calculating the start of the quarter (e.g. 2025.0),
        # but the data covers the full year, slider_max should just be sufficient.
        # Ensure slider_max is at least precise_max
        if slider_max < precise_max:
             slider_max = math.ceil(precise_max + 0.01)

        # Default value: show last 5 years up to the ACTUAL data end
        start_target = precise_max - 5
        start_value = max(slider_min, start_target)
        
        # Snap start to quarter grid as well to be safe
        start_value = math.floor(start_value * 4) / 4.0

        slider_value = [start_value, precise_max]
        
        # Generate marks - only 5-year intervals plus min and max years
        slider_marks = {}
        
        # Add year marks every 5 years
        for year in range(slider_min, slider_max + 1, 5):
            slider_marks[year] = str(year)
        
        # Always show min and max years
        if slider_min not in slider_marks:
            slider_marks[slider_min] = str(slider_min)
        if slider_max not in slider_marks:
            slider_marks[slider_max] = str(slider_max)
        
        logger.info(f"[Slider Update] min={slider_min}, max={slider_max}, value={slider_value}")
        
        return slider_min, slider_max, slider_value, slider_marks
        
    except Exception as e:
        logger.error(f"[Slider Update] Error: {e}")
        return default_min, default_max, default_value, default_marks


@app.callback(
    Output('zeitraum-display', 'children'),
    Input('zeitraum-slider', 'value')
)
def update_zeitraum_display(value):
    """
    Formats and displays the selected time range in readable format.
    Converts decimal years (e.g., 2024.75) to quarter format (e.g., "2024 Q4").
    """
    if not value or len(value) != 2:
        return ""
    
    def decimal_to_quarter(decimal_year):
        year = int(decimal_year)
        fraction = decimal_year - year
        
        # Determine quarter from fraction
        if fraction < 0.125:
            quarter = 1
        elif fraction < 0.375:
            quarter = 1
        elif fraction < 0.625:
            quarter = 2
        elif fraction < 0.875:
            quarter = 3
        else:
            quarter = 4
        
        return f"{year} Q{quarter}"
    
    start_label = decimal_to_quarter(value[0])
    end_label = decimal_to_quarter(value[1])
    
    return f"{start_label} — {end_label}"


# ==============================================================================
# DRILL-DOWN MANAGEMENT
# ==============================================================================

@app.callback(
    Output('drill-store', 'data'),
    [Input('kpi-card-gesamt', 'n_clicks'),
     Input('kpi-card-einlagen', 'n_clicks'),
     Input('kpi-card-wertpapiere', 'n_clicks'),
     Input('kpi-card-versicherungen', 'n_clicks'),
     Input('kpi-card-kredite', 'n_clicks'),
     Input('detail-ebene-dropdown', 'value')],
    [State('drill-store', 'data')]
)
def update_drill_context(n_all, n_einl, n_wp, n_vers, n_kred,
                         dropdown_level, store):
    """Persistiert Drill-Down-Kontext in dcc.Store('drill-store')"""
    def _next_lower_level(cur: str) -> str:
        if cur == 'ebene1': return 'ebene2'
        if cur == 'ebene2': return 'ebene3'
        return 'ebene3'

    # Init
    if not isinstance(store, dict):
        store = {"eff_level": (dropdown_level or 'ebene1'), "parent": None}

    ctx = callback_context
    trig = ctx.triggered[0]['prop_id'].split('.')[0] if (ctx and ctx.triggered) else None
    cur_lvl = store.get("eff_level", dropdown_level or 'ebene1')

    if trig == 'kpi-card-gesamt':
        return {"eff_level": 'ebene1', "parent": None}

    kpi_map = {
        'kpi-card-einlagen': 'Einlagen',
        'kpi-card-wertpapiere': 'Wertpapiere',
        'kpi-card-versicherungen': 'Versicherungen',
        'kpi-card-kredite': 'Kredite'
    }
    if trig in kpi_map:
        parent = kpi_map[trig]
        return {"eff_level": _next_lower_level(cur_lvl), "parent": parent}

    if trig == 'detail-ebene-dropdown' and dropdown_level:
        return {"eff_level": dropdown_level, "parent": None}

    return store



@app.callback(
    [Output('main-trend-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('distribution-metrics', 'children'),
     Output('detail-ebene-dropdown', 'value')],
    [Input('detail-ebene-dropdown', 'value'),
     Input('datenmodus-switch', 'value'),
     Input('glaettung-dropdown', 'value'),
     Input('zeitraum-slider', 'value'),
     Input('chart-type-area', 'n_clicks'),
     Input('chart-type-bar', 'n_clicks'),
     Input('kpi-card-gesamt', 'n_clicks'),
     Input('kpi-card-einlagen', 'n_clicks'),
     Input('kpi-card-wertpapiere', 'n_clicks'),
     Input('kpi-card-versicherungen', 'n_clicks'),
     Input('kpi-card-kredite', 'n_clicks'),
     Input('log-transform-switch', 'value'),
     Input('distribution-chart', 'clickData'),
     Input('performance-focus-dropdown', 'value'),
     Input('sektor-dropdown', 'value'),
     Input('data-metadata-store', 'modified_timestamp')],
    [State('gvb-data-store', 'data'),
     State('exog-data-store', 'data')]
)
def update_main_charts(
    detail_level, is_fluss_mode, smoothing, zeitraum,
    area_clicks, bar_clicks,
    kpi_gesamt, kpi_einlagen, kpi_wertpapiere, kpi_versicherungen, kpi_kredite,
    use_log, donut_click, focus_category, sektor_value, _ts,
    gvb_json, exog_json
):
    """Trend (Area/Bar) + Donut — mit Sektor-Filter, hierarchischer Sortierung und robuster Donut-Logik für negative Flüsse."""
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from dash import callback_context, html
    import dash
    import dash_bootstrap_components as dbc

    # ---------- Farben ----------
    def _get_theme():
        base_map = globals().get("GVB_COLORS", {
            'Gesamt GVB': '#14324E',
            'Einlagen': '#17a2b8',
            'Wertpapiere': '#28a745',
            'Versicherungen': '#ffc107',
            'Kredite': '#dc3545',
        })
        seq = globals().get("GVB_COLOR_SEQUENCE",
                            ['#17a2b8', '#28a745', '#ffc107', '#dc3545', '#14324E'])
        e1_order = list(globals().get("KPI_E1",
                         ("Einlagen", "Wertpapiere", "Versicherungen", "Kredite")))
        return base_map, seq, e1_order

    def _color_for(category: str, level: str, df_map: pd.DataFrame, idx: int = 0):
        base_map, seq, e1_order = _get_theme()
        if level == 'ebene1':
            if category in base_map:
                return base_map[category]
            if category in e1_order:
                i = e1_order.index(category)
                return base_map.get(category, seq[i % len(seq)])
            return seq[idx % len(seq)]
        # Unterebenen: Parent-Farbe
        if df_map is not None and not df_map.empty and level in df_map.columns:
            parent = df_map.loc[df_map[level] == category, 'ebene1']
            if not parent.empty:
                p = parent.iloc[0]
                return base_map.get(p, seq[0])
        return seq[0]

    def _placeholder(title: str):
        fig = go.Figure()
        fig.update_layout(template='plotly_white', title=title)
        return fig

    # ---------- Sektor-Mapping ----------
    def _map_sektor(val):
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in ("all", "alle", "gesamt", "both"):
            return None
        if s.startswith("haush"):               # "haushalte", "private haushalte", …
            return "PH"
        if s in ("nfk", "nku", "unternehmen", "nichtfinanzielle unternehmen"):
            return "NFK"
        return None

    # ---------- Stores laden ----------
    if not gvb_json:
        return (_placeholder("Daten werden geladen …"),
                _placeholder("Daten werden geladen …"),
                html.Div("Lade Daten…"),
                (detail_level or 'ebene1'))

    try:
        gvb_df = pd.read_json(gvb_json, orient='split')
    except Exception:
        return (_placeholder("GVB-Daten unlesbar"),
                _placeholder("GVB-Daten unlesbar"),
                html.Div("GVB-Daten unlesbar"),
                (detail_level or 'ebene1'))

    # Pflichtspalten auffüllen
    for lvl in ('ebene1', 'ebene2', 'ebene3', 'sektor'):
        if lvl not in gvb_df.columns:
            gvb_df[lvl] = None
    gvb_df['date'] = pd.to_datetime(gvb_df['date'], errors='coerce')
    gvb_df = gvb_df.dropna(subset=['date']).sort_values('date')

    # ---------- Sektor-Filter ----------
    sektor_code = _map_sektor(sektor_value)
    if sektor_code:
        gvb_df = gvb_df[gvb_df['sektor'] == sektor_code].copy()
        if gvb_df.empty:
            msg = "Keine Daten für gewählten Sektor"
            return (_placeholder(msg), _placeholder(msg), html.Div(msg), (detail_level or 'ebene1'))

    # ---------- Parameter ----------
    detail_level = (detail_level or 'ebene1')
    data_type = 'fluss' if is_fluss_mode else 'bestand'
    if data_type not in gvb_df.columns:
        return (_placeholder(f"Spalte '{data_type}' fehlt"),
                _placeholder(f"Spalte '{data_type}' fehlt"),
                html.Div(f"Spalte '{data_type}' fehlt"),
                detail_level)
    try:
        s = int(smoothing) if smoothing not in (None, "", False) else 1
    except Exception:
        s = 1

    # ---------- Zeitraum (robust) ----------
    def _safe_window(idx, slider_val, fallback_years=3):
        if idx.empty: return None, None
        dmin, dmax = idx.min(), idx.max()
        start = end = None
        if isinstance(slider_val, (list, tuple)) and len(slider_val) == 2:
            try:
                a, b = int(slider_val[0]), int(slider_val[1])
                y0, y1 = (a, b) if a <= b else (b, a)
                start = pd.Timestamp(year=y0, month=1, day=1)
                end   = pd.Timestamp(year=y1, month=12, day=31)
            except Exception:
                pass
        if start is None or end is None:
            end = dmax
            start = dmax - pd.DateOffset(years=fallback_years)
        start = max(start, dmin); end = min(end, dmax)
        if start >= end:
            start = end - pd.DateOffset(months=1)
            if start < dmin: start = dmin
            if start >= end: return None, None
        return start.normalize(), end.normalize()

    start_window, end_window = _safe_window(gvb_df['date'], zeitraum, 3)
    if start_window is None:
        return (_placeholder("Ungültiger Zeitraum"),
                _placeholder("Ungültiger Zeitraum"),
                html.Div("Ungültiger Zeitraum"),
                detail_level)

    # ---------- Trigger/Drill, Chart-Typ, Fokus ----------
    area_clicks = area_clicks or 0
    bar_clicks  = bar_clicks  or 0
    chart_type = 'bar' if bar_clicks > area_clicks else 'area'

    def _get_trigger_id():
        try:
            if callback_context and callback_context.triggered:
                return callback_context.triggered[0]['prop_id'].split('.')[0]
        except Exception:
            pass
        try:
            return getattr(dash, "ctx").triggered_id  # Dash >=2.17
        except Exception:
            return None

    trig = _get_trigger_id()

    selected_category = None
    kpi_map = {
        'kpi-card-gesamt': 'Gesamt GVB',
        'kpi-card-einlagen': 'Einlagen',
        'kpi-card-wertpapiere': 'Wertpapiere',
        'kpi-card-versicherungen': 'Versicherungen',
        'kpi-card-kredite': 'Kredite',
    }
    if trig in kpi_map:
        selected_category = kpi_map[trig]
    elif trig == 'distribution-chart' and donut_click and isinstance(donut_click, dict):
        selected_category = donut_click.get('points', [{}])[0].get('label')
    if selected_category == 'Gesamt GVB':
        selected_category = None
    if (selected_category is None) and focus_category and focus_category != 'all':
        selected_category = focus_category

    # ---------- Fenster + Aggregation ----------
    df_win = gvb_df[(gvb_df['date'] >= start_window) & (gvb_df['date'] <= end_window)].copy()
    if df_win.empty:
        return (_placeholder(f"Keine {data_type}-Daten im Zeitraum"),
                _placeholder("Keine Verteilung verfügbar"),
                html.Div(f"Keine {data_type}-Daten"),
                detail_level)

    grp = (df_win.groupby(['date', detail_level], dropna=False, as_index=False)[data_type]
                 .sum())
    if s and s > 1:
        grp = (grp.sort_values(['date', detail_level]).set_index('date'))
        grp[data_type] = (grp.groupby(detail_level)[data_type]
                               .transform(lambda x: x.rolling(s, min_periods=1).mean()))
        grp = grp.reset_index()

    # ---------- Pivot ----------
    pivot_df = (grp.pivot(index='date', columns=detail_level, values=data_type)
                    .sort_index().fillna(0.0))

    # ---------- Hierarchische Ordnung ----------
    def _ordered_columns(detail_lvl: str, cols: list, df_all: pd.DataFrame) -> list:
        base_map, _, e1_order = _get_theme()
        df_sorted = df_all.sort_values(['date', 'ebene1', 'ebene2', 'ebene3'])
        first_seen = {}
        for c in cols:
            try:
                if detail_lvl == 'ebene1':
                    t = df_sorted[df_sorted['ebene1'] == c]['date']
                elif detail_lvl == 'ebene2':
                    t = df_sorted[df_sorted['ebene2'] == c]['date']
                else:
                    t = df_sorted[df_sorted['ebene3'] == c]['date']
                first_seen[c] = t.iloc[0] if not t.empty else pd.Timestamp.max
            except Exception:
                first_seen[c] = pd.Timestamp.max

        if detail_lvl == 'ebene1':
            in_order = [c for c in e1_order if c in cols]
            rest = sorted([c for c in cols if c not in in_order])
            return in_order + rest

        map_df = df_sorted[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
        if detail_lvl == 'ebene2':
            def e1_rank(x):
                p = map_df.loc[map_df['ebene2'] == x, 'ebene1']
                p = p.iloc[0] if not p.empty else None
                return e1_order.index(p) if p in e1_order else len(e1_order)
            return sorted(cols, key=lambda x: (e1_rank(x), first_seen.get(x), str(x)))
        else:
            def parents(x):
                row = map_df.loc[map_df['ebene3'] == x, ['ebene1', 'ebene2']].head(1)
                return (row.iloc[0, 0], row.iloc[0, 1]) if not row.empty else (None, None)
            def e1_rank(x):
                p1, _ = parents(x)
                return e1_order.index(p1) if p1 in e1_order else len(e1_order)
            def e2_key(x):
                _, p2 = parents(x); return str(p2 or '')
            return sorted(cols, key=lambda x: (e1_rank(x), e2_key(x), first_seen.get(x), str(x)))

    ordered_cols = _ordered_columns(detail_level, pivot_df.columns.tolist(), df_win)
    pivot_df = pivot_df[ordered_cols]

    # ---------- Fokus anwenden ----------
    if selected_category:
        if selected_category in pivot_df.columns:
            pivot_df = pivot_df[[selected_category]]
        else:
            mapping_df = df_win[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
            if detail_level in ('ebene2', 'ebene3'):
                allowed = (mapping_df.loc[mapping_df['ebene1'] == selected_category, detail_level]
                                     .dropna().unique().tolist())
                allowed = [c for c in ordered_cols if c in allowed]
                if allowed:
                    pivot_df = pivot_df[allowed]

    if use_log:
        pivot_df = np.sign(pivot_df) * np.log1p(np.abs(pivot_df))

    # ---------- Hauptchart ----------
    main_fig = go.Figure()
    map_df = df_win[['ebene1','ebene2','ebene3']].drop_duplicates()

    if pivot_df.empty:
        main_fig = _placeholder("Keine Daten im Zeitraum")
    else:
        if chart_type == 'area':
            for i, cat in enumerate(pivot_df.columns):
                main_fig.add_trace(go.Scatter(
                    x=pivot_df.index, y=pivot_df[cat],
                    stackgroup='one', name=cat, mode='lines',
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(color=_color_for(cat, detail_level, map_df, idx=i), width=0),
                    showlegend=True,
                    hovertemplate="<b>%{fullData.name}</b><br>Wert: %{y:.1f}<extra></extra>"
                ))
        else:
            for i, cat in enumerate(pivot_df.columns):
                main_fig.add_trace(go.Bar(
                    x=pivot_df.index, y=pivot_df[cat], name=cat,
                    marker_color=_color_for(cat, detail_level, map_df, idx=i),
                    hovertemplate="<b>%{fullData.name}</b><br>Wert: %{y:.1f}<extra></extra>"
                ))
            main_fig.update_layout(barmode='stack')

        main_fig.update_layout(
            title=f"{'Fluss' if data_type=='fluss' else 'Bestand'}-Daten",
            xaxis_title='Jahr',
            yaxis_title=f"{data_type.title()}{' — Log' if use_log else ''}",
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15,
                        xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.2)", borderwidth=0),
            margin=dict(b=80),
            height=500
        )

        # X-Ticks: Formatierung auf Quartale
        # Alte biennal-Logik entfernt, neue Helper-Funktion genutzt
        format_axis_quarters(main_fig, pivot_df.index)


    # ---------- Donut (robust bei negativen Flüssen) ----------
    latest_date = grp['date'].max()
    latest = grp[grp['date'] == latest_date]
    if latest.empty:
        donut_fig = _placeholder("Keine Verteilung verfügbar")
        metrics = html.Div("Keine Daten")
    else:
        latest_cat = latest[detail_level].tolist()
        latest_vals_raw = latest[data_type].tolist()

        # Reihenfolge konsistent zu den Spalten im Hauptchart
        cat_order = ordered_cols
        order_map = {c: i for i, c in enumerate(cat_order)}
        order_idx = sorted(range(len(latest_cat)), key=lambda i: order_map.get(latest_cat[i], 10**9))
        cats = [latest_cat[i] for i in order_idx]
        vals_raw = [latest_vals_raw[i] for i in order_idx]

        # --- WICHTIG: Pie kann keine „negativen Anteile“ zeigen.
        #     Lösung: Bei Flussdaten mit negativen Werten -> absolute Beträge für den Anteil,
        #     Vorzeichen im Tooltip/Label beibehalten.
        use_abs_for_pie = bool(is_fluss_mode and any(v < 0 for v in vals_raw))
        if use_abs_for_pie:
            vals_for_pie = [abs(v) for v in vals_raw]
            labels = [f"{c} (−)" if v < 0 else c for c, v in zip(cats, vals_raw)]
            title_suffix = " — nach Betragshöhe (|Wert|)"
            hover_tmpl = "%{label}<br>Original: %{customdata:.1f}<br>%{percent}<extra></extra>"
            customdata = np.array(vals_raw, dtype=float)
        else:
            vals_for_pie = vals_raw
            labels = cats
            title_suffix = ""
            hover_tmpl = "%{label}<br>%{value:.1f}<br>%{percent}<extra></extra>"
            customdata = None

        total_for_pie = float(np.nansum(vals_for_pie))
        if total_for_pie <= 0.0:
            donut_fig = _placeholder("Keine Verteilung verfügbar")
            metrics = html.Div("Keine Daten")
        else:
            donut_colors = [_color_for(c, detail_level, map_df, idx=i) for i, c in enumerate(cats)]
            pie = go.Pie(
                labels=labels,
                values=vals_for_pie,
                hole=0.4,
                marker_colors=donut_colors,
                textinfo='label+percent', textposition='inside', insidetextorientation='auto',
                textfont=dict(size=10),
                hovertemplate=hover_tmpl,
                sort=False
            )
            if customdata is not None:
                pie.update(customdata=customdata)

            donut_fig = go.Figure(pie)
            donut_fig.update_layout(
                title=f"Verteilung ({data_type.title()}) — {latest_date.date()}{title_suffix}",
                template='plotly_white', showlegend=False,
                margin=dict(l=40, r=40, t=40, b=20), height=450
            )

            # KPI-Text: größte Kategorie (konsistent zur Darstellungslogik)
            i_max = int(np.nanargmax(vals_for_pie))
            largest_category = cats[i_max]
            largest_share = (vals_for_pie[i_max] / total_for_pie) * 100.0

            metrics = dbc.Row([
                html.H6(
                    f"Größte Kategorie: {largest_category} ({largest_share:.1f}%)",
                    className="text-muted", style={"margin": "8px 0"}
                )
            ])

    return main_fig, donut_fig, metrics, detail_level


@app.callback(
    [Output('quarterly-changes-chart', 'figure'),
     Output('seasonality-radar-chart', 'figure')],
    [Input('zeitraum-slider', 'value'),
     Input('detail-ebene-dropdown', 'value'),
     Input('datenmodus-switch', 'value'),
     Input('glaettung-dropdown', 'value'),
     Input('performance-focus-dropdown', 'value'),
     Input('change-metric', 'value'),
     Input('data-metadata-store', 'modified_timestamp'),
     Input('drill-store', 'data'),
     Input('quarterly-changes-chart', 'clickData'),
     Input('sektor-dropdown', 'value')],            # ← NEU: Sektor-Input
    [State('gvb-data-store', 'data'),
     State('exog-data-store', 'data')]
)
def update_quarterly_changes(
    zeitraum, detail_level, is_fluss_mode, smoothing,
    focus_category, change_metric, _ts, drill_state, bar_click, sektor_value,
    gvb_json, exog_json
):
    """Veränderungen (links) + Saisonalität (Radar). Nutzt Farben aus app.py und filtert nach Sektor."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from dash import callback_context

    # -------------------- Farb-Helper (holt Farben direkt aus app.py) --------------------
    def _get_theme():
        base_map = globals().get("GVB_COLORS", {
            'Gesamt GVB': '#14324E',
            'Einlagen': '#17a2b8',
            'Wertpapiere': '#28a745',
            'Versicherungen': '#ffc107',
            'Kredite': '#dc3545',
        })
        seq = globals().get("GVB_COLOR_SEQUENCE",
                            ['#17a2b8', '#28a745', '#ffc107', '#dc3545', '#14324E'])
        e1_order = list(globals().get("KPI_E1",
                         ("Einlagen", "Wertpapiere", "Versicherungen", "Kredite")))
        return base_map, seq, e1_order

    def _hex_to_rgb_tuple(hexcode: str):
        try:
            if isinstance(hexcode, str) and hexcode.startswith('#') and len(hexcode) == 7:
                return tuple(int(hexcode[i:i+2], 16) for i in (1, 3, 5))
        except Exception:
            pass
        return (108, 117, 125)  # fallback '#6c757d'

    def _color_for(category: str, level: str, df_map: pd.DataFrame, idx: int = 0):
        """E1 → feste GVB_COLORS / Reihenfolge; E2/E3 → Farbe des E1-Parents."""
        base_map, seq, e1_order = _get_theme()
        if level == 'ebene1':
            if category in base_map:
                return base_map[category]
            if category in e1_order:
                i = e1_order.index(category)
                return base_map.get(category, seq[i % len(seq)])
            return seq[idx % len(seq)]
        # Unterebenen → Parent-E1 suchen
        if df_map is not None and not df_map.empty and level in df_map.columns:
            parent = df_map.loc[df_map[level] == category, 'ebene1']
            if not parent.empty:
                p = parent.iloc[0]
                return base_map.get(p, seq[0])
        return seq[0]

    def _rgba_fill(hexcode: str, alpha: float = 0.25):
        r, g, b = _hex_to_rgb_tuple(hexcode)
        return f"rgba({r},{g},{b},{alpha})"

    # -------------------- Sektor-Mapping --------------------
    def _map_sektor(val):
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in ("all", "alle", "gesamt", "both"):
            return None
        if s.startswith("haush"):                        # „(Private) Haushalte …“
            return "PH"
        if s in ("nfk", "nku", "unternehmen", "nichtfinanzielle unternehmen"):
            return "NFK"
        return None

    # -------------------- Platzhalter/Error-Figs --------------------
    def _placeholder_fig(title_text: str):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=title_text,
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=10, r=10, t=60, b=40))
        return fig

    def _error_fig(msg: str):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Fehler",
                          annotations=[dict(text=str(msg), x=0.5, y=0.5,
                                            xref="paper", yref="paper",
                                            showarrow=False, font=dict(size=13))],
                          margin=dict(l=10, r=10, t=60, b=40))
        return fig

    # -------------------- Guards + Daten laden --------------------
    if not gvb_json:
        return _placeholder_fig("Veränderungen"), _placeholder_fig("Saisonalität (QoQ)")
    try:
        df = pd.read_json(gvb_json, orient='split')
    except Exception as e:
        return _error_fig(f"GVB-Daten unlesbar: {e}"), _error_fig(f"{e}")

    if df.empty or 'date' not in df.columns:
        return _placeholder_fig("Keine Daten"), _placeholder_fig("Keine Daten")

    # Pflichtspalten absichern
    for lvl in ('ebene1', 'ebene2', 'ebene3', 'sektor'):
        if lvl not in df.columns:
            df[lvl] = None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    if df.empty:
        return _placeholder_fig("Keine gültigen Datumswerte"), _placeholder_fig("Keine gültigen Datumswerte")

    # -------------------- Sektor-Filter (NEU) --------------------
    sektor_code = _map_sektor(sektor_value)
    if sektor_code:
        df = df[df['sektor'] == sektor_code].copy()
        if df.empty:
            msg = "Keine Daten für gewählten Sektor"
            return _placeholder_fig(msg), _placeholder_fig(msg)

    # -------------------- Parameter + Drill/Fokus --------------------
    data_type = 'fluss' if is_fluss_mode else 'bestand'
    if data_type not in df.columns:
        return _error_fig(f"Spalte '{data_type}' fehlt"), _error_fig(f"Spalte '{data_type}' fehlt")

    try:
        s = int(smoothing) if (smoothing not in (None, "", False)) else 1
    except Exception:
        s = 1

    if isinstance(drill_state, dict):
        eff_level = drill_state.get('eff_level') or (detail_level or 'ebene1')
        parent_filter = drill_state.get('parent')
    else:
        eff_level = detail_level or 'ebene1'
        parent_filter = None

    if focus_category and focus_category != 'all':
        eff_level = 'ebene2' if eff_level == 'ebene1' else eff_level
        parent_filter = focus_category

    # -------------------- Zeitraum aus Slider robust bestimmen --------------------
    def _safe_window(idx, slider_val, fallback_years=3):
        if idx.empty:
            return None, None
        dmin, dmax = idx.min(), idx.max()
        start = end = None
        if isinstance(slider_val, (list, tuple)) and len(slider_val) == 2:
            try:
                a, b = int(slider_val[0]), int(slider_val[1])
                y0, y1 = (a, b) if a <= b else (b, a)
                start = pd.Timestamp(year=y0, month=1, day=1)
                end   = pd.Timestamp(year=y1, month=12, day=31)
            except Exception:
                pass
        if start is None or end is None:
            end = dmax
            start = dmax - pd.DateOffset(years=fallback_years)
        start = max(start, dmin); end = min(end, dmax)
        if start >= end:
            start = end - pd.DateOffset(months=1)
            if start < dmin: start = dmin
            if start >= end: return None, None
        return start.normalize(), end.normalize()

    start_window, end_window = _safe_window(df['date'], zeitraum, 3)
    if start_window is None:
        return _placeholder_fig("Veränderungen"), _placeholder_fig("Saisonalität (QoQ)")

    # -------------------- Parent-/Fokus-Filter --------------------
    def _apply_parent_and_focus(df_in: pd.DataFrame) -> pd.DataFrame:
        d = df_in
        if not parent_filter:
            return d
        mapping = d[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
        if eff_level == 'ebene2':
            allowed = mapping.loc[mapping['ebene1'] == parent_filter, 'ebene2'].dropna().unique().tolist()
        elif eff_level == 'ebene3':
            allowed = mapping.loc[mapping['ebene1'] == parent_filter, 'ebene3'].dropna().unique().tolist()
        else:
            allowed = [parent_filter]
        return d[d[eff_level].isin(allowed)] if allowed else d

    df_win = df[(df['date'] >= start_window) & (df['date'] <= end_window)].copy()
    df_win = _apply_parent_and_focus(df_win)
    if df_win.empty:
        return _placeholder_fig("Keine Daten im Zeitraum"), _placeholder_fig("Keine Daten im Zeitraum")

    # -------------------- Aggregation (roh & ggf. geglättet) --------------------
    valcol = data_type
    df_rw = (df_win.groupby(['date', eff_level], dropna=False, as_index=False)[valcol]
                  .sum().rename(columns={valcol: data_type}))
    if s and s > 1:
        df_rw = (df_rw.sort_values(['date', eff_level]).set_index('date'))
        df_rw[data_type] = (df_rw.groupby(eff_level)[data_type]
                                 .transform(lambda x: x.rolling(s, min_periods=1).mean()))
        df_rw = df_rw.reset_index()

    # -------------------- Veränderungen berechnen (linkes Chart) --------------------
    pivot = df_rw.pivot(index='date', columns=eff_level, values=data_type).sort_index()
    series_vals = {}
    if not pivot.empty:
        for col in pivot.columns:
            s_col = pivot[col].dropna()
            s_win = s_col.loc[(s_col.index >= start_window) & (s_col.index <= end_window)]
            if s_win.empty:
                continue

            if change_metric == 'abs':
                value = float(s_win.sum()) if is_fluss_mode else float(s_win.iloc[-1] - s_win.iloc[0])
            elif change_metric == 'avg_pct_qoq':
                r = s_win.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                value = float(r.mean() * 100.0) if len(r) else np.nan
            elif change_metric == 'avg_pct_pa':
                r = s_win.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(4)
                if len(r):
                    factors = (1.0 + r.values)
                    value = float(np.prod(factors) - 1.0) * 100.0
                else:
                    value = np.nan
            else:
                value = float(s_win.iloc[-1] - s_win.iloc[0])

            series_vals[col] = value

    metric_series = pd.Series(series_vals, dtype=float)

    left_fig = go.Figure()
    if metric_series.empty:
        left_fig = _placeholder_fig("Veränderungen")
        selected_category = None
    else:
        srt = metric_series.sort_values(ascending=True)
        y_labels = list(srt.index[::-1])
        x_vals = list(srt.values[::-1])

        map_df = df[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
        bar_colors = [_color_for(cat, eff_level, map_df, idx=i)
                      for i, cat in enumerate(y_labels)]

        hover_tmpl = (
            "<b>%{y}</b><br>Δ: %{x:.2f}<extra></extra>"
            if change_metric == 'abs'
            else "<b>%{y}</b><br>Ø Δ: %{x:.2f} %<extra></extra>"
            if change_metric == 'avg_pct_qoq'
            else "<b>%{y}</b><br>Δ p.a.: %{x:.2f} %<extra></extra>"
        )

        left_fig.add_trace(go.Bar(
            x=x_vals, y=y_labels, orientation='h',
            marker_color=bar_colors, hovertemplate=hover_tmpl,
            showlegend=False
        ))

        trig_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context and callback_context.triggered else None
        if trig_id == 'quarterly-changes-chart' and bar_click and isinstance(bar_click, dict):
            try:
                cand = bar_click.get('points', [{}])[0].get('y')
                selected_category = cand if cand in y_labels else None
            except Exception:
                selected_category = None
        else:
            selected_category = None

        mode_label = {"abs": "Absolute Veränderung",
                      "avg_pct_qoq": "Ø % pro Quartal",
                      "avg_pct_pa": "Ø % p.a."}.get(change_metric, "Absolute Veränderung")

        left_fig.update_layout(
            title=f"Veränderungen — {mode_label}",
            template="plotly_white",
            margin=dict(l=10, r=10, t=60, b=40),
            height=420
        )

    # -------------------- Radar (rechte Karte, Durchschnitt QoQ) --------------------
    def _avg_qoq_by_quarter(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=['quarter', 'value', 'category'])
        d = df_in.sort_values('date').copy()
        d['quarter'] = d['date'].dt.quarter
        d['prev'] = d.groupby(eff_level)[data_type].shift(1)
        denom = d['prev'].replace(0, np.nan)
        d['qoq'] = (d[data_type] - d['prev']) / denom
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['qoq'])
        out = (d.groupby(['quarter', eff_level])['qoq']
                 .mean().reset_index()
                 .rename(columns={eff_level: 'category', 'qoq': 'value'}))
        out['value'] = out['value'] * 100.0
        return out

    radar_df = _avg_qoq_by_quarter(df_rw)
    if 'category' not in radar_df.columns:
        radar_df = pd.DataFrame(columns=['quarter', 'value', 'category'])

    if selected_category:
        radar_df = radar_df[radar_df['category'] == selected_category]

    right_fig = go.Figure()
    if radar_df.empty:
        right_fig = _placeholder_fig(f"Saisonalität (QoQ){' — Filter: '+selected_category if selected_category else ''}")
    else:
        theta_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        cats = sorted(radar_df['category'].unique())
        map_df = df[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()

        for i, cat in enumerate(cats):
            sub = radar_df[radar_df['category'] == cat].sort_values('quarter')
            q_vals = {int(q): float(v) for q, v in zip(sub['quarter'], sub['value'])}
            r_vals = [q_vals.get(q, 0.0) for q in [1, 2, 3, 4]]

            base_hex = _color_for(cat, eff_level, map_df, idx=i)
            right_fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=theta_labels, name=cat,
                line=dict(width=2, color=base_hex),
                fill='toself', fillcolor=_rgba_fill(base_hex, 0.25),
                opacity=1.0, showlegend=(eff_level == 'ebene1'),
                hovertemplate="<b>%{theta}</b><br>%{r:.2f} %<extra>"+cat+"</extra>"
            ))

        title = "Saisonalität (Durchschnittliche QoQ-Raten im Fenster)"
        if selected_category:
            title += f" — Filter: {selected_category}"

        right_fig.update_layout(
            title=title,
            template="plotly_white",
            polar=dict(
                radialaxis=dict(visible=True, tickformat=".1f"),
                angularaxis=dict(direction="clockwise", period=4)
            ),
            margin=dict(l=10, r=10, t=60, b=40),
            showlegend=(eff_level == 'ebene1'),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
                        title=None, bgcolor="rgba(0,0,0,0)")
        )

    return left_fig, right_fig







# ==============================================================================
# KPI- UND PERFORMANCE-CALLBACKS
# ==============================================================================
@app.callback(
    [
        Output('kpi-card-gesamt-value', 'children'),
        Output('kpi-card-gesamt-netto', 'children'),
        Output('kpi-card-gesamt-qoq', 'children'),
        Output('kpi-card-gesamt-yoy', 'children'),

        Output('kpi-card-einlagen-value', 'children'),
        Output('kpi-card-einlagen-qoq', 'children'),
        Output('kpi-card-einlagen-yoy', 'children'),

        Output('kpi-card-wertpapiere-value', 'children'),
        Output('kpi-card-wertpapiere-qoq', 'children'),
        Output('kpi-card-wertpapiere-yoy', 'children'),

        Output('kpi-card-versicherungen-value', 'children'),
        Output('kpi-card-versicherungen-qoq', 'children'),
        Output('kpi-card-versicherungen-yoy', 'children'),

        Output('kpi-card-kredite-value', 'children'),
        Output('kpi-card-kredite-qoq', 'children'),
        Output('kpi-card-kredite-yoy', 'children'),
    ],
    [
        Input('gvb-data-store', 'modified_timestamp'),
        Input('sektor-dropdown', 'value')
    ],
    [
        State('gvb-data-store', 'data'),
        State('exog-data-store', 'data')
    ],
    prevent_initial_call=False
)
def update_kpi_cards(timestamp, sektor_value, gvb_json, exog_json):
    """Aktualisiert alle KPI-Karten – nur mit Store-Daten (ohne DataManager)."""
    import pandas as pd
    import numpy as np
    from dash import html

    # ------- Platzhalter (Lade-/Fehlerzustand) -------
    placeholder_val = html.H3("...", className="mb-2 fw-bold text-muted")
    placeholder_netto = html.Small("Netto: ...", className="ms-2 text-muted")

    def _badge(label):
        return html.Span(
            f"±0.0% {label}",
            className="metric-badge",
            style={"backgroundColor": "var(--bs-secondary)", "color": "#212529"}
        )

    def _placeholders():
        return [
            placeholder_val, placeholder_netto, _badge("QoQ"), _badge("YoY"),
            placeholder_val, _badge("QoQ"), _badge("YoY"),
            placeholder_val, _badge("QoQ"), _badge("YoY"),
            placeholder_val, _badge("QoQ"), _badge("YoY"),
            placeholder_val, _badge("QoQ"), _badge("YoY"),
        ]

    if gvb_json is None:
        return _placeholders()

    # ------- JSON → DataFrame -------
    try:
        df = pd.read_json(gvb_json, orient='split')
    except Exception as e:
        logger.error(f"KPI-Fehler (read_json): {e}")
        return _placeholders()

    if df is None or df.empty:
        return _placeholders()

    # ------- Schema-Guards -------
    # Erwartet: Spalten 'date', 'ebene1', 'bestand', optional 'sektor'
    if 'date' not in df.columns or 'ebene1' not in df.columns or 'bestand' not in df.columns:
        logger.error("[KPI] fehlende Spalten: nötig sind 'date', 'ebene1', 'bestand'")
        return _placeholders()

    # Datums-Typ
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if df.empty:
        return _placeholders()

    # ------- Sektor-Mapping (robust) -------
    # Versuche vorhandenen Helper zu nutzen; sonst einfacher Fallback
    try:
        sektor = map_sektor(sektor_value)
    except Exception:
        # Fallback: akzeptiere 'PH' / 'NFK' direkt, sonst None
        val = (sektor_value or "").strip().lower()
        if val in ("ph", "haushalte", "privathaushalte"):
            sektor = "PH"
        elif val in ("nfk", "nichtfinanzielle unternehmen", "unternehmen"):
            sektor = "NFK"
        else:
            sektor = None

    if sektor is not None and 'sektor' in df.columns:
        df = df[df['sektor'] == sektor]

    if df.empty:
        return _placeholders()

    # ------- Aggregation: Ebene 1 je Datum -------
    # Summiere Bestand je Datum und ebene1
    agg = (df.groupby(['date', 'ebene1'], as_index=False)['bestand']
             .sum()
             .rename(columns={'bestand': 'wert'}))

    if agg.empty:
        return _placeholders()

    # Pivot: Zeilen = date, Spalten = ebene1 (Einlagen/Wertpapiere/Versicherungen/Kredite)
    pv = (agg.pivot(index='date', columns='ebene1', values='wert')
              .sort_index()
              .fillna(0.0))

    # ------- Aktuelles Datum / Werte -------
    if pv.empty:
        return _placeholders()

    last_date = pv.index.max()
    last_row = pv.loc[last_date]

    # Einzelsäulen (fehlende → 0.0)
    e = float(last_row.get('Einlagen', 0.0))
    w = float(last_row.get('Wertpapiere', 0.0))
    v = float(last_row.get('Versicherungen', 0.0))
    k = float(last_row.get('Kredite', 0.0))

    gesamt = e + w + v
    netto  = gesamt - k

    # ------- Hilfsfunktion: %-Änderungen über Perioden -------
    def _pct_change(series: pd.Series, periods: int):
        try:
            s = series.dropna()
            if len(s) <= abs(periods):
                return None
            base = s.iloc[-(periods+1)]
            curr = s.iloc[-1]
            if pd.isna(base) or pd.isna(curr) or base == 0:
                return None
            return (curr / base - 1.0) * 100.0
        except Exception:
            return None

    # Für QoQ/YoY brauchen wir serielle Reihen je Kategorie
    e_s = pv.get('Einlagen', pd.Series(dtype=float))
    w_s = pv.get('Wertpapiere', pd.Series(dtype=float))
    v_s = pv.get('Versicherungen', pd.Series(dtype=float))
    k_s = pv.get('Kredite', pd.Series(dtype=float))
    gesamt_s = e_s.add(w_s, fill_value=0.0).add(v_s, fill_value=0.0)

    qoq = {
        "Gesamt GVB": _pct_change(gesamt_s, 1),
        "Einlagen": _pct_change(e_s, 1),
        "Wertpapiere": _pct_change(w_s, 1),
        "Versicherungen": _pct_change(v_s, 1),
        "Kredite": _pct_change(k_s, 1),
    }
    yoy = {
        "Gesamt GVB": _pct_change(gesamt_s, 4),
        "Einlagen": _pct_change(e_s, 4),
        "Wertpapiere": _pct_change(w_s, 4),
        "Versicherungen": _pct_change(v_s, 4),
        "Kredite": _pct_change(k_s, 4),
    }

    # ------- Rendering Helpers -------
    def fmt_val(x):
        try:
            if x is None or pd.isna(x):
                return placeholder_val
            return html.H3(f"{x:.1f}", className="mb-2 fw-bold")
        except Exception:
            return placeholder_val

    def fmt_netto(x):
        try:
            if x is None or pd.isna(x):
                return placeholder_netto
            return html.Small(f"Netto: {x:.1f}", className="ms-2 text-muted")
        except Exception:
            return placeholder_netto

    def fmt_badge(pct, label):
        if pct is None or pd.isna(pct) or abs(pct) < 0.01:
            return html.Span(
                f"±0.0% {label}",
                className="metric-badge",
                style={"backgroundColor": "var(--bs-secondary)", "color": "#212529"}
            )
        if pct > 0:
            return html.Span(
                f"↑ {abs(pct):.1f}% {label}",
                className="metric-badge",
                style={"backgroundColor": "var(--bs-success)", "color": "#ffffff"}
            )
        return html.Span(
            f"↓ {abs(pct):.1f}% {label}",
            className="metric-badge",
            style={"backgroundColor": "var(--bs-danger)", "color": "#ffffff"}
        )

    # ------- Output zusammenstellen -------
    out = [
        fmt_val(gesamt),
        fmt_netto(netto),
        fmt_badge(qoq.get("Gesamt GVB"), "QoQ"),
        fmt_badge(yoy.get("Gesamt GVB"), "YoY"),
    ]

    for cat, series in [
        ("Einlagen", e_s),
        ("Wertpapiere", w_s),
        ("Versicherungen", v_s),
        ("Kredite", k_s),
    ]:
        val = float(series.iloc[-1]) if len(series) else np.nan
        out.append(fmt_val(val))
        out.append(fmt_badge(qoq.get(cat), "QoQ"))
        out.append(fmt_badge(yoy.get(cat), "YoY"))

    # Debug/Info-Log optional
    try:
        logger.info(
            f"[KPI] agg=ebene1 dtype=bestand sektor={sektor} | "
            f"Einlagen={e:.1f} Wertpapiere={w:.1f} Versicherungen={v:.1f} Kredite={k:.1f}"
        )
    except Exception:
        pass

    return out




@app.callback(
    Output('performance-focus-dropdown', 'value', allow_duplicate=True),
    [
        Input('kpi-card-gesamt', 'n_clicks'),
        Input('kpi-card-einlagen', 'n_clicks'),
        Input('kpi-card-wertpapiere', 'n_clicks'),
        Input('kpi-card-versicherungen', 'n_clicks'),
        Input('kpi-card-kredite', 'n_clicks'),
    ],
    prevent_initial_call=True
)
def sync_performance_dropdown(kpi_gesamt, kpi_einlagen, kpi_wertpapiere, kpi_versicherungen, kpi_kredite):
    """Synchronisiert den Performance-Fokus-Dropdown mit Klicks auf KPI-Karten."""
    from dash import no_update
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    mapping = {
        'kpi-card-gesamt': 'all',
        'kpi-card-einlagen': 'Einlagen',
        'kpi-card-wertpapiere': 'Wertpapiere',
        'kpi-card-versicherungen': 'Versicherungen',
        'kpi-card-kredite': 'Kredite'
    }
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    return mapping.get(trig, no_update)


@app.callback(
    [
        Output('performance-focus-dropdown', 'value', allow_duplicate=True),
        Output('selected-categories-store', 'data', allow_duplicate=True)
    ],
    Input('performance-chart', 'clickData'),
    State('performance-focus-dropdown', 'value'),
    State('selected-categories-store', 'data'),
    State('performance-chart', 'figure'),
    prevent_initial_call=True
)
def drilldown_performance_from_click(clickData, current_focus, path, perf_fig):
    """Interpretiert Klicks im Performance-Chart: Drilldown-Pfad aktualisieren oder Fokus setzen."""
    from dash import no_update
    if not clickData or not perf_fig:
        return no_update, no_update

    try:
        curve_idx = clickData['points'][0].get('curveNumber')
        if curve_idx is None:
            return no_update, no_update
        series_name = perf_fig['data'][curve_idx].get('name')
        if not series_name or series_name == 'Gesamt GVB':
            return no_update, no_update
    except Exception:
        return no_update, no_update

    path = path or []

    if current_focus == 'all':
        return series_name, []
    return no_update, [current_focus, series_name]


@app.callback(
    Output('performance-chart', 'figure'),
    [
        Input('performance-focus-dropdown', 'value'),
        Input('datenmodus-switch', 'value'),
        Input('glaettung-dropdown', 'value'),
        Input('kpi-card-gesamt', 'n_clicks'),
        Input('kpi-card-einlagen', 'n_clicks'),
        Input('kpi-card-wertpapiere', 'n_clicks'),
        Input('kpi-card-versicherungen', 'n_clicks'),
        Input('kpi-card-kredite', 'n_clicks'),
        Input('log-transform-switch', 'value'),
        Input('zeitraum-slider', 'value'),
        Input('sektor-dropdown', 'value'),          # ← NEU: Sektor als Input
        Input('data-metadata-store', 'modified_timestamp')
    ],
    [
        State('selected-categories-store', 'data'),
        State('gvb-data-store', 'data'),
        State('exog-data-store', 'data')
    ]
)
def update_performance_chart(
    focus_category, is_fluss_mode, smoothing,
    kpi_gesamt, kpi_einlagen, kpi_wertpapiere, kpi_versicherungen, kpi_kredite,
    use_log, zeitraum, sektor_value, _data_timestamp,
    drill_path, gvb_json, exog_json
):
    """Performance-Analyse — Drilldown bis Ebene 3 (Store-basiert) mit Glättung, Basis=100 und Sektor-Filter."""
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from dash import callback_context

    CHART_HEIGHT = 545

    # ===== Farben aus app.py (robust mit Fallbacks) =====
    _GVB_COLORS = globals().get("GVB_COLORS", {}) or {
        'Gesamt GVB': '#14324E',
        'Einlagen': '#17a2b8',
        'Wertpapiere': '#28a745',
        'Versicherungen': '#ffc107',
        'Kredite': '#dc3545'
    }
    _GVB_SEQ = globals().get("GVB_COLOR_SEQUENCE", []) or [
        '#17a2b8', '#28a745', '#ffc107', '#dc3545', '#14324E',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    def _get_category_color(cat: str) -> str:
        if isinstance(cat, str) and cat in _GVB_COLORS:
            return _GVB_COLORS[cat]
        return _GVB_SEQ[hash(cat) % len(_GVB_SEQ)] if _GVB_SEQ else '#1f77b4'

    def _hex_to_rgb_tuple(hexcode: str):
        try:
            if isinstance(hexcode, str) and hexcode.startswith('#') and len(hexcode) == 7:
                return tuple(int(hexcode[i:i+2], 16) for i in (1, 3, 5))
        except Exception:
            pass
        return (108, 117, 125)  # '#6c757d'

    # ===== Platzhalter- und Fehler-Figuren =====
    def _placeholder_fig(title_text: str):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white", title=title_text,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(
                text="Daten werden geladen …", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False, font=dict(size=14)
            )],
            height=CHART_HEIGHT, margin=dict(l=10, r=10, t=60, b=40),
        )
        return fig

    def _error_fig(msg: str):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white", title="Fehler",
            annotations=[dict(
                text=str(msg), x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False, font=dict(size=13)
            )],
            height=CHART_HEIGHT, margin=dict(l=10, r=10, t=60, b=40),
        )
        return fig

    # ===== Guards & Daten laden =====
    if not gvb_json:
        return _placeholder_fig("Daten werden geladen …")
    try:
        df = pd.read_json(gvb_json, orient='split')
    except Exception as e:
        return _error_fig(f"GVB-Daten unlesbar: {e}")

    if df is None or df.empty:
        return _placeholder_fig("Keine Daten verfügbar")

    # Erwartetes Schema absichern
    for col in ('date', 'ebene1'):
        if col not in df.columns:
            return _error_fig(f"Spalte '{col}' fehlt")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if df.empty:
        return _placeholder_fig("Keine gültigen Datumswerte")

    # ===== Sektor-Mapping & Filter =====
    def _map_sektor(val):
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in ("all", "alle", "gesamt", "both"):
            return None
        if s.startswith("haush"):          # "haushalte", "private haushalte" etc.
            return "PH"
        if s in ("nfk", "nku", "unternehmen", "nichtfinanzielle unternehmen"):
            return "NFK"
        return None

    sektor_code = _map_sektor(sektor_value)
    if sektor_code and 'sektor' in df.columns:
        df = df[df['sektor'] == sektor_code].copy()
        if df.empty:
            return _placeholder_fig("Keine Daten nach Sektor-Filter")

    # ===== Parameter =====
    data_type = 'fluss' if is_fluss_mode else 'bestand'
    if data_type not in df.columns:
        return _error_fig(f"Spalte '{data_type}' fehlt")

    try:
        s = int(smoothing) if (smoothing not in (None, "", False)) else 1
    except Exception:
        s = 1

    # ===== KPI-Klicks überschreiben Fokus =====
    drill_path = drill_path or []
    trig = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None
    kpi_override = {
        'kpi-card-gesamt': 'all',
        'kpi-card-einlagen': 'Einlagen',
        'kpi-card-wertpapiere': 'Wertpapiere',
        'kpi-card-versicherungen': 'Versicherungen',
        'kpi-card-kredite': 'Kredite'
    }.get(trig)
    if kpi_override is not None:
        focus_category = kpi_override
        drill_path = []

    # ===== Zeitraum bestimmen (robust, Slider → Fenster) =====
    def _safe_time_window_from_slider(df_all: pd.DataFrame, slider_val, fallback_years=5):
        dmin = pd.to_datetime(df_all['date'].min(), errors='coerce')
        dend = pd.to_datetime(df_all['date'].max(), errors='coerce')
        if pd.isna(dmin) or pd.isna(dend):
            return None, None

        start = end = None
        if isinstance(slider_val, (list, tuple)) and len(slider_val) == 2:
            try:
                ya = int(slider_val[0]) if slider_val[0] is not None else None
                yb = int(slider_val[1]) if slider_val[1] is not None else None
            except Exception:
                ya = yb = None
            if ya is not None and yb is not None:
                y0, y1 = (ya, yb) if ya <= yb else (yb, ya)
                start = pd.Timestamp(year=y0, month=1, day=1)
                end   = pd.Timestamp(year=y1, month=12, day=31)

        if start is None or end is None:
            end = dend
            start = dend - pd.DateOffset(years=fallback_years)

        start = max(start, dmin)
        end   = min(end, dend)
        if start >= end:
            start = end - pd.DateOffset(months=1)
            if start < dmin:
                start = dmin
            if start >= end:
                return None, None
        return start.normalize(), end.normalize()

    start_date, end_date = _safe_time_window_from_slider(df, zeitraum, fallback_years=5)
    if start_date is None or end_date is None:
        return _placeholder_fig("Ungültiger Zeitraum")

    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if df.empty:
        return _placeholder_fig("Keine Daten im Zeitraum")

    # ===== Ebene & Titel =====
    if focus_category == 'all' or not focus_category:
        eff_level = 'ebene1'
        title_suffix = "Alle Kategorien"
    elif len(drill_path) == 2 and drill_path[0] == focus_category:
        eff_level = 'ebene3'
        title_suffix = f"{focus_category} — {drill_path[1]}"
    else:
        eff_level = 'ebene2'
        title_suffix = focus_category

    # Fehlende Hierarchie-Spalten auffüllen
    for lvl in ('ebene2', 'ebene3'):
        if lvl not in df.columns:
            df[lvl] = None

    # ===== Aggregation je Datum x Ebene =====
    agg = (df.groupby(['date', eff_level], as_index=False)[data_type]
             .sum()
             .rename(columns={data_type: 'wert'}))

    if agg.empty:
        return _placeholder_fig("Keine Daten nach Aggregation")

    # ===== Glättung (rolling mean je Serie) =====
    if s and s > 1:
        agg = agg.sort_values(['date', eff_level]).set_index('date')
        agg['wert'] = (agg.groupby(eff_level)['wert']
                         .transform(lambda x: x.rolling(s, min_periods=1).mean()))
        agg = agg.reset_index()

    # ===== Unterebenen nur erlaubte Kinder zeigen =====
    if focus_category and focus_category != 'all':
        mapping = df[['ebene1', 'ebene2', 'ebene3']].drop_duplicates()
        if eff_level == 'ebene2':
            allowed = mapping.loc[mapping['ebene1'] == focus_category, 'ebene2'].dropna().unique().tolist()
            if allowed:
                agg = agg[agg['ebene2'].isin(allowed)]
        elif eff_level == 'ebene3':
            parent_e2 = drill_path[1] if (len(drill_path) == 2 and drill_path[0] == focus_category) else None
            if parent_e2:
                allowed = mapping.loc[
                    (mapping['ebene1'] == focus_category) &
                    (mapping['ebene2'] == parent_e2),
                    'ebene3'
                ].dropna().unique().tolist()
                if allowed:
                    agg = agg[agg['ebene3'].isin(allowed)]

    if agg.empty:
        return _placeholder_fig("Keine Daten nach Filter")

    # ===== Pivot & Basis=100 =====
    pivot_key = 'ebene1' if eff_level == 'ebene1' else ('ebene2' if eff_level == 'ebene2' else 'ebene3')
    pv = agg.pivot_table(index='date', columns=pivot_key, values='wert', aggfunc='sum').sort_index()
    if pv.empty:
        return _placeholder_fig("Keine Daten im Zeitraum")

    def _first_valid_nonzero(s_: pd.Series):
        s_ = s_.replace(0, np.nan).dropna()
        return s_.iloc[0] if len(s_) else np.nan

    base = pv.apply(_first_valid_nonzero)
    norm = pv.div(base).replace([np.inf, -np.inf], np.nan)
    norm = norm.fillna(method='bfill').fillna(method='ffill').fillna(1.0) * 100.0

    # ===== Figure bauen =====
    fig = go.Figure()
    if norm.empty:
        fig.update_layout(title="Keine Daten im Zeitraum", template="plotly_white", height=CHART_HEIGHT)
        return fig

    if eff_level == 'ebene1':
        for i, cat in enumerate(norm.columns):
            color_hex = _get_category_color(cat)
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[cat], name=cat, mode='lines',
                line=dict(color=color_hex, width=3),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{cat}: %{{y:.1f}}<extra></extra>"
            ))
    else:
        # Unterebenen: Parent-Farbe schattieren
        parent_hex = _get_category_color(focus_category) if focus_category else '#6c757d'
        r, g, b = _hex_to_rgb_tuple(parent_hex)
        for i, cat in enumerate(norm.columns):
            step = 0.10 if eff_level == 'ebene2' else 0.14
            opacity = max(0.25, 1.0 - (i * step))
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[cat], name=cat, mode='lines',
                line=dict(color=f"rgba({r},{g},{b},{opacity})", width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{cat}: %{{y:.1f}}<extra></extra>"
            ))

    # Log-Achse nur bei Beständen und durchweg positiven Werten
    use_log_axis = bool(use_log and not is_fluss_mode and (norm > 0).all().all())
    fig.update_yaxes(type='log' if use_log_axis else 'linear')

    # X-Ticks: Formatierung auf Quartale
    format_axis_quarters(fig, norm.index)

    fig.update_layout(
        title=f"Performance-Entwicklung: {'Alle Kategorien' if focus_category in (None, 'all') else title_suffix}",
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(
            title='Jahr'
        ),
        yaxis=dict(
            title=f'Performance (Basis = 100){" — Log" if use_log_axis else ""}'
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)", borderwidth=0
        ),
        margin=dict(b=80),
        height=CHART_HEIGHT
    )
    fig.add_hline(y=100, line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'))

    return fig


@app.callback(
    [
        Output('performance-metrics-table', 'children'),
        Output('momentum-kpi-table', 'data')
    ],
    [
        Input('performance-focus-dropdown', 'value'),
        Input('data-metadata-store', 'modified_timestamp'),
        Input('zeitraum-slider', 'value')  # Zeitraum macht die Metriken sensitiv
    ],
    [
        State('gvb-data-store', 'data'),
        State('exog-data-store', 'data'),
        State('selected-categories-store', 'data')  # optionaler Drill-Path: ["<ebene1>", "<ebene2>"]
    ]
)
def update_performance_metrics(focus_category, _ts, zeitraum, gvb_json, exog_json, drill_path):
    """
    Performance (CAGR/Volatilität/Drawdown) über den gewählten Zeitraum (Range-Slider)
    + Momentum-KPIs (immer YTD zum letzten Datum). Vollständig Store-basiert.
    """
    import numpy as np
    import pandas as pd
    from dash import dash_table, html
    from dash.dash_table import FormatTemplate
    from dash.dash_table.Format import Format, Scheme

    BRAND = globals().get("BRAND_COLOR", "#14324E")

    # --- Guards / Laden ---
    if not gvb_json:
        return html.Div("Daten werden geladen …"), []

    try:
        df = pd.read_json(gvb_json, orient='split')
    except Exception as e:
        logger.error(f"[PerformanceMetrics] GVB unlesbar: {e}")
        return html.Div("GVB-Daten unlesbar"), []

    if df is None or df.empty:
        return html.Div("Keine Daten verfügbar"), []

    # Pflichtspalten robust herstellen
    for col in ['date', 'ebene1', 'ebene2', 'ebene3', 'bestand', 'fluss']:
        if col not in df.columns:
            df[col] = np.nan

    # Datumsnormalisierung
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    if df.empty:
        return html.Div("Keine gültigen Datumswerte"), []

    # Welche Wertespalte?
    value_col = 'bestand' if df['bestand'].notna().any() else ('fluss' if df['fluss'].notna().any() else None)
    if value_col is None:
        return html.Div("Keine Wertespalte (weder 'bestand' noch 'fluss')"), []

    # --- Zeitraum wie in den Charts (macht Kennzahlen sensitiv) ---
    def _safe_window(idx, slider_val, fallback_years=3):
        if idx.empty:
            return None, None
        dmin, dmax = idx.min(), idx.max()
        start = end = None
        if isinstance(slider_val, (list, tuple)) and len(slider_val) == 2:
            try:
                a, b = int(slider_val[0]), int(slider_val[1])
                y0, y1 = (a, b) if a <= b else (b, a)
                start = pd.Timestamp(year=y0, month=1, day=1)
                end   = pd.Timestamp(year=y1, month=12, day=31)
            except Exception:
                pass
        if start is None or end is None:
            end = dmax
            start = dmax - pd.DateOffset(years=fallback_years)
        start = max(start, dmin); end = min(end, dmax)
        if start >= end:
            start = end - pd.DateOffset(months=1)
            if start < dmin: start = dmin
            if start >= end: return None, None
        return start.normalize(), end.normalize()

    start_window, end_window = _safe_window(df['date'], zeitraum, 5)
    if start_window is None:
        return html.Div("Ungültiger Zeitraum"), []

    # --- Performance-Tabelle (Zeitraum des Sliders) ---
    # Ebenen-/Parent-Logik analog zu den Charts
    if focus_category in (None, 'all'):
        level_perf = 'ebene1'
        df_perf = df.copy()
    else:
        level_perf = 'ebene2'
        df_perf = df[df['ebene1'] == focus_category].copy()

    # Pivot über Zeitraum
    pv = (df_perf
          .loc[(df_perf['date'] >= start_window) & (df_perf['date'] <= end_window)]
          .pivot_table(index='date', columns=level_perf, values=value_col, aggfunc='sum')
          .sort_index())

    if pv.empty:
        perf_component = html.Div("Keine Daten im gewählten Zeitraum")
    else:
        # Frequenz robust schätzen (für Annualisierung)
        def _periods_per_year(index: pd.Index) -> float:
            if len(index) < 3:
                return 4.0  # Default: Quartal
            idx = pd.to_datetime(index)
            diffs = (idx[1:] - idx[:-1]).days
            med = np.median(diffs) if len(diffs) else 90.0
            if med <= 0:
                return 4.0
            return float(np.clip(round(365.25 / med), 1, 365))  # z.B. 12, 4, 52, …

        k_annual = _periods_per_year(pv.index)

        rows = []
        for cat in pv.columns:
            s = pd.to_numeric(pv[cat], errors='coerce').dropna()
            if len(s) < 5:
                continue

            # CAGR über sichtbaren Zeitraum
            years = max(len(s) / k_annual, 1e-9)
            start_val, end_val = float(s.iloc[0]), float(s.iloc[-1])
            cagr = (end_val / start_val) ** (1.0 / years) - 1.0 if start_val > 0 else 0.0

            # Volatilität (annualisiert)
            rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            vol = (rets.std() * np.sqrt(k_annual)) if len(rets) > 1 else 0.0

            # Max Drawdown über sichtbaren Zeitraum
            peak = s.cummax()
            dd = (s - peak) / peak.replace(0, np.nan)
            max_dd = float(dd.min()) if len(dd) else 0.0

            rows.append({
                "kategorie": str(cat),
                "current_value": end_val,
                "cagr": float(cagr),
                "volatility": float(vol),
                "max_drawdown": float(max_dd)
            })

        if not rows:
            perf_component = html.Div("Nicht genügend Daten im gewählten Zeitraum")
        else:
            columns = [
                {"name": "Kategorie",        "id": "kategorie"},
                {"name": "Aktueller Wert",   "id": "current_value", "type": "numeric",
                 "format": Format(precision=0, scheme=Scheme.fixed).group(True)},
                {"name": "CAGR",             "id": "cagr",          "type": "numeric",
                 "format": FormatTemplate.percentage(1)},
                {"name": "Volatilität",      "id": "volatility",    "type": "numeric",
                 "format": FormatTemplate.percentage(1)},
                {"name": "Max. Drawdown",    "id": "max_drawdown",  "type": "numeric",
                 "format": FormatTemplate.percentage(1)},
            ]

            style_data_conditional = [
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'filter_query': '{max_drawdown} < 0',  'column_id': 'max_drawdown'},
                 'color': '#dc3545', 'fontWeight': '600'},
                {'if': {'filter_query': '{max_drawdown} >= 0', 'column_id': 'max_drawdown'},
                 'color': '#28a745', 'fontWeight': '600'},
            ]

            perf_component = dash_table.DataTable(
                data=rows,
                columns=columns,
                style_cell={'textAlign': 'center', 'fontSize': '11px', 'padding': '6px'},
                style_header={'backgroundColor': BRAND, 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditional,
                sort_action="native",
                page_action="none"
            )

    # --- Momentum-KPIs (immer YTD des letzten verfügbaren Jahres) ---
    drill_path = drill_path or []
    if focus_category in (None, 'all'):
        level_mom = 'ebene1'
        df_mom = df.copy()
        parent_e1, parent_e2 = None, None
    elif isinstance(drill_path, list) and len(drill_path) == 2 and drill_path[0] == focus_category:
        level_mom = 'ebene3'
        parent_e1, parent_e2 = focus_category, drill_path[1]
        df_mom = df[(df['ebene1'] == parent_e1) & (df['ebene2'] == parent_e2)].copy()
    else:
        level_mom = 'ebene2'
        parent_e1, parent_e2 = focus_category, None
        df_mom = df[df['ebene1'] == parent_e1].copy()

    if df_mom.empty:
        return perf_component, []

    pv_mom = (df_mom
              .pivot_table(index='date', columns=level_mom, values=value_col, aggfunc='sum')
              .sort_index()
              .fillna(0.0))
    if pv_mom.empty:
        return perf_component, []

    last_date = pv_mom.index.max()
    ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    pv_ytd = pv_mom.loc[(pv_mom.index >= ytd_start) & (pv_mom.index <= last_date)]
    if pv_ytd.empty:
        # Kein YTD im selben Jahr → nimm letztes volles Jahr
        last_year = int(last_date.year)
        pv_ytd = pv_mom[pv_mom.index.year == last_year]
        if pv_ytd.empty:
            return perf_component, []

    def _slope_last_n(s: pd.Series, n: int = 4) -> float:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s) < 2:
            return 0.0
        s = s.tail(n)
        x = np.arange(len(s), dtype=float)
        denom = np.dot(x - x.mean(), x - x.mean())
        if denom == 0:
            return 0.0
        slope = float(np.dot(x - x.mean(), s.values - s.mean()) / denom)
        return slope

    def _pct_change_ytd(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s) < 2:
            return 0.0
        base, curr = float(s.iloc[0]), float(s.iloc[-1])
        return ((curr / base - 1.0) * 100.0) if base != 0 else 0.0

    def _rsi_quarterly(s: pd.Series, period: int = 8) -> float:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s) < 3:
            return 50.0
        delta = s.diff().dropna()
        if len(delta) < 1:
            return 50.0
        gains = delta.clip(lower=0.0)
        losses = (-delta.clip(upper=0.0))
        roll_g = gains.ewm(alpha=1/period, adjust=False).mean()
        roll_l = losses.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_g.iloc[-1] / roll_l.iloc[-1] if roll_l.iloc[-1] != 0 else np.inf
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if not np.isfinite(rsi):
            rsi = 100.0
        return float(np.clip(rsi, 0.0, 100.0))

    momentum_records = []
    for cat in pv_ytd.columns:
        s_ytd = pv_ytd[cat]
        if s_ytd.dropna().empty:
            continue
        rec = {
            "Kategorie": str(cat),
            "Kurzfristiger Trend": round(_slope_last_n(s_ytd, n=4), 1),
            "Momentum Score (%)": round(_pct_change_ytd(s_ytd), 1) / 100.0,  # 0..1 für Dash %
            "RSI": round(_rsi_quarterly(pv_mom[cat].tail(12), period=8), 1)
        }
        momentum_records.append(rec)

    if momentum_records:
        momentum_records = sorted(momentum_records, key=lambda r: r["Momentum Score (%)"], reverse=True)

    return perf_component, momentum_records




def register_overview_callbacks(real_app, *, Log=_SafeLog(), DataManager=None, StoreSource=None, DiskSource=None):
    """
    Registriert alle in diesem Modul deklarierten Overview-Callbacks auf der echten Dash-App.
    Wird aus app.py aufgerufen und injiziert Log/DataManager/StoreSource/DiskSource.
    """
    # Injizierte Objekte global verfügbar machen
    globals().update({
        "Log": Log,
        "_DataManager": DataManager,
        "_StoreSource": StoreSource,
        "_DiskSource": DiskSource,
    })

    # Alle zuvor mit dem Proxy gesammelten Callback-Deklarationen auf real_app mappen
    regs = getattr(app, "_registrations", [])
    for args, kwargs, fn in regs:
        real_app.callback(*args, **kwargs)(fn)

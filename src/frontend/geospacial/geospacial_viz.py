# geospacial/geospacial_viz.py

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import pandas as pd
import folium
import branca.colormap as cm

# Geospacial App-Root-Logik
GEO_DIR = Path(__file__).resolve().parent  # .../geospacial

try:
    APP_ROOT: Path = GEO_DIR.parent        # Projektroot (typisch: dort liegt app.py)
except Exception:
    APP_ROOT = Path.cwd()

DATA_INPUT_DIR = GEO_DIR / "data_input"

# GeoJSON / Shapes
BASE_KRS = DATA_INPUT_DIR / "Shapefiles - KRS"
BASE_GEM = DATA_INPUT_DIR / "Shapefiles - GEM"
BASE_VBGEM = DATA_INPUT_DIR / "Shapefiles - VBGEM"

OUTPUT_MAP_DIR = GEO_DIR / "data_output"
OUTPUT_MAP_DIR.mkdir(parents=True, exist_ok=True)


# dieselben Keys wie im main
LEVEL_TO_SHAPE: Dict[str, Tuple[Path, str]] = {
    "krs": (BASE_KRS, "KRS"),
    "gem": (BASE_GEM, "GEM"),
    "vbgem": (BASE_VBGEM, "VBGEM"),
}

DEBUG = False

MISSING_VALUES = frozenset({
    None,
    "",
    "nan",
    "-9999", "-9999.0", "-9999,0",
    "-99999", "-99999.0", "-99999,0",
    "-999999", "-999999.0", "-999999,0",
})
GKZ_PROPERTIES = ("GKZ", "Gebietskennziffer", "VBG")
NAME_FIELD_CANDIDATES = ("Gemeindename", "Gemeindeverbandsname", "Kreisname", "NAME", "GEN")

# Caches
_GEOJSON_CACHE_RAW: Dict[Tuple[str, str], dict] = {}
_GEOJSON_CACHE_NORM: Dict[Tuple[str, str], dict] = {}
# (level, indicator, visible_gkz_tuple) -> html
_MAP_HTML_CACHE: Dict[Tuple[str, str, Optional[Tuple[str, ...]]], str] = {}

def build_empty_map(level: str = "krs", message: str = "Bitte wählen Sie einen Indikator und klicken Sie auf 'Filter anwenden'.") -> str:
    """
    Erstellt eine leere Basis-Karte von Deutschland mit Grenzen aber ohne Daten.
    Wird beim initialen Laden angezeigt.
    """
    shape_info = LEVEL_TO_SHAPE.get(level)
    if not shape_info:
        return f"<p style='text-align:center; padding:50px;'>{message}</p>"
    
    base_dir, prefix = shape_info
    gj = _get_normalized_geojson(base_dir, prefix)
    if not gj:
        return f"<p style='text-align:center; padding:50px;'>Karte kann nicht geladen werden.</p>"
    
    # Erstelle Karte
    m = folium.Map(location=[51.163, 10.447], zoom_start=6, tiles="cartodbpositron")
    
    # Style-Funktion: nur Grenzen, keine Füllung
    def empty_style_function(feature):
        return {
            "fillColor": "#f0f0f0",
            "color": "#666666",
            "weight": 0.5,
            "fillOpacity": 0.1,
        }
    
    # GeoJSON ohne Daten hinzufügen
    folium.GeoJson(
        gj,
        name="Deutschland",
        style_function=empty_style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["GKZ"],
            aliases=["Region"],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)
    
    # Info-Text auf der Karte
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 400px; 
                height: 60px; 
                background-color: white; 
                border: 2px solid grey; 
                z-index: 9999; 
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 5px 0;">Geo-Analyse</h4>
        <p style="margin: 0; color: #666;">{message}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m.get_root().render()



def log(*args):
    if DEBUG:
        print(*args)


def normalize_gkz(val, width=8) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    try:
        return str(int(float(s.replace(" ", "")))).zfill(width)
    except (ValueError, AttributeError):
        return s.split(".")[0].replace(" ", "").zfill(width)


def is_missing(val) -> bool:
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in MISSING_VALUES


def _load_geojson_raw(base_dir: Path, prefix: str) -> Optional[dict]:
    cache_key = (str(base_dir), prefix)
    if cache_key in _GEOJSON_CACHE_RAW:
        return _GEOJSON_CACHE_RAW[cache_key]

    geojson_path = base_dir / f"{prefix}_Map.geojson"
    if not geojson_path.exists():
        log(f"[viz] GeoJSON nicht gefunden: {geojson_path}")
        return None

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    _GEOJSON_CACHE_RAW[cache_key] = gj
    return gj


def _get_normalized_geojson(base_dir: Path, prefix: str) -> Optional[dict]:
    """
    GeoJSON laden und GKZ einmalig normalisieren
    """
    cache_key = (str(base_dir), prefix)
    if cache_key in _GEOJSON_CACHE_NORM:
        # defensive copy
        return json.loads(json.dumps(_GEOJSON_CACHE_NORM[cache_key]))

    gj = _load_geojson_raw(base_dir, prefix)
    if not gj:
        return None

    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        gkz = None
        for gkz_prop in GKZ_PROPERTIES:
            if gkz_prop in props:
                gkz = normalize_gkz(props[gkz_prop])
                break
        if gkz:
            props["GKZ"] = gkz
        feat["properties"] = props

    _GEOJSON_CACHE_NORM[cache_key] = gj
    return json.loads(json.dumps(gj))


def _merge_df_into_geojson_fast(
    geojson_data: dict,
    df: pd.DataFrame,
    indicator: str,
    display_name: Optional[str] = None,
    indicator_desc: Optional[str] = None,
    visible_gkz: Optional[Iterable[str]] = None,
) -> dict:
    """
    Schneller Merge:
    - df wird auf GKZ gemappt
    - wir schreiben den Indikator **immer** in die Properties,
      auch wenn er fehlend ist -> damit der Tooltip ihn anzeigen kann
    """
    feats = geojson_data.get("features", [])
    if not feats:
        return geojson_data

    df = df.copy()

    # GKZ sicherstellen
    if "GKZ" not in df.columns:
        for cand in ("gkz", "Gebietskennziffer", "AGS", "ags", "KRS", "VBG"):
            if cand in df.columns:
                df["GKZ"] = df[cand]
                break
    df["GKZ"] = df["GKZ"].apply(normalize_gkz)

    # Sichtbarkeitslogik: alles, was nicht in visible_gkz ist, bekommt NaN für den Indikator
    if visible_gkz:
        vis_set = {normalize_gkz(g) for g in visible_gkz}
        df.loc[~df["GKZ"].isin(vis_set), indicator] = pd.NA

    # Maps bauen
    val_map = dict(zip(df["GKZ"], df[indicator]))

    name_field = None
    for cand in NAME_FIELD_CANDIDATES:
        if cand in df.columns:
            name_field = cand
            break
    name_map = dict(zip(df["GKZ"], df[name_field])) if name_field else {}

    merged = 0
    for feat in feats:
        props = feat.get("properties", {})
        gkz = props.get("GKZ")
        if not gkz:
            continue

        v = val_map.get(gkz)
        nm = name_map.get(gkz)

        new_props = {
            "GKZ": gkz,
            "indicator_code": indicator,
            "indicator_name": display_name or indicator,
        }
        if indicator_desc:
            new_props["indicator_desc"] = indicator_desc
        if nm is not None and name_field:
            new_props[name_field] = nm

        # WICHTIG: Indikatorfeld **immer** schreiben, auch wenn fehlend
        if v is None or is_missing(v):
            # Platzhalter, damit Tooltip was hat
            new_props[indicator] = "–"
            feat["properties"] = new_props
            continue

        new_props[indicator] = v
        feat["properties"] = new_props
        merged += 1

    log(f"[viz] {merged} Features gemerged (fast).")
    return geojson_data


def build_map_from_df(
    level: str,
    df: pd.DataFrame,
    indicator: str,
    display_name: Optional[str] = None,
    indicator_desc: Optional[str] = None,
    visible_gkz: Optional[Iterable[str]] = None,
) -> str:
    """
    Baut eine Folium-Karte und cached sie nach (level, indicator, visible_gkz)
    """
    vis_key = tuple(sorted({normalize_gkz(g) for g in visible_gkz})) if visible_gkz else None
    cache_key = (level, indicator, vis_key)
    if cache_key in _MAP_HTML_CACHE:
        return _MAP_HTML_CACHE[cache_key]

    shape_info = LEVEL_TO_SHAPE.get(level)
    if not shape_info:
        return "<p>Kein Shape für diese Ebene vorhanden.</p>"

    base_dir, prefix = shape_info
    gj = _get_normalized_geojson(base_dir, prefix)
    if not gj:
        return "<p>GeoJSON konnte nicht geladen werden.</p>"

    # Data in GeoJSON schreiben
    gj = _merge_df_into_geojson_fast(
        gj,
        df,
        indicator=indicator,
        display_name=display_name,
        indicator_desc=indicator_desc,
        visible_gkz=visible_gkz,
    )

    feats = gj.get("features", [])
    cleaned_vals = []
    for f in feats:
        raw_v = f["properties"].get(indicator)
        if is_missing(raw_v) or raw_v == "–":
            continue
        try:
            cleaned_vals.append(float(str(raw_v).replace(",", ".")))
        except (ValueError, TypeError):
            continue

    vmin, vmax = (min(cleaned_vals), max(cleaned_vals)) if cleaned_vals else (0, 1)

    m = folium.Map(location=[51.163, 10.447], zoom_start=6, tiles="cartodbpositron")

    legend_caption = display_name or indicator
    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = legend_caption

    base_props = feats[0]["properties"] if feats else {}
    tooltip_fields = []
    tooltip_aliases = []
    for cand, alias in [
        ("GKZ", "GKZ"),
        ("Gemeindename", "Gemeinde"),
        ("Gemeindeverbandsname", "Gemeindeverbund"),
        ("Kreisname", "Kreis"),
        ("indicator_name", "Indikator"),
        (indicator, "Wert"),
    ]:
        if cand in base_props:
            tooltip_fields.append(cand)
            tooltip_aliases.append(alias)

    def style_function(feature):
        v = feature["properties"].get(indicator)
        if is_missing(v) or v == "–":
            # durchsichtig, nur Umriss
            return {
                "fillColor": "#ffffff00",
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.0,
            }
        try:
            vv = float(str(v).replace(",", "."))
            color = colormap(vv)
        except (ValueError, TypeError):
            color = "#cccccc"
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        gj,
        name=legend_caption,
        style_function=style_function,
        tooltip=(
            folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True,
            )
            if tooltip_fields
            else None
        ),
    ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    html_str = m.get_root().render()
    _MAP_HTML_CACHE[cache_key] = html_str
    return html_str
"""
exog_instructor.py
Einfache Routine zum Download von ECB-Indikatoren.

Ziel dieses Files:
- Möglichst unabhängig vom (evtl. fehlerhaften) loader.py funktionieren
- loader/ und loader/sources/ automatisch in sys.path hängen
- sources.ecb_client robust importieren
- transforms notfalls minimal bereitstellen (DataProcessor + format_date_for_ecb_api)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd
import asyncio

# ============================================================
# 1) Pfade robust setzen (loader/ und loader/sources/)
# ============================================================
# Annahme: dieses File liegt in .../loader/exog_instructor.py
_THIS_FILE = Path(__file__).resolve()
_LOADER_DIR = _THIS_FILE.parent               # .../loader
_SOURCES_DIR = _LOADER_DIR / "sources"        # .../loader/sources

# loader/ in sys.path aufnehmen, falls nicht drin
if str(_LOADER_DIR) not in sys.path:
    sys.path.insert(0, str(_LOADER_DIR))

# loader/sources/ in sys.path aufnehmen, falls vorhanden
if _SOURCES_DIR.exists() and str(_SOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCES_DIR))

# ============================================================
# 2) Optional: loader importieren – aber Fehler NICHT eskalieren
#    (dein aktuelles loader.py hatte Syntaxfehler, dann sollen wir
#     hier trotzdem weiterarbeiten können)
# ============================================================
LOADER_AVAILABLE = False
try:
    import loader  # noqa: F401
    LOADER_AVAILABLE = True
except Exception as e:
    print(
        "[exog_instructor] Hinweis: loader.py konnte nicht importiert werden "
        f"(wir machen trotzdem weiter): {e}"
    )

# ============================================================
# 3) transforms robust importieren / notfalls minimal nachrüsten
#    Dein ecb_client.py erwartet:
#       from transforms import DataProcessor, format_date_for_ecb_api
# ============================================================
try:
    from transforms import DataProcessor, format_date_for_ecb_api  # type: ignore
except Exception:
    # Minimal-Version bereitstellen
    import types
    import pandas as _pd

    print("[exog_instructor] Hinweis: 'transforms' nicht gefunden – minimale Fallback-Version wird erstellt.")

    transforms = types.ModuleType("transforms")

    class DataProcessor:
        """
        Sehr kleine Version deines DataProcessor:
        Erzwingt Spalten 'Datum' und 'value'
        """
        @staticmethod
        def standardize_dataframe(df: _pd.DataFrame) -> _pd.DataFrame:
            if df is None or df.empty:
                return _pd.DataFrame(columns=["Datum", "value"])

            dfc = df.copy()

            # Datumsspalte finden
            date_col = None
            for cand in ["Datum", "DATE", "TIME_PERIOD", "period", "Period", "date"]:
                if cand in dfc.columns:
                    date_col = cand
                    break
            if date_col is None:
                # fallback: erste Spalte
                date_col = dfc.columns[0]

            # Valuespalte finden
            value_col = None
            for cand in ["value", "VALUE", "OBS_VALUE", "Wert", "wert"]:
                if cand in dfc.columns and cand != date_col:
                    value_col = cand
                    break
            if value_col is None:
                # nimm eine andere Spalte
                for c in dfc.columns:
                    if c != date_col:
                        value_col = c
                        break

            out = _pd.DataFrame()
            out["Datum"] = _pd.to_datetime(dfc[date_col], errors="coerce")
            out["value"] = _pd.to_numeric(dfc[value_col], errors="coerce")
            out = out.dropna(subset=["Datum", "value"]).sort_values("Datum").reset_index(drop=True)
            return out

    def format_date_for_ecb_api(value: str) -> str:
        """
        ECB erwartet meistens YYYY-MM oder YYYY-MM-DD, im Zweifel geben wir denselben
        String zurück. Wir versuchen aber eine leichte Normalisierung.
        """
        value = (value or "").strip()
        if not value:
            return value
        # YYYY-MM → ok
        if len(value) == 7 and value[4] == "-":
            return value
        # YYYY-MM-DD → ok
        if len(value) == 10 and value[4] == "-" and value[7] == "-":
            return value[:7]
        # Versuch parse
        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%Y-%m")
        except Exception:
            return value

    transforms.DataProcessor = DataProcessor
    transforms.format_date_for_ecb_api = format_date_for_ecb_api

    # Imports im aktuellen Modulraum bereitstellen
    DataProcessor = DataProcessor
    format_date_for_ecb_api = format_date_for_ecb_api

    # auch ins sys.modules hängen, falls andere Module es brauchen
    sys.modules["transforms"] = transforms

# ============================================================
# 4) ECB-Client robust importieren
#    (du hast uns ecb_client.py gegeben → liegt vermutlich in loader/sources)
# ============================================================
_fetch_ecb_async = None
_fetch_ecb_sync = None

# mehrere Import-Pfade probieren
_import_errors = []

for mod_name in (
    "sources.ecb_client",   # loader/sources/ecb_client.py
    "ecb_client",           # direkt im loader/
    "scenario.data_sources.ecb_client",  # falls du es mal nach scenario/... kopiert hast
):
    if _fetch_ecb_sync is not None:
        break
    try:
        m = __import__(mod_name, fromlist=["fetch_ecb_async", "fetch_ecb_sync"])
        _fetch_ecb_async = getattr(m, "fetch_ecb_async", None)
        _fetch_ecb_sync = getattr(m, "fetch_ecb_sync", None)
    except Exception as e:
        _import_errors.append((mod_name, str(e)))

if _fetch_ecb_sync is None:
    # hier geben wir eine freundliche Fehlermeldung aus, aber lassen das Modul importierbar
    print("[exog_instructor] Konnte 'sources.ecb_client' nicht importieren. "
          "Stelle sicher, dass dein Ordner 'loader/sources' im PYTHONPATH liegt oder "
          "dass du ecb_client.py dorthin gelegt hast.")
    for name, err in _import_errors:
        print(f"  - Versuch {name}: {err}")

# ============================================================
# 5) Optional: aiohttp für Async-Download
# ============================================================
try:
    import aiohttp  # type: ignore
    ASYNC_AVAILABLE = True
except Exception:
    aiohttp = None  # type: ignore
    ASYNC_AVAILABLE = False


class ECBInstructor:
    """
    Klasse zum Download von ECB-Zeitreihen-Indikatoren.
    """

    def __init__(
        self,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        timeout_seconds: int = 30,
        min_response_size: int = 100,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.timeout_seconds = timeout_seconds
        self.min_response_size = min_response_size

        self.failed_indicators: List[str] = []
        self.success_indicators: List[str] = []

        if _fetch_ecb_sync is None:
            # schon beim Erzeugen warnen
            print(
                "[ECBInstructor] WARNUNG: Kein ECB-Client importierbar. "
                "Downloads werden fehlschlagen, bis 'sources/ecb_client.py' verfügbar ist."
            )

    # -------------------------------------------------------
    # interne Helfer
    # -------------------------------------------------------
    def _validate_ecb_code(self, code: str) -> bool:
        if not isinstance(code, str) or not code.strip():
            return False
        if "." not in code:
            return False
        ecb_prefixes = (
            "ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.",
            "STS.", "MNA.", "BOP.", "GFS.", "EXR.",
        )
        return code.upper().startswith(ecb_prefixes)

    # -------------------------------------------------------
    # async download eines einzelnen Codes
    # -------------------------------------------------------
    async def _fetch_indicator_async(
        self,
        session: "aiohttp.ClientSession",
        code: str,
    ) -> Optional[pd.DataFrame]:
        if _fetch_ecb_async is None:
            # kein async-client verfügbar → None
            print(f"[ECBInstructor] Async-Client nicht verfügbar, skip: {code}")
            self.failed_indicators.append(code)
            return None

        try:
            df = await _fetch_ecb_async(
                session=session,
                code=code,
                start=self.start_date,
                end=self.end_date,
                min_response_size=self.min_response_size,
                timeout_seconds=self.timeout_seconds,
            )
            if df is not None and not df.empty:
                if "Datum" in df.columns:
                    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
                self.success_indicators.append(code)
                return df
            else:
                self.failed_indicators.append(code)
                return None
        except Exception as e:
            print(f"[ECBInstructor] Fehler beim async Download von {code}: {e}")
            self.failed_indicators.append(code)
            return None

    # -------------------------------------------------------
    # sync download eines einzelnen Codes
    # -------------------------------------------------------
    def _fetch_indicator_sync(self, code: str) -> Optional[pd.DataFrame]:
        if _fetch_ecb_sync is None:
            print(f"[ECBInstructor] Kein sync-ECB-Client verfügbar, skip: {code}")
            self.failed_indicators.append(code)
            return None

        try:
            df = _fetch_ecb_sync(
                code=code,
                start=self.start_date,
                end=self.end_date,
                min_response_size=self.min_response_size,
                timeout_seconds=self.timeout_seconds,
            )
            if df is not None and not df.empty:
                if "Datum" in df.columns:
                    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
                self.success_indicators.append(code)
                return df
            else:
                self.failed_indicators.append(code)
                return None
        except Exception as e:
            print(f"[ECBInstructor] Fehler beim sync Download von {code}: {e}")
            self.failed_indicators.append(code)
            return None

    # -------------------------------------------------------
    # mehrere async
    # -------------------------------------------------------
    async def _download_async(self, indicators: List[str]) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        if not ASYNC_AVAILABLE or _fetch_ecb_async is None:
            # Fallback auf sync
            return self._download_sync(indicators)

        async with aiohttp.ClientSession() as session:  # type: ignore
            for code in indicators:
                df = await self._fetch_indicator_async(session, code)
                if df is not None:
                    results[code] = df
                await asyncio.sleep(0.25)  # kleine Pause
        return results

    # -------------------------------------------------------
    # mehrere sync
    # -------------------------------------------------------
    def _download_sync(self, indicators: List[str]) -> Dict[str, pd.DataFrame]:
        import time
        results: Dict[str, pd.DataFrame] = {}
        for code in indicators:
            df = self._fetch_indicator_sync(code)
            if df is not None:
                results[code] = df
            time.sleep(0.25)
        return results

    # -------------------------------------------------------
    # Mergen
    # -------------------------------------------------------
    def _merge_dataframes(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data_dict:
            return pd.DataFrame()

        series_list = []
        for code, df in data_dict.items():
            if df is None or df.empty:
                continue
            if "Datum" in df.columns and "value" in df.columns:
                s = df.set_index("Datum")["value"]
                s.name = code
                series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        merged = pd.concat(series_list, axis=1, sort=True)
        merged = merged.reset_index().rename(columns={"index": "Datum"})
        merged = merged.sort_values("Datum").reset_index(drop=True)
        return merged

    # -------------------------------------------------------
    # öffentliche Hauptmethode
    # -------------------------------------------------------
    def download(self, indicators: List[str], use_async: Optional[bool] = None) -> pd.DataFrame:
        # Status zurücksetzen
        self.failed_indicators = []
        self.success_indicators = []

        if not indicators:
            raise ValueError("Indikatorliste darf nicht leer sein")

        valid_indicators: List[str] = []
        invalid_indicators: List[str] = []

        for code in indicators:
            code = code.strip()
            if self._validate_ecb_code(code):
                valid_indicators.append(code)
            else:
                invalid_indicators.append(code)

        if invalid_indicators:
            print(f"[ECBInstructor] Warnung: {len(invalid_indicators)} ungültige ECB-Codes übersprungen:")
            for c in invalid_indicators[:5]:
                print(f"  - {c}")

        if not valid_indicators:
            raise ValueError("Keine gültigen ECB-Indikatorcodes gefunden")

        print(f"[ECBInstructor] Starte Download von {len(valid_indicators)} Indikatoren")
        print(f"[ECBInstructor] Zeitraum: {self.start_date} bis {self.end_date}")

        if use_async is None:
            use_async = ASYNC_AVAILABLE and (_fetch_ecb_async is not None)
        elif use_async and not ASYNC_AVAILABLE:
            print("[ECBInstructor] aiohttp nicht verfügbar – wechsle auf sync")
            use_async = False

        if use_async:
            try:
                data_dict = asyncio.run(self._download_async(valid_indicators))
            except RuntimeError as e:
                # z. B. „cannot be called from a running event loop“
                if "cannot be called from a running event loop" in str(e):
                    import nest_asyncio  # type: ignore
                    nest_asyncio.apply()
                    data_dict = asyncio.run(self._download_async(valid_indicators))
                else:
                    raise
        else:
            data_dict = self._download_sync(valid_indicators)

        result = self._merge_dataframes(data_dict)

        print("\n" + "=" * 60)
        print("[ECBInstructor] Download abgeschlossen:")
        print(f"  ✓ Erfolgreich:   {len(self.success_indicators)}/{len(valid_indicators)}")
        print(f"  ✗ Fehlgeschlagen: {len(self.failed_indicators)}/{len(valid_indicators)}")

        if self.failed_indicators:
            print("  Fehlgeschlagene Codes (max 10):")
            for c in self.failed_indicators[:10]:
                print(f"    - {c}")

        if not result.empty:
            print("  Resultierender DataFrame:")
            print(f"    - Zeitraum: {result['Datum'].min()} bis {result['Datum'].max()}")
            print(f"    - Beobachtungen: {len(result)}")
            print(f"    - Indikatoren:   {len(result.columns) - 1}")
        print("=" * 60 + "\n")

        return result


# ============================================================
# 6) Convenience-Funktion
# ============================================================
def download_ecb_indicators(
    indicators: List[str],
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    instructor = ECBInstructor(
        start_date=start_date,
        end_date=end_date,
        timeout_seconds=timeout_seconds,
    )
    return instructor.download(indicators)


# ============================================================
# 7) Manuelles Testen
# ============================================================
if __name__ == "__main__":
    test_indicators = [
        "ICP.M.U2.N.000000.4.ANR",
        "ICP.M.U2.N.010000.4.ANR",
    ]
    df = download_ecb_indicators(
        indicators=test_indicators,
        start_date="2015-01-01",
        end_date="2024-12-31",
    )
    print(df.head())

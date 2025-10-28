"""
instructor.py
Einfache Routine zum Download von ECB-Indikatoren unter Nutzung der loader.py-Funktionen
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

# WICHTIG: loader.py muss zuerst importiert werden, um die Module zu registrieren
import loader

# Jetzt können wir die Module importieren
from transforms import DataProcessor
from sources.ecb_client import fetch_ecb_async, fetch_ecb_sync

try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class ECBInstructor:
    """
    Klasse zum Download von ECB-Zeitreihen-Indikatoren.
    
    Attributes:
        start_date: Startdatum im Format 'YYYY-MM-DD' oder 'YYYY-MM'
        end_date: Enddatum im Format 'YYYY-MM-DD' oder 'YYYY-MM'
        timeout_seconds: Timeout für API-Anfragen
        min_response_size: Minimale Antwortgröße in Bytes
    """
    
    def __init__(
        self,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        timeout_seconds: int = 30,
        min_response_size: int = 100
    ):
        """
        Initialisiert den ECBInstructor.
        
        Args:
            start_date: Startdatum für den Download
            end_date: Enddatum für den Download (Standard: heute)
            timeout_seconds: Timeout für API-Requests
            min_response_size: Minimale erwartete Antwortgröße
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.timeout_seconds = timeout_seconds
        self.min_response_size = min_response_size
        self.failed_indicators: List[str] = []
        self.success_indicators: List[str] = []
    
    def _validate_ecb_code(self, code: str) -> bool:
        """
        Validiert, ob ein Code ein gültiger ECB-Code ist.
        
        Args:
            code: Der zu prüfende Indikatorcode
            
        Returns:
            True wenn gültig, False sonst
        """
        if not isinstance(code, str) or not code.strip():
            return False
        
        # ECB-Codes haben typischerweise ein Format wie "ICP.M.U2.N.000000.4.ANR"
        # oder "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"
        if "." not in code:
            return False
        
        # Prüfe auf bekannte ECB-Prefixes
        ecb_prefixes = (
            "ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", 
            "STS.", "MNA.", "BOP.", "GFS.", "EXR."
        )
        
        return code.upper().startswith(ecb_prefixes)
    
    async def _fetch_indicator_async(
        self, 
        session: 'aiohttp.ClientSession',
        code: str
    ) -> Optional[pd.DataFrame]:
        """
        Lädt einen einzelnen Indikator asynchron herunter.
        
        Args:
            session: aiohttp ClientSession
            code: ECB-Indikatorcode
            
        Returns:
            DataFrame mit den Zeitreihendaten oder None bei Fehler
        """
        try:
            df = await fetch_ecb_async(
                session=session,
                code=code,
                start=self.start_date,
                end=self.end_date,
                min_response_size=self.min_response_size,
                timeout_seconds=self.timeout_seconds
            )
            
            if df is not None and not df.empty:
                # Stelle sicher, dass Datum korrekt formatiert ist
                if 'Datum' in df.columns:
                    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
                
                self.success_indicators.append(code)
                return df
            else:
                self.failed_indicators.append(code)
                return None
                
        except Exception as e:
            print(f"Fehler beim Download von {code}: {str(e)}")
            self.failed_indicators.append(code)
            return None
    
    def _fetch_indicator_sync(self, code: str) -> Optional[pd.DataFrame]:
        """
        Lädt einen einzelnen Indikator synchron herunter.
        
        Args:
            code: ECB-Indikatorcode
            
        Returns:
            DataFrame mit den Zeitreihendaten oder None bei Fehler
        """
        try:
            df = fetch_ecb_sync(
                code=code,
                start=self.start_date,
                end=self.end_date,
                min_response_size=self.min_response_size,
                timeout_seconds=self.timeout_seconds
            )
            
            if df is not None and not df.empty:
                if 'Datum' in df.columns:
                    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
                
                self.success_indicators.append(code)
                return df
            else:
                self.failed_indicators.append(code)
                return None
                
        except Exception as e:
            print(f"Fehler beim Download von {code}: {str(e)}")
            self.failed_indicators.append(code)
            return None
    
    async def _download_async(
        self, 
        indicators: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Lädt mehrere Indikatoren asynchron herunter.
        
        Args:
            indicators: Liste von ECB-Indikatorcodes
            
        Returns:
            Dictionary mit Code als Key und DataFrame als Value
        """
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for code in indicators:
                df = await self._fetch_indicator_async(session, code)
                if df is not None:
                    results[code] = df
                # Kurze Pause zwischen Requests
                await asyncio.sleep(0.4)
        
        return results
    
    def _download_sync(
        self, 
        indicators: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Lädt mehrere Indikatoren synchron herunter.
        
        Args:
            indicators: Liste von ECB-Indikatorcodes
            
        Returns:
            Dictionary mit Code als Key und DataFrame als Value
        """
        import time
        results = {}
        
        for code in indicators:
            df = self._fetch_indicator_sync(code)
            if df is not None:
                results[code] = df
            time.sleep(0.4)
        
        return results
    
    def _merge_dataframes(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Führt mehrere Zeitreihen in einem DataFrame zusammen.
        
        Args:
            data_dict: Dictionary mit Code->DataFrame Mapping
            
        Returns:
            Merged DataFrame mit Datum als Index und Indikatoren als Spalten
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Erstelle Liste von Series mit code als Name
        series_list = []
        for code, df in data_dict.items():
            if 'Datum' in df.columns and 'value' in df.columns:
                series = df.set_index('Datum')['value']
                series.name = code
                series_list.append(series)
        
        if not series_list:
            return pd.DataFrame()
        
        # Merge alle Series
        merged = pd.concat(series_list, axis=1, sort=True)
        merged = merged.reset_index().rename(columns={'index': 'Datum'})
        
        # Sortiere nach Datum
        merged = merged.sort_values('Datum').reset_index(drop=True)
        
        return merged
    
    def download(
        self, 
        indicators: List[str],
        use_async: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Hauptmethode zum Download von ECB-Indikatoren.
        
        Args:
            indicators: Liste von ECB-Indikatorcodes
            use_async: Erzwinge async (True) oder sync (False). 
                      None = automatische Erkennung
            
        Returns:
            DataFrame mit Datum-Spalte und einer Spalte pro Indikator
            
        Raises:
            ValueError: Wenn die Indikatorliste leer oder ungültig ist
        """
        # Reset der Statuslisten
        self.failed_indicators = []
        self.success_indicators = []
        
        # Validierung
        if not indicators:
            raise ValueError("Indikatorliste darf nicht leer sein")
        
        # Filtere und validiere Codes
        valid_indicators = []
        invalid_indicators = []
        
        for code in indicators:
            code = code.strip()
            if self._validate_ecb_code(code):
                valid_indicators.append(code)
            else:
                invalid_indicators.append(code)
        
        if invalid_indicators:
            print(f"Warnung: {len(invalid_indicators)} ungültige ECB-Codes übersprungen:")
            for code in invalid_indicators[:5]:  # Zeige max. 5 Beispiele
                print(f"  - {code}")
        
        if not valid_indicators:
            raise ValueError("Keine gültigen ECB-Indikatorcodes gefunden")
        
        print(f"\nStarte Download von {len(valid_indicators)} Indikatoren...")
        print(f"Zeitraum: {self.start_date} bis {self.end_date}")
        
        # Bestimme Download-Methode
        if use_async is None:
            use_async = ASYNC_AVAILABLE
        elif use_async and not ASYNC_AVAILABLE:
            print("Warnung: aiohttp nicht verfügbar, nutze synchronen Download")
            use_async = False
        
        # Download
        if use_async:
            try:
                data_dict = asyncio.run(self._download_async(valid_indicators))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    import nest_asyncio
                    nest_asyncio.apply()
                    data_dict = asyncio.run(self._download_async(valid_indicators))
                else:
                    raise
        else:
            data_dict = self._download_sync(valid_indicators)
        
        # Merge zu einem DataFrame
        result = self._merge_dataframes(data_dict)
        
        # Status-Report
        print(f"\n{'='*60}")
        print(f"Download abgeschlossen:")
        print(f"  ✓ Erfolgreich: {len(self.success_indicators)}/{len(valid_indicators)}")
        print(f"  ✗ Fehlgeschlagen: {len(self.failed_indicators)}/{len(valid_indicators)}")
        
        if self.failed_indicators:
            print(f"\nFehlgeschlagene Indikatoren:")
            for code in self.failed_indicators[:10]:  # Max. 10 anzeigen
                print(f"  - {code}")
        
        if not result.empty:
            print(f"\nResultierender DataFrame:")
            print(f"  - Zeitraum: {result['Datum'].min()} bis {result['Datum'].max()}")
            print(f"  - Beobachtungen: {len(result)}")
            print(f"  - Indikatoren: {len(result.columns)-1}")
        
        print(f"{'='*60}\n")
        
        return result


# Convenience-Funktion für direkten Aufruf
def download_ecb_indicators(
    indicators: List[str],
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    timeout_seconds: int = 30
) -> pd.DataFrame:
    """
    Convenience-Funktion zum schnellen Download von ECB-Indikatoren.
    
    Args:
        indicators: Liste von ECB-Indikatorcodes
        start_date: Startdatum (Format: 'YYYY-MM-DD')
        end_date: Enddatum (Standard: heute)
        timeout_seconds: Timeout für API-Requests
        
    Returns:
        DataFrame mit den heruntergeladenen Zeitreihen
        
    Example:
        >>> indicators = ["ICP.M.U2.N.000000.4.ANR", "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"]
        >>> df = download_ecb_indicators(indicators, start_date="2015-01-01")
        >>> print(df.head())
    """
    instructor = ECBInstructor(
        start_date=start_date,
        end_date=end_date,
        timeout_seconds=timeout_seconds
    )
    
    return instructor.download(indicators)


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel-Indikatoren (HICP-bezogene Daten)
    test_indicators = [
        "ICP.M.U2.N.000000.4.ANR",  # HICP - All items
        "ICP.M.U2.N.010000.4.ANR",  # HICP - Food and non-alcoholic beverages
        "ICP.M.U2.N.020000.4.ANR",  # HICP - Alcoholic beverages and tobacco
    ]
    
    # Download mit der Convenience-Funktion
    df = download_ecb_indicators(
        indicators=test_indicators,
        start_date="2000-01-01",
        end_date="2024-12-31"
    )
    
    print("\nErste Zeilen des Ergebnisses:")
    print(df.head())
    
    print("\nLetzte Zeilen des Ergebnisses:")
    print(df.tail())
    
    print("\nDataFrame-Info:")
    print(df.info())
import sys, types


def _register_module(modname: str, source: str):
    """Create a module with name 'modname', execute 'source' in its namespace,
    and register it in sys.modules so that 'import modname' works."""
    if '.' in modname:
        pkg = modname.split('.')[0]
        if pkg not in sys.modules:
            pkg_mod = types.ModuleType(pkg)
            pkg_mod.__path__ = []
            sys.modules[pkg] = pkg_mod
    m = types.ModuleType(modname)
    m.__dict__['__name__'] = modname
    exec(source, m.__dict__)
    sys.modules[modname] = m


# 1) transforms ---------------------------------------------------------------
_register_module('transforms', r'''
import re
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

ECB_PREFIXES = ("ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", "STS.", "MNA.", "BOP.", "GFS.", "EXR.")
INDEX_SPEC_RE = re.compile(r'^\s*INDEX\s*\(\s*(.*?)\s*\)\s*', re.IGNORECASE)

def detect_data_source(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Invalid series code: {code}")
    code_upper = code.upper()
    if "." in code_upper and code_upper.startswith(ECB_PREFIXES):
        return "ECB"
    return "BUNDESBANK"

def parse_index_specification(spec: str) -> Optional[List[str]]:
    if not isinstance(spec, str):
        return None
    match = INDEX_SPEC_RE.match(spec.strip())
    if not match:
        return None
    inner = match.group(1)
    codes = [c.strip() for c in inner.split(",") if c.strip()]
    return list(dict.fromkeys(codes)) if codes else None

def format_date_for_ecb_api(date_str: str) -> str:
    if not date_str:
        return date_str
    try:
        if len(date_str) == 4:
            return f"{date_str}-01"
        elif len(date_str) == 7:
            return date_str
        elif len(date_str) == 10:
            return date_str[:7]
        else:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime("%Y-%m")
    except:
        return date_str

def get_excel_engine() -> str:
    try:
        import openpyxl  # noqa
        return 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter  # noqa
            return 'xlsxwriter'
        except ImportError:
            raise ImportError("Excel support requires openpyxl or xlsxwriter. Install with: pip install openpyxl")

class DataProcessor:
    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        date_candidates = ["TIME_PERIOD", "DATE", "Datum", "Period", "period"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None and len(df.columns) > 0:
            date_col = df.columns[0]

        value_candidates = ["OBS_VALUE", "VALUE", "Wert", "Value"]
        value_col = None
        for candidate in value_candidates:
            if candidate in df.columns and candidate != date_col:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = [c for c in df.columns if c != date_col and df[c].dtype in ['float64', 'int64']]
            if numeric_cols:
                value_col = numeric_cols[-1]
            else:
                raise ValueError("No value column found")

        result = pd.DataFrame()
        result["Datum"] = pd.to_datetime(df[date_col], errors='coerce')
        result["value"] = pd.to_numeric(df[value_col], errors='coerce')
        result = result.dropna(subset=["value", "Datum"])
        result = result.sort_values("Datum").reset_index(drop=True)
        return result

class CacheManager:
    def __init__(self, cache_dir: str, cache_max_age_days: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_days = cache_max_age_days

    def _cache_path(self, code: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in code)
        return self.cache_dir / f"{safe_name}.parquet"

    def is_fresh(self, code: str) -> bool:
        cache_path = self._cache_path(code)
        if not cache_path.exists():
            return False
        try:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_days = (dt.datetime.now() - mtime).days
            return age_days <= self.cache_max_age_days
        except OSError:
            return False

    def read_cache(self, code: str):
        if not self.is_fresh(code):
            return None
        cache_path = self._cache_path(code)
        try:
            df = pd.read_parquet(cache_path)
            return DataProcessor.standardize_dataframe(df)
        except Exception:
            return None

    def write_cache(self, code: str, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        cache_path = self._cache_path(code)
        tmp = cache_path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(tmp, index=False)
            tmp.replace(cache_path)
            return True
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

class IndexCreator:
    def __init__(self, *, index_base_year: int, index_base_value: float):
        self.index_base_year = index_base_year
        self.index_base_value = index_base_value

    def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
        if 'Datum' not in data_df.columns:
            raise ValueError("DataFrame must contain a 'Datum' column")

        available_codes = [code for code in series_codes if code in data_df.columns]
        if not available_codes:
            raise ValueError(f"No valid series found for index {index_name}")

        index_data = data_df[['Datum'] + available_codes].copy()
        index_data = index_data.set_index('Datum')

        has_any = index_data[available_codes].notna().any(axis=1)
        index_data = index_data.loc[has_any].copy()

        def _fill_inside(s: pd.Series) -> pd.Series:
            if s.notna().sum() == 0:
                return s
            first, last = s.first_valid_index(), s.last_valid_index()
            if first is None or last is None:
                return s
            filled = s.ffill().bfill()
            mask = (s.index >= first) & (s.index <= last)
            return filled.where(mask, s)

        index_data[available_codes] = index_data[available_codes].apply(_fill_inside)
        clean_data = index_data.dropna()

        import numpy as np
        if clean_data.empty:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result

        weights = {code: 1.0 / len(available_codes) for code in available_codes}
        weighted_values = [clean_data[c] * weights[c] for c in available_codes if c in clean_data.columns]

        if not weighted_values:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result

        aggregated = sum(weighted_values)

        try:
            base_year_int = int(self.index_base_year)
            base_year_mask = aggregated.index.year == base_year_int
            base_year_data = aggregated[base_year_mask]

            if base_year_data.empty or base_year_data.isna().all():
                first_valid = aggregated.dropna()
                base_value_actual = first_valid.iloc[0] if not first_valid.empty else 1.0
            else:
                base_value_actual = base_year_data.mean()

            if base_value_actual == 0 or pd.isna(base_value_actual):
                base_value_actual = 1.0

            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            mask = aggregated.notna()
            result[mask] = (aggregated[mask] / base_value_actual) * float(self.index_base_value)
            return result
        except Exception as e:
            print(f"Warning: Index normalization failed for {index_name}, using raw data: {e}")
            aggregated.name = index_name
            return aggregated
''')


# 2) sources.ecb_client -------------------------------------------------------
_register_module('sources.ecb_client', r'''
import io
import pandas as pd

ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data"

from transforms import DataProcessor, format_date_for_ecb_api

try:
    import aiohttp
except Exception:
    aiohttp = None


async def fetch_ecb_async(session: "aiohttp.ClientSession", code: str, start: str, end: str,
                          *, min_response_size: int, timeout_seconds: int) -> pd.DataFrame:
    flow, series = code.split(".", 1)
    url = f"{ECB_API_BASE_URL}/{flow}/{series}"
    fstart = format_date_for_ecb_api(start)
    fend = format_date_for_ecb_api(end)

    param_strategies = [
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
        {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
        {"format": "csvdata", "detail": "dataonly"},
    ]

    session_was_none = False
    if session is None and aiohttp is not None:
        session_was_none = True
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        session = aiohttp.ClientSession(timeout=timeout)

    headers = {"Accept": "text/csv"}
    last_error = None
    try:
        for params in param_strategies:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    last_error = f"Status {response.status}"
                    continue
                text = await response.text()
                if not text.strip() or len(text.strip()) < min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                try:
                    df = pd.read_csv(io.StringIO(text))
                    df = DataProcessor.standardize_dataframe(df)
                    if not df.empty:
                        return df
                except Exception as e:
                    last_error = f"CSV parse error: {e}"
                    continue
    finally:
        if session_was_none and session is not None:
            await session.close()

    raise Exception(f"ECB API failed for {code}. Last error: {last_error}")


def fetch_ecb_sync(code: str, start: str, end: str, *,
                   min_response_size: int, timeout_seconds: int, requests_module=None) -> pd.DataFrame:
    import io
    import pandas as pd
    requests = requests_module or __import__("requests")

    flow, series = code.split(".", 1)
    url = f"{ECB_API_BASE_URL}/{flow}/{series}"
    fstart = format_date_for_ecb_api(start)
    fend = format_date_for_ecb_api(end)

    param_strategies = [
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
        {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
        {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
        {"format": "csvdata", "detail": "dataonly"},
    ]
    headers = {"Accept": "text/csv"}
    last_error = None
    for params in param_strategies:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
            if resp.status_code != 200:
                last_error = f"Status {resp.status_code}"
                continue
            text = resp.text
            if not text.strip() or len(text.strip()) < min_response_size:
                last_error = f"Response too small: {len(text)}"
                continue
            df = pd.read_csv(io.StringIO(text))
            df = DataProcessor.standardize_dataframe(df)
            if not df.empty:
                return df
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
''')


# 3) sources.buba_client ------------------------------------------------------
_register_module('sources.buba_client', r'''
import io, ssl, asyncio
import pandas as pd
from typing import Dict, List
from transforms import DataProcessor

try:
    import aiohttp
except Exception:
    aiohttp = None


def _build_bundesbank_urls(code: str) -> List[str]:
    base_urls = [
        "https://api.statistiken.bundesbank.de/rest/download",
        "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
    ]
    urls = []
    if '.' in code:
        dataset, series = code.split('.', 1)
        urls.extend([
            f"{base_urls[0]}/{dataset}/{series}",
            f"{base_urls[0]}/{code.replace('.', '/')}",
            f"{base_urls[1]}/{dataset}/{series}",
            f"{base_urls[1]}/{code.replace('.', '/')}",
        ])
    urls.extend([
        f"{base_urls[0]}/{code}",
        f"{base_urls[1]}/{code}",
    ])
    if code.count('.') > 1:
        parts = code.split('.')
        for i in range(1, len(parts)):
            p1 = '.'.join(parts[:i])
            p2 = '.'.join(parts[i:])
            urls.extend([
                f"{base_urls[0]}/{p1}/{p2}",
                f"{base_urls[0]}/{p1.replace('.', '/')}/{p2.replace('.', '/')}",
            ])
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:12]


def _get_bundesbank_params(start: str, end: str) -> List[Dict[str, str]]:
    return [
        {"format": "csv", "lang": "en", "metadata": "false"},
        {"format": "csv", "lang": "de", "metadata": "false"},
        {"format": "csv", "lang": "en", "metadata": "false", "startPeriod": start, "endPeriod": end},
        {"format": "csv", "lang": "de", "metadata": "false", "startPeriod": start, "endPeriod": end},
        {"format": "tsv", "lang": "en", "metadata": "false"},
        {"format": "tsv", "lang": "de", "metadata": "false"},
        {"format": "csv"},
        {"lang": "en"},
        {"lang": "de"},
        {},
    ]


class BundesbankCSVParser:
    @staticmethod
    def parse(content: str, code: str) -> pd.DataFrame:
        lines = content.strip().split("\n")
        if not lines:
            raise ValueError("Empty CSV")
        start_idx = BundesbankCSVParser._find_data_start(lines, code)
        data_lines = lines[start_idx:]
        if not data_lines:
            raise ValueError("No data lines")
        delim = BundesbankCSVParser._detect_delimiter(data_lines[0])
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), delimiter=delim, skip_blank_lines=True)
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("No valid data after parsing")
        time_col, value_col = BundesbankCSVParser._identify_columns(df, code)
        res = pd.DataFrame()
        tvals = df[time_col].dropna()
        vvals = df[value_col].dropna()
        m = min(len(tvals), len(vvals))
        if m == 0:
            raise ValueError("No valid pairs")
        res["Datum"] = tvals.iloc[:m].astype(str)
        res["value"] = pd.to_numeric(vvals.iloc[:m], errors="coerce")
        res = res.dropna()
        if res.empty:
            raise ValueError("No numeric data")
        return res

    @staticmethod
    def _find_data_start(lines: list[str], code: str) -> int:
        for i, line in enumerate(lines):
            if code in line and ('BBAF3' in line or 'BBK' in line):
                return i
        for i, line in enumerate(lines):
            if code in line:
                return i
        for i, line in enumerate(lines):
            if ',' in line or ';' in line:
                if max(line.count(','), line.count(';')) >= 2:
                    return i
        return 0

    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        if header_line.count(',') > header_line.count(';'):
            return ','
        if header_line.count(';') > 0:
            return ';'
        if '\t' in header_line:
            return '\t'
        if '|' in header_line:
            return '|'
        return ','

    @staticmethod
    def _identify_columns(df: pd.DataFrame, code: str) -> tuple[str, str]:
        value_col = None
        for col in df.columns:
            c = str(col)
            if code in c and 'FLAG' not in c.upper() and 'ATTRIBUT' not in c.upper():
                value_col = col
                break
        if value_col is None:
            parts = code.split('.')
            for col in df.columns:
                c = str(col)
                if any(p for p in parts if p in c) and 'FLAG' not in c.upper():
                    value_col = col
                    break
        if value_col is None and len(df.columns) >= 2:
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0:
                    value_col = col
                    break
        if value_col is None:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
            else:
                raise ValueError("Could not identify value column")

        time_col = None
        date_keys = ['TIME', 'DATE', 'PERIOD', 'DATUM', 'ZEIT']
        for col in df.columns:
            c = str(col).upper()
            if any(k in c for k in date_keys):
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if col != value_col and 'FLAG' not in str(col).upper():
                    time_col = col
                    break
        if time_col is None:
            time_col = df.columns[0]
        return time_col, value_col


async def fetch_buba_async(code: str, start: str, end: str, *,
                           min_response_size: int, timeout_seconds: int) -> pd.DataFrame:
    if aiohttp is None:
        raise Exception("aiohttp not available")
    urls = _build_bundesbank_urls(code)
    params_list = _get_bundesbank_params(start, end)
    headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=10, limit_per_host=5)
    last_error = None
    attempts = 0
    max_attempts = min(len(urls) * len(params_list), 20)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for u in urls:
            for params in params_list:
                attempts += 1
                if attempts > max_attempts:
                    break
                try:
                    async with session.get(u, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            if text and len(text.strip()) > min_response_size:
                                df = BundesbankCSVParser.parse(text, code)
                                df = DataProcessor.standardize_dataframe(df)
                                if not df.empty:
                                    return df
                            else:
                                last_error = "Response too small"
                        elif resp.status == 404:
                            last_error = "Series not found (404)"
                        else:
                            txt = await resp.text()
                            last_error = f"Status {resp.status}: {txt[:80]}"
                except asyncio.TimeoutError:
                    last_error = f"Timeout after {timeout_seconds}s"
                except Exception as e:
                    last_error = f"Unexpected: {e}"
            if attempts > max_attempts:
                break
    raise Exception(f"Bundesbank API failed after {attempts} attempts. Last error: {last_error}")


def fetch_buba_sync(code: str, start: str, end: str, *,
                    min_response_size: int, timeout_seconds: int, requests_module=None) -> pd.DataFrame:
    requests = requests_module or __import__("requests")
    urls = _build_bundesbank_urls(code)
    params_list = _get_bundesbank_params(start, end)
    headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
    last_error = None
    attempts = 0
    max_attempts = min(len(urls) * len(params_list), 20)
    for u in urls:
        for params in params_list:
            attempts += 1
            if attempts > max_attempts:
                break
            try:
                resp = requests.get(u, params=params, headers=headers,
                                    timeout=timeout_seconds, verify=False)
                if resp.status_code == 200:
                    text = resp.text
                    if text and len(text.strip()) > min_response_size:
                        df = BundesbankCSVParser.parse(text, code)
                        df = DataProcessor.standardize_dataframe(df)
                        if not df.empty:
                            return df
                    else:
                        last_error = "Response too small"
                elif resp.status_code == 404:
                    last_error = "Series not found (404)"
                else:
                    last_error = f"Status {resp.status_code}: {resp.text[:80]}"
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout_seconds}s"
            except requests.exceptions.SSLError:
                last_error = "SSL verification failed"
            except Exception as e:
                last_error = f"Request failed: {e}"
        if attempts > max_attempts:
            break
    raise Exception(f"Bundesbank API failed after {attempts} attempts. Last error: {last_error}")
''')


# 4) main download / run_from_config -----------------------------------------
import asyncio
import yaml
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from transforms import (
    detect_data_source, parse_index_specification, CacheManager, IndexCreator, get_excel_engine
)
from sources.ecb_client import fetch_ecb_async, fetch_ecb_sync
from sources.buba_client import fetch_buba_async, fetch_buba_sync

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import aiohttp  # noqa
except Exception:
    aiohttp = None


def _parse_date_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if s.str.match(r"^\d{4}-\d{2}$").all():
        return pd.to_datetime(s + "-01", format="%Y-%m-%d", errors="coerce")
    if s.str.match(r"^\d{4}-Q[1-4]$").all():
        q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
        mapped = s.str.replace(
            r"^(\d{4})-(Q[1-4])$",
            lambda m: f"{m.group(1)}-{q_map[m.group(2)]}",
            regex=True
        )
        return pd.to_datetime(mapped, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


_ECB_PREFIXES = {
    "ICP", "RESR", "BP6", "MNA", "LFSI", "FM", "BSI", "BLS", "MIR",
    "PSS", "TRD", "STS", "SEC", "TRI", "EI", "CPI", "HICP", "ILM"
}
_BUBA_PREFIXES = {
    "BBAF3", "BBK", "BBEX", "BBK01", "BBK01U", "BB", "BBK0"
}

SIMPLE_TARGET_FALLBACKS = {
    "PH_KREDITE": "BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B",
    "PH_EINLAGEN": "BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B",
    "PH_WERTPAPIERE": "BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B",
    "PH_VERSICHERUNGEN": "BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B",
    "NF_KG_EINLAGEN": "BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_WERTPAPIERE": "BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_VERSICHERUNGEN": "BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_KREDITE": "BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B",
}

MIN_RESPONSE_SIZE_DEFAULT = 100


def _normalize_loader_path_str(path_str: str) -> str:
    if not path_str:
        return path_str
    path_str = path_str.replace("loader\\loader", "loader")
    path_str = path_str.replace("loader/loader", "loader")
    return path_str


async def _fetch_async(codes: List[str], start: str, end: str, *,
                       min_response_size: int, timeout_seconds: int,
                       overrides: Optional[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if aiohttp is None:
        return _fetch_sync(codes, start, end,
                           min_response_size=min_response_size,
                           timeout_seconds=timeout_seconds,
                           overrides=overrides)
    async with aiohttp.ClientSession() as session:
        for code in codes:
            try:
                # source is determined but we call the right fetcher anyway
                source_is_ecb = _prefix(code).upper() in _ECB_PREFIXES
                if overrides and code.upper() in overrides:
                    source_is_ecb = overrides[code.upper()].upper() == "ECB"
                if source_is_ecb:
                    df = await fetch_ecb_async(session, code, start, end,
                                               min_response_size=min_response_size,
                                               timeout_seconds=timeout_seconds)
                else:
                    df = await fetch_buba_async(code, start, end,
                                                min_response_size=min_response_size,
                                                timeout_seconds=timeout_seconds)
                if "Datum" in df.columns:
                    df["Datum"] = _parse_date_column(df["Datum"])
                results[code] = df
                print(f"  ✓ {code}: {len(df)} observations")
            except Exception as e:
                print(f"  ✗ {code}: {e}")
            await asyncio.sleep(0.4)
    return results


def _prefix(code: str) -> str:
    return code.split('.', 1)[0].upper().strip()


def _resolve_source(code: str, overrides: Optional[Dict[str, str]] = None) -> str:
    c_up = code.upper().strip()
    pfx = _prefix(c_up)
    if overrides and c_up in overrides:
        cand = overrides[c_up].upper()
        if cand in {"ECB", "BUBA"}:
            return cand
    if overrides:
        for k, v in overrides.items():
            ku = k.upper().strip()
            if ku.endswith(".*") and pfx == ku[:-2]:
                cand = v.upper()
                if cand in {"ECB", "BUBA"}:
                    return cand
    if pfx in _ECB_PREFIXES:
        return "ECB"
    if pfx in _BUBA_PREFIXES:
        return "BUBA"
    try:
        ds = detect_data_source(c_up)
        return "ECB" if str(ds).upper().startswith("ECB") else "BUBA"
    except Exception:
        return "BUBA" if pfx.startswith("BB") else "ECB"


def _fetch_sync(codes: List[str], start: str, end: str, *,
                min_response_size: int, timeout_seconds: int,
                overrides: Optional[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            source = _resolve_source(code, overrides)
            if source == "ECB":
                df = fetch_ecb_sync(code, start, end,
                                    min_response_size=min_response_size,
                                    timeout_seconds=timeout_seconds)
            else:
                df = fetch_buba_sync(code, start, end,
                                     min_response_size=min_response_size,
                                     timeout_seconds=timeout_seconds)
            if "Datum" in df.columns:
                df["Datum"] = _parse_date_column(df["Datum"])
            out[code] = df
            print(f"  ✓ {code} [{source}]: {len(df)} observations")
        except Exception as e:
            print(f"  ✗ {code}: {e}")
        import time; time.sleep(0.4)
    return out


def _merge_series_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_series = []
    for code, df in data_dict.items():
        if df is None or df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "datum" not in cols or "value" not in cols:
            continue
        dcol = cols["datum"]
        series_df = df.copy()
        series_df[dcol] = _parse_date_column(series_df[dcol])
        series_df = series_df.set_index(dcol)[["value"]].rename(columns={"value": code})
        all_series.append(series_df)
    if not all_series:
        return pd.DataFrame()
    merged_df = pd.concat(all_series, axis=1, sort=True)
    merged_df = merged_df.reset_index().rename(columns={"index": "Datum"})
    merged_df = merged_df.sort_values("Datum").reset_index(drop=True)
    return merged_df


def _build_calendar_index(start: str, end: str, freq: str = "MS") -> pd.DataFrame:
    s = pd.to_datetime(start + "-01" if len(start) == 7 else start)
    e = pd.to_datetime(end + "-01" if len(end) == 7 else end)
    rng = pd.date_range(s, e, freq=freq, inclusive="both")
    return pd.DataFrame({"Datum": rng})


def _align_to_calendar(merged: pd.DataFrame, start: str, end: str, *,
                       freq: str = "MS", fill: str = "none",
                       fill_limit: Optional[int] = None) -> pd.DataFrame:
    cal = _build_calendar_index(start, end, freq=freq)
    out = cal.merge(merged, on="Datum", how="left")
    if fill in {"ffill", "bfill"}:
        method = "ffill" if fill == "ffill" else "bfill"
        value_cols = [c for c in out.columns if c != "Datum"]
        out[value_cols] = out[value_cols].fillna(method=method, limit=fill_limit)
    return out


def run_from_config(config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    start_date: str = cfg.get("start_date")
    end_date: str = cfg.get("end_date")
    prefer_cache: bool = bool(cfg.get("prefer_cache", True))
    cache_cfg = cfg.get("cache", {}) or {}

    # default: cwd/financial_cache
    cache_dir_raw = cache_cfg.get("cache_dir", str((Path.cwd() / "financial_cache").resolve()))
    cache_dir = _normalize_loader_path_str(cache_dir_raw)
    cache_max_age_days = int(cache_cfg.get("cache_max_age_days", 60))

    anchor_var: Optional[str] = cfg.get("anchor_var")
    series_definitions: Dict[str, str] = cfg.get("series_definitions") or {}

    index_base_year = int(cfg.get("index_base_year", 2015))
    index_base_value = float(cfg.get("index_base_value", 100.0))

    timeout_seconds = int(cfg.get("download_timeout_seconds", 30))
    min_response_size = int(cfg.get("min_response_size", MIN_RESPONSE_SIZE_DEFAULT))

    source_overrides: Dict[str, str] = cfg.get("source_overrides") or {}
    min_populated_vars: int = int(cfg.get("min_populated_vars", 2))
    cal_cfg = cfg.get("calendar_index") or {}
    cal_freq: str = cal_cfg.get("freq", "MS")
    cal_fill: str = cal_cfg.get("fill", "none")
    cal_fill_limit = cal_cfg.get("fill_limit", None)

    regular_codes: Dict[str, str] = {}
    index_defs: Dict[str, List[str]] = {}
    for var_name, definition in series_definitions.items():
        idx_codes = parse_index_specification(definition)
        if idx_codes:
            index_defs[var_name] = idx_codes
        else:
            regular_codes[var_name] = definition

    all_codes = set(regular_codes.values())
    for codes in index_defs.values():
        all_codes.update(codes)
    all_codes = list(all_codes)
    print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
    print(f"Total series to download: {len(all_codes)}")

    cache = CacheManager(cache_dir=cache_dir, cache_max_age_days=cache_max_age_days)

    cached_data: Dict[str, pd.DataFrame] = {}
    missing = []
    if prefer_cache:
        for code in all_codes:
            dfc = cache.read_cache(code)
            if dfc is not None and not dfc.empty:
                if "Datum" in dfc.columns:
                    dfc["Datum"] = _parse_date_column(dfc["Datum"])
                cached_data[code] = dfc
            else:
                missing.append(code)
    else:
        missing = all_codes[:]

    downloaded_data: Dict[str, pd.DataFrame] = {}
    if missing:
        print(f"Downloading {len(missing)} missing series...")
        try:
            downloaded_data = asyncio.run(_fetch_async(
                missing, start_date, end_date,
                min_response_size=min_response_size,
                timeout_seconds=timeout_seconds,
                overrides=source_overrides
            ))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                import nest_asyncio; nest_asyncio.apply()
                downloaded_data = asyncio.run(_fetch_async(
                    missing, start_date, end_date,
                    min_response_size=min_response_size,
                    timeout_seconds=timeout_seconds,
                    overrides=source_overrides
                ))
            else:
                print("Async failed, using synchronous download mode...")
                downloaded_data = _fetch_sync(
                    missing, start_date, end_date,
                    min_response_size=min_response_size,
                    timeout_seconds=timeout_seconds,
                    overrides=source_overrides
                )
        except Exception:
            print("Download failed, switching to synchronous mode...")
            downloaded_data = _fetch_sync(
                missing, start_date, end_date,
                min_response_size=min_response_size,
                timeout_seconds=timeout_seconds,
                overrides=source_overrides
            )

        for code, df in downloaded_data.items():
            cache.write_cache(code, df)

    all_data = {**cached_data, **downloaded_data}
    if not all_data:
        raise RuntimeError("No series loaded successfully")

    merged = _merge_series_data(all_data)
    merged = _align_to_calendar(
        merged, start=start_date, end=end_date,
        freq=cal_freq, fill=cal_fill, fill_limit=cal_fill_limit
    )

    final_data = {"Datum": merged["Datum"]}
    for var_name, series_code in regular_codes.items():
        if series_code in merged.columns:
            final_data[var_name] = merged[series_code]

    indexer = IndexCreator(index_base_year=index_base_year, index_base_value=index_base_value)
    for var_name, idx_codes in index_defs.items():
        try:
            available = [c for c in idx_codes if c in merged.columns]
            if len(available) >= max(1, int(len(idx_codes) * 0.3)):
                idx_series = indexer.create_index(merged, available, var_name)
                aligned_idx = idx_series.reindex(pd.to_datetime(merged["Datum"]))
                final_data[var_name] = aligned_idx.values
                print(f"Created INDEX: {var_name} from {len(available)}/{len(idx_codes)} series")
            else:
                if var_name in SIMPLE_TARGET_FALLBACKS:
                    fb = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fb in merged.columns:
                        final_data[var_name] = merged[fb]
                        print(f"Using fallback for {var_name}: {fb}")
                    else:
                        print(f"Warning: Could not create {var_name} – fallback {fb} not available")
                else:
                    print(f"Warning: Could not create INDEX {var_name} – insufficient data")
        except Exception as e:
            print(f"Failed to create INDEX {var_name}: {e}")
            if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                fb = SIMPLE_TARGET_FALLBACKS[var_name]
                if fb in merged.columns:
                    final_data[var_name] = merged[fb]
                    print(f"Using fallback for {var_name} after failure: {fb}")

    final_df = pd.DataFrame(final_data)
    final_df["Datum"] = pd.to_datetime(final_df["Datum"])
    final_df = final_df.sort_values("Datum").reset_index(drop=True)

    value_cols = [c for c in final_df.columns if c != "Datum"]
    if value_cols:
        non_na = final_df[value_cols].notna().sum(axis=1)
        req = min_populated_vars if len(value_cols) >= min_populated_vars else 1
        keep_mask = non_na >= req
        if keep_mask.any():
            first_keep = keep_mask.idxmax()
            if first_keep > 0:
                before = len(final_df)
                final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                print(f"Trimmed leading rows with <{req} populated vars: {before} → {len(final_df)}")

    if anchor_var and anchor_var in final_df.columns:
        mask_anchor = final_df[anchor_var].notna()
        if mask_anchor.any():
            start_a = final_df.loc[mask_anchor, "Datum"].min()
            end_a = final_df.loc[mask_anchor, "Datum"].max()
            before = len(final_df)
            final_df = final_df[(final_df["Datum"] >= start_a) & (final_df["Datum"] <= end_a)].copy()
            final_df.reset_index(drop=True, inplace=True)
            print(f"Anchored final dataset to '{anchor_var}' window: {start_a.date()} → {end_a.date()} ({before} → {len(final_df)})")

    if anchor_var and anchor_var in final_df.columns:
        exog_cols = [c for c in final_df.columns if c not in ("Datum", anchor_var)]
        if exog_cols:
            tgt_notna = final_df[anchor_var].notna().values
            all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
            keep_start = 0
            for i in range(len(final_df)):
                if not (tgt_notna[i] and all_exog_nan[i]):
                    keep_start = i
                    break
            if keep_start > 0:
                before = len(final_df)
                final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                print(f"Trimmed leading target-only rows: {before} → {len(final_df)}")

    print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")

    out_path_raw = cfg.get("output_path", "output.xlsx")
    out_path_str = _normalize_loader_path_str(out_path_raw)
    out_path = Path(out_path_str)

    # parquet-branch: zusätzlich Multi-Sheet-Excel mit erwarteten Namen
    if out_path.suffix.lower() == ".parquet":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path.resolve()} (parquet, final_dataset only)")

        xlsx_path = out_path.with_suffix(".xlsx")
        try:
            with pd.ExcelWriter(xlsx_path, engine=get_excel_engine()) as writer:
                # 1) das eigentliche Dataset
                final_df.to_excel(writer, index=False, sheet_name="final_dataset")
                # 2) erwartete, aber im Loader nicht befüllte Sheets – leer anlegen
                pd.DataFrame().to_excel(writer, index=False, sheet_name="bestand_ph")
                pd.DataFrame().to_excel(writer, index=False, sheet_name="bestand_nfk")
                pd.DataFrame().to_excel(writer, index=False, sheet_name="fluss_ph")
                pd.DataFrame().to_excel(writer, index=False, sheet_name="fluss_nfk")
            print(f"Wrote {xlsx_path.resolve()} (excel, final_dataset + empty 4 sheets)")
        except Exception as e:
            print(f"WARNING: could not write Excel copy next to parquet: {e}")

        return final_df

    # excel-branch (voll)
    raw_ecb = {code: df for code, df in all_data.items() if _resolve_source(code, source_overrides) == "ECB"}
    raw_buba = {code: df for code, df in all_data.items() if _resolve_source(code, source_overrides) != "ECB"}

    with pd.ExcelWriter(out_path, engine=get_excel_engine()) as writer:
        if raw_ecb:
            ecb_merged = _merge_series_data(raw_ecb)
            ecb_merged = _align_to_calendar(ecb_merged, start=start_date, end=end_date, freq=cal_freq, fill="none")
            ecb_merged.to_excel(writer, index=False, sheet_name="raw_ecb")
        else:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_ecb")

        if raw_buba:
            buba_merged = _merge_series_data(raw_buba)
            buba_merged = _align_to_calendar(buba_merged, start=start_date, end=end_date, freq=cal_freq, fill="none")
            buba_merged.to_excel(writer, index=False, sheet_name="raw_buba")
        else:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_buba")

        final_df.to_excel(writer, index=False, sheet_name="final_dataset")

        # Zusatz-Sheets, damit Dashboard glücklich ist
        pd.DataFrame().to_excel(writer, index=False, sheet_name="bestand_ph")
        pd.DataFrame().to_excel(writer, index=False, sheet_name="bestand_nfk")
        pd.DataFrame().to_excel(writer, index=False, sheet_name="fluss_ph")
        pd.DataFrame().to_excel(writer, index=False, sheet_name="fluss_nfk")

        meta = pd.DataFrame({
            "param": [
                "start_date", "end_date", "prefer_cache",
                "index_base_year", "index_base_value",
                "anchor_var", "min_populated_vars",
                "calendar_freq", "calendar_fill", "calendar_fill_limit"
            ],
            "value": [
                start_date, end_date, prefer_cache,
                index_base_year, index_base_value,
                anchor_var, min_populated_vars,
                cal_freq, cal_fill, cal_fill_limit
            ]
        })
        meta.to_excel(writer, index=False, sheet_name="metadata")

    print(f"Wrote {out_path.resolve()}")
    return final_df


if __name__ == "__main__":
    run_from_config("config.yaml")

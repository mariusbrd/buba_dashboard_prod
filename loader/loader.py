import sys, types


def _register_module(modname: str, source: str):
    """Create a module with name 'modname', execute 'source' in its namespace,
    and register it in sys.modules so that 'import modname' works."""
    # Ensure parent package exists (e.g., 'sources' for 'sources.ecb_client')
    if '.' in modname:
        pkg = modname.split('.')[0]
        if pkg not in sys.modules:
            pkg_mod = types.ModuleType(pkg)
            pkg_mod.__path__ = []  # mark as package-like
            sys.modules[pkg] = pkg_mod
    m = types.ModuleType(modname)
    # Provide a minimal module-level global namespace
    m.__dict__['__name__'] = modname
    exec(source, m.__dict__)
    sys.modules[modname] = m

# 1) Register 'transforms' exactly as-is
_register_module('transforms', r"""
# transforms.py
# Shared helpers + transformations, copied 1:1 from your pipeline.

import re
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

# === Constants and helpers (1:1) ===
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
        import openpyxl
        return 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter
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
        return self.cache_dir / f"{safe_name}.xlsx"

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
            df = pd.read_excel(cache_path, sheet_name="data", engine=get_excel_engine())
            return DataProcessor.standardize_dataframe(df)
        except Exception:
            return None

    def write_cache(self, code: str, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        cache_path = self._cache_path(code)
        temp_path = cache_path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(temp_path, engine=get_excel_engine()) as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            temp_path.replace(cache_path)
            return True
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False

class IndexCreator:
    def __init__(self, *, index_base_year: int, index_base_value: float):
        self.index_base_year = index_base_year
        self.index_base_value = index_base_value

    def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
        # === 1:1 copied logic ===
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
        weighted_values = []
        for code in available_codes:
            if code in clean_data.columns:
                weighted_values.append(clean_data[code] * weights[code])

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
                if first_valid.empty:
                    base_value_actual = 1.0
                else:
                    base_value_actual = first_valid.iloc[0]
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

""")

# 2) Register 'sources.ecb_client' exactly as-is
_register_module('sources.ecb_client', r"""
# sources/ecb_client.py
# NOTE: Logic copied 1:1 from your pipeline's APIClient._fetch_ecb (async) and _fetch_ecb_sync.
import io
import aiohttp
import pandas as pd

ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data"

# Dependencies from transforms.py (imported):
from transforms import DataProcessor, format_date_for_ecb_api

async def fetch_ecb_async(session: aiohttp.ClientSession, code: str, start: str, end: str, *, min_response_size: int, timeout_seconds: int) -> pd.DataFrame:
    # === BEGIN 1:1 COPIED LOGIC ===
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

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {"Accept": "text/csv"}
    last_error = None

    for params in param_strategies:
        async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
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

    raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    # === END 1:1 COPIED LOGIC ===

def fetch_ecb_sync(code: str, start: str, end: str, *, min_response_size: int, timeout_seconds: int, requests_module=None) -> pd.DataFrame:
    import io
    import pandas as pd
    requests = requests_module
    if requests is None:
        import requests  # local import as in your original

    # === BEGIN 1:1 COPIED LOGIC ===
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
    # === END 1:1 COPIED LOGIC ===

""")

# 3) Register 'sources.buba_client' exactly as-is
_register_module('sources.buba_client', r"""
# sources/buba_client.py
# NOTE: Logic copied 1:1 from your pipeline's APIClient._fetch_bundesbank (async),
# _fetch_bundesbank_sync, and BundesbankCSVParser.
import io, ssl, asyncio
import aiohttp
import pandas as pd

from typing import Dict, List
from transforms import DataProcessor

# === BEGIN 1:1 COPIED HELPERS (URL builder & param variants) ===
def _build_bundesbank_urls(code: str) -> List[str]:
    base_urls = [
        "https://api.statistiken.bundesbank.de/rest/download",
        "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
    ]
    url_patterns = []
    if '.' in code:
        dataset, series = code.split('.', 1)
        url_patterns.extend([
            f"{base_urls[0]}/{dataset}/{series}",
            f"{base_urls[0]}/{code.replace('.', '/')}",
            f"{base_urls[1]}/{dataset}/{series}",
            f"{base_urls[1]}/{code.replace('.', '/')}"
        ])
    url_patterns.extend([
        f"{base_urls[0]}/{code}",
        f"{base_urls[1]}/{code}"
    ])
    if code.count('.') > 1:
        parts = code.split('.')
        for i in range(1, len(parts)):
            path1 = '.'.join(parts[:i])
            path2 = '.'.join(parts[i:])
            url_patterns.extend([
                f"{base_urls[0]}/{path1}/{path2}",
                f"{base_urls[0]}/{path1.replace('.', '/')}/{path2.replace('.', '/')}"
            ])
    seen = set()
    unique_patterns = []
    for pattern in url_patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)
    return unique_patterns[:12]

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
        {}
    ]
# === END 1:1 COPIED HELPERS ===

# === BEGIN 1:1 COPIED PARSER ===
class BundesbankCSVParser:
    @staticmethod
    def parse(content: str, code: str) -> pd.DataFrame:
        try:
            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("Empty CSV content")

            data_start_idx = BundesbankCSVParser._find_data_start(lines, code)
            csv_lines = lines[data_start_idx:]
            if not csv_lines:
                raise ValueError("No data lines found")

            delimiter = BundesbankCSVParser._detect_delimiter(csv_lines[0])
            df = pd.read_csv(io.StringIO('\n'.join(csv_lines)), delimiter=delimiter, skip_blank_lines=True)
            df = df.dropna(how='all')
            if df.empty:
                raise ValueError("No valid data after parsing")

            time_col, value_col = BundesbankCSVParser._identify_columns(df, code)
            result_df = pd.DataFrame()
            time_values = df[time_col].dropna()
            value_values = df[value_col].dropna()
            min_len = min(len(time_values), len(value_values))
            if min_len == 0:
                raise ValueError("No valid data pairs found")

            result_df['Datum'] = time_values.iloc[:min_len].astype(str)
            result_df['value'] = pd.to_numeric(value_values.iloc[:min_len], errors='coerce')
            result_df = result_df.dropna()
            if result_df.empty:
                raise ValueError("No valid numeric data after cleaning")
            return result_df
        except Exception as e:
            raise ValueError(f"Bundesbank CSV parsing failed: {e}")

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
                sep_count = max(line.count(','), line.count(';'))
                if sep_count >= 2:
                    return i
        return 0

    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        comma_count = header_line.count(',')
        semicolon_count = header_line.count(';')
        if comma_count > semicolon_count:
            return ','
        elif semicolon_count > 0:
            return ';'
        else:
            if '\t' in header_line:
                return '\t'
            elif '|' in header_line:
                return '|'
            else:
                return ','

    @staticmethod
    def _identify_columns(df: pd.DataFrame, code: str) -> tuple[str, str]:
        value_col = None
        for col in df.columns:
            col_str = str(col)
            if code in col_str and 'FLAG' not in col_str.upper() and 'ATTRIBUT' not in col_str.upper():
                value_col = col
                break
        if value_col is None:
            code_parts = code.split('.')
            for col in df.columns:
                col_str = str(col)
                if any(part in col_str for part in code_parts if len(part) > 3) and 'FLAG' not in col_str.upper():
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
        date_keywords = ['TIME', 'DATE', 'PERIOD', 'DATUM', 'ZEIT']
        for col in df.columns:
            col_str = str(col).upper()
            if any(keyword in col_str for keyword in date_keywords):
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
# === END 1:1 COPIED PARSER ===

async def fetch_buba_async(code: str, start: str, end: str, *, min_response_size: int, timeout_seconds: int) -> pd.DataFrame:
    # === BEGIN 1:1 COPIED LOGIC (async) ===
    url_patterns = _build_bundesbank_urls(code)
    params_variants = _get_bundesbank_params(start, end)
    headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    last_error = None
    attempt_count = 0
    max_attempts = min(len(url_patterns) * len(params_variants), 20)

    connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as bb_session:
            for url in url_patterns:
                for params in params_variants:
                    attempt_count += 1
                    if attempt_count > max_attempts:
                        break
                    try:
                        async with bb_session.get(url, params=params, headers=headers) as response:
                            if response.status == 200:
                                text = await response.text()
                                if text and len(text.strip()) > min_response_size:
                                    df = BundesbankCSVParser.parse(text, code)
                                    if df is not None and not df.empty:
                                        df = DataProcessor.standardize_dataframe(df)
                                        if not df.empty:
                                            return df
                                else:
                                    last_error = f"Response too small: {len(text)} bytes"
                                    continue
                            elif response.status == 404:
                                last_error = "Series not found (404)"
                                continue
                            else:
                                error_text = await response.text()
                                last_error = f"Status {response.status}: {error_text[:100]}"
                                continue
                    except asyncio.TimeoutError:
                        last_error = f"Timeout after {timeout_seconds}s"
                        continue
                    except Exception as e:
                        last_error = f"Unexpected error: {str(e)}"
                        continue
                if attempt_count > max_attempts:
                    break
    except Exception as e:
        last_error = f"Session creation failed: {e}"

    raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")
    # === END 1:1 COPIED LOGIC (async) ===

def fetch_buba_sync(code: str, start: str, end: str, *, min_response_size: int, timeout_seconds: int, requests_module=None) -> pd.DataFrame:
    import requests
    if requests_module is not None:
        requests = requests_module

    # === BEGIN 1:1 COPIED LOGIC (sync) ===
    url_patterns = _build_bundesbank_urls(code)
    params_variants = _get_bundesbank_params(start, end)
    headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
    last_error = None
    attempt_count = 0
    max_attempts = min(len(url_patterns) * len(params_variants), 20)

    for url in url_patterns:
        for params in params_variants:
            attempt_count += 1
            if attempt_count > max_attempts:
                break

            try:
                response = requests.get(
                    url, params=params, headers=headers,
                    timeout=timeout_seconds, verify=False
                )

                if response.status_code == 200:
                    text = response.text
                    if text and len(text.strip()) > min_response_size:
                        df = BundesbankCSVParser.parse(text, code)
                        if df is not None and not df.empty:
                            df = DataProcessor.standardize_dataframe(df)
                            if not df.empty:
                                return df
                    else:
                        last_error = f"Response too small: {len(text)} bytes"
                        continue
                elif response.status_code == 404:
                    last_error = "Series not found (404)"
                    continue
                else:
                    last_error = f"Status {response.status_code}: {response.text[:100]}"
                    continue

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout_seconds}s"
                continue
            except requests.exceptions.SSLError:
                last_error = "SSL verification failed"
                continue
            except Exception as e:
                last_error = f"Request failed: {str(e)}"
                continue

        if attempt_count > max_attempts:
            break

    raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")
    # === END 1:1 COPIED LOGIC (sync) ===

""")

# 4) Append the original 'download_data.py' code verbatim so this file can be executed directly.
#    It will import from the modules we just registered, with functions 1:1 preserved.
# ---- BEGIN download_data.py (verbatim) ----

# download_data_fixed.py  (v1.3)
# - Robust source resolution (ECB/BuBa) with config overrides & known prefixes
# - Faster, warning-free date parsing (handles YYYY-MM-DD, YYYY-MM, YYYY-Qn)
# - Configurable leading-row trim threshold: `min_populated_vars` (default=2)
# - NEW: Complete monthly calendar index from start→end (configurable) and merge-all series onto it
#        via `calendar_index.freq` (default "MS") and optional `calendar_index.fill` (none|ffill|bfill)
#
# Usage: python download_data_fixed.py  (expects config.yaml in same folder)

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
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

# -----------------------------------------------------------------------------
# Date parsing helpers (fast & no warnings)
# -----------------------------------------------------------------------------

def _parse_date_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # ISO date: YYYY-MM-DD
    mask_ymd = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if mask_ymd.all():
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # Year-month: YYYY-MM  -> map to first day of month
    mask_ym = s.str.match(r"^\d{4}-\d{2}$")
    if mask_ym.all():
        return pd.to_datetime(s + "-01", format="%Y-%m-%d", errors="coerce")

    # Year-quarter: YYYY-Qn -> map to quarter end
    mask_yq = s.str.match(r"^\d{4}-Q[1-4]$")
    if mask_yq.all():
        q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
        mapped = s.str.replace(
            r"^(\d{4})-(Q[1-4])$",
            lambda m: f"{m.group(1)}-{q_map[m.group(2)]}",
            regex=True
        )
        return pd.to_datetime(mapped, format="%Y-%m-%d", errors="coerce")

    # Fallback – slower but robust
    return pd.to_datetime(s, errors="coerce")


# -----------------------------------------------------------------------------
# Source resolver with (1) config overrides and (2) dataset prefixes
# -----------------------------------------------------------------------------

_ECB_PREFIXES = {
    "ICP", "RESR", "BP6", "MNA", "LFSI", "FM", "BSI", "BLS", "MIR",
    "PSS", "TRD", "STS", "SEC", "TRI", "EI", "CPI", "HICP", "ILM"
}

_BUBA_PREFIXES = {
    "BBAF3", "BBK", "BBEX", "BBK01", "BBK01U", "BB", "BBK0"
}

def _prefix(code: str) -> str:
    return code.split('.', 1)[0].upper().strip()

def _resolve_source(code: str, overrides: Optional[Dict[str, str]] = None) -> str:
    c_up = code.upper().strip()
    pfx = _prefix(c_up)

    # 1) Exact override
    if overrides and c_up in overrides:
        cand = overrides[c_up].upper()
        if cand in {"ECB", "BUBA"}:
            return cand

    # 2) Prefix override with '.*'
    if overrides:
        for k, v in overrides.items():
            ku = k.upper().strip()
            if ku.endswith(".*") and pfx == ku[:-2]:
                cand = v.upper()
                if cand in {"ECB", "BUBA"}:
                    return cand

    # 3) Known prefix lists
    if pfx in _ECB_PREFIXES:
        return "ECB"
    if pfx in _BUBA_PREFIXES:
        return "BUBA"

    # 4) Fallback to existing heuristic
    try:
        ds = detect_data_source(c_up)
        if ds and str(ds).upper().startswith("ECB"):
            return "ECB"
        else:
            return "BUBA"
    except Exception:
        if pfx.startswith("BBA") or pfx.startswith("BB"):
            return "BUBA"
        return "ECB"


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

async def _fetch_async(codes: List[str], start: str, end: str, *, min_response_size: int, timeout_seconds: int, overrides: Optional[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if aiohttp is None:
        return _fetch_sync(codes, start, end, min_response_size=min_response_size, timeout_seconds=timeout_seconds, overrides=overrides)

    async with aiohttp.ClientSession() as session:
        for code in codes:
            try:
                source = _resolve_source(code, overrides)
                if source == "ECB":
                    df = await fetch_ecb_async(session, code, start, end, min_response_size=min_response_size, timeout_seconds=timeout_seconds)
                else:
                    df = await fetch_buba_async(code, start, end, min_response_size=min_response_size, timeout_seconds=timeout_seconds)
                if "Datum" in df.columns:
                    df["Datum"] = _parse_date_column(df["Datum"])
                results[code] = df
                print(f"  ✓ {code} [{source}]: {len(df)} observations")
            except Exception as e:
                print(f"  ✗ {code}: {e}")
            await asyncio.sleep(0.4)
    return results

def _fetch_sync(codes: List[str], start: str, end: str, *, min_response_size: int, timeout_seconds: int, overrides: Optional[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            source = _resolve_source(code, overrides)
            if source == "ECB":
                df = fetch_ecb_sync(code, start, end, min_response_size=min_response_size, timeout_seconds=timeout_seconds)
            else:
                df = fetch_buba_sync(code, start, end, min_response_size=min_response_size, timeout_seconds=timeout_seconds)
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
        # Normalize date
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

# -----------------------------------------------------------------------------
# Build full calendar index and align all series to it
# -----------------------------------------------------------------------------

def _build_calendar_index(start: str, end: str, freq: str = "MS") -> pd.DataFrame:
    """
    Create a complete calendar of dates from start to end with given frequency.
    freq: e.g. 'MS' (month start, default) or 'M' (month end)
    Returns DataFrame with single 'Datum' column.
    """
    # Normalize start/end to Timestamp
    s = pd.to_datetime(start + "-01" if len(start) == 7 else start)
    e = pd.to_datetime(end + "-01" if len(end) == 7 else end)
    # Ensure end inclusive: if freq='M' or 'MS', use date_range with inclusive='both'
    rng = pd.date_range(s, e, freq=freq, inclusive="both")
    return pd.DataFrame({"Datum": rng})

def _align_to_calendar(merged: pd.DataFrame, start: str, end: str, freq: str = "MS", fill: str = "none", fill_limit: Optional[int] = None) -> pd.DataFrame:
    """
    Align merged wide DataFrame (Datum + series columns) to complete calendar.
    fill: 'none' | 'ffill' | 'bfill'
    """
    cal = _build_calendar_index(start, end, freq=freq)
    out = cal.merge(merged, on="Datum", how="left")
    if fill in {"ffill", "bfill"}:
        method = {"ffill": "ffill", "bfill": "bfill"}[fill]
        value_cols = [c for c in out.columns if c != "Datum"]
        out[value_cols] = out[value_cols].fillna(method=method, limit=fill_limit)
    return out

def run_from_config(config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    start_date: str = cfg.get("start_date")
    end_date: str = cfg.get("end_date")
    prefer_cache: bool = bool(cfg.get("prefer_cache", True))
    cache_cfg = cfg.get("cache", {}) or {}
    cache_dir = cache_cfg.get("cache_dir", "financial_cache")
    cache_max_age_days = int(cache_cfg.get("cache_max_age_days", 60))

    anchor_var: Optional[str] = cfg.get("anchor_var")
    series_definitions: Dict[str, str] = cfg.get("series_definitions") or {}

    index_base_year = int(cfg.get("index_base_year", 2015))
    index_base_value = float(cfg.get("index_base_value", 100.0))

    timeout_seconds = int(cfg.get("download_timeout_seconds", 30))
    min_response_size = int(cfg.get("min_response_size", MIN_RESPONSE_SIZE_DEFAULT))

    # New options
    source_overrides: Dict[str, str] = cfg.get("source_overrides") or {}
    min_populated_vars: int = int(cfg.get("min_populated_vars", 2))
    cal_cfg = cfg.get("calendar_index") or {}
    cal_freq: str = cal_cfg.get("freq", "MS")          # 'MS' or 'M'
    cal_fill: str = cal_cfg.get("fill", "none")        # 'none' | 'ffill' | 'bfill'
    cal_fill_limit = cal_cfg.get("fill_limit", None)   # int or None

    # Split definitions
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
            downloaded_data = asyncio.run(_fetch_async(missing, start_date, end_date, min_response_size=min_response_size, timeout_seconds=timeout_seconds, overrides=source_overrides))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                import nest_asyncio; nest_asyncio.apply()
                downloaded_data = asyncio.run(_fetch_async(missing, start_date, end_date, min_response_size=min_response_size, timeout_seconds=timeout_seconds, overrides=source_overrides))
            else:
                print("Async failed, using synchronous download mode...")
                downloaded_data = _fetch_sync(missing, start_date, end_date, min_response_size=min_response_size, timeout_seconds=timeout_seconds, overrides=source_overrides)
        except Exception:
            print("Download failed, switching to synchronous mode...")
            downloaded_data = _fetch_sync(missing, start_date, end_date, min_response_size=min_response_size, timeout_seconds=timeout_seconds, overrides=source_overrides)

        for code, df in downloaded_data.items():
            cache.write_cache(code, df)

    all_data = {**cached_data, **downloaded_data}
    if not all_data:
        raise RuntimeError("No series loaded successfully")

    # Merge (raw)
    merged = _merge_series_data(all_data)

    # Align to full monthly calendar index (no data loss)
    merged = _align_to_calendar(merged, start=start_date, end=end_date, freq=cal_freq, fill=cal_fill, fill_limit=cal_fill_limit)

    # Build final_data
    final_data = {"Datum": merged["Datum"]}
    for var_name, series_code in regular_codes.items():
        if series_code in merged.columns:
            final_data[var_name] = merged[series_code]

    indexer = IndexCreator(index_base_year=index_base_year, index_base_value=index_base_value)
    for var_name, idx_codes in index_defs.items():
        try:
            available = [c for c in idx_codes if c in merged.columns]
            if len(available) >= max(1, int(len(idx_codes) * 0.3)):
                index_series = indexer.create_index(merged, available, var_name)
                aligned_index = index_series.reindex(pd.to_datetime(merged['Datum']))
                final_data[var_name] = aligned_index.values
                print(f"Created INDEX: {var_name} from {len(available)}/{len(idx_codes)} series")
            else:
                if var_name in SIMPLE_TARGET_FALLBACKS:
                    fallback = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback in merged.columns:
                        final_data[var_name] = merged[fallback]
                        print(f"Using fallback for {var_name}: {fallback}")
                    else:
                        print(f"Warning: Could not create {var_name} - fallback series {fallback} not available")
                else:
                    print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available)}/{len(idx_codes)} series available)")
        except Exception as e:
            print(f"Failed to create INDEX {var_name}: {e}")
            if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                fallback = SIMPLE_TARGET_FALLBACKS[var_name]
                if fallback in merged.columns:
                    final_data[var_name] = merged[fallback]
                    print(f"Using fallback for {var_name} after INDEX creation failed: {fallback}")

    final_df = pd.DataFrame(final_data)
    final_df["Datum"] = pd.to_datetime(final_df["Datum"])
    final_df = final_df.sort_values("Datum").reset_index(drop=True)

    value_cols = [c for c in final_df.columns if c != 'Datum']
    if value_cols:
        non_na_count = final_df[value_cols].notna().sum(axis=1)
        required = min_populated_vars if len(value_cols) >= min_populated_vars else 1
        keep_mask = non_na_count >= required
        if keep_mask.any():
            first_keep = keep_mask.idxmax()
            if first_keep > 0:
                _before = len(final_df)
                final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

    if anchor_var and anchor_var in final_df.columns:
        mask_anchor = final_df[anchor_var].notna()
        if mask_anchor.any():
            start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
            end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
            _before_rows = len(final_df)
            final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
            final_df.reset_index(drop=True, inplace=True)
            print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

    if anchor_var and anchor_var in final_df.columns:
        exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
        if exog_cols:
            tgt_notna = final_df[anchor_var].notna().values
            all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
            keep_start = 0
            for i in range(len(final_df)):
                if not (tgt_notna[i] and all_exog_nan[i]):
                    keep_start = i
                    break
            if keep_start > 0:
                _before = len(final_df)
                final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

    print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")

    # --- Save XLSX with requested sheets ---
    out_path = Path(cfg.get("output_path", "output.xlsx"))
    raw_ecb = {code: df for code, df in all_data.items() if _resolve_source(code, source_overrides) == "ECB"}
    raw_buba = {code: df for code, df in all_data.items() if _resolve_source(code, source_overrides) != "ECB"}

    with pd.ExcelWriter(out_path, engine=get_excel_engine()) as writer:
        # Raw ECB
        if raw_ecb:
            ecb_merged = _merge_series_data(raw_ecb)
            ecb_merged = _align_to_calendar(ecb_merged, start=start_date, end=end_date, freq=cal_freq, fill="none")
            ecb_merged.to_excel(writer, index=False, sheet_name="raw_ecb")
        else:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_ecb")

        # Raw Bundesbank
        if raw_buba:
            buba_merged = _merge_series_data(raw_buba)
            buba_merged = _align_to_calendar(buba_merged, start=start_date, end=end_date, freq=cal_freq, fill="none")
            buba_merged.to_excel(writer, index=False, sheet_name="raw_buba")
        else:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="raw_buba")

        # Final
        final_df.to_excel(writer, index=False, sheet_name="final_dataset")

        # Optional metadata
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


# ---- END download_data.py ----

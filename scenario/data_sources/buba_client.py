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

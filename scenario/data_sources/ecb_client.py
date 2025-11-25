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

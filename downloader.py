# downloader.py

import os
import time
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ..config import SETTINGS

def download_and_process_funding_rate():
    """
    Downloads and processes funding rate data, handling multiple inconsistent column names.
    """
    FUNDING_DIR = SETTINGS.get_funding_rate_path()
    os.makedirs(FUNDING_DIR, exist_ok=True)
    BASE_URL_TEMPLATE = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{SETTINGS.ASSET}/{SETTINGS.ASSET}-fundingRate-{{year}}-{{month:02d}}.zip"

    start_date = min(SETTINGS.IN_SAMPLE_START, SETTINGS.OUT_OF_SAMPLE_START)
    end_date = max(SETTINGS.IN_SAMPLE_END, SETTINGS.OUT_OF_SAMPLE_END)

    print("\n--- Starting Funding Rate Data Download & Processing ---")
    print(f" -> Saving to: {FUNDING_DIR}")

    current_month = start_date
    while current_month <= end_date:
        year, month = current_month.year, current_month.month
        url = BASE_URL_TEMPLATE.format(year=year, month=month)
        base_filename = os.path.basename(url).replace('.zip', '')
        output_path = os.path.join(FUNDING_DIR, f'{base_filename}.parquet')

        if os.path.exists(output_path):
            print(f"  -> Skipping {base_filename}, file already exists.")
        else:
            print(f"  -> Processing: {base_filename}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f)
                        
                        def _find_col(df: pd.DataFrame, possible_names: list) -> str:
                            for name in possible_names:
                                if name in df.columns: return name
                            raise KeyError(f"None of the possible columns found: {possible_names}")

                        timestamp_col = _find_col(df, ['fundingTime', 'calc_time'])
                        funding_rate_col = _find_col(df, ['fundingRate', 'last_funding_rate', 'funding_rate'])

                        df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='ms', utc=True)
                        df.rename(columns={funding_rate_col: 'funding_rate'}, inplace=True)
                        df[['timestamp', 'funding_rate']].to_parquet(output_path, index=False)
                        print(f"    -> ✅ Success! Used '{timestamp_col}' and '{funding_rate_col}'.")

            except requests.exceptions.HTTPError:
                print(f"    -> ❌ FAILED. File not found at URL. Skipping.")
            except KeyError as e:
                print(f"    -> ❌ FAILED. Critical column missing: {e}. Skipping file.")
            except Exception as e:
                print(f"    -> ❌ FAILED. An unexpected error occurred: {e}")

        current_month += relativedelta(months=1)
        time.sleep(1)
    print("\n🎉 Funding Rate data download and processing complete. 🎉")

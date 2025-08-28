#processor.py

import os
import zipfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from ..config import SETTINGS
from ..utils import get_files_for_period

def _process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms a raw data chunk into the clean, simulation-ready format."""
    rename_map = {'id': 'trade_id', 'time': 'timestamp', 'qty': 'size'}
    chunk_df = chunk_df.rename(columns=rename_map)
    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms', utc=True)
    chunk_df['side'] = np.where(chunk_df['is_buyer_maker'] == False, 'BUY', 'SELL')
    chunk_df['asset'] = SETTINGS.ASSET
    return chunk_df[SETTINGS.FINAL_COLUMNS]

def process_raw_trades(period_name: str):
    """Reads raw .zip files, processes them, and saves clean Parquet files."""
    raw_dir = SETTINGS.get_raw_trades_path(period_name)
    processed_dir = SETTINGS.get_processed_trades_path(period_name)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"\n--- Starting Data Processing for Period: {period_name.upper()} ---")
    files_to_process = get_files_for_period(period_name, "raw_trades")

    if not files_to_process: return

    for raw_path in files_to_process:
        filename = os.path.basename(raw_path)
        output_path = os.path.join(processed_dir, filename.replace('.zip', '.parquet'))

        if os.path.exists(output_path):
            print(f"Skipping {filename}, processed Parquet file already exists.")
            continue

        print(f"Processing: {filename}")
        try:
            with zipfile.ZipFile(raw_path) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    first_line = f.readline().decode('utf-8').strip()
                    f.seek(0)
                    has_header = first_line.startswith(SETTINGS.BINANCE_RAW_COLUMNS[0])
                    params = {'chunksize': 10_000_000, 'dtype': SETTINGS.DTYPE_MAP,
                              'header': 0 if has_header else None,
                              'names': None if has_header else SETTINGS.BINANCE_RAW_COLUMNS}
                    
                    chunk_iterator = pd.read_csv(f, **params)
                    processed_chunk = _process_chunk(next(chunk_iterator))
                    table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
                    with pq.ParquetWriter(output_path, table.schema) as writer:
                        writer.write_table(table)
                        for chunk in chunk_iterator:
                            processed_chunk = _process_chunk(chunk)
                            writer.write_table(pa.Table.from_pandas(processed_chunk, preserve_index=False))
            print(f"  -> ✅ Success! Clean data saved.")
        except StopIteration:
             print(f"  -> ⚠️ WARNING. File {filename} was empty. Skipping.")
        except Exception as e:
            print(f"  -> ❌ FAILED. An unexpected error occurred: {e}")

def create_bars_from_trades() -> pd.DataFrame:
    """Loads all processed trade data, resamples it into bars, and calculates ATR."""
    print("\n--- Preparing bar data for labeling ---")
    all_trade_files = get_files_for_period("in_sample") + get_files_for_period("out_of_sample")

    if not all_trade_files:
        raise FileNotFoundError("No processed trade files found. Cannot generate bars.")

    all_bars = []
    for file_path in tqdm(all_trade_files, desc="Reading processed trade files"):
        df = pd.read_parquet(file_path, columns=['timestamp', 'price']).set_index('timestamp')
        all_bars.append(df['price'].resample(SETTINGS.labeling.BAR_TIMEFRAME).ohlc())

    bars_df = pd.concat(all_bars).sort_index().dropna()
    high_low = bars_df['high'] - bars_df['low']
    high_close = np.abs(bars_df['high'] - bars_df['close'].shift())
    low_close = np.abs(bars_df['low'] - bars_df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_lookback = SETTINGS.labeling.ATR_LOOKBACK
    bars_df['atr'] = tr.ewm(alpha=1/atr_lookback, min_periods=atr_lookback).mean()
    
    print(f"Prepared {len(bars_df):,} bars from {bars_df.index.min()} to {bars_df.index.max()}")
    return bars_df.reset_index()

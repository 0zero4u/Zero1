

#processor.py

import os
import glob
import zipfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from ..config import SETTINGS

def get_files_for_period(period_name: str, data_type: str = "processed_trades") -> list[str]:
    """
    Finds all data files for a given period (e.g., 'in_sample') and data type.
    Args:
        period_name: The name of the period ('in_sample', 'out_of_sample').
        data_type: The type of data ('raw_trades', 'processed_trades').
    """
    if data_type == "raw_trades":
        path_template = SETTINGS.get_raw_trades_path(period_name)
        file_extension = ".zip"
    elif data_type == "processed_trades":
        path_template = SETTINGS.get_processed_trades_path(period_name)
        file_extension = ".parquet"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    search_path = os.path.join(path_template, f"*{file_extension}")
    files = sorted(glob.glob(search_path))
    if not files:
        print(f"Warning: No files found for {data_type} in period {period_name} at {search_path}")
    return files

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

def create_bars_from_trades(period_name: str) -> pd.DataFrame:
    """Loads processed trade data and resamples it into OHLCV bars at the base timeframe."""
    print(f"\n--- Preparing bar data for period: {period_name} ---")
    all_trade_files = get_files_for_period(period_name, "processed_trades")

    if not all_trade_files:
        raise FileNotFoundError(f"No processed trade files found for period '{period_name}'. Cannot generate bars.")

    all_ohlcv = []
    for file_path in tqdm(all_trade_files, desc=f"Reading processed trade files for {period_name}"):
        # NEW: Read 'size' for volume and use more robust aggregation
        df = pd.read_parquet(file_path, columns=['timestamp', 'price', 'size']).set_index('timestamp')
        resample_freq = SETTINGS.BASE_BAR_TIMEFRAME
        
        # NEW: Aggregate to create OHLC and sum volume
        agg_dict = {'price': 'ohlc', 'size': 'sum'}
        ohlcv_df = df.resample(resample_freq).agg(agg_dict)
        all_ohlcv.append(ohlcv_df)

    if not all_ohlcv:
        print("Warning: No bar data was generated.")
        return pd.DataFrame()

    bars_df = pd.concat(all_ohlcv).sort_index()
    # NEW: Flatten the MultiIndex columns from the .agg() call
    bars_df.columns = ['open', 'high', 'low', 'close', 'volume']
    bars_df.dropna(inplace=True)
    
    print(f"Prepared {len(bars_df):,} bars from {bars_df.index.min()} to {bars_df.index.max()}")
    return bars_df.reset_index()

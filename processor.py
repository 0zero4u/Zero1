# Zero1-main/processor.py

import os
import glob
import zipfile
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ..config import SETTINGS

def get_files_for_period(period_name: str, data_type: str = "processed_trades") -> list[str]:
    # ... (code is unchanged)
    if data_type == "raw_trades": path_template = SETTINGS.get_raw_trades_path(period_name); file_extension = ".zip"
    elif data_type == "processed_trades": path_template = SETTINGS.get_processed_trades_path(period_name); file_extension = ".parquet"
    else: raise ValueError(f"Unknown data_type: {data_type}")
    search_path = os.path.join(path_template, f"*{file_extension}"); files = sorted(glob.glob(search_path))
    if not files: print(f"Warning: No files found for {data_type} in period {period_name} at {search_path}")
    return files

def _process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    # ... (code is unchanged)
    rename_map = {'id': 'trade_id', 'time': 'timestamp', 'qty': 'size'}; chunk_df = chunk_df.rename(columns=rename_map)
    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms', utc=True); chunk_df['side'] = np.where(chunk_df['is_buyer_maker'] == False, 'BUY', 'SELL')
    chunk_df['asset'] = SETTINGS.ASSET; return chunk_df[SETTINGS.FINAL_COLUMNS]

def process_raw_trades(period_name: str):
    # ... (code is unchanged)
    raw_dir, processed_dir = SETTINGS.get_raw_trades_path(period_name), SETTINGS.get_processed_trades_path(period_name)
    os.makedirs(processed_dir, exist_ok=True); print(f"\n--- Starting Data Processing for Period: {period_name.upper()} ---")
    files_to_process = get_files_for_period(period_name, "raw_trades");
    if not files_to_process: return
    for raw_path in files_to_process:
        filename = os.path.basename(raw_path); output_path = os.path.join(processed_dir, filename.replace('.zip', '.parquet'))
        if os.path.exists(output_path): print(f"Skipping {filename}, processed Parquet file already exists."); continue
        print(f"Processing: {filename}")
        try:
            with zipfile.ZipFile(raw_path) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # ... (rest of function is unchanged)
                    first_line = f.readline().decode('utf-8').strip(); f.seek(0)
                    has_header = first_line.startswith(SETTINGS.BINANCE_RAW_COLUMNS[0])
                    params = {'chunksize': 10_000_000, 'dtype': SETTINGS.DTYPE_MAP, 'header': 0 if has_header else None, 'names': None if has_header else SETTINGS.BINANCE_RAW_COLUMNS}
                    chunk_iterator = pd.read_csv(f, **params); processed_chunk = _process_chunk(next(chunk_iterator)); table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
                    with pa.parquet.ParquetWriter(output_path, table.schema) as writer:
                        writer.write_table(table)
                        for chunk in chunk_iterator: writer.write_table(pa.Table.from_pandas(_process_chunk(chunk), preserve_index=False))
            print(f"  -> ✅ Success! Clean data saved.")
        except StopIteration: print(f"  -> ⚠️ WARNING. File {filename} was empty. Skipping.")
        except Exception as e: print(f"  -> ❌ FAILED. An unexpected error occurred: {e}")


def create_feature_dataframe(period_name: str) -> pd.DataFrame:
    """
    Loads trade data and generates a multi-timeframe feature dataframe for the model.
    """
    print(f"\n--- Creating feature dataframe for period: {period_name} ---")
    all_trade_files = get_files_for_period(period_name, "processed_trades")
    if not all_trade_files: raise FileNotFoundError(f"No processed trade files found for '{period_name}'.")

    # 1. Load all trades
    print("Loading all trades for the period...")
    trades_df = pd.concat([pd.read_parquet(f) for f in tqdm(all_trade_files)]).sort_values('timestamp').set_index('timestamp')

    # 2. Create base bars at the environment's frequency
    resample_freq = SETTINGS.BASE_BAR_TIMEFRAME
    print(f"Resampling trades to {resample_freq} base bars...")
    bars_df = trades_df['price'].resample(resample_freq).ohlc()
    bars_df.dropna(subset=['close'], inplace=True)

    # 3. Create feature columns for each required timeframe
    print("Generating multi-timeframe price series...")
    feature_df = pd.DataFrame(index=bars_df.index)
    feature_df['close'] = bars_df['close'] # For environment price ticks
    feature_df['price_15m'] = bars_df['close'] # Base timeframe is 15m
    
    # Resample 15m prices to 1h and forward-fill to align indices
    price_1h = bars_df['close'].resample('1H').last()
    feature_df['price_1h'] = price_1h.reindex(feature_df.index, method='ffill')

    # 4. Final cleaning
    feature_df.fillna(method='ffill', inplace=True)
    feature_df.dropna(inplace=True)
    
    print(f"✅ Prepared {len(feature_df):,} bars with multi-timeframe data.")
    return feature_df.reset_index()

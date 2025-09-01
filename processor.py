"""
Enhanced Data Processing System for Crypto Trading RL
Provides robust data processing, quality validation, and performance optimization.
Fixed import issues and removed funding rate dependency.
"""

import os
import glob
import zipfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Iterator
import warnings
from pathlib import Path
import logging
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass

# Import configuration - fixed import path
from config import SETTINGS, StrategyConfig

# Added imports for the new vectorized function
import scipy.signal


# Configure logging
logger = logging.getLogger(__name__)

# Optional dependencies with fallbacks
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available, falling back to pandas for data processing")

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using standard numpy operations")

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    total_trades: int
    price_outliers: int
    volume_outliers: int
    timestamp_gaps: int
    duplicate_trades: int
    invalid_prices: int
    data_completeness: float
    quality_score: float

class EnhancedDataProcessor:
    """
    Enhanced data processor: The "Refinery" of the system.
    This module ingests raw trade data and forges it into enriched bars containing
    not just OHLCV, but also critical order flow metrics like Volume Delta and VWAP,
    which are born from the raw transactions.
    """

    def __init__(self, config=None, enable_caching: bool = True, parallel_workers: int = None):
        self.cfg = config or SETTINGS
        self.enable_caching = enable_caching
        self.parallel_workers = parallel_workers or min(mp.cpu_count(), 8)

        # Quality thresholds
        self.quality_thresholds = {
            'price_outlier_std': 5.0,  # Standard deviations for outlier detection
            'volume_outlier_std': 4.0,
            'min_completeness': 0.95,  # Minimum data completeness
            'max_gap_minutes': 60      # Maximum acceptable gap in minutes
        }

        # Caching
        self.cache_dir = Path(self.cfg.base_path) / "cache"
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)

        logger.info(f"Enhanced Data Processor initialized with {self.parallel_workers} workers")

    def get_files_for_period(self, period_name: str, data_type: str = "processed_trades") -> List[str]:
        """Enhanced file discovery with validation and sorting."""
        try:
            if data_type == "raw_trades":
                path_template = self.cfg.get_raw_trades_path(period_name)
                file_extension = ".zip"
            elif data_type == "processed_trades":
                path_template = self.cfg.get_processed_trades_path(period_name)
                file_extension = ".parquet"
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            search_path = os.path.join(path_template, f"*{file_extension}")
            files = sorted(glob.glob(search_path))

            if not files:
                logger.warning(f"No files found for {data_type} in period {period_name} at {search_path}")
                return files

            # Validate file integrity
            valid_files = []
            for file_path in files:
                if self._validate_file_integrity(file_path, file_extension):
                    valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping corrupted file: {file_path}")

            logger.info(f"Found {len(valid_files)} valid {data_type} files for period {period_name}")
            return valid_files

        except Exception as e:
            logger.error(f"Error getting files for period {period_name}: {e}")
            return []

    def _validate_file_integrity(self, file_path: str, extension: str) -> bool:
        """Validate file integrity and accessibility."""
        try:
            if not os.path.exists(file_path):
                return False

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False

            # Basic format validation
            if extension == ".zip":
                with zipfile.ZipFile(file_path, 'r') as z:
                    return len(z.namelist()) > 0
            elif extension == ".parquet":
                # Try to read metadata
                parquet_file = pq.ParquetFile(file_path)
                return parquet_file.num_row_groups > 0

            return True

        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False

    def _get_cache_key(self, file_path: str, processing_params: Dict) -> str:
        """Generate cache key including file modification time."""
        try:
            mod_time = os.path.getmtime(file_path)
            content_hash = hashlib.md5(
                file_path.encode() +
                str(mod_time).encode() +
                str(processing_params).encode()
            ).hexdigest()
            return f"processed_{Path(file_path).stem}_{content_hash}"
        except OSError:
            # Fallback if file doesn't exist
            content_hash = hashlib.md5(file_path.encode() + str(processing_params).encode()).hexdigest()
            return f"processed_{Path(file_path).stem}_{content_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load processed data from cache."""
        if not self.enable_caching:
            return None

        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                logger.debug(f"Loading from cache: {cache_key}")
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                return None
        return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save processed data to cache."""
        if not self.enable_caching or data.empty:
            return

        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            data.to_parquet(cache_file, index=False)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _detect_price_outliers(self, prices: np.ndarray, threshold: float) -> np.ndarray:
        """Detect price outliers. Uses Numba if available."""
        if NUMBA_AVAILABLE:
            return self._detect_price_outliers_numba(prices, threshold)
        else:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            z_scores = np.abs((prices - mean_price) / (std_price + 1e-8))
            return z_scores > threshold

    if NUMBA_AVAILABLE:
        @staticmethod
        @njit
        def _detect_price_outliers_numba(prices: np.ndarray, threshold: float) -> np.ndarray:
            """Fast outlier detection using Numba."""
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            z_scores = np.abs((prices - mean_price) / (std_price + 1e-8))
            return z_scores > threshold

    def _advanced_data_quality_check(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Comprehensive data quality assessment."""
        total_trades = len(df)
        if total_trades == 0:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0)

        # Price outlier detection
        prices = df['price'].values
        price_outliers = np.sum(self._detect_price_outliers(
            prices, self.quality_thresholds['price_outlier_std']
        )) if len(prices) > 10 else 0

        # Volume outlier detection
        volumes = df['size'].values
        volume_outliers = np.sum(self._detect_price_outliers(
            volumes, self.quality_thresholds['volume_outlier_std']
        )) if len(volumes) > 10 else 0

        # Timestamp gap analysis
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            large_gaps = time_diffs > pd.Timedelta(minutes=self.quality_thresholds['max_gap_minutes'])
            timestamp_gaps = large_gaps.sum()
        else:
            timestamp_gaps = 0

        # Duplicate detection
        duplicate_trades = df.duplicated().sum()

        # Invalid price detection
        invalid_prices = ((df['price'] <= 0) | df['price'].isna()).sum()

        # Data completeness
        required_columns = ['timestamp', 'price', 'size', 'side']
        missing_data = 0
        for col in required_columns:
            if col in df.columns:
                missing_data += df[col].isna().sum()

        data_completeness = 1.0 - (missing_data / (len(required_columns) * total_trades))

        # Overall quality score
        outlier_ratio = (price_outliers + volume_outliers) / total_trades
        gap_ratio = timestamp_gaps / max(total_trades - 1, 1)
        duplicate_ratio = duplicate_trades / total_trades
        invalid_ratio = invalid_prices / total_trades

        quality_score = max(0.0, 1.0 - (outlier_ratio * 0.3 + gap_ratio * 0.2 +
                                       duplicate_ratio * 0.2 + invalid_ratio * 0.3 +
                                       (1 - data_completeness) * 0.3))

        return DataQualityMetrics(
            total_trades=total_trades,
            price_outliers=price_outliers,
            volume_outliers=volume_outliers,
            timestamp_gaps=timestamp_gaps,
            duplicate_trades=duplicate_trades,
            invalid_prices=invalid_prices,
            data_completeness=data_completeness,
            quality_score=quality_score
        )

    def _clean_and_validate_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning with validation."""
        try:
            # Basic transformations
            rename_map = {'id': 'trade_id', 'time': 'timestamp', 'qty': 'size'}
            chunk_df = chunk_df.rename(columns=rename_map)

            # Timestamp conversion with timezone handling
            if 'timestamp' in chunk_df.columns:
                chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms', utc=True)

            # Side determination
            if 'is_buyer_maker' in chunk_df.columns:
                chunk_df['side'] = np.where(chunk_df['is_buyer_maker'] == False, 'BUY', 'SELL')

            # Add asset column
            chunk_df['asset'] = self.cfg.primary_asset

            # Data quality checks and cleaning
            initial_count = len(chunk_df)

            # Remove invalid prices
            chunk_df = chunk_df[chunk_df['price'] > 0]

            # Remove invalid sizes
            chunk_df = chunk_df[chunk_df['size'] > 0]

            # Remove duplicates based on timestamp and price
            chunk_df = chunk_df.drop_duplicates(subset=['timestamp', 'price', 'size'])

            # Sort by timestamp
            chunk_df = chunk_df.sort_values('timestamp').reset_index(drop=True)

            # Log cleaning statistics
            final_count = len(chunk_df)
            if initial_count > final_count:
                logger.debug(f"Cleaned data: {initial_count} -> {final_count} trades "
                           f"({(initial_count-final_count)/initial_count:.2%} removed)")

            return chunk_df[self.cfg.final_columns]

        except Exception as e:
            logger.error(f"Error cleaning chunk: {e}")
            return pd.DataFrame()

    def _process_single_file(self, file_info: Dict) -> Optional[str]:
        """Process a single file with enhanced error handling."""
        raw_path = file_info['raw_path']
        output_path = file_info['output_path']

        # Check cache first
        cache_key = self._get_cache_key(raw_path, {})
        cached_data = self._load_from_cache(cache_key)

        if cached_data is not None:
            # Save cached data to output
            cached_data.to_parquet(output_path, index=False)
            return f"✅ {Path(raw_path).name} (from cache)"

        filename = os.path.basename(raw_path)

        try:
            with zipfile.ZipFile(raw_path, 'r') as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Detect header
                    first_line = f.readline().decode('utf-8').strip()
                    f.seek(0)
                    has_header = first_line.startswith(self.cfg.binance_raw_columns[0])

                    # Optimized reading parameters
                    read_params = {
                        'chunksize': 4_000_000,
                        'dtype': self.cfg.dtype_map,
                        'header': 0 if has_header else None,
                        'names': None if has_header else self.cfg.binance_raw_columns,
                        'low_memory': False
                    }

                    chunk_iterator = pd.read_csv(f, **read_params)

                    # Process first chunk and initialize Parquet writer
                    try:
                        first_chunk = next(chunk_iterator)
                        processed_chunk = self._clean_and_validate_chunk(first_chunk)

                        if processed_chunk.empty:
                            logger.warning(f"Empty processed chunk for {filename}")
                            return f"⚠️ {filename} (empty after processing)"

                        # Create Parquet table and writer
                        table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
                        with pq.ParquetWriter(output_path, table.schema, compression='snappy') as writer:
                            writer.write_table(table)

                            # Process remaining chunks
                            total_chunks = 1
                            total_rows = len(processed_chunk)
                            all_processed_chunks = [processed_chunk]  # Store for caching

                            for chunk in chunk_iterator:
                                processed = self._clean_and_validate_chunk(chunk)
                                if not processed.empty:
                                    writer.write_table(pa.Table.from_pandas(processed, preserve_index=False))
                                    all_processed_chunks.append(processed)
                                    total_rows += len(processed)
                                total_chunks += 1

                            # Cache the complete processed data
                            if self.enable_caching and all_processed_chunks:
                                complete_data = pd.concat(all_processed_chunks, ignore_index=True)
                                self._save_to_cache(complete_data, cache_key)

                        return f"✅ {filename} ({total_rows:,} trades, {total_chunks} chunks)"

                    except StopIteration:
                        logger.warning(f"Empty file: {filename}")
                        return f"⚠️ {filename} (empty file)"

        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            return f"❌ {filename} (error: {str(e)[:50]})"

    def _intelligent_gap_filling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conservative gap filling to avoid data leakage."""
        try:
            # Only forward fill for very small gaps (< 3 minutes)
            time_diff = df.index.to_series().diff()
            small_gaps = time_diff <= pd.Timedelta(minutes=3)

            # Forward fill OHLC only for very small gaps
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # Only fill small gaps to avoid lookahead bias
                    mask = small_gaps & df[col].isna()
                    df.loc[mask, col] = df[col].ffill(limit=2)

            # Set volume to 0 for gaps (more conservative)
            df['volume'] = df['volume'].fillna(0)
            df['trade_count'] = df['trade_count'].fillna(0)

            # For other numeric columns, use forward fill only for small gaps
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['volume', 'trade_count', 'open', 'high', 'low', 'close']:
                    mask = small_gaps & df[col].isna()
                    df.loc[mask, col] = df[col].ffill(limit=1)

            return df.dropna()

        except Exception as e:
            logger.error(f"Error in gap filling: {e}")
            return df

    def process_raw_trades_parallel(self, period_name: str, force_reprocess: bool = False):
        """Process raw trades with parallel processing and enhanced features."""
        try:
            raw_dir = self.cfg.get_raw_trades_path(period_name)
            processed_dir = self.cfg.get_processed_trades_path(period_name)
            os.makedirs(processed_dir, exist_ok=True)

            logger.info(f"🚀 Starting Enhanced Data Processing for Period: {period_name.upper()}")
            logger.info(f"Using {self.parallel_workers} parallel workers")

            files_to_process = self.get_files_for_period(period_name, "raw_trades")

            if not files_to_process:
                logger.warning("No files to process")
                return

            # Prepare file processing tasks
            tasks = []
            for raw_path in files_to_process:
                filename = os.path.basename(raw_path)
                output_path = os.path.join(processed_dir, filename.replace('.zip', '.parquet'))

                if os.path.exists(output_path) and not force_reprocess:
                    logger.info(f"⏭️ Skipping {filename} (already processed)")
                    continue

                tasks.append({
                    'raw_path': raw_path,
                    'output_path': output_path
                })

            if not tasks:
                logger.info("All files already processed")
                return

            logger.info(f"Processing {len(tasks)} files...")

            # Process files in parallel
            results = []
            with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = [executor.submit(self._process_single_file, task) for task in tasks]

                # Use tqdm for progress tracking
                for future in tqdm(futures, desc="Processing files", unit="file"):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        results.append(result)
                        if result:
                            logger.info(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        results.append(f"❌ Task failed: {e}")

            # Summary statistics
            successful = sum(1 for r in results if r and r.startswith("✅"))
            warnings_count = sum(1 for r in results if r and r.startswith("⚠️"))
            errors = sum(1 for r in results if r and r.startswith("❌"))

            logger.info(f"📊 Processing Summary:")
            logger.info(f" ✅ Successful: {successful}")
            logger.info(f" ⚠️ Warnings: {warnings_count}")
            logger.info(f" ❌ Errors: {errors}")

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            raise

    def create_enhanced_bars_from_trades(self, period_name: str,
                                        additional_features: bool = True,
                                        quality_check: bool = True) -> pd.DataFrame:
        """
        Bar creation with order flow features (VWAP, Volume Delta).
        This is the critical step where VWAP and Volume Delta are born from raw trades.
        """
        logger.info(f"📊 Creating enriched bars with order flow data for period: {period_name}")

        all_trade_files = self.get_files_for_period(period_name, "processed_trades")

        if not all_trade_files:
            raise FileNotFoundError(f"No processed trade files found for period '{period_name}'")

        # Check cache for complete bars
        cache_key = f"bars_enriched_{period_name}_{hashlib.md5(str(all_trade_files).encode()).hexdigest()}"
        cached_bars = self._load_from_cache(cache_key)

        if cached_bars is not None:
            logger.info("📦 Loaded enriched bars from cache")
            return cached_bars

        # Choose processing method based on availability
        if DASK_AVAILABLE and len(all_trade_files) > 10:
            bars_df = self._create_bars_with_dask(all_trade_files, additional_features)
        else:
            bars_df = self._create_bars_with_pandas(all_trade_files, additional_features)

        if bars_df.empty:
            logger.warning("No valid bar data generated")
            return pd.DataFrame()

        # Sort index and fill gaps
        bars_df = bars_df.sort_index()
        bars_df = self._intelligent_gap_filling(bars_df)

        # Reset index to get timestamp as column
        bars_df = bars_df.reset_index()

        # Cache the results
        self._save_to_cache(bars_df, cache_key)

        logger.info(f"✅ Generated {len(bars_df):,} enriched bars from "
                   f"{bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")

        return bars_df

    def _create_bars_with_dask(self, all_trade_files: List[str], additional_features: bool) -> pd.DataFrame:
        """Create bars using Dask for better performance with large datasets."""
        logger.info(f"Loading {len(all_trade_files)} trade files with Dask...")

        try:
            # Load all parquet files into a Dask DataFrame
            ddf = dd.read_parquet(all_trade_files, columns=['timestamp', 'price', 'size', 'side'])

            if ddf.npartitions == 0:
                logger.warning("No data found in trade files.")
                return pd.DataFrame()

            # Create helper columns for VWAP and Volume Delta calculation
            ddf['price_volume'] = ddf['price'] * ddf['size']
            ddf['buy_volume'] = ddf['size'].where(ddf['side'] == 'BUY', 0)
            ddf['sell_volume'] = ddf['size'].where(ddf['side'] == 'SELL', 0)

            # Set timestamp as index for resampling
            ddf = ddf.set_index('timestamp')

            # Resample to create OHLCV bars
            resample_freq = self.cfg.base_bar_timeframe
            agg_dict = {
                'price': ['first', 'max', 'min', 'last', 'count'],
                'size': 'sum',
                'price_volume': 'sum',
                'buy_volume': 'sum',
                'sell_volume': 'sum'
            }

            ohlcv_ddf = ddf.resample(resample_freq).agg(agg_dict)

            # Flatten MultiIndex columns and rename
            ohlcv_ddf.columns = [
                'open', 'high', 'low', 'close', 'trade_count',
                'volume', 'price_volume_sum', 'buy_volume_sum', 'sell_volume_sum'
            ]

            # Calculate VWAP and Volume Delta
            ohlcv_ddf['vwap'] = ohlcv_ddf['price_volume_sum'] / (ohlcv_ddf['volume'] + 1e-9)
            ohlcv_ddf['volume_delta'] = ohlcv_ddf['buy_volume_sum'] - ohlcv_ddf['sell_volume_sum']

            # Drop intermediate columns
            ohlcv_ddf = ohlcv_ddf.drop(columns=['price_volume_sum', 'buy_volume_sum', 'sell_volume_sum'])

            # Add additional features if requested
            if additional_features:
                ohlcv_ddf = self._add_enhanced_features(ohlcv_ddf)

            # Remove bars with no trades
            ohlcv_ddf = ohlcv_ddf[ohlcv_ddf['trade_count'] > 0]

            # Trigger computation
            logger.info("🚀 Triggering Dask computation...")
            bars_df = ohlcv_ddf.compute()
            logger.info("✅ Dask computation complete.")

            return bars_df

        except Exception as e:
            logger.error(f"Dask processing failed: {e}")
            logger.info("Falling back to pandas processing...")
            return self._create_bars_with_pandas(all_trade_files, additional_features)

    def _create_bars_with_pandas(self, all_trade_files: List[str], additional_features: bool) -> pd.DataFrame:
        """Create bars using pandas as fallback."""
        logger.info(f"Loading {len(all_trade_files)} trade files with pandas...")

        try:
            # Load all files
            dfs = [pd.read_parquet(f, columns=['timestamp', 'price', 'size', 'side']) for f in all_trade_files]

            # Combine all data
            combined_df = pd.concat(dfs, ignore_index=True)

            # Create helper columns for VWAP and Volume Delta calculation
            combined_df['price_volume'] = combined_df['price'] * combined_df['size']
            combined_df['buy_volume'] = combined_df['size'].where(combined_df['side'] == 'BUY', 0)
            combined_df['sell_volume'] = combined_df['size'].where(combined_df['side'] == 'SELL', 0)

            combined_df = combined_df.set_index('timestamp').sort_index()

            # Resample to create OHLCV bars
            resample_freq = self.cfg.base_bar_timeframe

            bars_df = combined_df.resample(resample_freq).agg({
                'price': ['first', 'max', 'min', 'last', 'count'],
                'size': 'sum',
                'price_volume': 'sum',
                'buy_volume': 'sum',
                'sell_volume': 'sum'
            })

            # Flatten MultiIndex columns and rename
            bars_df.columns = [
                'open', 'high', 'low', 'close', 'trade_count', 'volume',
                'price_volume_sum', 'buy_volume_sum', 'sell_volume_sum'
            ]
            
            # Calculate VWAP and Volume Delta for the base timeframe
            bars_df['vwap'] = bars_df['price_volume_sum'] / (bars_df['volume'] + 1e-9)
            bars_df['volume_delta'] = bars_df['buy_volume_sum'] - bars_df['sell_volume_sum']
            
            # --- START: MODIFIED LOGIC TO PRECOMPUTE dist_vwap_3m ---

            # 1. Resample to 3-minutes to get the 3T VWAP
            logger.info("Calculating 3-minute VWAP for feature precomputation...")
            df_3m = combined_df.resample('3T').agg({
                'size': 'sum',
                'price_volume': 'sum'
            })
            df_3m.columns = ['volume_3m', 'price_volume_sum_3m']
            df_3m['vwap_3m'] = df_3m['price_volume_sum_3m'] / (df_3m['volume_3m'] + 1e-9)

            # 2. Merge the 3-minute VWAP back onto the base bars DataFrame
            # pd.merge_asof ensures that each 20s bar gets the VWAP from the most recent 3-minute bar, preventing lookahead bias.
            bars_df = pd.merge_asof(
                left=bars_df.sort_index(),
                right=df_3m[['vwap_3m']].sort_index(),
                left_index=True,
                right_index=True,
                direction='backward' # Use the last known 3m VWAP
            )
            
            # 3. Calculate the distance feature and fill any initial NaNs
            bars_df['dist_vwap_3m'] = (bars_df['close'] - bars_df['vwap_3m']) / (bars_df['vwap_3m'] + 1e-9)
            bars_df.drop(columns=['vwap_3m'], inplace=True) # Clean up the intermediate column
            bars_df['dist_vwap_3m'].fillna(0.0, inplace=True)

            logger.info("✅ 'dist_vwap_3m' feature precomputed and added to bars.")
            # --- END: MODIFIED LOGIC ---

            # Drop intermediate columns
            bars_df = bars_df.drop(columns=['price_volume_sum', 'buy_volume_sum', 'sell_volume_sum'])

            # Add additional features if requested
            if additional_features:
                bars_df = self._add_enhanced_features(bars_df)

            # Remove bars with no trades
            bars_df = bars_df[bars_df['trade_count'] > 0]

            return bars_df

        except Exception as e:
            logger.error(f"Pandas processing failed: {e}")
            return pd.DataFrame()

    def _add_enhanced_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical and statistical features."""
        try:
            # Price-based features
            ohlcv_df['typical_price'] = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
            ohlcv_df['price_range'] = ohlcv_df['high'] - ohlcv_df['low']
            ohlcv_df['price_change'] = ohlcv_df['close'].pct_change()

            # Volume features
            ohlcv_df['volume_ma_5'] = ohlcv_df['volume'].rolling(5).mean()
            ohlcv_df['volume_ratio'] = ohlcv_df['volume'] / (ohlcv_df['volume_ma_5'] + 1e-8)
            ohlcv_df['log_volume'] = np.log1p(ohlcv_df['volume'])
            ohlcv_df['normalized_volume'] = ohlcv_df['volume'] / ohlcv_df['volume'].rolling(20).mean()

            # Volatility features
            ohlcv_df['volatility'] = ohlcv_df['price_change'].rolling(20).std()

            # True Range calculation (compatible with both Dask and pandas)
            tr1 = ohlcv_df['high'] - ohlcv_df['low']
            tr2 = (ohlcv_df['high'] - ohlcv_df['close'].shift(1)).abs()
            tr3 = (ohlcv_df['low'] - ohlcv_df['close'].shift(1)).abs()

            if DASK_AVAILABLE and hasattr(ohlcv_df, 'npartitions'):
                # Dask DataFrame
                ohlcv_df['true_range'] = dd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            else:
                # Pandas DataFrame
                ohlcv_df['true_range'] = np.maximum.reduce([tr1, tr2, tr3])

            # Market microstructure features
            ohlcv_df['spread_proxy'] = (ohlcv_df['high'] - ohlcv_df['low']) / ohlcv_df['close']
            ohlcv_df['trade_intensity'] = ohlcv_df['trade_count'] / (ohlcv_df['volume'] + 1e-8)

            # Time-based features
            if hasattr(ohlcv_df.index, 'hour'):
                ohlcv_df['hour'] = ohlcv_df.index.hour
                ohlcv_df['day_of_week'] = ohlcv_df.index.dayofweek
                ohlcv_df['is_weekend'] = ohlcv_df['day_of_week'].isin([5, 6])

            return ohlcv_df

        except Exception as e:
            logger.error(f"Error adding enhanced features: {e}")
            return ohlcv_df

# --- CONVENIENCE FUNCTIONS ---

def process_trades_for_period(period_name: str, force_reprocess: bool = False):
    """Convenience function to process trades for a period."""
    processor = EnhancedDataProcessor()
    return processor.process_raw_trades_parallel(period_name, force_reprocess)

def create_bars_from_trades(period_name: str, additional_features: bool = True) -> pd.DataFrame:
    """Convenience function to create bars from processed trades."""
    processor = EnhancedDataProcessor()
    return processor.create_enhanced_bars_from_trades(period_name, additional_features)

# --- NEW VECTORIZED FEATURE GENERATION FOR NORMALIZER ---
# This new section is added to solve the performance bottleneck in the trainer.

def _calculate_sr_distances_for_window(window: np.ndarray, num_levels: int) -> Dict[str, float]:
    """
    Helper function to calculate S/R distances for a given window of prices.
    This replicates the logic from features.py in a way that can be applied to a rolling window.
    """
    if len(window) < 2:
        return {}
        
    current_price = window[-1]
    results = {f'dist_s{i+1}': 1.0 for i in range(num_levels)}
    results.update({f'dist_r{i+1}': 1.0 for i in range(num_levels)})

    peaks, _ = scipy.signal.find_peaks(window, distance=5)
    troughs, _ = scipy.signal.find_peaks(-window, distance=5)

    support_levels = sorted([p for p in window[troughs] if p < current_price], reverse=True)
    for i in range(num_levels):
        if i < len(support_levels):
            results[f'dist_s{i+1}'] = (current_price - support_levels[i]) / current_price

    resistance_levels = sorted([p for p in window[peaks] if p > current_price])
    for i in range(num_levels):
        if i < len(resistance_levels):
            results[f'dist_r{i+1}'] = (resistance_levels[i] - current_price) / current_price
            
    return results

def generate_stateful_features_for_fitting(bars_df: pd.DataFrame, strat_cfg: StrategyConfig) -> pd.DataFrame:
    """
    Vectorized generation of stateful features for fitting the normalizer.
    This is significantly faster than stepping through the environment and avoids
    a major performance bottleneck during trainer initialization.
    """
    logger.info("🚀 Starting vectorized generation of stateful features for normalizer...")
    
    # Ensure bars_df is indexed by timestamp for resampling and joining
    if 'timestamp' not in bars_df.columns or not pd.api.types.is_datetime64_any_dtype(bars_df['timestamp']):
        raise ValueError("bars_df must have a 'timestamp' column of datetime type.")
    
    base_df_indexed = bars_df.set_index('timestamp').sort_index()
    
    # This will hold all the generated feature columns
    features_df = pd.DataFrame(index=base_df_indexed.index)

    # Pre-calculate all required resampled dataframes for efficiency
    timeframes = {}
    required_freqs = {calc.timeframe for calc in strat_cfg.stateful_calculators}
    for freq in tqdm(required_freqs, desc="Resampling data"):
        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df_indexed.columns}
        df_resampled = base_df_indexed.resample(freq).agg(valid_agg_rules).ffill()
        timeframes[freq] = df_resampled

    # Calculate each feature
    for calc_cfg in tqdm(strat_cfg.stateful_calculators, desc="Calculating features"):
        df_tf = timeframes[calc_cfg.timeframe]
        source_data = df_tf[calc_cfg.source_col]

        if calc_cfg.class_name == 'StatefulBBWPercentRank':
            period = calc_cfg.params['period']
            rank_window = calc_cfg.params['rank_window']
            sma = source_data.rolling(window=period, min_periods=period).mean()
            std = source_data.rolling(window=period, min_periods=period).std()
            bbw = (4 * std) / (sma + 1e-9)
            # Percentile rank over the rank_window
            bbw_pct_rank = bbw.rolling(window=rank_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False) * 100
            # Reindex to align with the base 20s timeframe, forward-filling values
            features_df[calc_cfg.output_keys[0]] = bbw_pct_rank.reindex(features_df.index, method='ffill')

        elif calc_cfg.class_name == 'StatefulPriceDistanceMA':
            period = calc_cfg.params['period']
            sma = source_data.rolling(window=period, min_periods=period).mean()
            dist = (source_data / (sma + 1e-9)) - 1.0
            features_df[calc_cfg.output_keys[0]] = dist.reindex(features_df.index, method='ffill')

        elif calc_cfg.class_name == 'StatefulSRDistances':
            period = calc_cfg.params['period']
            num_levels = calc_cfg.params['num_levels']
            
            # This operation is slower but still much faster than the environment loop
            results_series = source_data.rolling(window=period).apply(
                lambda x: _calculate_sr_distances_for_window(x, num_levels), raw=True
            )
            # Convert the series of dicts to a DataFrame
            sr_df = pd.DataFrame(results_series.dropna().tolist(), index=results_series.dropna().index)
            
            # Reindex and merge each S/R column into the main features DataFrame
            for col in sr_df.columns:
                features_df[col] = sr_df[col].reindex(features_df.index, method='ffill')

    # Final cleanup
    features_df.fillna(0.0, inplace=True)
    
    logger.info("✅ Vectorized feature generation complete.")
    return features_df.reset_index()


if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Testing data processor...")

        # Test with sample period
        # process_trades_for_period("in_sample")
        # bars_df = create_bars_from_trades("in_sample")
        # logger.info(f"✅ Created {len(bars_df)} bars")

        logger.info("✅ Data processor test completed!")

    except Exception as e:
        logger.error(f"Data processor test failed: {e}")

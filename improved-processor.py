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
import dask.dataframe as dd
from numba import jit, njit
import logging
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass
from ..config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Enhanced data processor with quality checks and optimizations."""

    def __init__(self, config=None, enable_caching: bool = True, parallel_workers: int = None):
        self.cfg = config or SETTINGS
        self.enable_caching = enable_caching
        self.parallel_workers = parallel_workers or min(mp.cpu_count(), 8)

        # Quality thresholds
        self.quality_thresholds = {
            'price_outlier_std': 5.0,  # Standard deviations for outlier detection
            'volume_outlier_std': 4.0,
            'min_completeness': 0.95,  # Minimum data completeness
            'max_gap_minutes': 60  # Maximum acceptable gap in minutes
        }

        # Caching
        self.cache_dir = Path(self.cfg.base_path) / "cache"
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)

        logger.info(f"Enhanced Data Processor initialized with {self.parallel_workers} workers")

    def get_files_for_period(self, period_name: str, data_type: str = "processed_trades") -> List[str]:
        """Enhanced file discovery with validation and sorting."""
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
        """IMPROVED: Generate cache key including file modification time."""
        try:
            mod_time = os.path.getmtime(file_path)
            content_hash = hashlib.md5(
                file_path.encode() + 
                str(mod_time).encode() + 
                str(processing_params).encode()
            ).hexdigest()
            return f"processed_{Path(file_path).stem}_{content_hash}"
        except OSError:
            # Fallback to original method if file doesn't exist
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

    @njit
    def _detect_price_outliers_numba(self, prices: np.ndarray, threshold: float) -> np.ndarray:
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
        price_outliers = np.sum(self._detect_price_outliers_numba(
            prices, self.quality_thresholds['price_outlier_std']
        )) if len(prices) > 10 else 0

        # Volume outlier detection
        volumes = df['size'].values
        volume_outliers = np.sum(self._detect_price_outliers_numba(
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
                        'chunksize': 5_000_000,  # Reduced chunk size for better memory management
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
        """IMPROVED: Conservative gap filling to avoid data leakage."""
        # Only forward fill for very small gaps (< 3 minutes) to avoid creating artificial data
        time_diff = df.index.to_series().diff()
        small_gaps = time_diff <= pd.Timedelta(minutes=3)

        # Forward fill OHLC only for very small gaps
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Only fill small gaps to avoid lookahead bias
                mask = small_gaps & df[col].isna()
                df.loc[mask, col] = df[col].fillna(method='ffill', limit=2)

        # Set volume to 0 for gaps (more conservative)
        df['volume'] = df['volume'].fillna(0)
        df['trade_count'] = df['trade_count'].fillna(0)

        # For other features, use backward fill only for small gaps to avoid lookahead
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['volume', 'trade_count', 'open', 'high', 'low', 'close']:
                # Use a combination of forward and backward fill, but limit to very small gaps
                mask = small_gaps & df[col].isna()
                df.loc[mask, col] = df[col].fillna(method='ffill', limit=1)

        return df.dropna()

    def process_raw_trades_parallel(self, period_name: str, force_reprocess: bool = False):
        """Process raw trades with parallel processing and enhanced features."""
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
        warnings = sum(1 for r in results if r and r.startswith("⚠️"))
        errors = sum(1 for r in results if r and r.startswith("❌"))

        logger.info(f"📊 Processing Summary:")
        logger.info(f" ✅ Successful: {successful}")
        logger.info(f" ⚠️ Warnings: {warnings}")
        logger.info(f" ❌ Errors: {errors}")

    def create_enhanced_bars_from_trades(self, period_name: str,
                                       additional_features: bool = True,
                                       quality_check: bool = True) -> pd.DataFrame:
        """Enhanced bar creation with additional features and quality checks."""
        logger.info(f"📊 Creating enhanced bars for period: {period_name}")

        all_trade_files = self.get_files_for_period(period_name, "processed_trades")
        if not all_trade_files:
            raise FileNotFoundError(f"No processed trade files found for period '{period_name}'")

        # Check cache for complete bars
        cache_key = f"bars_{period_name}_{hashlib.md5(str(all_trade_files).encode()).hexdigest()}"
        cached_bars = self._load_from_cache(cache_key)

        if cached_bars is not None:
            logger.info("📦 Loaded bars from cache")
            return cached_bars

        all_ohlcv = []
        quality_metrics = []

        logger.info(f"Processing {len(all_trade_files)} trade files...")

        for file_path in tqdm(all_trade_files, desc="Processing trade files"):
            try:
                # Read trade data
                df = pd.read_parquet(file_path, columns=['timestamp', 'price', 'size'])

                if df.empty:
                    logger.warning(f"Empty trade file: {file_path}")
                    continue

                # Quality check
                if quality_check:
                    metrics = self._advanced_data_quality_check(df)
                    quality_metrics.append(metrics)

                    if metrics.quality_score < 0.8:
                        logger.warning(f"Low quality data in {file_path}: {metrics.quality_score:.2%}")

                # Set timestamp as index for resampling
                df = df.set_index('timestamp')

                # Resample to create OHLCV bars
                resample_freq = self.cfg.base_bar_timeframe

                # Enhanced aggregation with additional statistics
                agg_dict = {
                    'price': ['first', 'max', 'min', 'last', 'count'],  # OHLC + count
                    'size': ['sum', 'mean', 'std']  # Volume + statistics
                }

                ohlcv_df = df.resample(resample_freq).agg(agg_dict)

                # Flatten MultiIndex columns
                ohlcv_df.columns = [
                    'open', 'high', 'low', 'close', 'trade_count',
                    'volume', 'avg_trade_size', 'trade_size_std'
                ]

                # Additional features if requested
                if additional_features:
                    ohlcv_df = self._add_enhanced_features(ohlcv_df)

                # Remove bars with no trades
                ohlcv_df = ohlcv_df[ohlcv_df['trade_count'] > 0]

                all_ohlcv.append(ohlcv_df)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_ohlcv:
            logger.warning("No valid bar data generated")
            return pd.DataFrame()

        # Combine all OHLCV data
        bars_df = pd.concat(all_ohlcv).sort_index()

        # Fill missing values intelligently
        bars_df = self._intelligent_gap_filling(bars_df)

        # Final quality report
        if quality_check and quality_metrics:
            self._generate_quality_report(quality_metrics, period_name)

        # Reset index to get timestamp as column
        bars_df = bars_df.reset_index()

        # Cache the results
        self._save_to_cache(bars_df, cache_key)

        logger.info(f"✅ Generated {len(bars_df):,} bars from "
                   f"{bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")

        return bars_df

    def _add_enhanced_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical and statistical features."""
        # Price-based features
        ohlcv_df['typical_price'] = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
        ohlcv_df['price_range'] = ohlcv_df['high'] - ohlcv_df['low']
        ohlcv_df['price_change'] = ohlcv_df['close'].pct_change()

        # IMPROVED: Normalize volume features to avoid scale issues
        ohlcv_df['volume_ma_5'] = ohlcv_df['volume'].rolling(5).mean()
        ohlcv_df['volume_ratio'] = ohlcv_df['volume'] / (ohlcv_df['volume_ma_5'] + 1e-8)
        
        # Log-normalize volume for better neural network training
        ohlcv_df['log_volume'] = np.log1p(ohlcv_df['volume'])
        ohlcv_df['normalized_volume'] = ohlcv_df['volume'] / ohlcv_df['volume'].rolling(20).mean()

        # Volatility features
        ohlcv_df['volatility'] = ohlcv_df['price_change'].rolling(20).std()
        ohlcv_df['true_range'] = np.maximum.reduce([
            ohlcv_df['high'] - ohlcv_df['low'],
            np.abs(ohlcv_df['high'] - ohlcv_df['close'].shift(1)),
            np.abs(ohlcv_df['low'] - ohlcv_df['close'].shift(1))
        ])

        # Market microstructure features
        ohlcv_df['spread_proxy'] = (ohlcv_df['high'] - ohlcv_df['low']) / ohlcv_df['close']
        ohlcv_df['trade_intensity'] = ohlcv_df['trade_count'] / (ohlcv_df['volume'] + 1e-8)

        # Time-based features
        ohlcv_df['hour'] = ohlcv_df.index.hour
        ohlcv_df['day_of_week'] = ohlcv_df.index.dayofweek
        ohlcv_df['is_weekend'] = ohlcv_df['day_of_week'].isin([5, 6])

        return ohlcv_df

    def _generate_quality_report(self, quality_metrics: List[DataQualityMetrics], period_name: str):
        """Generate comprehensive data quality report."""
        if not quality_metrics:
            return

        # Aggregate metrics
        total_trades = sum(m.total_trades for m in quality_metrics)
        total_outliers = sum(m.price_outliers + m.volume_outliers for m in quality_metrics)
        total_gaps = sum(m.timestamp_gaps for m in quality_metrics)
        total_duplicates = sum(m.duplicate_trades for m in quality_metrics)
        avg_quality_score = np.mean([m.quality_score for m in quality_metrics])
        avg_completeness = np.mean([m.data_completeness for m in quality_metrics])

        # Create quality report
        report = {
            'period': period_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_trades': total_trades,
                'total_files_processed': len(quality_metrics),
                'avg_quality_score': avg_quality_score,
                'avg_data_completeness': avg_completeness
            },
            'issues': {
                'total_outliers': total_outliers,
                'outlier_rate': total_outliers / total_trades if total_trades > 0 else 0,
                'timestamp_gaps': total_gaps,
                'duplicate_trades': total_duplicates,
                'duplicate_rate': total_duplicates / total_trades if total_trades > 0 else 0
            },
            'file_details': []
        }

        # Add per-file details
        for i, metrics in enumerate(quality_metrics):
            report['file_details'].append({
                'file_index': i,
                'trades': metrics.total_trades,
                'quality_score': metrics.quality_score,
                'completeness': metrics.data_completeness,
                'outliers': metrics.price_outliers + metrics.volume_outliers,
                'gaps': metrics.timestamp_gaps,
                'duplicates': metrics.duplicate_trades
            })

        # Save report
        report_dir = Path(self.cfg.base_path) / "quality_reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"quality_report_{period_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Log summary
        logger.info(f"📋 Data Quality Report for {period_name}:")
        logger.info(f" Total Trades: {total_trades:,}")
        logger.info(f" Average Quality Score: {avg_quality_score:.1%}")
        logger.info(f" Data Completeness: {avg_completeness:.1%}")
        logger.info(f" Outlier Rate: {(total_outliers/total_trades):.2%}")
        logger.info(f" Report saved: {report_file}")

    def load_and_prepare_funding_data_enhanced(self) -> pd.DataFrame:
        """Enhanced funding rate data loading with validation."""
        funding_path = self.cfg.get_funding_rate_path()

        if not os.path.exists(funding_path):
            logger.warning(f"Funding rate path not found: {funding_path}")
            return pd.DataFrame()

        try:
            # Look for funding rate files
            funding_files = glob.glob(os.path.join(funding_path, "*.parquet"))
            if not funding_files:
                # Try CSV files as fallback
                funding_files = glob.glob(os.path.join(funding_path, "*.csv"))

            if not funding_files:
                logger.warning("No funding rate files found")
                return pd.DataFrame()

            funding_data = []
            for file_path in funding_files:
                try:
                    if file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    else:
                        df = pd.read_csv(file_path)

                    # Standardize column names
                    if 'fundingTime' in df.columns:
                        df = df.rename(columns={'fundingTime': 'timestamp', 'fundingRate': 'funding_rate'})

                    # Convert timestamp
                    if 'timestamp' in df.columns:
                        if df['timestamp'].dtype in ['int64', 'int32']:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

                    # Validate funding rates (should be between -1% and 1%)
                    if 'funding_rate' in df.columns:
                        df = df[(df['funding_rate'] >= -0.01) & (df['funding_rate'] <= 0.01)]

                    funding_data.append(df)

                except Exception as e:
                    logger.warning(f"Failed to load funding file {file_path}: {e}")
                    continue

            if funding_data:
                combined_funding = pd.concat(funding_data, ignore_index=True)
                combined_funding = combined_funding.drop_duplicates(subset=['timestamp'])
                combined_funding = combined_funding.sort_values('timestamp').set_index('timestamp')

                logger.info(f"✅ Loaded {len(combined_funding)} funding rate records")
                return combined_funding

        except Exception as e:
            logger.error(f"Error loading funding data: {e}")

        return pd.DataFrame()


# --- BACKWARDS COMPATIBILITY FUNCTIONS ---

def get_files_for_period(period_name: str, data_type: str = "processed_trades") -> List[str]:
    """Backwards compatible function."""
    processor = EnhancedDataProcessor()
    return processor.get_files_for_period(period_name, data_type)


def process_raw_trades(period_name: str):
    """Backwards compatible function with enhanced processing."""
    processor = EnhancedDataProcessor()
    processor.process_raw_trades_parallel(period_name)


def create_bars_from_trades(period_name: str) -> pd.DataFrame:
    """Backwards compatible function with enhanced features."""
    processor = EnhancedDataProcessor()
    return processor.create_enhanced_bars_from_trades(period_name)


def load_and_prepare_funding_data() -> pd.DataFrame:
    """Backwards compatible function."""
    processor = EnhancedDataProcessor()
    return processor.load_and_prepare_funding_data_enhanced()


if __name__ == "__main__":
    # Example usage
    processor = EnhancedDataProcessor(parallel_workers=4, enable_caching=True)

    # Process raw data with quality checks
    processor.process_raw_trades_parallel("in_sample", force_reprocess=False)

    # Create enhanced bars with additional features
    bars = processor.create_enhanced_bars_from_trades("in_sample", additional_features=True)

    print("✅ Enhanced data processing completed!")

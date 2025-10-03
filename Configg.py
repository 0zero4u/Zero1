import os
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Any, Union
# FIXED: Updated Pydantic imports for V2 compatibility
from pydantic import BaseModel, field_validator, model_validator, Field, ConfigDict
from enum import Enum
import warnings
from dotenv import load_dotenv
import logging
import math

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- UTILITY FUNCTIONS ---

def get_bool_env(var_name: str, default: str = 'false') -> bool:
    """Safely get a boolean value from an environment variable."""
    return os.getenv(var_name, default).lower() in ('true', '1', 't', 'yes')

def setup_environment():
    """Setup logging and environment validation."""
    # Setup logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create required directories
    base_path = os.getenv('BASE_PATH', './data')
    required_dirs = [
        os.path.join(base_path, 'models'),
        os.path.join(base_path, 'logs'),
        os.path.join(base_path, 'cache'),
        os.path.join(base_path, 'backtest_results'),
        os.path.join(base_path, 'optimization') # <-- ADDED THIS LINE
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info("Environment setup completed")

# --- ENUMS FOR TYPE SAFETY ---

class AssetType(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"

### FIX 1: UPDATED TIMEFRAMETYPE ENUM ###
class TimeFrameType(str, Enum):
    SECOND_20 = "20s"
    MINUTE_1 = "1min"
    MINUTE_3 = "3min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    # --- START OF FIX: Use modern pandas frequency strings ---
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    # --- END OF FIX ---

class Environment(str, Enum):
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

### FIX 2: UPDATED FEATUREKEYS ENUM FOR CONSISTENCY ###
class FeatureKeys(str, Enum):
    """Enum for feature keys to prevent typos and enable autocompletion."""
    PRICE_20S = 'price_20s'
    OHLCV_20S = 'ohlcv_20s'
    PRICE_1M = 'price_1min'
    OHLC_1M = 'ohlc_1min'
    PRICE_3M = 'price_3min'
    OHLCV_3M = 'ohlcv_3min'
    PRICE_5M = 'price_5min'
    PRICE_15M = 'price_15min'
    OHLC_15M = 'ohlc_15min'
    PRICE_1H = 'price_1h'
    CONTEXT = 'context'
    
    # Keys for Volume Delta data streams
    VOLUME_DELTA_20S = 'volume_delta_20s'
    VOLUME_DELTA_1M = 'volume_delta_1min'
    
    # Key for the agent's internal portfolio state
    PORTFOLIO_STATE = 'portfolio_state'
    
    # Key for pre-calculated features from the data processor
    PRECOMPUTED_FEATURES = 'precomputed_features'

# --- TRANSFORMER ARCHITECTURE CONFIG ---

class ModelArchitectureConfig(BaseModel):
    """Configuration for the Transformer-based Neural Architecture"""
    
    # Transformer Architecture Parameters
    transformer_d_model: int = Field(default=64, ge=32, le=512, description="Transformer model dimension")
    transformer_n_heads: int = Field(default=4, ge=1, le=16, description="Number of attention heads")
    transformer_dim_feedforward: int = Field(default=256, ge=64, le=2048, description="Transformer feed-forward dimension")
    transformer_num_layers: int = Field(default=2, ge=1, le=8, description="Number of Transformer encoder layers")
    
    # Expert and attention parameters
    expert_output_dim: int = Field(default=32, ge=8, le=256, description="Expert head output dimension")
    attention_head_features: int = Field(default=64, ge=16, le=256, description="Attention head output features")
    
    # Regularization parameters
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.8, description="Dropout rate for regularization")
    use_batch_norm: bool = Field(default=True, description="Whether to use batch normalization")
    use_residual_connections: bool = Field(default=True, description="Whether to use residual connections")
    
    # FIXED: Replaced deprecated @validator with @model_validator for Pydantic V2
    @model_validator(mode='after')
    def validate_transformer_d_model(self) -> 'ModelArchitectureConfig':
        """Ensure d_model is divisible by n_heads."""
        if self.transformer_d_model % self.transformer_n_heads != 0:
            # Auto-adjust to nearest valid value
            adjusted_d_model = ((self.transformer_d_model // self.transformer_n_heads) + 1) * self.transformer_n_heads
            warnings.warn(f"Adjusted transformer_d_model from {self.transformer_d_model} to {adjusted_d_model} to be divisible by n_heads ({self.transformer_n_heads})")
            self.transformer_d_model = adjusted_d_model
        return self

class IndicatorConfig(BaseModel):
    """Declarative configuration for a single technical indicator cell."""
    name: str = Field(description="Unique name for this indicator instance, e.g., '20s_roc'")
    cell_class_name: str = Field(description="The name of the class to instantiate from tins.py, e.g., 'ROCCell'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the cell's constructor, e1000., {'period': 12}")
    input_key: FeatureKeys = Field(description="The key from the observation space to use as input for this cell")
    expert_group: Literal['flow', 'volatility', 'value_trend', 'context', 'precomputed'] = Field(description="The expert group this indicator's output feeds into")
    input_type: Literal['price', 'ohlc', 'feature'] = Field(default='price', description="Type of input data expected by the cell")

class StatefulCalculatorConfig(BaseModel):
    """Declarative configuration for a single stateful feature calculator."""
    name: str = Field(description="Unique name for the calculator instance, e.g., 'sr_15m'")
    class_name: str = Field(description="The name of the class to instantiate from features.py, e.g., 'StatefulSRDistances'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the calculator's constructor")
    # --- START OF FIX: Example uses modern pandas frequency string ---
    timeframe: str = Field(description="The data timeframe this calculator operates on, e.g., '1h'")
    # --- END OF FIX ---
    # Explicitly declare the output feature keys. This is crucial for decoupling.
    output_keys: List[str] = Field(description="List of feature keys this calculator produces, order must be preserved.")
    source_col: str = Field(default='close', description="The column from the resampled dataframe to use as input")

class StrategyConfig(BaseModel):
    """REFINED: Strategy configuration reflecting the advanced, weighted reward component system."""
    
    # FIXED: Removed path-aware reward horizon system that broke PPO credit assignment
    
    # Configurable leverage with validation
    leverage: float = Field(
        default_factory=lambda: float(os.getenv('LEVERAGE', 10.0)),
        ge=1.0, le=25.0, description="Trading leverage (1.0 to 25.0x maximum)"
    )
    
    # Risk management parameters
    max_position_size: float = Field(
        default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', 1.0)),
        ge=0.1, le=2.0, description="Maximum position size"
    )
    
    max_drawdown_threshold: float = Field(
        default_factory=lambda: float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.2)),
        ge=0.05, le=0.5, description="Maximum drawdown before stopping"
    )
    
    maintenance_margin_rate: float = Field(
        default_factory=lambda: max(0.005, 0.1 / float(os.getenv('LEVERAGE', 10.0))),
        ge=0.001, le=0.1, description="Maintenance margin rate for liquidation (adjusted for leverage)"
    )
    
    max_margin_allocation_pct: float = Field(
        default=0.04, ge=0.0001, le=0.2,
        description="Maximum margin for a single position as a percentage of total equity"
    )
    
    # Feature engineering parameters
    volatility_window: int = Field(default=20, ge=10, le=100, description="Window for volatility calculation")
    trend_window: int = Field(default=50, ge=20, le=200, description="Window for trend calculation")
    support_resistance_window: int = Field(default=100, ge=50, le=500, description="Window for S/R detection")
    
    # FIXED: Precomputed features (restored from original design)
    precomputed_feature_keys: List[str] = Field(
        default=[
            'typical_price', 'price_range', 'price_change', 'volume_ma_5',
            'volume_ratio', 'log_volume', 'normalized_volume', 'volatility',
            'true_range', 'spread_proxy', 'trade_intensity', 'hour',
            'day_of_week', 'is_weekend'
        ],
        description="List of precomputed feature names from processor.py, order must be preserved."
    )
    
    ### FIX 3: UPDATED STATEFULCALCULATORCONFIG DEFAULTS ###
    stateful_calculators: List[StatefulCalculatorConfig] = Field(
        default_factory=lambda: [
            StatefulCalculatorConfig(
                name='bbw_1h_pct',
                class_name='StatefulBBWPercentRank',
                params={'period': 20, 'rank_window': 250},
                # --- START OF FIX: Use modern pandas frequency strings ---
                timeframe='1h',
                # --- END OF FIX ---
                output_keys=['bbw_1h_pct']
            ),
            StatefulCalculatorConfig(
                name='price_dist_ma_4h',
                class_name='StatefulPriceDistanceMA',
                params={'period': 50},
                # --- START OF FIX: Use modern pandas frequency strings ---
                timeframe='4h',
                # --- END OF FIX ---
                output_keys=['price_dist_ma_4h']
            ),
            # FIXED: Restored dist_vwap_3m to declarative pattern
            StatefulCalculatorConfig(
        name='vwap_dist_3m',
                class_name='StatefulVWAPDistance',
                params={'period': 9}, # 9 bars of 20s = 3 minutes
                timeframe='20s', # Calculate on base timeframe but use for 3m VWAP
                output_keys=['dist_vwap_3m'],
                source_col='close' # Will also need volume, handled in engine
            ),
            StatefulCalculatorConfig(
                name='sr_3m',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 1},
                timeframe='3min',
                output_keys=['dist_s1_3m', 'dist_r1_3m']
            ),
            StatefulCalculatorConfig(
                name='sr_15m',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 2},
                timeframe='15min',
                output_keys=['dist_s1_15m', 'dist_r1_15m', 'dist_s2_15m', 'dist_r2_15m']
            ),
            StatefulCalculatorConfig(
                name='sr_1h',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 2},
                # --- START OF FIX: Use modern pandas frequency strings ---
                timeframe='1h',
                # --- END OF FIX ---
                output_keys=['dist_s1_1h', 'dist_r1_1h', 'dist_s2_1h', 'dist_r2_1h']
            ),
        ]
    )
    
    @property
    def context_feature_keys(self) -> List[str]:
        """Dynamically generates the list of context feature keys from the stateful calculators."""
        return [key for calc in self.stateful_calculators for key in calc.output_keys]
    
    @property
    def lookback_periods(self) -> Dict[FeatureKeys, int]:
        """
        Dynamically generate lookback periods.
        The context and precomputed lookbacks are derived from their respective key lists.
        """
        return {
            FeatureKeys.PRICE_20S: 120,
            FeatureKeys.OHLCV_20S: 90,
            FeatureKeys.PRICE_1M: 80,
            FeatureKeys.OHLC_1M: 50,
            FeatureKeys.PRICE_3M: 80,
            FeatureKeys.OHLCV_3M: 50,
            FeatureKeys.PRICE_5M: 70,
            FeatureKeys.PRICE_15M: 50,
            FeatureKeys.OHLC_15M: 50,
            FeatureKeys.PRICE_1H: 70,
            FeatureKeys.CONTEXT: len(self.context_feature_keys), # Dynamically set
            FeatureKeys.VOLUME_DELTA_20S: 120,
            FeatureKeys.VOLUME_DELTA_1M: 80,
            # Portfolio state features
            FeatureKeys.PORTFOLIO_STATE: 4,
            # Precomputed features
            FeatureKeys.PRECOMPUTED_FEATURES: len(self.precomputed_feature_keys),
        }
    
    # Declarative Indicator Configuration
    indicators: List[IndicatorConfig] = Field(
        default_factory=lambda: [
            # Flow/Momentum Expert Group
            IndicatorConfig(name='20s_roc', cell_class_name='ROCCell', params={'period': 12}, input_key=FeatureKeys.PRICE_20S, expert_group='flow', input_type='price'),
            IndicatorConfig(name='1m_roc', cell_class_name='ROCCell', params={'period': 10}, input_key=FeatureKeys.PRICE_1M, expert_group='flow', input_type='price'),
            IndicatorConfig(name='20s_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_20S, expert_group='flow', input_type='feature'),
            IndicatorConfig(name='1m_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_1M, expert_group='flow', input_type='feature'),
            
            # Volatility Expert Group
            IndicatorConfig(name='1m_atr', cell_class_name='EnhancedATRCell', params={'period': 20}, input_key=FeatureKeys.OHLCV_20S, expert_group='volatility', input_type='ohlc'),
            IndicatorConfig(name='3m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLCV_3M, expert_group='volatility', input_type='ohlc'),
            IndicatorConfig(name='15m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLC_15M, expert_group='volatility', input_type='ohlc'),
            
            # Value/Trend Expert Group
            IndicatorConfig(name='5m_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend', input_type='price'),
            IndicatorConfig(name='5m_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend', input_type='price'),
            IndicatorConfig(name='1h_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend', input_type='price'),
            IndicatorConfig(name='1h_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend', input_type='price'),
        ]
    )
    
    # --- START OF FIX ---
    # CRITICAL: Reduced default sequence length to prevent multiprocessing pipe errors
    # and corrected the invalid Pydantic validation rule (le=100).
    sequence_length: int = Field(default=80, ge=5, le=256, description="Sequence length for temporal processing")
    # --- END OF FIX ---
    
    # Model architecture reference (Transformer-based)
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)
    
    # FIXED: Replaced deprecated class Config with model_config for Pydantic V2
    model_config = ConfigDict(from_attributes=True, frozen=True)

# --- START OF MODIFICATION: Replaced PPOTrainingConfig with SACTrainingConfig ---
class SACTrainingConfig(BaseModel):
    """Enhanced SAC training configuration with modern hyperparameters."""
    
    # Core parameters
    total_timesteps: int = Field(default=500000, ge=10000)
    buffer_size: int = Field(default=200_000, ge=10_000)
    learning_starts: int = Field(default=10_000, ge=1_000)
    batch_size: int = Field(default=256, ge=32)
    
    # SAC-specific parameters
    tau: float = Field(default=0.005, ge=0.001, le=0.1)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999, description="Discount factor")
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2, description="Learning rate")
    ent_coef: Union[str, float] = Field(default="auto")
    train_freq: Tuple[int, str] = Field(default=(1, "step"))
    gradient_steps: int = Field(default=1, ge=1)
    
    # Optimization and logging parameters
    optimization_trials: int = Field(
        default_factory=lambda: int(os.getenv('OPTIMIZATION_TRIALS', 20)),
        ge=0, description="Number of Optuna hyperparameter optimization trials"
    )
    checkpoint_frequency: int = Field(default=10000, ge=1000, description="Save model every N steps")
    validation_frequency: int = Field(default=5000, ge=1000, description="Validate every N steps")
# --- END OF MODIFICATION ---

class GlobalConfig(BaseModel):
    """FIXED: Enhanced global configuration with immediate reward system for proper RL training."""
    
    # Configuration loaded from .env
    environment: Environment = Field(
        default_factory=lambda: Environment(os.getenv('ENVIRONMENT', 'dev')),
        description="Deployment environment (dev, staging, prod)"
    )
    
    base_path: str = Field(
        default_factory=lambda: os.getenv('BASE_PATH', "./data"),
        description="Base data path - set via BASE_PATH environment variable"
    )
    
    primary_asset: str = Field(
        default_factory=lambda: os.getenv('PRIMARY_ASSET', 'BTCUSDT'),
        description="Primary trading asset"
    )
    
    asset_type: AssetType = Field(default=AssetType.CRYPTO, description="Type of asset")
    additional_assets: List[str] = Field(default_factory=list, description="Additional assets for multi-asset strategies")
    
    # Hardware configuration loaded from .env
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_gpu_memory_gb: Optional[float] = Field(
        default_factory=lambda: float(os.getenv('MAX_GPU_MEMORY_GB')) if os.getenv('MAX_GPU_MEMORY_GB') else None,
        ge=1.0, description="Maximum GPU memory to use"
    )
    
    num_workers: int = Field(
        default_factory=lambda: int(os.getenv('PARALLEL_WORKERS', 1)),
        ge=1, le=16, description="Number of parallel CPU workers for data processing and environment simulation."
    )
    
    # Time periods with validation
    in_sample_start: datetime = datetime(2025, 1, 1)
    in_sample_end: datetime = datetime(2025, 5, 31)
    out_of_sample_start: datetime = datetime(2025, 6, 1)
    out_of_sample_end: datetime = datetime(2025, 7, 31)
    
    # Trading simulation loaded from .env
    base_bar_timeframe: TimeFrameType = TimeFrameType.SECOND_20
    transaction_fee_pct: float = Field(
        default_factory=lambda: float(os.getenv('TRANSACTION_FEE', 0.000472)),
        ge=0.0, le=0.01, description="Transaction fee percentage"
    )
    
    slippage_pct: float = Field(
        default_factory=lambda: float(os.getenv('SLIPPAGE', 0.0001)),
        ge=0.0, le=0.004, description="Slippage percentage"
    )
    
    # Data schema
    binance_raw_columns: List[str] = Field(
        default_factory=lambda: ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
    )
    
    final_columns: List[str] = Field(
        default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset']
    )
    
    dtype_map: Dict[str, str] = Field(
        default_factory=lambda: {
            'id': 'int64', 'price': 'float64', 'qty': 'float64',
            'quote_qty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'
        }
    )
    
    # Logging and monitoring loaded from .env
    log_level:

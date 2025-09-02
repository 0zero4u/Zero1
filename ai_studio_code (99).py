"""
Enhanced Configuration System for Crypto Trading RL
Provides robust configuration management with validation and environment support.
Fixed import issues and enhanced validation.
"""

import os
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Any
from pydantic import BaseModel, validator, Field
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
        os.path.join(base_path, 'backtest_results')
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

class TimeFrameType(str, Enum):
    SECOND_20 = "20S"
    MINUTE_1 = "1T"
    MINUTE_3 = "3T"
    MINUTE_5 = "5T"
    MINUTE_15 = "15T"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"

class Environment(str, Enum):
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

class FeatureKeys(str, Enum):
    """Enum for feature keys to prevent typos and enable autocompletion."""
    PRICE_20S = 'price_20s'
    OHLCV_20S = 'ohlcv_20s'
    PRICE_1M = 'price_1m'
    OHLC_1M = 'ohlc_1m'
    PRICE_3M = 'price_3m'
    OHLCV_3M = 'ohlcv_3m'
    PRICE_5M = 'price_5m'
    PRICE_15M = 'price_15m'
    OHLC_15M = 'ohlc_15m'
    PRICE_1H = 'price_1h'
    CONTEXT = 'context'
    # NEW: Keys for Volume Delta data streams
    VOLUME_DELTA_20S = 'volume_delta_20s'
    VOLUME_DELTA_1M = 'volume_delta_1m'
    # NEW: Key for the agent's internal portfolio state
    PORTFOLIO_STATE = 'portfolio_state'
    # NEW: Key for pre-calculated features from the data processor
    PRECOMPUTED_FEATURES = 'precomputed_features'


# --- VALIDATION MODELS WITH PYDANTIC ---

class ModelArchitectureConfig(BaseModel):
    """Configuration for the Hierarchical Attention Model structure with validation."""

    lstm_layers: int = Field(default=2, ge=1, le=5, description="Number of LSTM layers")
    lstm_global_hidden_size: int = Field(default=64, ge=16, le=512, description="Global LSTM hidden size")
    expert_lstm_hidden_size: int = Field(default=32, ge=8, le=256, description="Expert LSTM hidden size")
    attention_head_features: int = Field(default=64, ge=16, le=256, description="Attention head output features")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.8, description="Dropout rate for regularization")
    use_batch_norm: bool = Field(default=True, description="Whether to use batch normalization")
    use_residual_connections: bool = Field(default=True, description="Whether to use residual connections")

    @validator('attention_head_features')
    def validate_attention_features(cls, v, values):
        if 'expert_lstm_hidden_size' in values and v < values['expert_lstm_hidden_size']:
            warnings.warn(f"Attention features ({v}) smaller than expert hidden size ({values['expert_lstm_hidden_size']})")
        return v

class IndicatorConfig(BaseModel):
    """Declarative configuration for a single technical indicator cell."""
    name: str = Field(description="Unique name for this indicator instance, e.g., '20s_roc'")
    cell_class_name: str = Field(description="The name of the class to instantiate from tins.py, e.g., 'ROCCell'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the cell's constructor, e.g., {'period': 12}")
    input_key: FeatureKeys = Field(description="The key from the observation space to use as input for this cell")
    expert_group: Literal['flow', 'volatility', 'value_trend', 'context', 'precomputed'] = Field(description="The expert group this indicator's output feeds into")
    input_type: Literal['price', 'ohlc', 'feature'] = Field(default='price', description="Type of input data expected by the cell")

class StatefulCalculatorConfig(BaseModel):
    """Declarative configuration for a single stateful feature calculator."""
    name: str = Field(description="Unique name for the calculator instance, e.g., 'sr_15m'")
    class_name: str = Field(description="The name of the class to instantiate from features.py, e.g., 'StatefulSRDistances'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the calculator's constructor")
    timeframe: str = Field(description="The data timeframe this calculator operates on, e.g., '1H'")
    # NEW: Explicitly declare the output feature keys. This is crucial for decoupling.
    output_keys: List[str] = Field(description="List of feature keys this calculator produces, order must be preserved.")
    source_col: str = Field(default='close', description="The column from the resampled dataframe to use as input")

class StrategyConfig(BaseModel):
    """Enhanced strategy configuration with validation and updated features."""

    # REFACTORED: Define precomputed features from processor.py as a single source of truth.
    # Note: 'dist_vwap_3m' is now considered a precomputed feature, as it's a direct calculation
    # from the close and vwap of a given bar, which are generated in processor.py.
    precomputed_feature_keys: List[str] = Field(
        default=[
            'dist_vwap_3m', # Moved here from context features
            'typical_price', 'price_range', 'price_change', 'volume_ma_5',
            'volume_ratio', 'log_volume', 'normalized_volume', 'volatility',
            'true_range', 'spread_proxy', 'trade_intensity', 'hour',
            'day_of_week', 'is_weekend'
        ],
        description="List of precomputed feature names from processor.py, order must be preserved."
    )
    
    # REFACTORED: context_feature_keys is now a dynamic property derived from stateful_calculators.
    # This makes stateful_calculators the single source of truth for context features.
    @property
    def context_feature_keys(self) -> List[str]:
        """Dynamically generates the list of context feature keys from the stateful calculators."""
        return [key for calc in self.stateful_calculators for key in calc.output_keys]

    # UPDATED: Use FeatureKeys enum for type safety and added new features
    @property
    def lookback_periods(self) -> Dict[FeatureKeys, int]:
        """
        Dynamically generate lookback periods.
        The context and precomputed lookbacks are derived from their respective key lists.
        The portfolio state lookback defines the number of state features.
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
            FeatureKeys.CONTEXT: len(self.context_feature_keys),  # Dynamically set
            FeatureKeys.VOLUME_DELTA_20S: 120,
            FeatureKeys.VOLUME_DELTA_1M: 80,
            # NEW: Add portfolio state. The 'lookback' value here represents the
            # number of features in the state vector (e.g., position size, margin, pnl).
            FeatureKeys.PORTFOLIO_STATE: 3,
            # NEW: Add precomputed features. The 'lookback' value here is the
            # number of features in the vector, similar to context.
            FeatureKeys.PRECOMPUTED_FEATURES: len(self.precomputed_feature_keys),
        }

    # NEW: Declarative Indicator Configuration
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
    
    # REFACTORED: Declarative configuration for stateful feature calculators, now with output_keys.
    # This is now the single source of truth for context features.
    stateful_calculators: List[StatefulCalculatorConfig] = Field(
        default_factory=lambda: [
            StatefulCalculatorConfig(
                name='bbw_1h_pct',
                class_name='StatefulBBWPercentRank',
                params={'period': 20, 'rank_window': 250},
                timeframe='1H',
                output_keys=['bbw_1h_pct']
            ),
            StatefulCalculatorConfig(
                name='price_dist_ma_4h',
                class_name='StatefulPriceDistanceMA',
                params={'period': 50},
                timeframe='4H',
                output_keys=['price_dist_ma_4h']
            ),
            StatefulCalculatorConfig(
                name='sr_3m',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 1},
                timeframe='3T',
                output_keys=['dist_s1_3m', 'dist_r1_3m']
            ),
            StatefulCalculatorConfig(
                name='sr_15m',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 2},
                timeframe='15T',
                output_keys=['dist_s1_15m', 'dist_r1_15m', 'dist_s2_15m', 'dist_r2_15m']
            ),
            StatefulCalculatorConfig(
                name='sr_1h',
                class_name='StatefulSRDistances',
                params={'period': 100, 'num_levels': 2},
                timeframe='1H',
                output_keys=['dist_s1_1h', 'dist_r1_1h', 'dist_s2_1h', 'dist_r2_1h']
            ),
        ]
    )

    sequence_length: int = Field(default=10, ge=5, le=100, description="LSTM sequence length")

    # Risk management parameters loaded from .env
    max_position_size: float = Field(
        default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', 1.0)),
        ge=0.1, le=2.0, description="Maximum position size"
    )

    max_drawdown_threshold: float = Field(
        default_factory=lambda: float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.2)),
        ge=0.05, le=0.5, description="Maximum drawdown before stopping"
    )

    leverage: float = Field(default=10.0, ge=1.0, le=100.0, description="Trading leverage for calculating margin")

    maintenance_margin_rate: float = Field(default=0.005, ge=0.001, le=0.1, description="Maintenance margin rate for liquidation")

    max_margin_allocation_pct: float = Field(
        default=0.04, ge=0.0001, le=0.2,
        description="Maximum margin for a single position as a percentage of total equity, capping per-trade risk."
    )

    # Feature engineering parameters
    volatility_window: int = Field(default=20, ge=10, le=100, description="Window for volatility calculation")
    trend_window: int = Field(default=50, ge=20, le=200, description="Window for trend calculation")
    support_resistance_window: int = Field(default=100, ge=50, le=500, description="Window for S/R detection")

    # Model architecture reference
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)

    class Config:
        orm_mode = True 
        allow_mutation = False
        
    @validator('lookback_periods', pre=True, always=True)
    def validate_lookback_periods(cls, v, values):
        if not v:
             v = cls().lookback_periods

        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"Lookback period for {key} must be positive, got {value}")
            if value > 1000:
                warnings.warn(f"Large lookback period for {key}: {value}. Consider reducing for memory efficiency.")
        return v

class PPOTrainingConfig(BaseModel):
    """Enhanced PPO training configuration with modern hyperparameters."""

    model_output_file: str = "ppo_hierarchical_attention_tin.zip"

    # Core PPO parameters loaded from .env
    total_timesteps: int = Field(
        default_factory=lambda: int(os.getenv('TOTAL_TIMESTEPS', 500_000)),
        ge=10_000, description="Total training timesteps"
    )

    optimization_trials: int = Field(
        default_factory=lambda: int(os.getenv('OPTIMIZATION_TRIALS', 20)),
        ge=0, description="Number of Optuna hyperparameter optimization trials"
    )

    n_steps: int = Field(default=2048, ge=64, description="Steps per rollout")
    batch_size: int = Field(default=64, ge=16, description="Batch size for training")
    n_epochs: int = Field(default=10, ge=1, le=20, description="Epochs per training iteration")

    # Learning parameters
    gamma: float = Field(default=0.99, ge=0.9, le=0.999, description="Discount factor")
    gae_lambda: float = Field(default=0.95, ge=0.8, le=0.999, description="GAE lambda")
    clip_range: float = Field(default=0.2, ge=0.1, le=0.5, description="PPO clip range")
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.1, description="Entropy coefficient")
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2, description="Learning rate")

    # Advanced parameters
    learning_rate_schedule: Literal["constant", "linear", "cosine"] = "linear"
    max_grad_norm: float = Field(default=0.5, ge=0.1, le=2.0, description="Gradient clipping norm")
    target_kl: Optional[float] = Field(default=None, ge=0.001, le=0.1, description="Target KL divergence")

    # Early stopping and checkpointing
    early_stopping_patience: int = Field(default=50, ge=10, description="Early stopping patience")
    checkpoint_frequency: int = Field(default=10000, ge=1000, description="Save model every N steps")
    validation_frequency: int = Field(default=5000, ge=1000, description="Validate every N steps")

    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        if 'n_steps' in values and v > values['n_steps']:
            raise ValueError(f"Batch size ({v}) cannot be larger than n_steps ({values['n_steps']})")
        return v

class GlobalConfig(BaseModel):
    """Enhanced global configuration with environment support."""

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
        default_factory=lambda: int(os.getenv('PARALLEL_WORKERS', 4)),
        ge=1, le=16, description="Number of data loading workers"
    )

    # Time periods with validation
    in_sample_start: datetime = datetime(2024, 1, 1)
    in_sample_end: datetime = datetime(2024, 5, 31)
    out_of_sample_start: datetime = datetime(2024, 6, 1)
    out_of_sample_end: datetime = datetime(2024, 7, 31)

    # Trading simulation loaded from .env
    base_bar_timeframe: TimeFrameType = TimeFrameType.SECOND_20

    transaction_fee_pct: float = Field(
        default_factory=lambda: float(os.getenv('TRANSACTION_FEE', 0.001)),
        ge=0.0, le=0.01, description="Transaction fee percentage"
    )

    slippage_pct: float = Field(
        default_factory=lambda: float(os.getenv('SLIPPAGE', 0.0005)),
        ge=0.0, le=0.005, description="Slippage percentage"
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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper()
    )

    enable_tensorboard: bool = True
    enable_wandb: bool = Field(default_factory=lambda: get_bool_env('ENABLE_WANDB', 'false'))
    wandb_project: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_PROJECT'))
    wandb_entity: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_ENTITY'))

    # Sub-configurations
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: PPOTrainingConfig = Field(default_factory=PPOTrainingConfig)

    @validator('out_of_sample_start')
    def validate_sample_periods(cls, v, values):
        if 'in_sample_end' in values and v <= values['in_sample_end']:
            raise ValueError("Out-of-sample start must be after in-sample end")
        return v

    @validator('additional_assets')
    def validate_additional_assets(cls, v, values):
        if 'primary_asset' in values and values['primary_asset'] in v:
            raise ValueError("Primary asset cannot be in additional assets list")
        return v

    def get_processed_trades_path(self, period_name: str) -> str:
        """Get path for processed trades."""
        return os.path.join(self.base_path, self.environment.value, period_name, "processed", "trades")

    def get_model_path(self) -> str:
        """Get path for saving/loading models."""
        model_name = f"{self.environment.value}_{self.training.model_output_file}"
        return os.path.join(self.base_path, "models", model_name)
    
    def get_normalizer_path(self) -> str:
        """Get path for saving/loading the normalizer."""
        return os.path.join(self.base_path, "models", f"{self.environment.value}_normalizer.json")

    def get_raw_trades_path(self, period_name: str) -> str:
        """Get path for raw trades."""
        return os.path.join(self.base_path, self.environment.value, period_name, "raw", "trades")

    def get_logs_path(self) -> str:
        """Get path for logs."""
        return os.path.join(self.base_path, "logs", self.environment.value)

    def get_tensorboard_path(self) -> str:
        """Get path for TensorBoard logs."""
        return os.path.join(self.base_path, "tensorboard_logs", self.environment.value)

    def get_timeframe_seconds(self, timeframe: TimeFrameType | str) -> int:
        """
        Converts a timeframe string (e.g., '1T', '4H') or enum to seconds.
        This centralized method eliminates code duplication.
        """
        timeframe_str = timeframe.value if isinstance(timeframe, Enum) else str(timeframe)
        timeframe_str = timeframe_str.upper()
        
        # A direct mapping for clarity and safety
        timeframe_to_seconds_map = {
            "20S": 20, "1T": 60, "3T": 180, "5T": 300, "15T": 900,
            "1H": 3600, "4H": 14400, "1D": 86400
        }
        
        # Handle common alternative notations like '1M' for '1T'
        timeframe_key = timeframe_str.replace('M', 'T')

        if timeframe_key in timeframe_to_seconds_map:
            return timeframe_to_seconds_map[timeframe_key]
        
        # Fallback for simple parsing if not in map
        try:
            if 'S' in timeframe_key:
                return int(timeframe_key.replace('S', ''))
            if 'T' in timeframe_key:
                return int(timeframe_key.replace('T', '')) * 60
            if 'H' in timeframe_key:
                return int(timeframe_key.replace('H', '')) * 3600
            if 'D' in timeframe_key:
                return int(timeframe_key.replace('D', '')) * 86400
        except (ValueError, TypeError):
            pass # Ignore parsing errors and proceed to warning

        logger.warning(f"Could not parse timeframe '{timeframe_str}', defaulting to 20 seconds.")
        return 20 # Default to base bar timeframe's duration

    def get_required_warmup_period(self) -> int:
        """Calculate required warm-up period based on number of base bars."""
        try:
            # Time required for technical indicators, in seconds
            vol_seconds = self.strategy.volatility_window * self.get_timeframe_seconds("1H")
            trend_seconds = self.strategy.trend_window * self.get_timeframe_seconds("4H")
            sr_seconds = self.strategy.support_resistance_window * self.get_timeframe_seconds("1H")
            technical_seconds = max(vol_seconds, trend_seconds, sr_seconds)

            # Time required for model lookbacks, in seconds
            lookback_seconds = 0
            for key, lookback in self.strategy.lookback_periods.items():
                if key in [FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES]:
                    continue
                
                # Extract timeframe string like '20s', '1m', '1h' from the key
                timeframe_str = key.value.split('_')[-1]
                seconds_per_bar = self.get_timeframe_seconds(timeframe_str)
                
                seconds_needed = lookback * seconds_per_bar
                if seconds_needed > lookback_seconds:
                    lookback_seconds = seconds_needed
            
            max_seconds_needed = max(technical_seconds, lookback_seconds)
            
            # Convert to number of base bars
            base_bar_seconds = self.get_timeframe_seconds(self.base_bar_timeframe)
            required_bars = math.ceil(max_seconds_needed / base_bar_seconds)
            
            return max(required_bars, self.strategy.sequence_length)

        except Exception as e:
            logger.warning(f"Error calculating warmup period: {e}")
            return 36000  # Safe default (from trend_window: 50 * 4 * 3600 / 20)

# Configuration factory with environment-specific overrides
def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    """Create configuration with environment-specific settings."""
    try:
        # If env is not provided, use the one from .env or the default
        base_env = env or Environment(os.getenv('ENVIRONMENT', 'dev'))

        base_config = {
            Environment.DEVELOPMENT: {
                "training": {"total_timesteps": 50_000, "checkpoint_frequency": 1000},
                "log_level": "DEBUG"
            },
            Environment.STAGING: {
                "training": {"total_timesteps": 200_000, "checkpoint_frequency": 5000},
                "log_level": "INFO"
            },
            Environment.PRODUCTION: {
                "training": {"total_timesteps": 1_000_000, "checkpoint_frequency": 10000},
                "log_level": "WARNING",
                "enable_wandb": True
            }
        }

        env_config = base_config.get(base_env, {})
        env_config.update(overrides)
        env_config["environment"] = base_env

        return GlobalConfig(**env_config)

    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        # Return basic config as fallback
        return GlobalConfig()

# Global instance - can be overridden for testing
SETTINGS = create_config()

# Configuration validation function
def validate_configuration(config: GlobalConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings_list = []

    try:
        # Hardware validation
        if config.device.type == "cuda" and not torch.cuda.is_available():
            warnings_list.append("CUDA requested but not available, falling back to CPU")

        if config.max_gpu_memory_gb and torch.cuda.is_available():
            try:
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if config.max_gpu_memory_gb > available_memory:
                    warnings_list.append(f"Requested GPU memory ({config.max_gpu_memory_gb}GB) exceeds available ({available_memory:.1f}GB)")
            except Exception as e:
                warnings_list.append(f"Could not check GPU memory: {e}")

        # Path validation
        if not os.path.exists(config.base_path):
            warnings_list.append(f"Base path does not exist: {config.base_path}")

        # Training parameter validation
        if config.training.batch_size > config.training.n_steps:
            warnings_list.append(f"Batch size ({config.training.batch_size}) larger than n_steps ({config.training.n_steps})")

        # Strategy validation
        if config.strategy.max_position_size > 1.5:
            warnings_list.append(f"High max position size ({config.strategy.max_position_size}) may increase risk")

        if config.strategy.leverage > 20:
            warnings_list.append(f"High leverage ({config.strategy.leverage}x) significantly increases liquidation risk")

    except Exception as e:
        warnings_list.append(f"Configuration validation error: {e}")

    return warnings_list

# Enhanced validation with automatic fixes
def validate_and_fix_configuration(config: GlobalConfig) -> Tuple[GlobalConfig, List[str]]:
    """Validate configuration and automatically fix common issues."""
    warnings_list = []

    try:
        # Fix batch size if too large
        if config.training.batch_size > config.training.n_steps:
            old_batch_size = config.training.batch_size
            config.training.batch_size = config.training.n_steps // 4
            warnings_list.append(f"Auto-fixed batch size: {old_batch_size} -> {config.training.batch_size}")

        # Ensure directories exist
        required_dirs = [
            config.get_logs_path(),
            config.get_tensorboard_path(),
            os.path.dirname(config.get_model_path())
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                warnings_list.append(f"Created missing directory: {dir_path}")

        # Validate and get remaining warnings
        remaining_warnings = validate_configuration(config)
        warnings_list.extend(remaining_warnings)

    except Exception as e:
        warnings_list.append(f"Configuration fix error: {e}")

    return config, warnings_list

if __name__ == "__main__":
    print("--- Loading Configuration from .env and defaults ---")

    try:
        # Setup environment first
        setup_environment()

        print(f"Environment: {SETTINGS.environment.value}")
        print(f"Primary Asset: {SETTINGS.primary_asset}")
        print(f"Base Path: {SETTINGS.base_path}")
        print(f"Log Level: {SETTINGS.log_level}")
        print(f"W&B Enabled: {SETTINGS.enable_wandb}")
        print(f"Total Timesteps: {SETTINGS.training.total_timesteps}")
        print(f"Max Drawdown Threshold: {SETTINGS.strategy.max_drawdown_threshold}")

        # Validate the loaded configuration
        config_warnings = validate_configuration(SETTINGS)

        if config_warnings:
            print("\n--- Configuration Warnings ---")
            for warning in config_warnings:
                print(f" - {warning}")

        print("\n✅ Configuration loading and validation complete.")

        # Display key feature dimensions
        print("\n--- Feature Configuration ---")
        print("Context Features (dynamically generated):")
        for key in SETTINGS.strategy.context_feature_keys:
            print(f" - {key}")

        print("\nLookback Periods:")
        for key, value in SETTINGS.strategy.lookback_periods.items():
            print(f"{key.value}: {value}")

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        print(f"❌ Configuration error: {e}")
# FIXED: config.py with Immediate Reward System

"""
FIXED: Configuration with Immediate Reward System for proper PPO training

KEY FIXES:
1. REMOVED path-aware reward horizon system that broke PPO credit assignment
2. ADDED reward scaling factor for better gradient signals
3. Simplified reward configuration for immediate rewards
4. Preserved all other functionality (features, architecture, etc.)
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
    
    # Keys for Volume Delta data streams
    VOLUME_DELTA_20S = 'volume_delta_20s'
    VOLUME_DELTA_1M = 'volume_delta_1m'
    
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
    
    @validator('transformer_d_model')
    def validate_transformer_d_model(cls, v, values):
        """Ensure d_model is divisible by n_heads."""
        if 'transformer_n_heads' in values:
            n_heads = values['transformer_n_heads']
            if v % n_heads != 0:
                # Auto-adjust to nearest valid value
                adjusted_d_model = ((v // n_heads) + 1) * n_heads
                warnings.warn(f"Adjusted transformer_d_model from {v} to {adjusted_d_model} to be divisible by n_heads ({n_heads})")
                return adjusted_d_model
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
    # Explicitly declare the output feature keys. This is crucial for decoupling.
    output_keys: List[str] = Field(description="List of feature keys this calculator produces, order must be preserved.")
    source_col: str = Field(default='close', description="The column from the resampled dataframe to use as input")

class StrategyConfig(BaseModel):
    """FIXED: Enhanced strategy configuration with immediate reward system for proper PPO training."""
    
    # FIXED: Removed path-aware reward horizon system that broke PPO credit assignment
    # ADDED: Immediate reward scaling for better gradient signals
    
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
    
    # FIXED: Added reward scaling factor for immediate rewards
    reward_scaling_factor: float = Field(
        default=200.0, ge=50.0, le=500.0,
        description="Scaling factor for immediate rewards to improve gradient signals"
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
    
    # FIXED: Restored declarative calculators with VWAP distance
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
            # FIXED: Restored dist_vwap_3m to declarative pattern
            StatefulCalculatorConfig(
                name='vwap_dist_3m',
                class_name='StatefulVWAPDistance',
                params={'period': 9}, # 9 bars of 20s = 3 minutes
                timeframe='20S', # Calculate on base timeframe but use for 3m VWAP
                output_keys=['dist_vwap_3m'],
                source_col='close' # Will also need volume, handled in engine
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
            FeatureKeys.PORTFOLIO_STATE: 5,
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
    
    sequence_length: int = Field(default=10, ge=5, le=100, description="Sequence length for temporal processing")
    
    # Model architecture reference (Transformer-based)
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)
    
    class Config:
        orm_mode = True
        allow_mutation = False

class PPOTrainingConfig(BaseModel):
    """Enhanced PPO training configuration with modern hyperparameters."""
    
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
    
    # Enhanced parameters for tunable learning dynamics
    learning_rate_schedule: Literal["constant", "linear", "cosine"] = "linear"
    max_grad_norm: float = Field(default=0.5, ge=0.1, le=2.0, description="Gradient clipping norm")
    target_kl: Optional[float] = Field(default=None, ge=0.001, le=0.1, description="Target KL divergence for policy stability")
    
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
    """FIXED: Enhanced global configuration with immediate reward system for proper PPO training."""
    
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
        ge=0.0, le=0.003, description="Slippage percentage"
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
    
    class Config:
        arbitrary_types_allowed = True
    
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
        """
        Get the full path for saving/loading the model with a timestamp for versioning.
        This method is the single source of truth for the model's filename and location.
        """
        # Create a unique identifier for this training run to prevent overwriting
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        # The model filename reflects the PPO algorithm and Transformer architecture
        model_filename = f"ppo_hierarchical_attention_transformer_{timestamp}.zip"
        # The final model name is prefixed with the environment (e.g., 'dev', 'prod').
        model_name = f"{self.environment.value}_{model_filename}"
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
        Converts a timeframe string to seconds.
        """
        timeframe_str = timeframe.value if isinstance(timeframe, Enum) else str(timeframe)
        timeframe_str = timeframe_str.upper()
        
        # Direct mapping for clarity and safety
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
            pass
        
        logger.warning(f"Could not parse timeframe '{timeframe_str}', defaulting to 20 seconds.")
        return 20
    
    def get_required_warmup_period(self) -> int:
        """Calculate required warm-up period with leverage awareness."""
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
            return 36000  # Safe default

# Configuration factory with environment-specific overrides

def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    """
    FIXED: Create configuration with environment-specific settings for immediate reward system.
    """
    try:
        # If env is not provided, use the one from .env or the default
        base_env = env or Environment(os.getenv('ENVIRONMENT', 'dev'))
        
        base_config = {
            Environment.DEVELOPMENT: {
                "training": {"total_timesteps": 50_000, "checkpoint_frequency": 1000},
                "log_level": "DEBUG",
                "strategy": {
                    "reward_scaling_factor": 150.0,  # FIXED: Appropriate scaling for dev
                }
            },
            Environment.STAGING: {
                "training": {"total_timesteps": 200_000, "checkpoint_frequency": 5000},
                "log_level": "INFO",
                "strategy": {
                    "reward_scaling_factor": 200.0,  # FIXED: Standard scaling for staging
                }
            },
            Environment.PRODUCTION: {
                "training": {"total_timesteps": 1_000_000, "checkpoint_frequency": 10000},
                "log_level": "WARNING",
                "enable_wandb": True,
                "strategy": {
                    "reward_scaling_factor": 250.0,  # FIXED: Higher scaling for production
                }
            }
        }
        
        env_config = base_config.get(base_env, {})
        env_config.update(overrides)
        env_config["environment"] = base_env
        
        config = GlobalConfig(**env_config)
        
        # FIXED: Log immediate reward system configuration
        logger.info(f"FIXED: Immediate reward system enabled with scaling factor: {config.strategy.reward_scaling_factor}")
        
        # Log Transformer architecture configuration
        arch = config.strategy.architecture
        logger.info(f"Transformer architecture: d_model={arch.transformer_d_model}, n_heads={arch.transformer_n_heads}, layers={arch.transformer_num_layers}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        # Return basic config as fallback
        return GlobalConfig()

# Global instance - can be overridden for testing
SETTINGS = create_config()

def validate_configuration(config: GlobalConfig) -> List[str]:
    """FIXED: Validate configuration with immediate reward system awareness."""
    warnings_list = []
    
    try:
        # Hardware validation
        if config.device.type == "cuda" and not torch.cuda.is_available():
            warnings_list.append("CUDA requested but not available, falling back to CPU")
        
        # Path validation
        if not os.path.exists(config.base_path):
            warnings_list.append(f"Base path does not exist: {config.base_path}")
        
        # FIXED: Validate reward scaling factor
        reward_scaling = config.strategy.reward_scaling_factor
        if reward_scaling < 50.0:
            warnings_list.append(f"Low reward scaling factor ({reward_scaling}) may cause weak gradients")
        elif reward_scaling > 400.0:
            warnings_list.append(f"High reward scaling factor ({reward_scaling}) may cause unstable training")
        
        # Transformer architecture validation
        arch = config.strategy.architecture
        if arch.transformer_n_heads > 8:
            warnings_list.append(f"Large number of attention heads ({arch.transformer_n_heads}) may impact training speed")
        
        if arch.transformer_num_layers > 4:
            warnings_list.append(f"Deep Transformer ({arch.transformer_num_layers} layers) may require more training time")
        
        # Validate dimensionality relationships
        if arch.transformer_dim_feedforward < arch.transformer_d_model:
            warnings_list.append(f"Feed-forward dim ({arch.transformer_dim_feedforward}) smaller than d_model ({arch.transformer_d_model})")
        
        if arch.attention_head_features < arch.expert_output_dim:
            warnings_list.append(f"Attention head features ({arch.attention_head_features}) smaller than expert output dim ({arch.expert_output_dim})")
        
    except Exception as e:
        warnings_list.append(f"Configuration validation error: {e}")
    
    return warnings_list

if __name__ == "__main__":
    print("--- Loading FIXED Configuration with Immediate Reward System ---")
    
    try:
        setup_environment()
        print(f"Environment: {SETTINGS.environment.value}")
        print(f"Primary Asset: {SETTINGS.primary_asset}")
        print(f"Base Path: {SETTINGS.base_path}")
        
        # Display immediate reward system configuration
        print(f"\n--- FIXED: Immediate Reward System Configuration ---")
        print(f"Reward Scaling Factor: {SETTINGS.strategy.reward_scaling_factor}")
        print(f"Leverage: {SETTINGS.strategy.leverage}x")
        print(f"Max Drawdown: {SETTINGS.strategy.max_drawdown_threshold:.1%}")
        print(f"Immediate Rewards: ON (PPO-compatible)")
        
        # Display declarative features
        print(f"\n--- FIXED: Declarative Feature Configuration ---")
        print("Context Features:")
        for key in SETTINGS.strategy.context_feature_keys:
            print(f" - {key}")
        
        # Validate configuration
        config_warnings = validate_configuration(SETTINGS)
        if config_warnings:
            print("\n--- Configuration Warnings ---")
            for warning in config_warnings:
                print(f" - {warning}")
        
        print("\n✅ FIXED configuration loading complete with immediate reward system for proper PPO training.")
        print("🔧 KEY FIXES APPLIED:")
        print(" - Removed path-aware reward horizon system that broke PPO credit assignment")
        print(" - Added reward scaling factor for better gradient signals")
        print(" - Preserved all other functionality (features, architecture, etc.)")
        print(" - PPO now gets proper (state, action) -> immediate reward mapping")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        print(f"❌ Configuration error: {e}")

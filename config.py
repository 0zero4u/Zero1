# FIXED: config.py with Centralized Reward Weights and Immediate Reward System

"""
FIXED: Configuration with Immediate Reward System for proper PPO training

KEY FIXES:
1. CENTRALIZED all reward weights into a `RewardWeightsConfig` Pydantic model.
   This establishes a single source of truth for all default parameters.
2. REMOVED path-aware reward horizon system that broke PPO credit assignment.
3. ADDED reward scaling factor for better gradient signals.
4. Simplified reward configuration for immediate rewards.
5. Preserved all other functionality (features, architecture, etc.)
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
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    base_path = os.getenv('BASE_PATH', './data')
    required_dirs = [os.path.join(base_path, d) for d in ['models', 'logs', 'cache', 'backtest_results']]
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
    PRICE_20S, OHLCV_20S, PRICE_1M, OHLC_1M, PRICE_3M, OHLCV_3M, PRICE_5M, PRICE_15M, OHLC_15M, PRICE_1H, CONTEXT, VOLUME_DELTA_20S, VOLUME_DELTA_1M, PORTFOLIO_STATE, PRECOMPUTED_FEATURES = [f.lower() for f in ["PRICE_20S", "OHLCV_20S", "PRICE_1M", "OHLC_1M", "PRICE_3M", "OHLCV_3M", "PRICE_5M", "PRICE_15M", "OHLC_15M", "PRICE_1H", "CONTEXT", "VOLUME_DELTA_20S", "VOLUME_DELTA_1M", "PORTFOLIO_STATE", "PRECOMPUTED_FEATURES"]]

# --- TRANSFORMER ARCHITECTURE CONFIG ---

class ModelArchitectureConfig(BaseModel):
    transformer_d_model: int = Field(64, ge=32, le=512)
    transformer_n_heads: int = Field(4, ge=1, le=16)
    transformer_dim_feedforward: int = Field(256, ge=64, le=2048)
    transformer_num_layers: int = Field(2, ge=1, le=8)
    expert_output_dim: int = Field(32, ge=8, le=256)
    attention_head_features: int = Field(64, ge=16, le=256)
    dropout_rate: float = Field(0.1, ge=0.0, le=0.8)
    use_batch_norm: bool = True
    use_residual_connections: bool = True

    @validator('transformer_d_model')
    def validate_transformer_d_model(cls, v, values):
        if 'transformer_n_heads' in values:
            n_heads = values['transformer_n_heads']
            if v % n_heads != 0:
                adjusted_d_model = ((v // n_heads) + 1) * n_heads
                warnings.warn(f"Adjusted transformer_d_model from {v} to {adjusted_d_model} to be divisible by n_heads ({n_heads})")
                return adjusted_d_model
        return v

class IndicatorConfig(BaseModel):
    name: str
    cell_class_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    input_key: FeatureKeys
    expert_group: Literal['flow', 'volatility', 'value_trend', 'context', 'precomputed']
    input_type: Literal['price', 'ohlc', 'feature'] = 'price'

class StatefulCalculatorConfig(BaseModel):
    name: str
    class_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    timeframe: str
    output_keys: List[str]
    source_col: str = 'close'

# --- NEW: CENTRALIZED REWARD WEIGHTS CONFIG ---
class RewardWeightsConfig(BaseModel):
    """Configuration for the components of the immediate reward function."""
    base_return: float = 1.4
    risk_adjusted: float = 0.15
    stability: float = 0.1
    transaction_penalty: float = -0.05
    drawdown_penalty: float = -0.2
    position_penalty: float = -0.01
    risk_bonus: float = 0.2
    exploration_bonus: float = 0.08
    inactivity_penalty: float = -0.001

class StrategyConfig(BaseModel):
    leverage: float = Field(default_factory=lambda: float(os.getenv('LEVERAGE', 10.0)), ge=1.0, le=25.0)
    max_position_size: float = Field(default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', 1.0)), ge=0.1, le=2.0)
    max_drawdown_threshold: float = Field(default_factory=lambda: float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.2)), ge=0.05, le=0.5)
    maintenance_margin_rate: float = Field(default_factory=lambda: max(0.005, 0.1 / float(os.getenv('LEVERAGE', 10.0))), ge=0.001, le=0.1)
    max_margin_allocation_pct: float = Field(0.04, ge=0.0001, le=0.2)
    reward_scaling_factor: float = Field(200.0, ge=50.0, le=500.0)
    
    # --- NEW: Use the centralized reward weights model ---
    reward_weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)

    precomputed_feature_keys: List[str] = Field(default=[
        'typical_price', 'price_range', 'price_change', 'volume_ma_5', 'volume_ratio',
        'log_volume', 'normalized_volume', 'volatility', 'true_range', 'spread_proxy',
        'trade_intensity', 'hour', 'day_of_week', 'is_weekend'
    ])
    
    stateful_calculators: List[StatefulCalculatorConfig] = Field(default_factory=lambda: [
        StatefulCalculatorConfig(name='bbw_1h_pct', class_name='StatefulBBWPercentRank', params={'period': 20, 'rank_window': 250}, timeframe='1H', output_keys=['bbw_1h_pct']),
        StatefulCalculatorConfig(name='price_dist_ma_4h', class_name='StatefulPriceDistanceMA', params={'period': 50}, timeframe='4H', output_keys=['price_dist_ma_4h']),
        StatefulCalculatorConfig(name='vwap_dist_3m', class_name='StatefulVWAPDistance', params={'period': 9}, timeframe='20S', output_keys=['dist_vwap_3m']),
        StatefulCalculatorConfig(name='sr_3m', class_name='StatefulSRDistances', params={'period': 100, 'num_levels': 1}, timeframe='3T', output_keys=['dist_s1_3m', 'dist_r1_3m']),
        StatefulCalculatorConfig(name='sr_15m', class_name='StatefulSRDistances', params={'period': 100, 'num_levels': 2}, timeframe='15T', output_keys=['dist_s1_15m', 'dist_r1_15m', 'dist_s2_15m', 'dist_r2_15m']),
        StatefulCalculatorConfig(name='sr_1h', class_name='StatefulSRDistances', params={'period': 100, 'num_levels': 2}, timeframe='1H', output_keys=['dist_s1_1h', 'dist_r1_1h', 'dist_s2_1h', 'dist_r2_1h']),
    ])

    @property
    def context_feature_keys(self) -> List[str]:
        return [key for calc in self.stateful_calculators for key in calc.output_keys]

    @property
    def lookback_periods(self) -> Dict[FeatureKeys, int]:
        return {
            FeatureKeys.PRICE_20S: 120, FeatureKeys.OHLCV_20S: 90, FeatureKeys.PRICE_1M: 80,
            FeatureKeys.OHLC_1M: 50, FeatureKeys.PRICE_3M: 80, FeatureKeys.OHLCV_3M: 50,
            FeatureKeys.PRICE_5M: 70, FeatureKeys.PRICE_15M: 50, FeatureKeys.OHLC_15M: 50,
            FeatureKeys.PRICE_1H: 70, FeatureKeys.CONTEXT: len(self.context_feature_keys),
            FeatureKeys.VOLUME_DELTA_20S: 120, FeatureKeys.VOLUME_DELTA_1M: 80,
            FeatureKeys.PORTFOLIO_STATE: 5, FeatureKeys.PRECOMPUTED_FEATURES: len(self.precomputed_feature_keys),
        }

    indicators: List[IndicatorConfig] = Field(default_factory=lambda: [
        IndicatorConfig(name='20s_roc', cell_class_name='ROCCell', params={'period': 12}, input_key=FeatureKeys.PRICE_20S, expert_group='flow'),
        IndicatorConfig(name='1m_roc', cell_class_name='ROCCell', params={'period': 10}, input_key=FeatureKeys.PRICE_1M, expert_group='flow'),
        IndicatorConfig(name='20s_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_20S, expert_group='flow', input_type='feature'),
        IndicatorConfig(name='1m_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_1M, expert_group='flow', input_type='feature'),
        IndicatorConfig(name='1m_atr', cell_class_name='EnhancedATRCell', params={'period': 20}, input_key=FeatureKeys.OHLCV_20S, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='3m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLCV_3M, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='15m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLC_15M, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='5m_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend'),
        IndicatorConfig(name='5m_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend'),
        IndicatorConfig(name='1h_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend'),
        IndicatorConfig(name='1h_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend'),
    ])
    
    sequence_length: int = Field(10, ge=5, le=100)
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)
    
    class Config:
        orm_mode = True
        allow_mutation = False

class PPOTrainingConfig(BaseModel):
    total_timesteps: int = Field(default_factory=lambda: int(os.getenv('TOTAL_TIMESTEPS', 500_000)), ge=10_000)
    optimization_trials: int = Field(default_factory=lambda: int(os.getenv('OPTIMIZATION_TRIALS', 20)), ge=0)
    n_steps: int = Field(2048, ge=64)
    batch_size: int = Field(64, ge=16)
    n_epochs: int = Field(10, ge=1, le=20)
    gamma: float = Field(0.99, ge=0.9, le=0.999)
    gae_lambda: float = Field(0.95, ge=0.8, le=0.999)
    clip_range: float = Field(0.2, ge=0.1, le=0.5)
    ent_coef: float = Field(0.01, ge=0.0, le=0.1)
    learning_rate: float = Field(3e-4, ge=1e-5, le=1e-2)
    max_grad_norm: float = Field(0.5, ge=0.1, le=2.0)
    
    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        if 'n_steps' in values and v > values['n_steps']:
            raise ValueError(f"Batch size ({v}) cannot be larger than n_steps ({values['n_steps']})")
        return v

class GlobalConfig(BaseModel):
    environment: Environment = Field(default_factory=lambda: Environment(os.getenv('ENVIRONMENT', 'dev')))
    base_path: str = Field(default_factory=lambda: os.getenv('BASE_PATH', "./data"))
    primary_asset: str = Field(default_factory=lambda: os.getenv('PRIMARY_ASSET', 'BTCUSDT'))
    asset_type: AssetType = AssetType.CRYPTO
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_workers: int = Field(default_factory=lambda: int(os.getenv('PARALLEL_WORKERS', 4)), ge=1, le=16)
    in_sample_start: datetime = datetime(2025, 1, 1)
    in_sample_end: datetime = datetime(2025, 5, 31)
    out_of_sample_start: datetime = datetime(2025, 6, 1)
    out_of_sample_end: datetime = datetime(2025, 7, 31)
    base_bar_timeframe: TimeFrameType = TimeFrameType.SECOND_20
    transaction_fee_pct: float = Field(default_factory=lambda: float(os.getenv('TRANSACTION_FEE', 0.000472)), ge=0.0, le=0.01)
    slippage_pct: float = Field(default_factory=lambda: float(os.getenv('SLIPPAGE', 0.0001)), ge=0.0, le=0.003)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())
    enable_wandb: bool = Field(default_factory=lambda: get_bool_env('ENABLE_WANDB', 'false'))
    wandb_project: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_PROJECT'))
    wandb_entity: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_ENTITY'))
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: PPOTrainingConfig = Field(default_factory=PPOTrainingConfig)

    class Config:
        arbitrary_types_allowed = True
    
    @validator('out_of_sample_start')
    def validate_sample_periods(cls, v, values):
        if 'in_sample_end' in values and v <= values['in_sample_end']:
            raise ValueError("Out-of-sample start must be after in-sample end")
        return v
    
    def get_model_path(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_filename = f"ppo_hierarchical_attention_transformer_{timestamp}.zip"
        model_name = f"{self.environment.value}_{model_filename}"
        return os.path.join(self.base_path, "models", model_name)

    def get_normalizer_path(self) -> str:
        return os.path.join(self.base_path, "models", f"{self.environment.value}_normalizer.json")
    
    def get_logs_path(self) -> str:
        return os.path.join(self.base_path, "logs", self.environment.value)

    def get_timeframe_seconds(self, timeframe: Union[TimeFrameType, str]) -> int:
        timeframe_str = timeframe.value if isinstance(timeframe, Enum) else str(timeframe).upper().replace('M', 'T')
        timeframe_map = {"20S": 20, "1T": 60, "3T": 180, "5T": 300, "15T": 900, "1H": 3600, "4H": 14400, "1D": 86400}
        return timeframe_map.get(timeframe_str, 20)
    
    def get_required_warmup_period(self) -> int:
        try:
            max_lookback_seconds = 0
            for key, lookback in self.strategy.lookback_periods.items():
                if key in [FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES]: continue
                timeframe_str = key.value.split('_')[-1]
                seconds_per_bar = self.get_timeframe_seconds(timeframe_str)
                max_lookback_seconds = max(max_lookback_seconds, lookback * seconds_per_bar)
            
            base_bar_seconds = self.get_timeframe_seconds(self.base_bar_timeframe)
            required_bars = math.ceil(max_lookback_seconds / base_bar_seconds)
            return max(required_bars, self.strategy.sequence_length)
        except Exception as e:
            logger.warning(f"Error calculating warmup period: {e}")
            return 36000

def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    base_env = env or Environment(os.getenv('ENVIRONMENT', 'dev'))
    config = GlobalConfig(environment=base_env, **overrides)
    logger.info(f"FIXED: Immediate reward system enabled with scaling factor: {config.strategy.reward_scaling_factor}")
    arch = config.strategy.architecture
    logger.info(f"Transformer architecture: d_model={arch.transformer_d_model}, n_heads={arch.transformer_n_heads}, layers={arch.transformer_num_layers}")
    return config

SETTINGS = create_config()

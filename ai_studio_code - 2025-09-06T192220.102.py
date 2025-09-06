# REFINED: config.py with restored declarative pattern

"""
REFINEMENT: Configuration with StatefulVWAPDistance restored to declarative pattern
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
    VOLUME_DELTA_20S = 'volume_delta_20s'
    VOLUME_DELTA_1M = 'volume_delta_1m'
    PORTFOLIO_STATE = 'portfolio_state'
    PRECOMPUTED_FEATURES = 'precomputed_features'

# --- TRANSFORMER ARCHITECTURE CONFIG ---
class ModelArchitectureConfig(BaseModel):
    transformer_d_model: int = Field(default=64, ge=32, le=512)
    transformer_n_heads: int = Field(default=4, ge=1, le=16)
    transformer_dim_feedforward: int = Field(default=256, ge=64, le=2048)
    transformer_num_layers: int = Field(default=2, ge=1, le=8)
    expert_output_dim: int = Field(default=32, ge=8, le=256)
    attention_head_features: int = Field(default=64, ge=16, le=256)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.8)
    use_batch_norm: bool = Field(default=True)
    use_residual_connections: bool = Field(default=True)

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
    input_type: Literal['price', 'ohlc', 'feature'] = Field(default='price')

class StatefulCalculatorConfig(BaseModel):
    name: str
    class_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    timeframe: str
    output_keys: List[str]
    source_col: str = Field(default='close')

class StrategyConfig(BaseModel):
    reward_horizon_steps: int = Field(default=1, ge=1, le=20)
    reward_horizon_decay: float = Field(default=0.95, ge=0.7, le=1.0)
    use_path_aware_rewards: bool = Field(default=True)
    leverage: float = Field(default_factory=lambda: float(os.getenv('LEVERAGE', 10.0)), ge=1.0, le=25.0)
    max_position_size: float = Field(default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', 1.0)), ge=0.1, le=2.0)
    max_drawdown_threshold: float = Field(default_factory=lambda: float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.2)), ge=0.05, le=0.5)
    maintenance_margin_rate: float = Field(default_factory=lambda: max(0.005, 0.1 / float(os.getenv('LEVERAGE', 10.0))), ge=0.001, le=0.1)
    max_margin_allocation_pct: float = Field(default=0.04, ge=0.0001, le=0.2)
    volatility_window: int = Field(default=20, ge=10, le=100)
    trend_window: int = Field(default=50, ge=20, le=200)
    support_resistance_window: int = Field(default=100, ge=50, le=500)
    precomputed_feature_keys: List[str] = Field(default=[
        'typical_price', 'price_range', 'price_change', 'volume_ma_5',
        'volume_ratio', 'log_volume', 'normalized_volume', 'volatility',
        'true_range', 'spread_proxy', 'trade_intensity', 'hour',
        'day_of_week', 'is_weekend'
    ])
    stateful_calculators: List[StatefulCalculatorConfig] = Field(default_factory=lambda: [
        StatefulCalculatorConfig(name='bbw_1h_pct', class_name='StatefulBBWPercentRank', params={'period': 20, 'rank_window': 250}, timeframe='1H', output_keys=['bbw_1h_pct']),
        StatefulCalculatorConfig(name='price_dist_ma_4h', class_name='StatefulPriceDistanceMA', params={'period': 50}, timeframe='4H', output_keys=['price_dist_ma_4h']),
        StatefulCalculatorConfig(name='vwap_dist_3m', class_name='StatefulVWAPDistance', params={'period': 9}, timeframe='20S', output_keys=['dist_vwap_3m'], source_col='close'),
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
            FeatureKeys.PORTFOLIO_STATE: 5,
            FeatureKeys.PRECOMPUTED_FEATURES: len(self.precomputed_feature_keys),
        }

    indicators: List[IndicatorConfig] = Field(default_factory=lambda: [
        IndicatorConfig(name='20s_roc', cell_class_name='ROCCell', params={'period': 12}, input_key=FeatureKeys.PRICE_20S, expert_group='flow', input_type='price'),
        IndicatorConfig(name='1m_roc', cell_class_name='ROCCell', params={'period': 10}, input_key=FeatureKeys.PRICE_1M, expert_group='flow', input_type='price'),
        IndicatorConfig(name='20s_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_20S, expert_group='flow', input_type='feature'),
        IndicatorConfig(name='1m_vol_delta', cell_class_name='PrecomputedFeatureCell', input_key=FeatureKeys.VOLUME_DELTA_1M, expert_group='flow', input_type='feature'),
        IndicatorConfig(name='1m_atr', cell_class_name='EnhancedATRCell', params={'period': 20}, input_key=FeatureKeys.OHLCV_20S, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='3m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLCV_3M, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='15m_atr', cell_class_name='EnhancedATRCell', params={'period': 14}, input_key=FeatureKeys.OHLC_15M, expert_group='volatility', input_type='ohlc'),
        IndicatorConfig(name='5m_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend', input_type='price'),
        IndicatorConfig(name='5m_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_5M, expert_group='value_trend', input_type='price'),
        IndicatorConfig(name='1h_macd_fast', cell_class_name='EnhancedMACDCell', params={'fast_period': 6, 'slow_period': 13, 'signal_period': 5}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend', input_type='price'),
        IndicatorConfig(name='1h_macd_slow', cell_class_name='EnhancedMACDCell', params={'fast_period': 24, 'slow_period': 52, 'signal_period': 18}, input_key=FeatureKeys.PRICE_1H, expert_group='value_trend', input_type='price'),
    ])
    
    sequence_length: int = Field(default=10, ge=5, le=100)
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)

    # FIX: Updated to Pydantic V2 syntax
    model_config = {
        "from_attributes": True,
        "frozen": True
    }

    @validator('reward_horizon_steps')
    def validate_reward_horizon(cls, v):
        if v > 1:
            logger.info(f"Using {v * 20 / 60:.1f}-minute reward horizon ({v} steps)")
        return v

    def get_reward_horizon_info(self) -> Dict[str, Any]:
        horizon_minutes = self.reward_horizon_steps * 20 / 60
        return {
            'steps': self.reward_horizon_steps,
            'minutes': horizon_minutes,
            'decay_factor': self.reward_horizon_decay,
            'use_path_aware': self.use_path_aware_rewards,
            'description': f"{horizon_minutes:.1f}-minute path-aware reward horizon"
        }

class PPOTrainingConfig(BaseModel):
    total_timesteps: int = Field(default_factory=lambda: int(os.getenv('TOTAL_TIMESTEPS', 500_000)), ge=10_000)
    optimization_trials: int = Field(default_factory=lambda: int(os.getenv('OPTIMIZATION_TRIALS', 20)), ge=0)
    n_steps: int = Field(default=2048, ge=64)
    batch_size: int = Field(default=64, ge=16)
    n_epochs: int = Field(default=10, ge=1, le=20)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    gae_lambda: float = Field(default=0.95, ge=0.8, le=0.999)
    clip_range: float = Field(default=0.2, ge=0.1, le=0.5)
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.1)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2)
    learning_rate_schedule: Literal["constant", "linear", "cosine"] = "linear"
    max_grad_norm: float = Field(default=0.5, ge=0.1, le=2.0)
    target_kl: Optional[float] = Field(default=None, ge=0.001, le=0.1)
    early_stopping_patience: int = Field(default=50, ge=10)
    checkpoint_frequency: int = Field(default=10000, ge=1000)
    validation_frequency: int = Field(default=5000, ge=1000)

    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        if 'n_steps' in values and v > values['n_steps']:
            raise ValueError(f"Batch size ({v}) cannot be larger than n_steps ({values['n_steps']})")
        return v

class GlobalConfig(BaseModel):
    environment: Environment = Field(default_factory=lambda: Environment(os.getenv('ENVIRONMENT', 'dev')))
    base_path: str = Field(default_factory=lambda: os.getenv('BASE_PATH', "./data"))
    primary_asset: str = Field(default_factory=lambda: os.getenv('PRIMARY_ASSET', 'BTCUSDT'))
    asset_type: AssetType = Field(default=AssetType.CRYPTO)
    additional_assets: List[str] = Field(default_factory=list)
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_gpu_memory_gb: Optional[float] = Field(default_factory=lambda: float(os.getenv('MAX_GPU_MEMORY_GB')) if os.getenv('MAX_GPU_MEMORY_GB') else None, ge=1.0)
    num_workers: int = Field(default_factory=lambda: int(os.getenv('PARALLEL_WORKERS', 4)), ge=1, le=16)
    in_sample_start: datetime = datetime(2025, 1, 1)
    in_sample_end: datetime = datetime(2025, 5, 31)
    out_of_sample_start: datetime = datetime(2025, 6, 1)
    out_of_sample_end: datetime = datetime(2025, 7, 31)
    base_bar_timeframe: TimeFrameType = TimeFrameType.SECOND_20
    transaction_fee_pct: float = Field(default_factory=lambda: float(os.getenv('TRANSACTION_FEE', 0.001)), ge=0.0, le=0.01)
    slippage_pct: float = Field(default_factory=lambda: float(os.getenv('SLIPPAGE', 0.0005)), ge=0.0, le=0.005)
    binance_raw_columns: List[str] = Field(default_factory=lambda: ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'])
    final_columns: List[str] = Field(default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset'])
    dtype_map: Dict[str, str] = Field(default_factory=lambda: {'id': 'int64', 'price': 'float64', 'qty': 'float64', 'quote_qty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'})
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())
    enable_tensorboard: bool = True
    enable_wandb: bool = Field(default_factory=lambda: get_bool_env('ENABLE_WANDB', 'false'))
    wandb_project: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_PROJECT'))
    wandb_entity: Optional[str] = Field(default_factory=lambda: os.getenv('WANDB_ENTITY'))
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: PPOTrainingConfig = Field(default_factory=PPOTrainingConfig)

    # FIX: Updated to Pydantic V2 syntax
    model_config = {
        "arbitrary_types_allowed": True
    }

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

    def get_tensorboard_path(self) -> str:
        return os.path.join(self.base_path, "tensorboard_logs", self.environment.value)

    def get_timeframe_seconds(self, timeframe: TimeFrameType | str) -> int:
        timeframe_str = timeframe.value if isinstance(timeframe, Enum) else str(timeframe).upper()
        timeframe_map = {"20S": 20, "1T": 60, "3T": 180, "5T": 300, "15T": 900, "1H": 3600, "4H": 14400, "1D": 86400}
        return timeframe_map.get(timeframe_str.replace('M', 'T'), 20)

    def get_required_warmup_period(self) -> int:
        try:
            tech_seconds = max(
                self.strategy.volatility_window * 3600,
                self.strategy.trend_window * 14400,
                self.strategy.support_resistance_window * 3600
            )
            lookback_seconds = 0
            for key, lookback in self.strategy.lookback_periods.items():
                if key in [FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES]: continue
                timeframe_str = key.value.split('_')[-1]
                lookback_seconds = max(lookback_seconds, lookback * self.get_timeframe_seconds(timeframe_str))
            
            required_bars = math.ceil(max(tech_seconds, lookback_seconds) / self.get_timeframe_seconds(self.base_bar_timeframe))
            return max(required_bars, self.strategy.sequence_length)
        except Exception:
            return 36000

    def get_reward_horizon_bars(self) -> int:
        return self.strategy.reward_horizon_steps

    def validate_reward_horizon_data(self, total_bars: int) -> bool:
        required = self.get_required_warmup_period() + self.get_reward_horizon_bars() + 100
        if total_bars < required:
            logger.warning(f"Dataset may be too small. Have {total_bars}, need at least {required}")
            return False
        return True

def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    base_env = env or Environment(os.getenv('ENVIRONMENT', 'dev'))
    base_config = {
        Environment.DEVELOPMENT: {"training": {"total_timesteps": 50_000}, "log_level": "DEBUG", "strategy": {"reward_horizon_steps": 1}},
        Environment.STAGING: {"training": {"total_timesteps": 200_000}, "log_level": "INFO", "strategy": {"reward_horizon_steps": 3}},
        Environment.PRODUCTION: {"training": {"total_timesteps": 1_000_000}, "log_level": "WARNING", "enable_wandb": True, "strategy": {"reward_horizon_steps": 9}},
    }
    env_config = base_config.get(base_env, {})
    env_config.update(overrides)
    env_config["environment"] = base_env
    config = GlobalConfig(**env_config)
    logger.info(f"Reward horizon configured: {config.strategy.get_reward_horizon_info()['description']}")
    arch = config.strategy.architecture
    logger.info(f"Transformer architecture: d_model={arch.transformer_d_model}, n_heads={arch.transformer_n_heads}, layers={arch.transformer_num_layers}")
    return config

SETTINGS = create_config()

if __name__ == "__main__":
    setup_environment()
    print(f"Environment: {SETTINGS.environment.value}")
    reward_info = SETTINGS.strategy.get_reward_horizon_info()
    print(f"Reward Horizon: {reward_info['description']}")
    print("Context Features:", SETTINGS.strategy.context_feature_keys)
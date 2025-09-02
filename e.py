

from __future__ import annotations
import os, math, logging, warnings
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Literal, Tuple

import torch
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────
#  Init
# ──────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────
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
    """Keys for observation dict – guarantees type-safety & autocompletion."""
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


# ──────────────────────────────────────────────────────────
#  Model / indicator configuration
# ──────────────────────────────────────────────────────────
class ModelArchitectureConfig(BaseModel):
    lstm_layers: int = Field(2, ge=1, le=5)
    lstm_global_hidden_size: int = Field(64, ge=16, le=512)
    expert_lstm_hidden_size: int = Field(32, ge=8, le=256)
    attention_head_features: int = Field(64, ge=16, le=256)
    dropout_rate: float = Field(0.1, ge=0.0, le=0.8)
    use_batch_norm: bool = True
    use_residual_connections: bool = True

    @validator("attention_head_features")
    def _attn_vs_expert(cls, v, values):
        if 'expert_lstm_hidden_size' in values and v < values['expert_lstm_hidden_size']:
            warnings.warn(
                f"Attention features {v} < expert hidden size {values['expert_lstm_hidden_size']}"
            )
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


# ──────────────────────────────────────────────────────────
#  Strategy parameters
# ──────────────────────────────────────────────────────────
class StrategyConfig(BaseModel):
    # Leverage & risk
    leverage: float = Field(float(os.getenv("LEVERAGE", 10)), ge=1.0, le=25.0)
    max_position_size: float = Field(float(os.getenv("MAX_POSITION_SIZE", 1.0)), ge=0.1, le=2.0)
    max_drawdown_threshold: float = Field(float(os.getenv("MAX_DRAWDOWN_THRESHOLD", 0.2)), ge=0.05, le=0.5)
    maintenance_margin_rate: float = Field(
        default_factory=lambda: max(0.005, 0.1 / float(os.getenv("LEVERAGE", 10))),
        ge=0.001, le=0.1
    )
    max_margin_allocation_pct: float = Field(0.04, ge=0.0001, le=0.2)

    # Feature engineering
    volatility_window: int = 20
    trend_window: int = 50
    support_resistance_window: int = 100

    # Model-sequence
    sequence_length: int = Field(10, ge=5, le=100)

    # Indicator + stateful calculators will be injected below
    indicators: List[IndicatorConfig] = Field(default_factory=list)
    stateful_calculators: List[StatefulCalculatorConfig] = Field(default_factory=list)

    # Pre-computed & context features
    precomputed_feature_keys: List[str] = Field(
        default_factory=lambda: [
            'dist_vwap_3m', 'typical_price', 'price_range', 'price_change',
            'volume_ma_5', 'volume_ratio', 'log_volume', 'normalized_volume',
            'volatility', 'true_range', 'spread_proxy', 'trade_intensity',
            'hour', 'day_of_week', 'is_weekend'
        ]
    )

    # Derived properties  ────────────────────────────────
    @property
    def context_feature_keys(self) -> List[str]:
        return [k for calc in self.stateful_calculators for k in calc.output_keys]

    @property
    def lookback_periods(self) -> Dict[FeatureKeys, int]:
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
            FeatureKeys.CONTEXT: len(self.context_feature_keys),
            FeatureKeys.VOLUME_DELTA_20S: 120,
            FeatureKeys.VOLUME_DELTA_1M: 80,
            FeatureKeys.PORTFOLIO_STATE: 5,
            FeatureKeys.PRECOMPUTED_FEATURES: len(self.precomputed_feature_keys),
        }

    # Validators  ────────────────────────────────────────
    @validator("maintenance_margin_rate")
    def _margin_vs_leverage(cls, v, values):
        lev = values.get("leverage", 10)
        min_rate = 0.02 / lev
        if v < min_rate:
            warnings.warn(f"Maintenance margin {v:.4f} too low for {lev}x leverage")
        return v


# ──────────────────────────────────────────────────────────
#  Training hyper-params
# ──────────────────────────────────────────────────────────
class PPOTrainingConfig(BaseModel):
    model_output_file: str = "ppo_hierarchical_attention_tin.zip"
    total_timesteps: int = Field(int(os.getenv("TOTAL_TIMESTEPS", 500_000)), ge=10_000)
    optimization_trials: int = Field(int(os.getenv("OPTIMIZATION_TRIALS", 20)), ge=0)
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    learning_rate_schedule: Literal["constant", "linear", "cosine"] = "linear"
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    early_stopping_patience: int = 50
    checkpoint_frequency: int = 10_000
    validation_frequency: int = 5_000

    @validator("batch_size")
    def _batch_vs_steps(cls, v, values):
        if v > values.get("n_steps", 2048):
            raise ValueError("batch_size cannot exceed n_steps")
        return v


# ──────────────────────────────────────────────────────────
#  Global configuration root
# ──────────────────────────────────────────────────────────
class GlobalConfig(BaseModel):
    environment: Environment = Field(default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "dev")))
    base_path: str = Field(default_factory=lambda: os.getenv("BASE_PATH", "./data"))
    primary_asset: str = Field(default_factory=lambda: os.getenv("PRIMARY_ASSET", "BTCUSDT"))
    asset_type: AssetType = AssetType.CRYPTO
    additional_assets: List[str] = Field(default_factory=list)

    # Hardware
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_gpu_memory_gb: Optional[float] = Field(
        default_factory=lambda: float(os.getenv("MAX_GPU_MEMORY_GB", "0")) or None
    )
    num_workers: int = Field(int(os.getenv("PARALLEL_WORKERS", 4)), ge=1, le=16)

    # Sample windows
    in_sample_start: datetime = datetime(2024, 1, 1)
    in_sample_end:   datetime = datetime(2024, 5, 31)
    out_of_sample_start: datetime = datetime(2024, 6, 1)
    out_of_sample_end:   datetime = datetime(2024, 7, 31)

    # Trading costs
    base_bar_timeframe: TimeFrameType = TimeFrameType.SECOND_20
    transaction_fee_pct: float = Field(float(os.getenv("TRANSACTION_FEE", 0.001)), ge=0.0, le=0.01)
    slippage_pct: float = Field(float(os.getenv("SLIPPAGE", 0.0005)), ge=0.0, le=0.005)

    # Data schema
    binance_raw_columns: List[str] = Field(
        default_factory=lambda: ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
    )
    final_columns: List[str] = Field(
        default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset']
    )
    dtype_map: Dict[str, str] = Field(
        default_factory=lambda: {'id': 'int64', 'price': 'float64', 'qty': 'float64',
                                 'quote_qty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'}
    )

    # Logging / tracking
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(os.getenv("LOG_LEVEL", "INFO").upper())
    enable_tensorboard: bool = True
    enable_wandb: bool = Field(os.getenv("ENABLE_WANDB", "false").lower() in ('true', '1', 't', 'yes'))
    wandb_project: Optional[str] = os.getenv("WANDB_PROJECT")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY")

    # Sub-configs
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    training: PPOTrainingConfig = Field(default_factory=PPOTrainingConfig)

    # ───────────── helper paths ─────────────
    def _sub(self, *parts): return os.path.join(self.base_path, *parts)

    def get_processed_trades_path(self, period): return self._sub(self.environment.value, period, "processed", "trades")
    def get_raw_trades_path(self, period):       return self._sub(self.environment.value, period, "raw", "trades")
    def get_model_path(self):                    return self._sub("models", f"{self.environment.value}_{self.training.model_output_file}")
    def get_normalizer_path(self):               return self._sub("models", f"{self.environment.value}_normalizer.json")
    def get_logs_path(self):                     return self._sub("logs", self.environment.value)
    def get_tensorboard_path(self):              return self._sub("tensorboard_logs", self.environment.value)

    # ───────────── timeframe helpers ─────────────
    @staticmethod
    def get_timeframe_seconds(tf: TimeFrameType | str) -> int:
        m = {
            "20S": 20, "1T": 60, "3T": 180, "5T": 300, "15T": 900,
            "1H": 3600, "4H": 14_400, "1D": 86_400
        }
        tf_str = (tf.value if isinstance(tf, Enum) else str(tf)).upper().replace('M', 'T')
        if tf_str in m:
            return m[tf_str]
        # simple fallback
        for suf, mult in (("S", 1), ("T", 60), ("H", 3600), ("D", 86_400)):
            if tf_str.endswith(suf):
                return int(tf_str.rstrip(suf)) * mult
        logger.warning(f"Unrecognized timeframe {tf_str}, default 20s")
        return 20

    # ───────────── warm-up calc ─────────────
    def get_required_warmup_period(self) -> int:
        try:
            tech_secs = max(
                self.strategy.volatility_window * self.get_timeframe_seconds("1H"),
                self.strategy.trend_window * self.get_timeframe_seconds("4H"),
                self.strategy.support_resistance_window * self.get_timeframe_seconds("1H"),
            )
            lb_secs = 0
            for k, look in self.strategy.lookback_periods.items():
                if k in (FeatureKeys.CONTEXT, FeatureKeys.PORTFOLIO_STATE, FeatureKeys.PRECOMPUTED_FEATURES):
                    continue
                tf = k.value.split('_')[-1]
                lb_secs = max(lb_secs, look * self.get_timeframe_seconds(tf))
            need = max(tech_secs, lb_secs)
            base_secs = self.get_timeframe_seconds(self.base_bar_timeframe)
            return max(math.ceil(need / base_secs), self.strategy.sequence_length)
        except Exception as e:
            logger.warning(f"Warm-up calc failed: {e}")
            return 36_000  # safe default


# ──────────────────────────────────────────────────────────
#  Factory & Validation helpers
# ──────────────────────────────────────────────────────────
def get_bool_env(name, default='false'): return os.getenv(name, default).lower() in ('true', '1', 't', 'yes')


def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    base_env = env or Environment(os.getenv("ENVIRONMENT", "dev"))
    preset: Dict[str, Any] = {
        Environment.DEVELOPMENT: {"training": {"total_timesteps": 50_000}, "log_level": "DEBUG"},
        Environment.STAGING:     {"training": {"total_timesteps": 200_000}},
        Environment.PRODUCTION:  {"training": {"total_timesteps": 1_000_000}, "enable_wandb": True},
    }.get(base_env, {})
    preset.update(overrides)
    preset["environment"] = base_env
    return GlobalConfig(**preset)


def setup_environment():
    """Create needed folders & configure root logger once at startup."""
    log_lvl = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    logging.basicConfig(level=log_lvl,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    base = SETTINGS.base_path
    for p in ("models", "logs", "cache", "backtest_results", "tensorboard_logs"):
        os.makedirs(os.path.join(base, p, SETTINGS.environment.value), exist_ok=True)
    logger.info("Environment folders ready")


def validate_configuration(cfg: GlobalConfig) -> List[str]:
    warns: List[str] = []
    try:
        if cfg.device.type == "cuda" and not torch.cuda.is_available():
            warns.append("CUDA requested but not available – using CPU")
        if cfg.strategy.leverage > 20:
            warns.append(f"High leverage {cfg.strategy.leverage}x increases liquidation risk")
        if cfg.training.batch_size > cfg.training.n_steps:
            warns.append("batch_size larger than n_steps")
    except Exception as e:
        warns.append(f"Validation exception: {e}")
    return warns


# ──────────────────────────────────────────────────────────
#  Global instance
# ──────────────────────────────────────────────────────────
SETTINGS = create_config()

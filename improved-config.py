import os
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, validator, Field
from enum import Enum
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- ENUMS FOR TYPE SAFETY ---

class AssetType(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"

class TimeFrameType(str, Enum):
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

# NEW: Enum for feature keys to prevent typos and enable autocompletion
class FeatureKeys(str, Enum):
    PRICE_1M = 'price_1m'
    OHLC_1M = 'ohlc_1m'
    PRICE_3M = 'price_3m'
    OHLCV_3M = 'ohlcv_3m'
    PRICE_5M = 'price_5m'
    PRICE_15M = 'price_15m'
    OHLC_15M = 'ohlc_15m'
    PRICE_1H = 'price_1h'
    CONTEXT = 'context'

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

class StrategyConfig(BaseModel):
    """Enhanced strategy configuration with validation and new features."""
    # IMPROVED: Use FeatureKeys enum for type safety
    lookback_periods: Dict[FeatureKeys, int] = Field(
        default_factory=lambda: {
            FeatureKeys.PRICE_1M: 80,
            FeatureKeys.OHLC_1M: 50,
            FeatureKeys.PRICE_3M: 80,
            FeatureKeys.OHLCV_3M: 50,
            FeatureKeys.PRICE_5M: 70,
            FeatureKeys.PRICE_15M: 50,
            FeatureKeys.OHLC_15M: 50,
            FeatureKeys.PRICE_1H: 70,
            FeatureKeys.CONTEXT: 5
        }
    )

    sequence_length: int = Field(default=10, ge=5, le=100, description="LSTM sequence length")

    # New risk management parameters
    max_position_size: float = Field(default=1.0, ge=0.1, le=2.0, description="Maximum position size")
    max_drawdown_threshold: float = Field(default=0.2, ge=0.05, le=0.5, description="Maximum drawdown before stopping")
    leverage_limit: float = Field(default=1.0, ge=1.0, le=5.0, description="Maximum leverage allowed")

    # Feature engineering parameters
    volatility_window: int = Field(default=20, ge=10, le=100, description="Window for volatility calculation")
    trend_window: int = Field(default=50, ge=20, le=200, description="Window for trend calculation")
    support_resistance_window: int = Field(default=100, ge=50, le=500, description="Window for S/R detection")

    # Model architecture reference
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)

    @validator('lookback_periods')
    def validate_lookback_periods(cls, v):
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"Lookback period for {key} must be positive, got {value}")
            if value > 1000:
                warnings.warn(f"Large lookback period for {key}: {value}. Consider reducing for memory efficiency.")
        return v

class PPOTrainingConfig(BaseModel):
    """Enhanced PPO training configuration with modern hyperparameters."""
    model_output_file: str = "ppo_hierarchical_attention_tin.zip"

    # Core PPO parameters
    total_timesteps: int = Field(default=500_000, ge=10_000, description="Total training timesteps")
    n_steps: int = Field(default=2048, ge=64, description="Steps per rollout")
    batch_size: int = Field(default=64, ge=16, description="Batch size for training")
    n_epochs: int = Field(default=10, ge=1, le=20, description="Epochs per training iteration")

    # Learning parameters
    gamma: float = Field(default=0.99, ge=0.9, le=0.999, description="Discount factor")
    gae_lambda: float = Field(default=0.95, ge=0.8, le=0.999, description="GAE lambda")
    clip_range: float = Field(default=0.2, ge=0.1, le=0.5, description="PPO clip range")
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.1, description="Entropy coefficient")
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2, description="Learning rate")

    # New advanced parameters
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
    # IMPROVED: Use environment variables for paths
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    base_path: str = Field(
        default_factory=lambda: os.getenv('BASE_PATH', "./data"),
        description="Base data path - set via BASE_PATH environment variable"
    )

    # Asset configuration
    primary_asset: str = Field(
        default_factory=lambda: os.getenv('PRIMARY_ASSET', 'BTCUSDT'),
        description="Primary trading asset"
    )
    asset_type: AssetType = Field(default=AssetType.CRYPTO, description="Type of asset")
    additional_assets: List[str] = Field(default_factory=list, description="Additional assets for multi-asset strategies")

    # Hardware configuration
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_gpu_memory_gb: Optional[float] = Field(default=None, ge=1.0, description="Maximum GPU memory to use")
    num_workers: int = Field(default=4, ge=1, le=16, description="Number of data loading workers")

    # Time periods with validation
    in_sample_start: datetime = datetime(2024, 1, 1)
    in_sample_end: datetime = datetime(2024, 5, 31)
    out_of_sample_start: datetime = datetime(2024, 6, 1)
    out_of_sample_end: datetime = datetime(2024, 7, 31)

    # Trading simulation
    base_bar_timeframe: TimeFrameType = TimeFrameType.MINUTE_1
    transaction_fee_pct: float = Field(default=0.001, ge=0.0, le=0.01, description="Transaction fee percentage")
    slippage_pct: float = Field(default=0.0005, ge=0.0, le=0.005, description="Slippage percentage")

    # Data schema
    binance_raw_columns: List[str] = Field(
        default_factory=lambda: ['id', 'price', 'qty', 'quoteQty', 'time', 'is_buyer_maker']
    )

    final_columns: List[str] = Field(
        default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset']
    )

    dtype_map: Dict[str, str] = Field(
        default_factory=lambda: {
            'id': 'int64', 'price': 'float64', 'qty': 'float64',
            'quoteQty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'
        }
    )

    # Logging and monitoring
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    experiment_name: Optional[str] = None

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
        return os.path.join(self.base_path, self.environment.value, period_name, "processed", "trades")

    def get_model_path(self) -> str:
        model_name = f"{self.environment.value}_{self.training.model_output_file}"
        return os.path.join(self.base_path, "models", model_name)

    def get_raw_trades_path(self, period_name: str) -> str:
        return os.path.join(self.base_path, self.environment.value, period_name, "raw", "trades")

    def get_funding_rate_path(self) -> str:
        return os.path.join(self.base_path, self.environment.value, "funding_rate")

    def get_logs_path(self) -> str:
        return os.path.join(self.base_path, "logs", self.environment.value)

    def get_tensorboard_path(self) -> str:
        return os.path.join(self.base_path, "tensorboard_logs", self.environment.value)

    # NEW: Method to calculate dynamic warm-up period
    def get_required_warmup_period(self) -> int:
        """Calculate required warm-up period based on configuration."""
        max_lookback = max(self.strategy.lookback_periods.values())
        
        # Add extra buffer for technical indicators
        # BBW percentile needs 250 bars on 1H data = 250 * 60 = 15,000 minutes
        # Plus volatility and trend windows
        technical_buffer = max(
            self.strategy.volatility_window * 60,  # Convert to minutes if hourly
            self.strategy.trend_window * 60,
            self.strategy.support_resistance_window
        )
        
        return max(max_lookback * 60, technical_buffer, self.strategy.sequence_length)

# Configuration factory with environment-specific overrides
def create_config(env: Environment = Environment.DEVELOPMENT, **overrides) -> GlobalConfig:
    """Create configuration with environment-specific settings."""
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

    env_config = base_config.get(env, {})
    env_config.update(overrides)
    env_config["environment"] = env

    return GlobalConfig(**env_config)

# Global instance - can be overridden for testing
SETTINGS = create_config()

# Configuration validation function
def validate_configuration(config: GlobalConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []

    # Check device availability
    if config.device.type == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA requested but not available, falling back to CPU")

    # Check memory requirements
    if config.max_gpu_memory_gb and torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if config.max_gpu_memory_gb > available_memory:
            warnings.append(f"Requested GPU memory ({config.max_gpu_memory_gb}GB) exceeds available ({available_memory:.1f}GB)")

    # Check path accessibility
    if not os.path.exists(config.base_path):
        warnings.append(f"Base path does not exist: {config.base_path}")

    return warnings

# Example usage and testing
if __name__ == "__main__":
    # Create different environment configs
    dev_config = create_config(Environment.DEVELOPMENT)
    prod_config = create_config(Environment.PRODUCTION, primary_asset="ETHUSDT")

    # Validate configurations
    for config in [dev_config, prod_config]:
        warnings = validate_configuration(config)
        if warnings:
            print(f"Configuration warnings for {config.environment}:")
            for warning in warnings:
                print(f" - {warning}")

    print("✅ Configuration validation complete")
    print(f"Dynamic warm-up period: {dev_config.get_required_warmup_period()} minutes")
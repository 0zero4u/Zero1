import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import torch

# --- TRADING STRATEGY CONFIGURATION (Multi-Timeframe Hybrid TIN) ---

### --- REFINEMENT --- ###
# New configuration class for the model's architecture, inspired by the hierarchical concept.
@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Configuration for the Hierarchical Attention Model structure."""
    LSTM_LAYERS: int = 2
    # Hidden size for the final LSTM head in the original model
    LSTM_GLOBAL_HIDDEN_SIZE: int = 64
    # Hidden size for the specialized LSTMs in the new hierarchical model
    EXPERT_LSTM_HIDDEN_SIZE: int = 32
    # The final feature dimension produced by the attention mechanism
    ATTENTION_HEAD_FEATURES: int = 64

@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the Multi-Timeframe Hybrid TIN."""
    # Define the lookback window (in bars) for each required input series.
    LOOKBACK_PERIODS: Dict[str, int] = field(default_factory=lambda: {
        # --- Ultra-Short Term Tactical Layer ---
        'price_1m': 80,
        'ohlc_1m': 50,
        'price_3m': 80,
        ### --- REFINEMENT --- ###
        # Explicitly require OHLCV for the VWAP calculation.
        'ohlcv_3m': 50,

        # --- Original Tactical & Short-Term Layers ---
        'price_5m': 70,
        'price_15m': 50,
        'ohlc_15m': 50,
        'price_1h': 70,

        ### --- REFINEMENT --- ###
        # Context vector now includes funding rate.
        'context': 5,      # Context: volatility, trend, dist_support, dist_resistance, funding_rate
    })
    # Defines how many past time steps the LSTM will look at for each decision.
    SEQUENCE_LENGTH: int = 10

    # Sub-configuration for model architecture
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)


# --- MODEL TRAINING CONFIGURATION (Stable-Baselines3 PPO) ---

@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration for the Proximal Policy Optimization (PPO) training process."""
    MODEL_OUTPUT_FILE: str = "ppo_hierarchical_attention_tin.zip" # New model name

    # --- PPO Hyperparameters ---
    TOTAL_TIMESTEPS: int = 200_000
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01
    LEARNING_RATE: float = 3e-4

# --- GLOBAL SYSTEM & DATA CONFIGURATION ---

@dataclass(frozen=True)
class GlobalConfig:
    # --- Paths & Asset ---
    BASE_PATH: str = os.getenv('BASE_PATH', "/content/drive/MyDrive/crypto_data/alpha_proof")
    ASSET: str = "BTCUSDT"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Time Periods ---
    IN_SAMPLE_START: datetime = datetime(2025, 1, 1); IN_SAMPLE_END: datetime = datetime(2025, 5, 31)
    OUT_OF_SAMPLE_START: datetime = datetime(2025, 6, 1); OUT_OF_SAMPLE_END: datetime = datetime(2025, 7, 31)

    # --- Base bar data for environment ---
    BASE_BAR_TIMEFRAME: str = "1T"

    # --- Trading Simulation ---
    TRANSACTION_FEE_PCT: float = 0.001

    # --- Data Schema Attributes ---
    BINANCE_RAW_COLUMNS: List[str] = field(default_factory=lambda: ['id', 'price', 'qty', 'quoteQty', 'time', 'is_buyer_maker'])
    FINAL_COLUMNS: List[str] = field(default_factory=lambda: ['trade_id', 'timestamp', 'price', 'size', 'side', 'asset'])
    DTYPE_MAP: Dict[str, str] = field(default_factory=lambda: {'id': 'int64', 'price': 'float64', 'qty': 'float64', 'quoteQty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'})

    # --- Sub-configurations ---
    strategy: StrategyConfig = field(default_factory=StrategyConfig); training: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)

    # --- Directory Methods ---
    def get_processed_trades_path(self, period_name: str) -> str: return os.path.join(self.BASE_PATH, period_name, "processed", "trades")
    def get_model_path(self) -> str: return os.path.join(self.BASE_PATH, self.training.MODEL_OUTPUT_FILE)
    def get_raw_trades_path(self, period_name: str) -> str: return os.path.join(self.BASE_PATH, period_name, "raw", "trades")
    def get_funding_rate_path(self) -> str: return os.path.join(self.BASE_PATH, "funding_rate")

# --- Singleton Instance ---
SETTINGS = GlobalConfig()

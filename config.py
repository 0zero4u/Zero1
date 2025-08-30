
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import torch

# --- TRADING STRATEGY CONFIGURATION (Multi-Timeframe Hybrid TIN) ---

@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the Multi-Timeframe Hybrid TIN."""
    # Define the lookback window (in bars) for each required input series.
    LOOKBACK_PERIODS: Dict[str, int] = field(default_factory=lambda: {
        # --- NEW: Ultra-Short Term Tactical Layer ---
        'price_1m': 80,    # For momentum/flow analysis (ROC, MACD)
        'ohlc_1m': 50,     # For volatility analysis (ATR)
        'price_3m': 80,    # For VWAP signal generation
        'ohlc_3m': 50,     # For trend/value analysis (VWAP, ATR)

        # --- Original Tactical & Short-Term Layers ---
        'price_5m': 70,    # For tactical cells (fast/slow MACD, fast/slow ROC)
        'price_15m': 50,   # For short-term cells (RSI, BBands %B)
        'ohlc_15m': 50,    # For volatility cells (ATR)
        'price_1h': 70,    # For strategic cells (fast/slow MACD)
        
        # --- Context Layer (Unchanged) ---
        'context': 4,      # Context vector: volatility, trend, dist_to_support, dist_to_resistance
    })
    # Defines how many past time steps the LSTM will look at for each decision.
    SEQUENCE_LENGTH: int = 10
    # ACTION_SPACE_SIZE is removed as the action space is now a continuous Box space
    # defined directly in the environment for more nuanced control.

# --- MODEL TRAINING CONFIGURATION (Stable-Baselines3 PPO) ---

@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration for the Proximal Policy Optimization (PPO) training process."""
    MODEL_OUTPUT_FILE: str = "ppo_multi_timeframe_hybrid_tin.zip"

    # --- PPO Hyperparameters ---
    TOTAL_TIMESTEPS: int = 200_000 # Total steps for the entire training process
    N_STEPS: int = 2048          # (Rollout Buffer Size) Steps collected per agent per update
    BATCH_SIZE: int = 64           # Mini-batch size for PPO updates
    N_EPOCHS: int = 10             # Number of optimization epochs per update
    GAMMA: float = 0.99            # Discount factor
    GAE_LAMBDA: float = 0.95       # Factor for Generalized Advantage Estimation
    CLIP_RANGE: float = 0.2        # Clipping parameter for PPO
    ENT_COEF: float = 0.01         # Entropy coefficient for exploration
    LEARNING_RATE: float = 3e-4    # Learning rate for the optimizer

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
    # NEW: The environment's "heartbeat" is now 1 minute to support 1m and 3m analysis.
    BASE_BAR_TIMEFRAME: str = "1T"

    # --- Trading Simulation ---
    # Realistic transaction fee (e.g., 0.001 for 0.1%) applied to every trade.
    # This is critical for preventing the agent from learning to over-trade.
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

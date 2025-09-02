# Enhanced config.py with reward horizon support

"""
Enhanced Configuration with Reward Horizon System

Key Addition: reward_horizon_steps parameter for configurable reward calculation periods
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

# ... [Previous utility functions and enums remain the same] ...

class StrategyConfig(BaseModel):
    """✅ ENHANCED: Strategy configuration with reward horizon system"""
    
    # ✅ NEW: Configurable Reward Horizon System
    reward_horizon_steps: int = Field(
        default=1, 
        ge=1, 
        le=20, 
        description="Number of steps to look ahead for reward calculation. 1=immediate (20s), 9=3min horizon"
    )
    
    reward_horizon_decay: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Decay factor for multi-step rewards to balance immediate vs delayed gratification"
    )
    
    # ✅ ENHANCED: Leverage with validation (existing code)
    leverage: float = Field(
        default_factory=lambda: float(os.getenv('LEVERAGE', 10.0)),
        ge=1.0, le=25.0, description="Trading leverage (1.0 to 25.0x maximum)"
    )
    
    # ... [Rest of existing StrategyConfig parameters remain the same] ...
    
    # REFACTORED: Define precomputed features from processor.py as a single source of truth.
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
    
    # ... [Rest of existing configuration remains the same] ...
    
    @validator('reward_horizon_steps')
    def validate_reward_horizon(cls, v):
        """Validate reward horizon and provide guidance."""
        if v == 1:
            logger.info("Using immediate reward horizon (20s)")
        elif v <= 9:
            horizon_minutes = v * 20 / 60
            logger.info(f"Using {horizon_minutes:.1f}-minute reward horizon ({v} steps)")
        else:
            horizon_minutes = v * 20 / 60
            logger.warning(f"Long reward horizon ({horizon_minutes:.1f} minutes) may slow learning")
        return v
    
    @validator('reward_horizon_decay')
    def validate_reward_decay(cls, v, values):
        """Validate reward decay factor."""
        if 'reward_horizon_steps' in values and values['reward_horizon_steps'] > 1:
            if v < 0.8:
                logger.warning(f"Low decay factor ({v}) may overweight distant rewards")
        return v

    def get_reward_horizon_info(self) -> Dict[str, Any]:
        """Get detailed information about reward horizon configuration."""
        horizon_seconds = self.reward_horizon_steps * 20
        horizon_minutes = horizon_seconds / 60
        
        return {
            'steps': self.reward_horizon_steps,
            'seconds': horizon_seconds,
            'minutes': horizon_minutes,
            'decay_factor': self.reward_horizon_decay,
            'description': f"{horizon_minutes:.1f}-minute reward horizon" if self.reward_horizon_steps > 1 else "Immediate reward"
        }

# Enhanced Global Configuration (additions only shown)
class GlobalConfig(BaseModel):
    """✅ ENHANCED: Global configuration with reward horizon support"""
    
    # ... [All existing GlobalConfig parameters remain the same] ...
    
    def get_reward_horizon_bars(self) -> int:
        """Get the number of bars needed for reward horizon calculation."""
        return self.strategy.reward_horizon_steps
    
    def validate_reward_horizon_data(self, total_bars: int) -> bool:
        """Validate that we have enough data for reward horizon calculation."""
        required_bars = self.get_reward_horizon_bars()
        warmup_bars = self.get_required_warmup_period()
        
        min_required = warmup_bars + required_bars + 100  # Safety margin
        
        if total_bars < min_required:
            logger.warning(f"Dataset may be too small for reward horizon {required_bars}. "
                         f"Have {total_bars}, need at least {min_required}")
            return False
        return True

# Configuration factory with reward horizon validation
def create_config(env: Optional[Environment] = None, **overrides) -> GlobalConfig:
    """✅ ENHANCED: Create configuration with reward horizon validation"""
    try:
        # If env is not provided, use the one from .env or the default
        base_env = env or Environment(os.getenv('ENVIRONMENT', 'dev'))
        
        # Base configurations for different environments
        base_config = {
            Environment.DEVELOPMENT: {
                "training": {"total_timesteps": 50_000, "checkpoint_frequency": 1000},
                "log_level": "DEBUG",
                "strategy": {"reward_horizon_steps": 1}  # Conservative for dev
            },
            Environment.STAGING: {
                "training": {"total_timesteps": 200_000, "checkpoint_frequency": 5000},
                "log_level": "INFO",
                "strategy": {"reward_horizon_steps": 3}  # Moderate for staging
            },
            Environment.PRODUCTION: {
                "training": {"total_timesteps": 1_000_000, "checkpoint_frequency": 10000},
                "log_level": "WARNING",
                "enable_wandb": True,
                "strategy": {"reward_horizon_steps": 9}  # Long-term for production
            }
        }
        
        env_config = base_config.get(base_env, {})
        env_config.update(overrides)
        env_config["environment"] = base_env
        
        config = GlobalConfig(**env_config)
        
        # Validate reward horizon configuration
        reward_info = config.strategy.get_reward_horizon_info()
        logger.info(f"Reward horizon configured: {reward_info['description']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        # Return basic config as fallback
        return GlobalConfig()

# Enhanced validation function
def validate_configuration(config: GlobalConfig) -> List[str]:
    """✅ ENHANCED: Validate configuration with reward horizon checks"""
    warnings_list = []
    
    try:
        # ... [All existing validation logic remains the same] ...
        
        # ✅ NEW: Reward horizon validation
        reward_horizon = config.strategy.reward_horizon_steps
        if reward_horizon > 15:
            warnings_list.append(f"Very long reward horizon ({reward_horizon} steps = {reward_horizon*20/60:.1f} min) may slow learning")
        
        if reward_horizon > 1 and config.strategy.reward_horizon_decay < 0.9:
            warnings_list.append(f"Low reward decay ({config.strategy.reward_horizon_decay}) with multi-step horizon may cause instability")
        
        # Check data requirements
        if hasattr(config, 'total_timesteps'):
            min_required_data = config.get_required_warmup_period() + reward_horizon + 1000
            logger.info(f"Reward horizon requires at least {min_required_data} data points")
        
    except Exception as e:
        warnings_list.append(f"Configuration validation error: {e}")
    
    return warnings_list

# Global instance with enhanced configuration
SETTINGS = create_config()

if __name__ == "__main__":
    print("--- Enhanced Configuration with Reward Horizon System ---")
    try:
        print(f"Environment: {SETTINGS.environment.value}")
        print(f"Primary Asset: {SETTINGS.primary_asset}")
        
        # ✅ NEW: Display reward horizon configuration
        reward_info = SETTINGS.strategy.get_reward_horizon_info()
        print(f"\n--- Reward Horizon Configuration ---")
        print(f"Horizon Steps: {reward_info['steps']}")
        print(f"Horizon Time: {reward_info['description']}")
        print(f"Decay Factor: {reward_info['decay_factor']}")
        
        # Display leverage-related settings
        print(f"\n--- Trading Configuration ---")
        print(f"Leverage: {SETTINGS.strategy.leverage}x")
        print(f"Maintenance Margin Rate: {SETTINGS.strategy.maintenance_margin_rate:.4f}")
        
        # Validate configuration
        config_warnings = validate_configuration(SETTINGS)
        if config_warnings:
            print("\n--- Configuration Warnings ---")
            for warning in config_warnings:
                print(f" - {warning}")
        
        print("\n✅ Enhanced configuration loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        print(f"❌ Configuration error: {e}")
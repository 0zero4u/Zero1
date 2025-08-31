"""
Main entry point for the Enhanced Crypto Trading RL System
Demonstrates proper usage of the improved modules.
"""

import logging
import sys
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_system.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating the improved trading system."""
    try:
        logger.info("🚀 Starting Enhanced Crypto Trading RL System")

        # Import and setup configuration
        from config import SETTINGS, setup_environment, validate_configuration

        # Setup environment and validate configuration
        setup_environment()
        warnings = validate_configuration(SETTINGS)

        if warnings:
            logger.warning("Configuration warnings detected:")
            for warning in warnings:
                logger.warning(f" - {warning}")

        logger.info(f"Environment: {SETTINGS.environment.value}")
        logger.info(f"Primary Asset: {SETTINGS.primary_asset}")
        logger.info(f"Base Path: {SETTINGS.base_path}")

        # Example 1: Data Processing
        logger.info("\n--- Example 1: Data Processing ---")
        from processor import process_trades_for_period, create_bars_from_trades

        # Note: This assumes you have raw trade data files
        # process_trades_for_period("in_sample")
        # bars_df = create_bars_from_trades("in_sample")
        # logger.info(f"Created {len(bars_df)} bars")

        # Example 2: Training (commented out to avoid actual training)
        logger.info("\n--- Example 2: Training Setup ---")
        from trainer import train_model_advanced

        # Uncomment to run actual training:
        # model = train_model_advanced(
        #     optimization_trials=5,
        #     final_training_steps=50000,
        #     use_wandb=False,
        #     use_ensemble=False
        # )

        # Example 3: Backtesting (requires trained model)
        logger.info("\n--- Example 3: Backtesting Setup ---")
        from generator import run_backtest, run_ensemble_backtest

        # Uncomment to run actual backtesting:
        # results = run_backtest()
        # logger.info("Backtest completed successfully!")

        logger.info("\n✅ System demonstration completed!")
        logger.info("\nTo use the system:")
        logger.info("1. Ensure you have raw trade data in the configured directories")
        logger.info("2. Run data processing: python -c 'from processor import process_trades_for_period; process_trades_for_period("in_sample")'")
        logger.info("3. Run training: python -c 'from trainer import train_model_advanced; train_model_advanced()'")
        logger.info("4. Run backtesting: python -c 'from generator import run_backtest; run_backtest()'")

    except Exception as e:
        logger.error(f"System failed: {e}")
        raise

if __name__ == "__main__":
    main()



import logging
import sys
import argparse
from pathlib import Path
import multiprocessing as mp

# --- Initial Logging Setup ---
# A basic logger is configured here to catch any issues during early imports.
# It will be enhanced later in the main function.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def setup_cli():
    """Configures the command-line interface using argparse."""
    parser = argparse.ArgumentParser(
        description="Zero1: An Enhanced Reinforcement Learning System for Crypto Trading.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- Process Command ---
    process_parser = subparsers.add_parser(
        'process',
        help="Process raw trade data into enriched Parquet files.",
        description="Ingests raw trade data (zip files) and converts it into a clean, processed format (Parquet) with features like side and asset."
    )
    process_parser.add_argument(
        '--period',
        type=str,
        default='in_sample',
        choices=['in_sample', 'out_of_sample'],
        help="The data period to process (default: in_sample)."
    )
    process_parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing of files even if they already exist."
    )

    # --- Train Command ---
    train_parser = subparsers.add_parser(
        'train',
        help="Train a new PPO agent using the processed data.",
        description="Handles the full training pipeline, including optional hyperparameter optimization with Optuna and final model training."
    )
    train_parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help="Number of Optuna hyperparameter optimization trials to run (set to 0 to skip)."
    )
    train_parser.add_argument(
        '--steps',
        type=int,
        default=500000,
        help="Number of timesteps for the final training phase after optimization."
    )
    train_parser.add_argument(
        '--wandb',
        action='store_true',
        help="Enable experiment tracking with Weights & Biases (requires WANDB_API_KEY)."
    )

    # --- Backtest Command ---
    backtest_parser = subparsers.add_parser(
        'backtest',
        help="Evaluate a trained model's performance.",
        description="Runs a comprehensive backtest on the out-of-sample data, generating performance metrics and visualizations."
    )
    backtest_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to the model .zip file to evaluate (defaults to the path in config)."
    )
    backtest_parser.add_argument(
        '--period',
        type=str,
        default='out_of_sample',
        choices=['in_sample', 'out_of_sample'],
        help="The data period to backtest on (default: out_of_sample)."
    )

    # --- Run Pipeline Command ---
    pipeline_parser = subparsers.add_parser(
        'run-pipeline',
        help="Run the full end-to-end pipeline: process -> train -> backtest.",
        description="A convenience command to execute the standard workflow in sequence."
    )
    pipeline_parser.add_argument(
        '--trials',
        type=int,
        default=5,
        help="Number of optimization trials for the training step."
    )
    pipeline_parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help="Number of timesteps for the final training step."
    )

    return parser


def main():
    """Main function to orchestrate the trading system's operations via CLI."""
    parser = setup_cli()
    args = parser.parse_args()

    # --- System Initialization ---
    try:
        print("🚀 Starting Zero1 Crypto Trading RL System 🚀")
        
        # Set multiprocessing start method for compatibility (especially on macOS/Windows)
        # This is crucial for SubprocVecEnv in the trainer.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        # Import and setup configuration
        from config import SETTINGS, setup_environment, validate_configuration

        # Configure file-based logging
        log_file = Path(SETTINGS.get_logs_path()) / "system.log"
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Setup environment (create directories) and validate configuration
        setup_environment()
        warnings = validate_configuration(SETTINGS)
        if warnings:
            logger.warning("Configuration warnings detected:")
            for warning in warnings:
                logger.warning(f" - {warning}")

        logger.info(f"Environment: {SETTINGS.environment.value} | Primary Asset: {SETTINGS.primary_asset}")

    except Exception as e:
        logger.error(f"Fatal error during system initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- Command Dispatcher ---
    if args.command == 'process':
        run_processing(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'backtest':
        run_evaluation(args)
    elif args.command == 'run-pipeline':
        logger.info("Starting End-to-End Pipeline...")
        run_processing(argparse.Namespace(period='in_sample', force=True))
        run_training(args)
        run_evaluation(argparse.Namespace(model_path=None, period='out_of_sample'))
        logger.info("✅ End-to-End Pipeline Completed Successfully!")


def run_processing(args):
    """Handles the data processing command."""
    try:
        logger.info(f"--- Starting Data Processing for '{args.period}' period ---")
        from processor import process_trades_for_period
        process_trades_for_period(args.period, force_reprocess=args.force)
        logger.info(f"✅ Data processing for '{args.period}' completed successfully.")
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        sys.exit(1)


def run_training(args):
    """Handles the model training command."""
    try:
        logger.info("--- Starting Model Training ---")
        logger.info(f"Optimization trials: {args.trials}, Final steps: {args.steps}, W&B: {args.wandb}")
        from trainer import train_model_advanced
        train_model_advanced(
            optimization_trials=args.trials,
            final_training_steps=args.steps,
            use_wandb=args.wandb
        )
        logger.info("✅ Model training completed successfully.")
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)


def run_evaluation(args):
    """Handles the model evaluation (backtesting) command."""
    try:
        logger.info(f"--- Starting Model Evaluation on '{args.period}' period ---")
        # CRITICAL FIX: Importing from the correct module 'evaluator'
        from evaluator import run_backtest
        
        # Check if the model exists if a custom path is not provided
        from config import SETTINGS
        model_path = args.model_path or SETTINGS.get_model_path()
        if not Path(model_path).exists():
            logger.error(f"Model file not found at '{model_path}'.")
            logger.error("Please train a model first using the 'train' command or provide a valid path with --model-path.")
            sys.exit(1)

        logger.info(f"Using model: {model_path}")
        run_backtest(model_path=model_path)
        logger.info("✅ Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

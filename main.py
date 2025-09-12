

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
        help="Process raw trade data into enriched Parquet files."
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
        help="Train a new PPO agent using the processed data."
    )
    train_parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help="Number of Optuna hyperparameter optimization trials to run."
    )
    train_parser.add_argument(
        '--steps',
        type=int,
        default=500000,
        help="Number of timesteps for the final training phase."
    )
    train_parser.add_argument(
        '--wandb',
        action='store_true',
        help="Enable experiment tracking with Weights & Biases."
    )

    # --- Backtest Command ---
    backtest_parser = subparsers.add_parser(
        'backtest',
        help="Evaluate a trained model's performance."
    )
    backtest_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to the model .zip file to evaluate."
    )
    backtest_parser.add_argument(
        '--period',
        type=str,
        default='out_of_sample',
        choices=['in_sample', 'out_of_sample'],
        help="The data period to backtest on."
    )

    # --- Run Pipeline Command ---
    pipeline_parser = subparsers.add_parser(
        'run-pipeline',
        help="Run the full end-to-end pipeline: process -> train -> backtest."
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
    pipeline_parser.add_argument(
        '--wandb',
        action='store_true',
        help="Enable experiment tracking with Weights & Biases for the pipeline."
    )
    pipeline_parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing of data files during the pipeline run."
    )

    return parser


def main():
    """Main function to orchestrate the trading system's operations via CLI."""
    parser = setup_cli()
    args = parser.parse_args()

    # --- System Initialization ---
    try:
        print("🚀 Starting Zero1 Crypto Trading RL System 🚀")
        
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        from config import SETTINGS, setup_environment, validate_configuration

        log_file = Path(SETTINGS.get_logs_path()) / "system.log"
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
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
        logger.info("--- Starting End-to-End Pipeline ---")
        
        logger.info("Step 1/3: Processing in-sample data...")
        run_processing(argparse.Namespace(period='in_sample', force=args.force))
        logger.info("Step 1/3: Processing out-of-sample data...")
        run_processing(argparse.Namespace(period='out_of_sample', force=args.force))
        
        logger.info("Step 2/3: Training model...")
        new_model_path = run_training(args)
        
        logger.info("Step 3/3: Evaluating model on out-of-sample data...")
        run_evaluation(argparse.Namespace(model_path=new_model_path, period='out_of_sample'))
        
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
    """Handles the model training command and returns the path to the trained model."""
    try:
        logger.info("--- Starting Model Training ---")
        logger.info(f"Optimization trials: {args.trials}, Final steps: {args.steps}, W&B: {args.wandb}")
        
        # --- START OF FIX ---
        # The main training function was renamed from 'train_model_advanced'
        # to 'train_model_fixed' to reflect the new anti-turtling logic.
        # The import path was corrected from 'trainer' to 'enhanced_trainer_verbose'.
        from enhanced_trainer_verbose import train_model_fixed
        model_path = train_model_fixed(
            optimization_trials=args.trials,
            final_training_steps=args.steps,
            use_wandb=args.wandb
        )
        # --- END OF FIX ---

        logger.info("✅ Model training completed successfully.")
        return model_path
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)


def run_evaluation(args):
    """Handles the model evaluation (backtesting) command."""
    try:
        logger.info(f"--- Starting Model Evaluation on '{args.period}' period ---")
        from evaluator import run_backtest
        
        if not args.model_path:
            logger.error("Model path is required for backtesting.")
            logger.error("Please provide the path using the --model-path argument.")
            sys.exit(1)
        
        model_path = args.model_path
        if not Path(model_path).exists():
            logger.error(f"Model file not found at '{model_path}'.")
            sys.exit(1)

        logger.info(f"Using model: {model_path}")
        run_backtest(model_path=model_path, period=args.period)
        logger.info("✅ Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

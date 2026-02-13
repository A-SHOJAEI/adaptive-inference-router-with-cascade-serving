#!/usr/bin/env python3
"""
Training script for Adaptive Inference Router with Cascade Serving.

This script trains the multi-objective reinforcement learning model for
adaptive inference routing across model cascades.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
from adaptive_inference_router_with_cascade_serving.training.trainer import MultiObjectiveTrainer
from adaptive_inference_router_with_cascade_serving.utils.config import (
    Config,
    get_default_config_path,
    load_config,
    setup_device,
    setup_logging,
    setup_seeds,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train Adaptive Inference Router with Cascade Serving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (uses default if not specified)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override device specification"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - setup everything but don't train"
    )

    return parser.parse_args()


def load_and_override_config(args: argparse.Namespace) -> Config:
    """Load configuration and apply command line overrides.

    Args:
        args: Command line arguments.

    Returns:
        Configuration with overrides applied.
    """
    # Load base configuration
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = get_default_config_path()

    if config_path.exists():
        config = load_config(config_path)
    else:
        # Create default config
        config = Config()
        print(f"Warning: Config file {config_path} not found, using defaults")

    # Apply command line overrides
    if args.device is not None:
        config.system.device = args.device

    if args.epochs is not None:
        config.training.epochs = args.epochs

    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    if args.learning_rate is not None:
        config.training.rl["learning_rate"] = args.learning_rate

    if args.seed is not None:
        config.system.seed = args.seed

    if args.debug:
        config.experiment.logging["level"] = "DEBUG"

    return config


def setup_data_loaders(config: Config) -> tuple:
    """Setup data loaders for training.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, data_loader).
    """
    print("Setting up data loaders...")

    # Create data loader
    data_loader = AdaptiveDataLoader(config)

    # Create datasets
    train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # Setup preprocessing
    data_loader.setup_preprocessing(train_dataset)

    print(f"Data loaders created:")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, data_loader


def create_trainer(config: Config, device: torch.device, resume_path: str = None) -> MultiObjectiveTrainer:
    """Create and initialize trainer.

    Args:
        config: Configuration object.
        device: Training device.
        resume_path: Optional path to checkpoint to resume from.

    Returns:
        Initialized trainer.
    """
    print("Initializing trainer...")

    # Create trainer
    trainer = MultiObjectiveTrainer(config, device)

    # Resume from checkpoint if specified
    if resume_path:
        checkpoint_path = Path(resume_path)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
            print(f"Resumed training from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting fresh")

    # Print model summary
    model_summary = trainer.model.get_model_summary()
    print(f"Model initialized:")
    print(f"  - Input dimension: {model_summary['input_dim']}")
    print(f"  - Hidden dimension: {model_summary['hidden_dim']}")
    print(f"  - Number of routes: {model_summary['num_routes']}")
    print(f"  - Total parameters: {model_summary['total_parameters']:,}")
    print(f"  - Trainable parameters: {model_summary['trainable_parameters']:,}")

    return trainer


def train_model(
    trainer: MultiObjectiveTrainer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    dry_run: bool = False
) -> None:
    """Train the model.

    Args:
        trainer: Initialized trainer.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration object.
        dry_run: Whether this is a dry run.
    """
    if dry_run:
        print("Dry run mode - skipping actual training")
        return

    print("Starting training...")
    print(f"Training configuration:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Learning rate: {trainer.ppo_trainer.learning_rate}")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Early stopping patience: {config.training.patience}")

    try:
        trainer.train(train_loader, val_loader)
        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def evaluate_final_model(
    trainer: MultiObjectiveTrainer,
    test_loader: torch.utils.data.DataLoader
) -> dict:
    """Evaluate the final trained model.

    Args:
        trainer: Trained trainer.
        test_loader: Test data loader.

    Returns:
        Dictionary containing evaluation metrics.
    """
    print("Evaluating final model...")

    # Run evaluation
    eval_metrics = trainer.evaluate(test_loader)

    # Print results
    print("Final evaluation results:")
    for metric_name, value in eval_metrics.items():
        if isinstance(value, float):
            print(f"  - {metric_name}: {value:.4f}")
        else:
            print(f"  - {metric_name}: {value}")

    return eval_metrics


def save_final_artifacts(
    trainer: MultiObjectiveTrainer,
    config: Config,
    eval_metrics: dict
) -> None:
    """Save final training artifacts.

    Args:
        trainer: Trained trainer.
        config: Configuration object.
        eval_metrics: Final evaluation metrics.
    """
    print("Saving final artifacts...")

    # Create output directories
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Also ensure standard directories exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # Save final checkpoint to multiple locations
    final_checkpoint_path = output_dir / "final_model.pth"
    trainer.save_checkpoint(final_checkpoint_path, is_best=True)

    # Also save to models/ directory
    models_checkpoint_path = models_dir / "final_model.pth"
    trainer.save_checkpoint(models_checkpoint_path, is_best=True)

    # Save best model to checkpoints/ directory
    best_checkpoint_path = checkpoints_dir / "best_model.pth"
    trainer.save_checkpoint(best_checkpoint_path, is_best=True)

    # Save configuration
    from adaptive_inference_router_with_cascade_serving.utils.config import save_config
    config_path = output_dir / "config.yaml"
    save_config(config, config_path)

    # Save evaluation metrics
    import json
    metrics_path = output_dir / "final_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Artifacts saved to {output_dir}/")
    print(f"Model checkpoints saved to: {final_checkpoint_path}, {models_checkpoint_path}, {best_checkpoint_path}")


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Load and override configuration
    config = load_and_override_config(args)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting training script")

    # Setup device and seeds
    device_str = setup_device(config)
    device = torch.device(device_str)
    setup_seeds(config)

    print(f"Training on device: {device}")

    # Ensure GPU is being used if available and verify CUDA
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        # Set memory growth
        torch.cuda.empty_cache()

    # Create necessary directories
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Setup data loaders
        train_loader, val_loader, test_loader, data_loader = setup_data_loaders(config)

        # Create trainer
        trainer = create_trainer(config, device, args.resume)

        # Train model
        train_model(trainer, train_loader, val_loader, config, args.dry_run)

        # Evaluate final model
        if not args.dry_run:
            eval_metrics = evaluate_final_model(trainer, test_loader)
            save_final_artifacts(trainer, config, eval_metrics)

        logger.info("Training script completed successfully")

    except Exception as e:
        logger.error(f"Training script failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        if 'trainer' in locals() and hasattr(trainer, 'tb_writer'):
            trainer.tb_writer.close()


if __name__ == "__main__":
    main()
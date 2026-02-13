"""
Multi-objective reinforcement learning trainer for adaptive inference routing.

This module implements the training loop for the adaptive routing policy using
Proximal Policy Optimization (PPO) with multi-objective optimization.
"""

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data.preprocessing import SystemLoadMonitor
from ..models.model import AdaptiveInferenceRouter
from ..utils.config import Config


logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Tracks and manages training metrics."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize training metrics tracker.

        Args:
            window_size: Moving average window size.
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.step_count = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric name to value.
        """
        for name, value in metrics.items():
            self.metrics[name].append(value)
        self.step_count += 1

    def get_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """Get moving average of a metric.

        Args:
            metric_name: Name of the metric.
            window: Window size (uses default if None).

        Returns:
            Moving average value.
        """
        if metric_name not in self.metrics:
            return 0.0

        values = list(self.metrics[metric_name])
        if window is not None:
            values = values[-window:]

        return np.mean(values) if values else 0.0

    def get_latest(self, metric_name: str) -> float:
        """Get latest value of a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Latest metric value.
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return self.metrics[metric_name][-1]

    def get_all_averages(self) -> Dict[str, float]:
        """Get moving averages for all metrics.

        Returns:
            Dictionary of metric names to moving averages.
        """
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step_count = 0


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for routing policies.

    Implements PPO with multi-objective optimization for learning
    adaptive inference routing strategies.
    """

    def __init__(
        self,
        model: AdaptiveInferenceRouter,
        config: Config,
        device: torch.device
    ) -> None:
        """Initialize PPO trainer.

        Args:
            model: Adaptive inference router model.
            config: Training configuration.
            device: Training device.
        """
        self.model = model
        self.config = config
        self.device = device
        self.rl_config = config.training.rl

        # PPO hyperparameters
        self.learning_rate = self.rl_config.get("learning_rate", 3e-4)
        self.clip_epsilon = self.rl_config.get("clip_epsilon", 0.2)
        self.entropy_coeff = self.rl_config.get("entropy_coeff", 0.01)
        self.value_loss_coeff = self.rl_config.get("value_loss_coeff", 0.5)
        self.max_grad_norm = self.rl_config.get("max_grad_norm", 0.5)
        self.gamma = self.rl_config.get("gamma", 0.99)
        self.lambda_gae = self.rl_config.get("lambda_gae", 0.95)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.training.optimizer.get("weight_decay", 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.scheduler.get("min_lr", 1e-6)
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        logger.info(f"Initialized PPO trainer with lr={self.learning_rate}")

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and value targets.

        Args:
            rewards: Reward sequence [num_episodes, batch_size].
            values: Value predictions [num_episodes, batch_size].
            dones: Done flags [num_episodes, batch_size].

        Returns:
            Tuple of (advantages, value_targets).
        """
        # For single-step episodic decisions, we can compute advantages directly
        # without temporal dependencies since each routing decision is independent
        # Simply use: advantage = reward - value_prediction

        advantages = rewards - values
        value_targets = rewards  # Target is the actual reward received

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, value_targets

    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        value_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss components.

        Args:
            states: State features [batch_size, state_dim].
            actions: Selected actions [batch_size].
            old_log_probs: Old action log probabilities [batch_size].
            advantages: GAE advantages [batch_size].
            value_targets: Value function targets [batch_size].

        Returns:
            Dictionary containing loss components.
        """
        # Forward pass
        predictions = self.model(states)

        # Current policy log probabilities
        action_dist = torch.distributions.Categorical(logits=predictions["route_logits"])
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        # PPO policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        # Value function loss
        value_loss = nn.MSELoss()(predictions["values"], value_targets)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coeff * value_loss +
            self.entropy_coeff * entropy_loss
        )

        # Compute additional metrics
        approx_kl = ((old_log_probs - log_probs) ** 2).mean()
        clipfrac = ((ratio - clipped_ratio).abs() > 1e-6).float().mean()

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "entropy": entropy.mean(),
        }

    def update_policy(
        self,
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Update policy using PPO algorithm.

        Args:
            batch_data: Batch of training data.

        Returns:
            Dictionary containing training metrics.
        """
        # Compute advantages
        advantages, value_targets = self.compute_advantages(
            batch_data["rewards"],
            batch_data["values"],
            batch_data["dones"]
        )

        # Flatten sequences for training
        states = batch_data["states"].view(-1, batch_data["states"].shape[-1])
        actions = batch_data["actions"].view(-1)
        old_log_probs = batch_data["log_probs"].view(-1)
        advantages = advantages.view(-1)
        value_targets = value_targets.view(-1)

        # Compute loss
        loss_dict = self.compute_ppo_loss(
            states, actions, old_log_probs, advantages, value_targets
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        self.optimizer.step()
        self.global_step += 1

        # Convert to float for logging
        metrics = {k: v.item() for k, v in loss_dict.items()}
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return metrics


class MultiObjectiveTrainer:
    """
    Main trainer for multi-objective adaptive inference routing.

    Coordinates the training process including data generation, policy updates,
    evaluation, and experiment tracking.
    """

    def __init__(self, config: Config, device: torch.device) -> None:
        """Initialize multi-objective trainer.

        Args:
            config: Training configuration.
            device: Training device.
        """
        self.config = config
        self.device = device

        # Initialize model
        self.model = AdaptiveInferenceRouter(config).to(device)

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(self.model, config, device)

        # System monitoring
        self.system_monitor = SystemLoadMonitor(config)

        # Training metrics
        self.metrics = TrainingMetrics()

        # Experiment tracking
        self.setup_experiment_tracking()

        # Training state
        self.best_reward = float('-inf')
        self.episodes_trained = 0
        self.early_stopping_counter = 0

        logger.info("Initialized MultiObjectiveTrainer")

    def setup_experiment_tracking(self) -> None:
        """Setup MLflow and TensorBoard tracking."""
        # MLflow setup
        mlflow_config = self.config.experiment.mlflow
        mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "./mlruns"))

        experiment_name = mlflow_config.get("experiment_name", "adaptive_inference_router")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
            experiment_id = "0"

        # Start MLflow run
        mlflow.start_run(experiment_id=experiment_id)
        mlflow.log_params({
            "model_hidden_dim": self.config.model.router.get("hidden_dim", 256),
            "learning_rate": self.ppo_trainer.learning_rate,
            "batch_size": self.config.data.batch_size,
            "epochs": self.config.training.epochs,
        })

        # TensorBoard setup
        log_dir = Path(self.config.experiment.logging.get("log_dir", "./logs"))
        self.tb_writer = SummaryWriter(log_dir / "tensorboard")

        logger.info("Setup experiment tracking with MLflow and TensorBoard")

    def generate_episode_data(
        self,
        data_loader: DataLoader,
        num_episodes: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Generate episode data for training.

        Args:
            data_loader: Data loader for sampling experiences.
            num_episodes: Number of episodes to generate.

        Returns:
            Dictionary containing episode data.
        """
        episode_data = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

        self.model.eval()
        with torch.no_grad():
            for episode in range(num_episodes):
                # Sample batch from data loader
                try:
                    batch = next(iter(data_loader))
                except StopIteration:
                    break

                # Move to device
                features = batch["features"].to(self.device)
                sla_constraints = batch["sla_constraints"].to(self.device)

                # Note: features already contains all 14 features:
                # - 5 difficulty features
                # - 4 SLA features (including target_latency_ms and accuracy_threshold)
                # - 5 system features
                # No need to concatenate additional features

                # Select actions
                actions, pred_info = self.model.select_route(features, deterministic=False)

                # Simulate performance and compute rewards
                actual_performance = self._simulate_performance(actions, features)
                rewards = self.model.compute_route_rewards(
                    pred_info["performance_preds"],
                    actual_performance,
                    sla_constraints,
                    actions
                )

                # Store episode data
                episode_data["states"].append(features)
                episode_data["actions"].append(actions)
                episode_data["log_probs"].append(pred_info["log_probs"])
                episode_data["rewards"].append(rewards)
                episode_data["values"].append(pred_info["values"])
                episode_data["dones"].append(torch.ones_like(actions, dtype=torch.float32))

        # Convert to tensors
        for key in episode_data:
            if episode_data[key]:
                episode_data[key] = torch.stack(episode_data[key])
            else:
                # Handle empty case
                batch_size = self.config.data.batch_size
                if key in ["states"]:
                    # Use model's input_dim for the feature dimension
                    episode_data[key] = torch.zeros((1, batch_size, self.model.input_dim), device=self.device)
                else:
                    episode_data[key] = torch.zeros((1, batch_size), device=self.device)

        return episode_data

    def _simulate_performance(
        self,
        actions: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Simulate actual performance for selected routes.

        Args:
            actions: Selected route actions [batch_size].
            features: Input features [batch_size, feature_dim].

        Returns:
            Simulated performance metrics [batch_size, 4].
        """
        batch_size = actions.shape[0]

        # Base performance from cascade model
        base_latency = 100.0 + features[:, 0] * 50.0  # Based on complexity
        base_accuracy = 0.95 - features[:, 4] * 0.1   # Based on uncertainty
        base_throughput = 50.0 + torch.randn(batch_size, device=self.device) * 10.0

        # Apply route-specific multipliers
        cascade_configs = [
            {"latency_mult": 0.3, "accuracy_mult": 0.95},  # quantized
            {"latency_mult": 0.5, "accuracy_mult": 0.97},  # pruned
            {"latency_mult": 0.4, "accuracy_mult": 0.94},  # distilled
            {"latency_mult": 1.0, "accuracy_mult": 1.0},   # full
        ]

        actual_latency = torch.zeros_like(base_latency)
        actual_accuracy = torch.zeros_like(base_accuracy)

        for i, config in enumerate(cascade_configs):
            mask = actions == i
            actual_latency[mask] = base_latency[mask] * config["latency_mult"]
            actual_accuracy[mask] = base_accuracy[mask] * config["accuracy_mult"]

        # Add noise to simulate real-world variance
        actual_latency += torch.randn_like(actual_latency) * 5.0
        actual_accuracy += torch.randn_like(actual_accuracy) * 0.01

        # Clamp values to realistic ranges
        actual_latency = torch.clamp(actual_latency, 10.0, 1000.0)
        actual_accuracy = torch.clamp(actual_accuracy, 0.5, 1.0)
        actual_throughput = torch.clamp(base_throughput, 10.0, 200.0)

        # SLA violation probability (simplified)
        sla_violation_prob = (actual_latency > 100.0).float() * 0.8 + (actual_accuracy < 0.95).float() * 0.6
        sla_violation_prob = torch.clamp(sla_violation_prob, 0.0, 1.0)

        return torch.stack([
            actual_latency,
            actual_accuracy,
            actual_throughput,
            sla_violation_prob
        ], dim=1)

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            data_loader: Training data loader.

        Returns:
            Dictionary containing epoch metrics.
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_updates = 0

        # Generate episode data
        episode_data = self.generate_episode_data(data_loader, num_episodes=50)

        # PPO updates
        num_ppo_epochs = 4  # Multiple epochs on same data
        for ppo_epoch in range(num_ppo_epochs):
            update_metrics = self.ppo_trainer.update_policy(episode_data)

            for key, value in update_metrics.items():
                epoch_metrics[key] += value
            num_updates += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_updates

        # Update learning rate
        self.ppo_trainer.scheduler.step()

        # Additional metrics
        epoch_metrics["episodes_trained"] = self.episodes_trained
        epoch_metrics["avg_reward"] = episode_data["rewards"].mean().item()
        epoch_metrics["avg_episode_length"] = episode_data["states"].shape[0]

        return dict(epoch_metrics)

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            data_loader: Evaluation data loader.

        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        eval_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                features = batch["features"].to(self.device)
                sla_constraints = batch["sla_constraints"].to(self.device)
                optimal_routes = batch["optimal_route"].to(self.device)

                # Note: features already contains all 14 features:
                # - 5 difficulty features
                # - 4 SLA features (including target_latency_ms and accuracy_threshold)
                # - 5 system features
                # No need to concatenate additional features

                # Predict routes
                predicted_routes, pred_info = self.model.select_route(features, deterministic=True)

                # Compute metrics
                accuracy = (predicted_routes == optimal_routes).float().mean()
                eval_metrics["route_accuracy"].append(accuracy.item())

                # Performance prediction metrics
                actual_perf = self._simulate_performance(predicted_routes, features)
                pred_perf = pred_info["performance_preds"].gather(
                    1, predicted_routes.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4)
                ).squeeze(1)

                mse = nn.MSELoss()(pred_perf, actual_perf)
                eval_metrics["performance_mse"].append(mse.item())

                # Route distribution
                for i in range(self.model.num_routes):
                    route_count = (predicted_routes == i).sum().item()
                    eval_metrics[f"route_{i}_usage"].append(route_count / predicted_routes.shape[0])

        # Average metrics
        return {key: np.mean(values) for key, values in eval_metrics.items()}

    def save_checkpoint(self, save_path: Path, is_best: bool = False) -> None:
        """Save training checkpoint.

        Args:
            save_path: Path to save checkpoint.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.ppo_trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.ppo_trainer.scheduler.state_dict(),
            "epoch": self.ppo_trainer.epoch,
            "global_step": self.ppo_trainer.global_step,
            "best_reward": self.best_reward,
            "config": self.config.dict(),
        }

        torch.save(checkpoint, save_path)

        if is_best:
            best_path = save_path.parent / "best_model.pth"
            torch.save(checkpoint, best_path)

        logger.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, load_path: Path) -> None:
        """Load training checkpoint.

        Args:
            load_path: Path to load checkpoint from.
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.ppo_trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.ppo_trainer.epoch = checkpoint["epoch"]
        self.ppo_trainer.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]

        logger.info(f"Loaded checkpoint from {load_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
        """
        logger.info("Starting training")

        # Setup checkpoint directory
        checkpoint_dir = Path(self.config.system.checkpoint.get("save_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Also create models directory
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.training.epochs):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Evaluation
            if epoch % self.config.evaluation.eval_frequency == 0:
                eval_metrics = self.evaluate(val_loader)

                # Check for improvement
                current_reward = train_metrics["avg_reward"]
                is_best = current_reward > self.best_reward
                if is_best:
                    self.best_reward = current_reward
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                # Log metrics
                all_metrics = {**train_metrics, **eval_metrics}
                self._log_metrics(all_metrics, epoch)

                # Save checkpoint
                if epoch % self.config.system.checkpoint.get("save_frequency", 5) == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                    self.save_checkpoint(checkpoint_path, is_best)

                # Early stopping
                patience = self.config.training.patience
                if self.early_stopping_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break

            # Update trainer state
            self.ppo_trainer.epoch = epoch
            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{self.config.training.epochs} "
                f"completed in {epoch_time:.2f}s, "
                f"avg_reward={train_metrics['avg_reward']:.4f}"
            )

        logger.info("Training completed")
        mlflow.end_run()

    def _log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to tracking systems.

        Args:
            metrics: Metrics to log.
            epoch: Current epoch.
        """
        # Update metrics tracker
        self.metrics.update(metrics)

        # Log to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=epoch)

        # Log to TensorBoard
        for name, value in metrics.items():
            self.tb_writer.add_scalar(name, value, epoch)

        # Log model weights histogram
        for name, param in self.model.named_parameters():
            self.tb_writer.add_histogram(f"weights/{name}", param.data, epoch)

        self.tb_writer.flush()


def main() -> None:
    """Main training function."""
    try:
        from ..data.loader import AdaptiveDataLoader
        from ..utils.config import load_config, setup_device, setup_logging, setup_seeds, get_default_config_path
    except ImportError:
        # Handle case when running as script directly
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        from adaptive_inference_router_with_cascade_serving.utils.config import load_config, setup_device, setup_logging, setup_seeds, get_default_config_path

    # Load configuration
    config_path = get_default_config_path()
    config = load_config(config_path)

    # Setup logging and device
    setup_logging(config)
    device_str = setup_device(config)
    device = torch.device(device_str)
    setup_seeds(config)

    logger.info(f"Starting training on device: {device}")

    # Create data loaders
    data_loader = AdaptiveDataLoader(config)
    train_dataset, val_dataset, test_dataset = data_loader.create_datasets()
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # Setup preprocessing
    data_loader.setup_preprocessing(train_dataset)

    # Initialize trainer
    trainer = MultiObjectiveTrainer(config, device)

    # Start training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if hasattr(trainer, 'tb_writer'):
            trainer.tb_writer.close()


if __name__ == "__main__":
    main()
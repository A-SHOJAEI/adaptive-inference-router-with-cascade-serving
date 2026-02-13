"""Tests for training modules and PPO implementation."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from adaptive_inference_router_with_cascade_serving.training.trainer import (
    TrainingMetrics,
    PPOTrainer,
    MultiObjectiveTrainer
)
from adaptive_inference_router_with_cascade_serving.models.model import AdaptiveInferenceRouter


class TestTrainingMetrics:
    """Test cases for TrainingMetrics."""

    @pytest.mark.unit
    def test_metrics_initialization(self):
        """Test metrics tracker initialization."""
        metrics = TrainingMetrics(window_size=50)

        assert metrics.window_size == 50
        assert metrics.step_count == 0
        assert len(metrics.metrics) == 0

    @pytest.mark.unit
    def test_update_metrics(self):
        """Test metrics update."""
        metrics = TrainingMetrics(window_size=50)

        test_metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "reward": 1.2
        }

        metrics.update(test_metrics)

        assert metrics.step_count == 1
        assert metrics.get_latest("loss") == 0.5
        assert metrics.get_latest("accuracy") == 0.8
        assert metrics.get_latest("reward") == 1.2

    @pytest.mark.unit
    def test_moving_average(self):
        """Test moving average calculation."""
        metrics = TrainingMetrics(window_size=5)

        # Add some values
        for i in range(10):
            metrics.update({"test_metric": float(i)})

        # Should average last 5 values: 5, 6, 7, 8, 9 -> mean = 7
        avg = metrics.get_average("test_metric")
        assert abs(avg - 7.0) < 1e-6

    @pytest.mark.unit
    def test_get_all_averages(self):
        """Test getting all metric averages."""
        metrics = TrainingMetrics(window_size=10)

        for i in range(5):
            metrics.update({
                "loss": float(i),
                "accuracy": float(i + 1)
            })

        averages = metrics.get_all_averages()

        assert "loss" in averages
        assert "accuracy" in averages
        assert averages["loss"] == 2.0  # Average of 0, 1, 2, 3, 4
        assert averages["accuracy"] == 3.0  # Average of 1, 2, 3, 4, 5

    @pytest.mark.unit
    def test_reset_metrics(self):
        """Test metrics reset."""
        metrics = TrainingMetrics(window_size=10)

        metrics.update({"test": 1.0})
        assert metrics.step_count == 1

        metrics.reset()
        assert metrics.step_count == 0
        assert len(metrics.metrics) == 0

    @pytest.mark.unit
    def test_window_size_limit(self):
        """Test that metrics window respects size limit."""
        metrics = TrainingMetrics(window_size=3)

        # Add more values than window size
        for i in range(10):
            metrics.update({"test": float(i)})

        # Should only keep last 3 values
        assert len(metrics.metrics["test"]) == 3
        assert list(metrics.metrics["test"]) == [7.0, 8.0, 9.0]


class TestPPOTrainer:
    """Test cases for PPOTrainer."""

    @pytest.fixture
    def mock_model(self, test_config, device):
        """Create mock model for testing."""
        return AdaptiveInferenceRouter(test_config).to(device)

    @pytest.mark.unit
    def test_ppo_trainer_initialization(self, mock_model, test_config, device):
        """Test PPO trainer initialization."""
        trainer = PPOTrainer(mock_model, test_config, device)

        assert trainer.model == mock_model
        assert trainer.config == test_config
        assert trainer.device == device
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')

    @pytest.mark.unit
    def test_compute_advantages(self, mock_model, test_config, device, setup_seeds):
        """Test GAE advantage computation."""
        trainer = PPOTrainer(mock_model, test_config, device)

        batch_size, seq_len = 4, 10
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len + 1)  # One extra for bootstrapping
        dones = torch.zeros(batch_size, seq_len)  # No episodes end

        advantages, value_targets = trainer.compute_advantages(rewards, values, dones)

        assert advantages.shape == (batch_size, seq_len)
        assert value_targets.shape == (batch_size, seq_len)

        # Check that advantages are normalized (approximately zero mean, unit std)
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.1

    @pytest.mark.unit
    def test_compute_ppo_loss(self, mock_model, test_config, device, setup_seeds):
        """Test PPO loss computation."""
        trainer = PPOTrainer(mock_model, test_config, device)

        batch_size = 16
        state_dim = mock_model.input_dim

        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, 4, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        value_targets = torch.randn(batch_size)

        loss_dict = trainer.compute_ppo_loss(
            states, actions, old_log_probs, advantages, value_targets
        )

        # Check that all loss components are present
        assert "total_loss" in loss_dict
        assert "policy_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "entropy_loss" in loss_dict
        assert "approx_kl" in loss_dict
        assert "clipfrac" in loss_dict
        assert "entropy" in loss_dict

        # Check that losses are scalars
        for key, value in loss_dict.items():
            assert value.dim() == 0, f"{key} should be a scalar"

    @pytest.mark.unit
    def test_update_policy(self, mock_model, test_config, device, setup_seeds):
        """Test policy update step."""
        trainer = PPOTrainer(mock_model, test_config, device)

        batch_size, seq_len = 4, 8
        state_dim = mock_model.input_dim

        # Create batch data
        batch_data = {
            "states": torch.randn(batch_size, seq_len, state_dim),
            "actions": torch.randint(0, 4, (batch_size, seq_len)),
            "log_probs": torch.randn(batch_size, seq_len),
            "rewards": torch.randn(batch_size, seq_len),
            "values": torch.randn(batch_size, seq_len + 1),
            "dones": torch.zeros(batch_size, seq_len)
        }

        initial_step = trainer.global_step

        metrics = trainer.update_policy(batch_data)

        # Check that update happened
        assert trainer.global_step == initial_step + 1

        # Check returned metrics
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "learning_rate" in metrics

    @pytest.mark.unit
    def test_gradient_clipping(self, mock_model, test_config, device, setup_seeds):
        """Test gradient clipping during updates."""
        trainer = PPOTrainer(mock_model, test_config, device)

        batch_size, seq_len = 4, 8
        state_dim = mock_model.input_dim

        # Create batch data with extreme values to trigger clipping
        batch_data = {
            "states": torch.randn(batch_size, seq_len, state_dim),
            "actions": torch.randint(0, 4, (batch_size, seq_len)),
            "log_probs": torch.randn(batch_size, seq_len),
            "rewards": torch.randn(batch_size, seq_len) * 1000,  # Large rewards
            "values": torch.randn(batch_size, seq_len + 1),
            "dones": torch.zeros(batch_size, seq_len)
        }

        # Should not raise any errors
        metrics = trainer.update_policy(batch_data)
        assert "total_loss" in metrics


class TestMultiObjectiveTrainer:
    """Test cases for MultiObjectiveTrainer."""

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_trainer_initialization(self, mock_log_params, mock_start_run, test_config, device):
        """Test multi-objective trainer initialization."""
        trainer = MultiObjectiveTrainer(test_config, device)

        assert trainer.config == test_config
        assert trainer.device == device
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'ppo_trainer')
        assert hasattr(trainer, 'system_monitor')
        assert hasattr(trainer, 'metrics')

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_experiment_tracking_setup(self, mock_log_params, mock_start_run, test_config, device):
        """Test experiment tracking setup."""
        trainer = MultiObjectiveTrainer(test_config, device)

        # Check that MLflow was called
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_simulate_performance(self, mock_log_params, mock_start_run, test_config, device, setup_seeds):
        """Test performance simulation."""
        trainer = MultiObjectiveTrainer(test_config, device)

        batch_size = 8
        actions = torch.randint(0, 4, (batch_size,))
        features = torch.randn(batch_size, 14)  # Expected feature dimension

        actual_performance = trainer._simulate_performance(actions, features)

        assert actual_performance.shape == (batch_size, 4)

        # Check realistic value ranges
        latencies = actual_performance[:, 0]
        accuracies = actual_performance[:, 1]
        throughputs = actual_performance[:, 2]

        assert torch.all(latencies >= 10)
        assert torch.all(latencies <= 500)
        assert torch.all(accuracies >= 0.7)
        assert torch.all(accuracies <= 1.0)
        assert torch.all(throughputs >= 20)
        assert torch.all(throughputs <= 150)

    @pytest.mark.integration
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    def test_train_epoch(self, mock_log_metric, mock_log_params, mock_start_run, test_config, device, setup_seeds):
        """Test single training epoch."""
        # Use smaller config for faster test
        test_config.training.epochs = 1
        test_config.data.batch_size = 4

        trainer = MultiObjectiveTrainer(test_config, device)

        # Create mock data loader
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, _, _ = data_loader.create_datasets()
        train_loader, _, _ = data_loader.create_dataloaders(train_dataset, train_dataset, train_dataset)

        # Run single epoch
        metrics = trainer.train_epoch(train_loader)

        assert isinstance(metrics, dict)
        assert "avg_reward" in metrics
        assert "total_loss" in metrics

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_evaluate(self, mock_log_params, mock_start_run, test_config, device, setup_seeds):
        """Test model evaluation."""
        test_config.data.batch_size = 4
        trainer = MultiObjectiveTrainer(test_config, device)

        # Create mock data loader
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        data_loader = AdaptiveDataLoader(test_config)
        _, val_dataset, _ = data_loader.create_datasets()
        _, val_loader, _ = data_loader.create_dataloaders(val_dataset, val_dataset, val_dataset)

        # Run evaluation
        eval_metrics = trainer.evaluate(val_loader)

        assert isinstance(eval_metrics, dict)
        assert "route_accuracy" in eval_metrics
        assert "performance_mse" in eval_metrics

        # Check route usage metrics
        for i in range(trainer.model.num_routes):
            assert f"route_{i}_usage" in eval_metrics

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_save_load_checkpoint(self, mock_log_params, mock_start_run, test_config, device, temp_dir):
        """Test checkpoint saving and loading."""
        trainer = MultiObjectiveTrainer(test_config, device)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path, is_best=True)

        assert checkpoint_path.exists()

        # Check that best model was also saved
        best_path = temp_dir / "best_model.pth"
        assert best_path.exists()

        # Test loading
        initial_state = trainer.model.state_dict()

        # Modify model state
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.fill_(0.0)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Check that state was restored
        loaded_state = trainer.model.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], loaded_state[key])

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_generate_episode_data(self, mock_log_params, mock_start_run, test_config, device, setup_seeds):
        """Test episode data generation."""
        test_config.data.batch_size = 4
        trainer = MultiObjectiveTrainer(test_config, device)

        # Create mock data loader
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, _, _ = data_loader.create_datasets()
        train_loader, _, _ = data_loader.create_dataloaders(train_dataset, train_dataset, train_dataset)

        # Generate episode data
        episode_data = trainer.generate_episode_data(train_loader, num_episodes=5)

        # Check structure
        assert isinstance(episode_data, dict)
        required_keys = ["states", "actions", "log_probs", "rewards", "values", "dones"]
        for key in required_keys:
            assert key in episode_data
            assert isinstance(episode_data[key], torch.Tensor)

    @pytest.mark.slow
    @pytest.mark.integration
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.end_run')
    def test_full_training_loop(self, mock_end_run, mock_log_metric, mock_log_params, mock_start_run, test_config, device, temp_dir):
        """Test complete training loop."""
        # Configure for fast test
        test_config.training.epochs = 2
        test_config.training.patience = 1
        test_config.evaluation.eval_frequency = 1
        test_config.data.batch_size = 4
        test_config.system.checkpoint.save_dir = str(temp_dir)

        trainer = MultiObjectiveTrainer(test_config, device)

        # Create data loaders
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, val_dataset, _ = data_loader.create_datasets()
        train_loader, val_loader, _ = data_loader.create_dataloaders(
            train_dataset, val_dataset, val_dataset
        )
        data_loader.setup_preprocessing(train_dataset)

        # Run training (should complete without errors)
        trainer.train(train_loader, val_loader)

        # Check that MLflow was called
        mock_end_run.assert_called_once()

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    def test_log_metrics(self, mock_log_metric, mock_log_params, mock_start_run, test_config, device):
        """Test metrics logging."""
        trainer = MultiObjectiveTrainer(test_config, device)

        test_metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "reward": 1.2
        }

        trainer._log_metrics(test_metrics, epoch=5)

        # Check that MLflow was called for each metric
        assert mock_log_metric.call_count == len(test_metrics)

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_early_stopping(self, mock_log_params, mock_start_run, test_config, device):
        """Test early stopping mechanism."""
        # Set very strict early stopping
        test_config.training.patience = 0
        trainer = MultiObjectiveTrainer(test_config, device)

        # Simulate no improvement
        trainer.best_reward = 1.0
        trainer.early_stopping_counter = 1  # Should trigger early stopping

        # Check early stopping condition
        assert trainer.early_stopping_counter >= test_config.training.patience
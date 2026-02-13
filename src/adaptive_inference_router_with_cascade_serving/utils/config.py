"""Configuration management for adaptive inference routing."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration."""

    router: Dict[str, Any] = Field(default_factory=dict)
    cascade: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training configuration."""

    rl: Dict[str, Any] = Field(default_factory=dict)
    objectives: Dict[str, float] = Field(default_factory=dict)
    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    optimizer: Dict[str, Any] = Field(default_factory=dict)
    scheduler: Dict[str, Any] = Field(default_factory=dict)

    @validator("epochs")
    def validate_epochs(cls, v: int) -> int:
        """Validate training epochs."""
        if v <= 0:
            raise ValueError("epochs must be positive")
        return v

    @validator("patience")
    def validate_patience(cls, v: int) -> int:
        """Validate early stopping patience."""
        if v < 0:
            raise ValueError("patience must be non-negative")
        return v


class DataConfig(BaseModel):
    """Data configuration."""

    dataset: str = "mlperf_inference"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    difficulty_features: list = Field(default_factory=list)
    sla_features: list = Field(default_factory=list)
    system_features: list = Field(default_factory=list)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    @validator("batch_size")
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    @validator("train_split", "val_split", "test_split")
    def validate_split(cls, v: float) -> float:
        """Validate data split ratios."""
        if not 0 <= v <= 1:
            raise ValueError("split ratios must be between 0 and 1")
        return v


class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    serving: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)
    sla: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    target_metrics: Dict[str, float] = Field(default_factory=dict)
    eval_frequency: int = 5
    eval_batch_size: int = 64
    num_eval_episodes: int = 100
    statistical_significance_alpha: float = 0.05

    @validator("eval_frequency", "eval_batch_size", "num_eval_episodes")
    def validate_positive(cls, v: int) -> int:
        """Validate positive integers."""
        if v <= 0:
            raise ValueError("value must be positive")
        return v


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""

    name: str = "adaptive_router_research"
    tags: list = Field(default_factory=list)
    mlflow: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """System configuration."""

    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    checkpoint: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)

    @validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        if v not in ["auto", "cpu", "cuda"]:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        return v


class Config(BaseModel):
    """Complete configuration for adaptive inference routing."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Loaded and validated configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
        ValueError: If configuration validation fails.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {e}") from e

    if config_dict is None:
        config_dict = {}

    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        config_path: Path to save configuration file.

    Raises:
        OSError: If unable to write to file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
    except OSError as e:
        raise OSError(f"Unable to save config to {config_path}: {e}") from e

    logger.info(f"Saved configuration to {config_path}")


def get_default_config_path() -> Path:
    """Get default configuration file path.

    Returns:
        Path to default configuration file.
    """
    return Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"


def setup_logging(config: Config) -> None:
    """Setup logging based on configuration.

    Args:
        config: Configuration containing logging settings.
    """
    log_config = config.experiment.logging

    # Create log directory
    log_dir = Path(log_config.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = getattr(logging, log_config.get("level", "INFO").upper())
    log_format = log_config.get(
        "log_format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "adaptive_router.log"),
        ]
    )

    logger.info(f"Logging configured with level {log_level}")


def setup_device(config: Config) -> str:
    """Setup and validate device configuration.

    Args:
        config: Configuration containing device settings.

    Returns:
        Resolved device string.
    """
    import torch

    device_config = config.system.device

    if device_config == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    else:
        device = device_config
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    return device


def setup_seeds(config: Config) -> None:
    """Setup random seeds for reproducibility.

    Args:
        config: Configuration containing seed settings.
    """
    import random
    import numpy as np
    import torch

    seed = config.system.seed

    # Set Python random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure deterministic behavior
    if config.system.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        torch.backends.cudnn.benchmark = config.system.benchmark

    logger.info(f"Random seeds set to {seed}")
    logger.info(f"Deterministic mode: {config.system.deterministic}")
    logger.info(f"CUDNN benchmark: {config.system.benchmark}")
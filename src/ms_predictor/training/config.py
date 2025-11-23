"""
Configuration dataclass for MS predictor.

This module provides dataclass definitions for configuration management.
When using Hydra, configs are loaded as OmegaConf DictConfig objects,
but these dataclasses serve as documentation and type hints.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    vocab_size: int = 22  # 20 amino acids + PAD + UNK
    hidden_dim: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    num_predictions: int = 100
    max_length: int = 50
    max_charge: int = 10
    dropout: float = 0.1
    activation: str = 'gelu'


@dataclass
class DataConfig:
    """Data configuration."""
    
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    max_mz: float = 2000.0
    top_k: int = 200
    use_dummy_data: bool = True  # Use dummy data if no real data available
    dummy_train_samples: int = 1000
    dummy_val_samples: int = 200
    use_parquet: bool = False  # Use Parquet format for data
    cache_dataframes: bool = False  # Cache dataframes in memory
    max_files: Optional[int] = None  # Maximum number of parquet files to load
    metadata_file: Optional[str] = None  # Path to metadata file


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Hungarian matching costs
    cost_mz: float = 1.0
    cost_intensity: float = 1.0
    cost_confidence: float = 1.0
    
    # Loss weights
    loss_mz_weight: float = 1.0
    loss_intensity_weight: float = 1.0
    loss_confidence_weight: float = 1.0
    background_confidence_weight: float = 0.1
    
    # Cosine similarity loss
    use_cosine_loss: bool = True
    cosine_loss_weight: float = 0.5  # Tunable scaler
    cosine_bin_size: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # For StepLR
    step_size: int = 10
    gamma: float = 0.1
    
    # For ReduceLROnPlateau
    patience: int = 5
    factor: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    num_epochs: int = 100
    gradient_clip: float = 1.0
    log_interval: int = 10
    val_interval: int = 1
    save_interval: int = 5
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = False
    
    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'
    mixed_precision: bool = True  # Use automatic mixed precision


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    confidence_threshold: float = 0.5
    max_mz: float = 2000.0
    batch_size: int = 64


@dataclass
class WandbConfig:
    """Wandb logging configuration."""
    
    enabled: bool = True
    project: str = 'ms-predictor'
    entity: Optional[str] = None
    mode: str = 'online'  # 'online', 'offline', or 'disabled'
    log_interval: int = 10  # Log every N steps


@dataclass
class Config:
    """Main configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Experiment settings
    experiment_name: str = 'ms_predictor'
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file (legacy support).
        
        Note: When using Hydra, this method is not needed as Hydra
        handles config loading. This is kept for backward compatibility.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
            experiment_name=config_dict.get('experiment_name', 'ms_predictor'),
            seed=config_dict.get('seed', 42)
        )
    
    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> 'Config':
        """
        Convert Hydra DictConfig to Config dataclass.
        
        This is useful when you need typed access to config values
        or when passing config to code that expects the Config dataclass.
        
        Args:
            cfg: Hydra DictConfig object
            
        Returns:
            Config object
        """
        # Convert OmegaConf to plain dict then to dataclass
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
            experiment_name=config_dict.get('experiment_name', 'ms_predictor'),
            seed=config_dict.get('seed', 42)
        )
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'loss': self.loss.__dict__,
            'optimizer': {
                k: v for k, v in self.optimizer.__dict__.items()
            },
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'wandb': self.wandb.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }
        
        # Convert tuples to lists for YAML serialization
        if 'betas' in config_dict['optimizer']:
            config_dict['optimizer']['betas'] = list(config_dict['optimizer']['betas'])
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


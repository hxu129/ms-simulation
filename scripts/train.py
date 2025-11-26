#!/usr/bin/env python
"""
Training script for MS spectrum predictor using Hydra configuration.

Usage examples:
  # Train with default config
  python train.py
  
  # Override specific parameters
  python train.py data.batch_size=64 model.hidden_dim=1024
  
  # Use config group variants
  python train.py experiment=small
  python train.py data=parquet
  
  # Combine multiple configs
  python train.py experiment=large data=parquet data.batch_size=32
  
  # Resume from checkpoint
  python train.py +resume_path=checkpoints/best_model.pt
"""

import torch
import numpy as np
import random
import os
import logging
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor.model.ms_predictor import MSPredictor, count_parameters
from ms_predictor.data.parquet_dataset import ParquetMSDataset, create_parquet_dataloaders, collate_fn
from ms_predictor.data.tokenizer import AminoAcidTokenizer
from ms_predictor.data.preprocessing import SpectrumPreprocessor
from ms_predictor.training.config import Config
from ms_predictor.training.trainer import Trainer


class TqdmFilter(logging.Filter):
    """Filter out tqdm progress bar output from logs."""
    
    def filter(self, record):
        # Filter out log records that contain tqdm-related messages
        # We look for common tqdm patterns in the message
        msg = record.getMessage()
        # Filter out lines that look like progress bars
        if any(pattern in msg for pattern in ['%|', 'it/s', 's/it', 'Epoch', '/s]']):
            # Check if it's from tqdm (simple heuristic)
            if hasattr(record, 'name') and 'tqdm' in record.name.lower():
                return False
        return True


def setup_logging(log_file: str = 'train.log'):
    """
    Set up logging to both file and console.
    
    File logging: saves all output except tqdm progress bars
    Console logging: shows everything including tqdm
    
    Args:
        log_file: Path to log file
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler - saves detailed logs without tqdm
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(TqdmFilter())  # Filter out tqdm from file
    logger.addHandler(file_handler)
    
    # Console handler - shows everything
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(cfg: DictConfig):
    """
    Create training and validation dataloaders.
    
    Only Parquet format is supported.
    
    Args:
        cfg: Hydra DictConfig object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = logging.getLogger(__name__)
    
    tokenizer = AminoAcidTokenizer()
    preprocessor = SpectrumPreprocessor(
        max_mz=cfg.data.max_mz,
        top_k=cfg.data.top_k,
        num_predictions=cfg.model.num_predictions
    )
    
    # Use Parquet dataset (only supported format)
    logger.info("Loading Parquet data")
    logger.info(f"Data directory: {cfg.data.train_data_path}")
    
    train_loader, val_loader, _ = create_parquet_dataloaders(
        data_dir=cfg.data.train_data_path,
        metadata_file=cfg.data.get('metadata_file', None),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=cfg.model.max_length,
        max_mz=cfg.data.max_mz,
        top_k=cfg.data.top_k,
        num_predictions=cfg.model.num_predictions,
        cache_dataframes=cfg.data.get('cache_dataframes', False),
        max_files=cfg.data.get('max_files', None)
    )
    
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object (DictConfig)
    """
    # Set up logging (saves to train.log in current output directory)
    logger = setup_logging('train.log')
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)
    
    # Get the original working directory (Hydra changes cwd to outputs/)
    original_cwd = hydra.utils.get_original_cwd()
    
    # Convert config to typed dataclass for better IDE support (optional)
    # config = Config.from_dictconfig(cfg)
    # For this implementation, we'll work directly with DictConfig
    
    logger.info(f"Experiment name: {cfg.experiment_name}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Original directory: {original_cwd}")
    
    # Set seed
    set_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = MSPredictor(
        vocab_size=cfg.model.vocab_size,
        hidden_dim=cfg.model.hidden_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_heads=cfg.model.num_heads,
        dim_feedforward=cfg.model.dim_feedforward,
        num_predictions=cfg.model.num_predictions,
        max_length=cfg.model.max_length,
        max_charge=cfg.model.max_charge,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation
    )
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    logger.info("\nCreating trainer...")
    # Convert DictConfig to Config dataclass for Trainer
    config = Config.from_dictconfig(cfg)
    
    # DEBUG: Check device configuration
    logger.info("="*80)
    logger.info("DEVICE CONFIGURATION CHECK:")
    logger.info(f"  Config device setting: {config.training.device}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  CUDA device count: {torch.cuda.device_count()}")
    logger.info("="*80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        raw_config=cfg  # Pass raw Hydra config for complete wandb logging
    )
    
    # DEBUG: Verify trainer device and model location
    logger.info("="*80)
    logger.info("TRAINER DEVICE VERIFICATION:")
    logger.info(f"  Trainer device: {trainer.device}")
    logger.info(f"  Model device: {next(trainer.model.parameters()).device}")
    logger.info(f"  Model is on CUDA: {next(trainer.model.parameters()).is_cuda}")
    logger.info("="*80)
    
    # Resume from checkpoint if specified
    resume_path = cfg.get('resume_path', None)
    if resume_path:
        # Handle relative paths from original working directory
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(original_cwd, resume_path)
        logger.info(f"\nResuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    
    # Train
    logger.info("\nStarting training...\n")
    trainer.train()
    
    logger.info("\nTraining complete!")
    logger.info(f"Outputs saved to: {os.getcwd()}")


if __name__ == '__main__':
    main()


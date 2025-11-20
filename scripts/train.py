#!/usr/bin/env python
"""
Training script for MS spectrum predictor.
"""

import argparse
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor.model.ms_predictor import MSPredictor, count_parameters
from ms_predictor.data.dataset import MSDataset, DummyMSDataset, collate_fn
from ms_predictor.data.hdf5_dataset import HDF5MSDataset, create_hdf5_dataloaders
from ms_predictor.data.tokenizer import AminoAcidTokenizer
from ms_predictor.data.preprocessing import SpectrumPreprocessor
from ms_predictor.training.config import Config
from ms_predictor.training.trainer import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(config: Config):
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    tokenizer = AminoAcidTokenizer()
    preprocessor = SpectrumPreprocessor(
        max_mz=config.data.max_mz,
        top_k=config.data.top_k,
        num_predictions=config.model.num_predictions
    )
    
    if config.data.use_dummy_data:
        print("Using dummy data for training (real data not available)")
        train_dataset = DummyMSDataset(
            num_samples=config.data.dummy_train_samples,
            max_length=config.model.max_length,
            num_predictions=config.model.num_predictions
        )
        val_dataset = DummyMSDataset(
            num_samples=config.data.dummy_val_samples,
            max_length=config.model.max_length,
            num_predictions=config.model.num_predictions
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    elif config.data.get('use_hdf5', False) or (config.data.train_data_path and os.path.isdir(config.data.train_data_path)):
        # Use HDF5 dataset
        print("Using HDF5 data from OBS")
        print(f"Data directory: {config.data.train_data_path}")
        
        train_loader, val_loader, _ = create_hdf5_dataloaders(
            data_dir=config.data.train_data_path,
            metadata_file=config.data.get('metadata_file', None),
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            max_length=config.model.max_length,
            cache_in_memory=config.data.get('cache_in_memory', False)
        )
    
    else:
        train_dataset = MSDataset(
            data_path=config.data.train_data_path,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            max_length=config.model.max_length,
            split='train'
        )
        val_dataset = MSDataset(
            data_path=config.data.val_data_path,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            max_length=config.model.max_length,
            split='val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train MS spectrum predictor')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Experiment name: {config.experiment_name}")
    
    # Set seed
    set_seed(config.seed)
    print(f"Random seed set to {config.seed}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = MSPredictor(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        num_heads=config.model.num_heads,
        dim_feedforward=config.model.dim_feedforward,
        num_predictions=config.model.num_predictions,
        max_length=config.model.max_length,
        max_charge=config.model.max_charge,
        dropout=config.model.dropout,
        activation=config.model.activation
    )
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...\n")
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()


#!/usr/bin/env python
"""
Training script for binned MS spectrum predictor using Hydra configuration.

Usage examples:
  # Train with binned config
  python train_binned.py --config-name binned_config
  
  # Override specific parameters
  python train_binned.py --config-name binned_config data.batch_size=64 model.hidden_dim=1024
  
  # Resume from checkpoint
  python train_binned.py --config-name binned_config +resume_path=checkpoints/best_model.pt
"""

import torch
import numpy as np
import random
import os
import logging
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor.model.binned_predictor import BinnedMSPredictor, count_parameters
from ms_predictor.loss.binned_cosine_loss import BinnedCosineLoss
from ms_predictor.data.parquet_dataset import create_parquet_dataloaders
from ms_predictor.data.tokenizer import AminoAcidTokenizer
from ms_predictor.data.preprocessing import SpectrumPreprocessor
from ms_predictor.data.data_prefetcher import DataPrefetcher

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(log_file: str = 'train_binned.log'):
    """Set up logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
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
    """Create training and validation dataloaders."""
    logger = logging.getLogger(__name__)
    
    # Check if data path is set
    if cfg.data.train_data_path is None:
        raise ValueError(
            "train_data_path is not set in the config file.\n"
            "Please set it using:\n"
            "  python scripts/train_binned.py --config-name binned_config data.train_data_path=/path/to/data"
        )
    
    # Check if data path exists
    if not os.path.exists(cfg.data.train_data_path):
        raise ValueError(
            f"Data path does not exist: {cfg.data.train_data_path}\n"
            "Please provide a valid path to your training data:\n"
            "  python scripts/train_binned.py --config-name binned_config data.train_data_path=/actual/path/to/data"
        )
    
    tokenizer = AminoAcidTokenizer()
    preprocessor = SpectrumPreprocessor(
        max_mz=cfg.data.max_mz,
        top_k=cfg.data.top_k,
        num_predictions=200  # Not used for binned model, but required by preprocessor
    )
    
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
        num_predictions=200,  # Not used for binned model
        cache_dataframes=cfg.data.get('cache_dataframes', False),
        max_files=cfg.data.get('max_files', None)
    )
    
    # Check if datasets are not empty
    if len(train_loader.dataset) == 0:
        raise ValueError(
            f"No training data found in {cfg.data.train_data_path}\n"
            "Make sure the directory contains .parquet files with MS/MS spectra data."
        )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler, epoch, cfg):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    # Use DataPrefetcher for faster data loading
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(DataPrefetcher(train_loader, device)):
        # Forward pass
        with autocast(enabled=use_amp):
            pred_binned = model(
                batch['sequence_tokens'],
                batch['sequence_mask'],
                batch['precursor_mz'],
                batch['charge']
            )
            
            loss = criterion(
                pred_binned,
                batch['target_mz'],
                batch['target_intensity'],
                batch['target_mask']
            )
        
        # Backward pass
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % cfg.training.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.update(1)
    
    pbar.close()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    for batch in DataPrefetcher(val_loader, device):
        with autocast(enabled=use_amp):
            pred_binned = model(
                batch['sequence_tokens'],
                batch['sequence_mask'],
                batch['precursor_mz'],
                batch['charge']
            )
            
            loss = criterion(
                pred_binned,
                batch['target_mz'],
                batch['target_intensity'],
                batch['target_mask']
            )
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def create_optimizer(model, cfg):
    """Create optimizer."""
    if cfg.optimizer.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps
        )
    elif cfg.optimizer.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, cfg, last_epoch=-1):
    """Create learning rate scheduler."""
    if cfg.optimizer.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_epochs,
            eta_min=cfg.optimizer.min_lr,
            last_epoch=last_epoch
        )
    elif cfg.optimizer.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.optimizer.step_size,
            gamma=cfg.optimizer.gamma,
            last_epoch=last_epoch
        )
    elif cfg.optimizer.scheduler.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.optimizer.patience,
            factor=cfg.optimizer.factor
        )
    else:
        scheduler = None
    
    return scheduler


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # List all checkpoint files
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
            epoch_num = int(f.replace('checkpoint_epoch_', '').replace('.pt', ''))
            checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, f)))
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and return the latest
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return the starting epoch."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']}, will start training from epoch {start_epoch}")
    return start_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="binned_config")
def main(cfg: DictConfig):
    """Main training function."""
    logger = setup_logging('train_binned.log')
    
    logger.info("=" * 80)
    logger.info("Binned Model Training Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)
    
    # Set seed
    set_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")
    
    # Device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("\nCreating binned model...")
    model = BinnedMSPredictor(
        vocab_size=cfg.model.vocab_size,
        hidden_dim=cfg.model.hidden_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_heads=cfg.model.num_heads,
        dim_feedforward=cfg.model.dim_feedforward,
        num_bins=cfg.model.num_bins,
        max_length=cfg.model.max_length,
        max_charge=cfg.model.max_charge,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation
    )
    model.to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = BinnedCosineLoss(
        weight=cfg.loss.cosine_loss_weight,
        bin_size=cfg.loss.cosine_bin_size,
        max_mz=cfg.data.max_mz,
        normalize_bins=cfg.loss.get('normalize_bins', True)
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, cfg)
    
    # Check for resume_path or find latest checkpoint
    start_epoch = 1
    resume_path = cfg.get('resume_path', None)
    auto_resume = cfg.get('auto_resume', False)
    
    if resume_path:
        # Explicit resume path provided
        if os.path.exists(resume_path):
            start_epoch = load_checkpoint(resume_path, model, optimizer, device)
        else:
            logger.warning(f"Resume path {resume_path} does not exist. Starting from scratch.")
    elif auto_resume:
        # Try to find latest checkpoint automatically
        latest_checkpoint = find_latest_checkpoint(cfg.training.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            logger.info("Auto-resume enabled, loading checkpoint...")
            start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, device)
        else:
            logger.info("No existing checkpoints found. Starting training from scratch.")
    else:
        logger.info("Auto-resume not enabled. Starting training from scratch.")
        latest_checkpoint = find_latest_checkpoint(cfg.training.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Note: Found checkpoint at {latest_checkpoint}. Use +auto_resume=true to resume.")
    
    # Create scheduler with last_epoch set correctly
    scheduler = create_scheduler(optimizer, cfg, last_epoch=start_epoch-2)
    
    # Mixed precision
    use_amp = cfg.training.mixed_precision and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    logger.info(f"Mixed precision training: {use_amp}")
    
    # Initialize wandb
    use_wandb = cfg.wandb.enabled and WANDB_AVAILABLE and cfg.wandb.mode != 'disabled'
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get('entity', None),
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            dir=cfg.wandb.get('dir', None),
            resume='allow'
        )
        wandb.watch(model, log='all', log_freq=100)
    
    # Training loop
    logger.info("\nStarting training...\n")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp, scaler, epoch, cfg
        )
        
        # Validate
        if epoch % cfg.training.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device, use_amp)
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(cfg.training.checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })
        
        # Save checkpoint
        if epoch % cfg.training.save_interval == 0:
            checkpoint_path = os.path.join(cfg.training.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if epoch % cfg.training.val_interval == 0 else train_loss)
            else:
                scheduler.step()
    
    logger.info("\nTraining complete!")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()


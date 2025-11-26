"""
Trainer for MS spectrum predictor.
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .config import Config
from ..model.ms_predictor import MSPredictor
from ..loss.set_loss import SetPredictionLoss
from ..loss.cosine_loss import CosineSimilarityLoss
from ..data.data_prefetcher import DataPrefetcher

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """
    Trainer for MS spectrum predictor.
    """
    
    def __init__(
        self,
        model: MSPredictor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
        device: Optional[torch.device] = None,
        raw_config: Optional[DictConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: MSPredictor model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration
            device: Device to use (defaults to config.training.device)
            raw_config: Optional raw Hydra DictConfig for complete wandb logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.raw_config = raw_config
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Device
        if device is None:
            device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(self.device)
        
        # Initialize wandb
        self.use_wandb = config.wandb.enabled and WANDB_AVAILABLE
        if self.use_wandb:
            if config.wandb.mode == 'disabled':
                self.use_wandb = False
            else:
                self._init_wandb()
        
        # Loss functions
        self.set_loss = SetPredictionLoss(
            cost_mz=config.loss.cost_mz,
            cost_intensity=config.loss.cost_intensity,
            cost_confidence=config.loss.cost_confidence,
            loss_mz_weight=config.loss.loss_mz_weight,
            loss_mz_l1_weight=config.loss.loss_mz_l1_weight,
            loss_intensity_weight=config.loss.loss_intensity_weight,
            loss_confidence_weight=config.loss.loss_confidence_weight,
            background_confidence_weight=config.loss.background_confidence_weight,
            temperature=config.loss.temperature
        )
        
        if config.loss.use_cosine_loss:
            self.cosine_loss = CosineSimilarityLoss(
                weight=config.loss.cosine_loss_weight,
                bin_size=config.loss.cosine_bin_size,
                max_mz=config.data.max_mz
            )
        else:
            self.cosine_loss = None
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = config.training.mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize wandb for experiment tracking."""
        # Count model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Prepare config dict for wandb
        if self.raw_config is not None:
            # Use complete runtime configuration from Hydra
            wandb_config = OmegaConf.to_container(self.raw_config, resolve=True)
            # Add model parameter counts
            if 'model' not in wandb_config:
                wandb_config['model'] = {}
            wandb_config['model']['num_parameters'] = num_params
            wandb_config['model']['num_trainable_parameters'] = num_trainable
        else:
            # Fallback to manual config selection (for backward compatibility)
            wandb_config = {
                'experiment_name': self.config.experiment_name,
                'model': {
                    'vocab_size': self.config.model.vocab_size,
                    'hidden_dim': self.config.model.hidden_dim,
                    'num_encoder_layers': self.config.model.num_encoder_layers,
                    'num_decoder_layers': self.config.model.num_decoder_layers,
                    'num_heads': self.config.model.num_heads,
                    'dim_feedforward': self.config.model.dim_feedforward,
                    'num_predictions': self.config.model.num_predictions,
                    'max_length': self.config.model.max_length,
                    'max_charge': self.config.model.max_charge,
                    'dropout': self.config.model.dropout,
                    'activation': self.config.model.activation,
                    'num_parameters': num_params,
                    'num_trainable_parameters': num_trainable,
                },
                'optimizer': {
                    'optimizer': self.config.optimizer.optimizer,
                    'learning_rate': self.config.optimizer.learning_rate,
                    'weight_decay': self.config.optimizer.weight_decay,
                    'scheduler': self.config.optimizer.scheduler,
                },
                'loss': {
                    'loss_mz_weight': self.config.loss.loss_mz_weight,
                    'loss_mz_l1_weight': self.config.loss.loss_mz_l1_weight,
                    'loss_intensity_weight': self.config.loss.loss_intensity_weight,
                    'loss_confidence_weight': self.config.loss.loss_confidence_weight,
                    'background_confidence_weight': self.config.loss.background_confidence_weight,
                    'use_cosine_loss': self.config.loss.use_cosine_loss,
                    'cosine_loss_weight': self.config.loss.cosine_loss_weight,
                },
                'training': {
                    'num_epochs': self.config.training.num_epochs,
                    'batch_size': self.config.data.batch_size,
                    'gradient_clip': self.config.training.gradient_clip,
                    'mixed_precision': self.config.training.mixed_precision,
                }
            }
        
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.experiment_name,
            config=wandb_config,
            mode=self.config.wandb.mode,
        )
        
        # Watch model (log gradients and parameters)
        wandb.watch(self.model, log='all', log_freq=self.config.wandb.log_interval * 10)
        
        self.logger.info(f"Wandb initialized: project={self.config.wandb.project}, mode={self.config.wandb.mode}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        cfg = self.config.optimizer
        
        if cfg.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        cfg = self.config.optimizer
        
        if cfg.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs - cfg.warmup_epochs,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.gamma
            )
        elif cfg.scheduler.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=cfg.patience,
                factor=cfg.factor,
                min_lr=cfg.min_lr
            )
        elif cfg.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_set_loss = 0.0
        total_mz_loss = 0.0
        total_mz_l1_loss = 0.0
        total_intensity_loss = 0.0
        total_confidence_matched_loss = 0.0
        total_confidence_background_loss = 0.0
        total_cosine_loss = 0.0
        num_batches = 0
        
        # Use DataPrefetcher for async data loading (batch already on GPU)
        prefetcher = DataPrefetcher(self.train_loader, self.device)
        pbar = tqdm(prefetcher, desc=f"Epoch {self.epoch + 1}", total=len(self.train_loader))
        
        for batch_idx, batch in enumerate(pbar):
            # Batch is already on GPU thanks to DataPrefetcher!
            # No need for: batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                pred_mz, pred_intensity, pred_confidence_logits = self.model(
                    batch['sequence_tokens'],
                    batch['sequence_mask'],
                    batch['precursor_mz'],
                    batch['charge']
                )
                
                # Compute set prediction loss
                loss_dict = self.set_loss(
                    pred_mz, pred_intensity, pred_confidence_logits,
                    batch['target_mz'], batch['target_intensity'], batch['target_mask']
                )
                
                loss = loss_dict['loss']
                
                # Add cosine similarity loss if enabled
                cosine_loss_val = 0.0
                if self.cosine_loss is not None:
                    matched_indices = loss_dict.get('matched_indices', None)
                    cosine_loss_val = self.cosine_loss(
                        pred_mz, pred_intensity,
                        batch['target_mz'], batch['target_intensity'], batch['target_mask'],
                        matched_indices=matched_indices
                    )
                    loss = loss + cosine_loss_val
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_set_loss += loss_dict['loss'].item()
            total_mz_loss += loss_dict['loss_mz'].item()
            total_mz_l1_loss += loss_dict['loss_mz_l1'].item()
            total_intensity_loss += loss_dict['loss_intensity'].item()
            total_confidence_matched_loss += loss_dict['loss_confidence_matched'].item()
            total_confidence_background_loss += loss_dict['loss_confidence_background'].item()
            if self.cosine_loss is not None:
                total_cosine_loss += cosine_loss_val.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and self.global_step % self.config.wandb.log_interval == 0:
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/set_loss': loss_dict['loss'].item(),
                    'train/mz_loss': loss_dict['loss_mz'].item(),
                    'train/mz_l1_loss': loss_dict['loss_mz_l1'].item(),
                    'train/intensity_loss': loss_dict['loss_intensity'].item(),
                    'train/confidence_loss_matched': loss_dict['loss_confidence_matched'].item(),
                    'train/confidence_loss_background': loss_dict['loss_confidence_background'].item(),
                    'train/cosine_loss': cosine_loss_val.item() if self.cosine_loss else 0.0,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                }, step=self.global_step)
            
            # Update progress bar
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                pbar.set_postfix({
                    'loss': total_loss / num_batches,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        return {
            'loss': total_loss / num_batches,
            'set_loss': total_set_loss / num_batches,
            'mz_loss': total_mz_loss / num_batches,
            'mz_l1_loss': total_mz_l1_loss / num_batches,
            'intensity_loss': total_intensity_loss / num_batches,
            'confidence_loss_matched': total_confidence_matched_loss / num_batches,
            'confidence_loss_background': total_confidence_background_loss / num_batches,
            'cosine_loss': total_cosine_loss / num_batches if self.cosine_loss else 0.0
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of average losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_set_loss = 0.0
        total_mz_loss = 0.0
        total_mz_l1_loss = 0.0
        total_intensity_loss = 0.0
        total_confidence_matched_loss = 0.0
        total_confidence_background_loss = 0.0
        total_cosine_loss = 0.0
        num_batches = 0
        
        # Use DataPrefetcher for async data loading (batch already on GPU)
        prefetcher = DataPrefetcher(self.val_loader, self.device)
        for batch in tqdm(prefetcher, desc="Validation", total=len(self.val_loader)):
            # Batch is already on GPU thanks to DataPrefetcher!
            
            pred_mz, pred_intensity, pred_confidence = self.model(
                batch['sequence_tokens'],
                batch['sequence_mask'],
                batch['precursor_mz'],
                batch['charge']
            )
            
            loss_dict = self.set_loss(
                pred_mz, pred_intensity, pred_confidence,
                batch['target_mz'], batch['target_intensity'], batch['target_mask']
            )
            
            loss = loss_dict['loss']
            
            cosine_loss_val = 0.0
            if self.cosine_loss is not None:
                cosine_loss_val = self.cosine_loss(
                    pred_mz, pred_intensity,
                    batch['target_mz'], batch['target_intensity'], batch['target_mask']
                )
                loss = loss + cosine_loss_val
            
            total_loss += loss.item()
            total_set_loss += loss_dict['loss'].item()
            total_mz_loss += loss_dict['loss_mz'].item()
            total_mz_l1_loss += loss_dict['loss_mz_l1'].item()
            total_intensity_loss += loss_dict['loss_intensity'].item()
            total_confidence_matched_loss += loss_dict['loss_confidence_matched'].item()
            total_confidence_background_loss += loss_dict['loss_confidence_background'].item()
            if self.cosine_loss is not None:
                total_cosine_loss += cosine_loss_val.item()
            num_batches += 1
        
        # Calculate averages
        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_set_loss': total_set_loss / num_batches,
            'val_mz_loss': total_mz_loss / num_batches,
            'val_mz_l1_loss': total_mz_l1_loss / num_batches,
            'val_intensity_loss': total_intensity_loss / num_batches,
            'val_confidence_loss_matched': total_confidence_matched_loss / num_batches,
            'val_confidence_loss_background': total_confidence_background_loss / num_batches,
            'val_cosine_loss': total_cosine_loss / num_batches if self.cosine_loss else 0.0
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/total_loss': avg_metrics['val_loss'],
                'val/set_loss': avg_metrics['val_set_loss'],
                'val/mz_loss': avg_metrics['val_mz_loss'],
                'val/mz_l1_loss': avg_metrics['val_mz_l1_loss'],
                'val/intensity_loss': avg_metrics['val_intensity_loss'],
                'val/confidence_loss_matched': avg_metrics['val_confidence_loss_matched'],
                'val/confidence_loss_background': avg_metrics['val_confidence_loss_background'],
                'val/cosine_loss': avg_metrics['val_cosine_loss'],
                'epoch': self.epoch,
            }, step=self.global_step)
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded from {filename}")
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            self.logger.info(f"  Train loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"    - Set loss: {train_metrics['set_loss']:.4f}")
            self.logger.info(f"    - M/Z loss (contrastive): {train_metrics['mz_loss']:.4f}")
            self.logger.info(f"    - M/Z loss (L1): {train_metrics['mz_l1_loss']:.4f}")
            self.logger.info(f"    - Intensity loss: {train_metrics['intensity_loss']:.4f}")
            self.logger.info(f"    - Confidence loss (matched): {train_metrics['confidence_loss_matched']:.4f}")
            self.logger.info(f"    - Confidence loss (background): {train_metrics['confidence_loss_background']:.4f}")
            if self.cosine_loss is not None:
                self.logger.info(f"    - Cosine loss: {train_metrics['cosine_loss']:.4f}")
            
            # Log epoch summary to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_set_loss': train_metrics['set_loss'],
                    'train/epoch_mz_loss': train_metrics['mz_loss'],
                    'train/epoch_mz_l1_loss': train_metrics['mz_l1_loss'],
                    'train/epoch_intensity_loss': train_metrics['intensity_loss'],
                    'train/epoch_confidence_loss_matched': train_metrics['confidence_loss_matched'],
                    'train/epoch_confidence_loss_background': train_metrics['confidence_loss_background'],
                    'train/epoch_cosine_loss': train_metrics['cosine_loss'],
                }, step=self.global_step)
            
            # Validate
            if (epoch + 1) % self.config.training.val_interval == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}")
                    self.logger.info(f"    - Set loss: {val_metrics['val_set_loss']:.4f}")
                    self.logger.info(f"    - M/Z loss (contrastive): {val_metrics['val_mz_loss']:.4f}")
                    self.logger.info(f"    - M/Z loss (L1): {val_metrics['val_mz_l1_loss']:.4f}")
                    self.logger.info(f"    - Intensity loss: {val_metrics['val_intensity_loss']:.4f}")
                    self.logger.info(f"    - Confidence loss (matched): {val_metrics['val_confidence_loss_matched']:.4f}")
                    self.logger.info(f"    - Confidence loss (background): {val_metrics['val_confidence_loss_background']:.4f}")
                    if self.cosine_loss is not None:
                        self.logger.info(f"    - Cosine loss: {val_metrics['val_cosine_loss']:.4f}")
                    
                    # Check for improvement
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.epochs_without_improvement = 0
                        
                        # Save best model
                        best_path = os.path.join(
                            self.config.training.checkpoint_dir,
                            f"{self.config.experiment_name}_best.pt"
                        )
                        self.save_checkpoint(best_path)
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Early stopping
                    if (self.config.training.early_stopping and
                        self.epochs_without_improvement >= self.config.training.early_stopping_patience):
                        self.logger.info(f"Early stopping after {epoch + 1} epochs")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                if not self.config.training.save_best_only:
                    checkpoint_path = os.path.join(
                        self.config.training.checkpoint_dir,
                        f"{self.config.experiment_name}_epoch_{epoch + 1}.pt"
                    )
                    self.save_checkpoint(checkpoint_path)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    # Apply warmup if needed
                    if epoch < self.config.optimizer.warmup_epochs:
                        warmup_factor = (epoch + 1) / self.config.optimizer.warmup_epochs
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.config.optimizer.learning_rate * warmup_factor
                    else:
                        self.scheduler.step()
        
        self.logger.info("Training complete!")
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
            self.logger.info("Wandb run finished")


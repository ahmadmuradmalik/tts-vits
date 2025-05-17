import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from pathlib import Path
import os
import logging
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import json
import numpy as np

from ..models.vits import VITS
from ..data.dataset import TTSDataset, AudioProcessor, TextProcessor
from ..utils.loss import compute_loss
from ..utils.visualization import visualize_training_step

logger = logging.getLogger(__name__)

class TTSTrainer:
    """Handles training and inference for the VITS model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.sample_dir = self.output_dir / 'samples'
        for dir_path in [self.checkpoint_dir, self.log_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=config['data']['sample_rate'],
            n_mels=config['data']['n_mels']
        )
        self.text_processor = TextProcessor(config.get('vocab_path'))
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.steps = 0
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log configuration to tensorboard"""
        config_str = json.dumps(self.config, indent=2)
        self.writer.add_text('config', config_str)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Log learning rate changes
            if new_lr != old_lr:
                logger.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            
            # Log epoch results
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {new_lr:.6f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(is_best)
        
        # Save final checkpoint
        self.save_checkpoint(is_best)
        
        # Log training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        self.writer.close()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_adv_loss = 0
        epoch_fm_loss = 0
        epoch_logdet_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['training'].get('fp16_run', False)):
                y = {
                    'mel_spec': batch['mel_spec'],
                    'audio_lengths': batch['audio_lengths'],
                    'audio': batch['audio'],
                }
                output = self.model(
                    batch['text'],
                    batch['text_lengths'],
                    y,
                    batch['audio_lengths']
                )
                loss, loss_dict = compute_loss(
                    output,
                    batch['mel_spec'],
                    batch['audio_lengths']
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('grad_clip_thresh', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update progress
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['recon_loss']
            epoch_kl_loss += loss_dict['kl_loss']
            epoch_adv_loss += loss_dict['adv_loss']
            epoch_fm_loss += loss_dict['fm_loss']
            epoch_logdet_loss += loss_dict['logdet_loss']
            self.steps += 1
            
            # Log training progress
            if self.steps % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.steps)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'],
                                     self.steps)
                
                # Log model parameters
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'params/{name}', param.data, self.steps)
                        self.writer.add_histogram(f'grads/{name}', param.grad.data, self.steps)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
            f"Recon: {epoch_recon_loss/len(train_loader):.4f}, "
            f"KL: {epoch_kl_loss/len(train_loader):.4f}, "
            f"Adv: {epoch_adv_loss/len(train_loader):.4f}, "
            f"FM: {epoch_fm_loss/len(train_loader):.4f}, "
            f"Logdet: {epoch_logdet_loss/len(train_loader):.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config['training'].get('fp16_run', False)):
                    output = self.model(
                        batch['text'],
                        batch['text_lengths'],
                        batch['mel_spec'],
                        batch['audio_lengths']
                    )
                    loss, loss_dict = compute_loss(
                        output,
                        batch['audio'],
                        batch['audio_lengths']
                    )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'steps': self.steps,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss: {self.best_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint['best_loss']
        self.steps = checkpoint['steps']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def synthesize(self, text: str, output_path: str):
        """Generate speech from text."""
        self.model.eval()
        
        # Process text
        text_tensor = self.text_processor.text_to_sequence(text)
        text_length = torch.tensor([len(text_tensor)], device=self.device)
        text_tensor = text_tensor.unsqueeze(0).to(self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            output = self.model(text_tensor, text_length)
            mel_spec = output['y_hat']
        
        # Convert mel-spectrogram to audio (requires vocoder)
        # This is a placeholder - you'll need to implement the vocoder
        audio = torch.randn(1, mel_spec.shape[2] * 256)  # Placeholder
        
        # Save audio
        torchaudio.save(output_path, audio.cpu(), self.config['data']['sample_rate']) 
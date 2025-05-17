import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
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
import csv

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
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01,
            betas=(0.8, 0.99)
        )
        
        # Initialize learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['epochs'],
            steps_per_epoch=config['training'].get('steps_per_epoch', 1000),
            pct_start=0.1,  # 10% of training for warmup
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1e4  # Final lr = max_lr/10000
        )
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler(enabled=config['training'].get('fp16_run', True))
        
        # Training state
        self.current_epoch = 0
        self.steps = 0
        self.best_loss = float('inf')
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # CSV Logging Setup
        self.epoch_log_path = self.log_dir / 'epoch_metrics.csv'
        self._initialize_epoch_log()
        
        # Load checkpoint if exists (this will be handled more explicitly by main.py)
        # self.load_checkpoint() # Initial call can be removed if main.py handles it
    
    def _initialize_epoch_log(self):
        """Initializes the epoch log CSV file with headers if it doesn't exist."""
        if not self.epoch_log_path.exists():
            with open(self.epoch_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'epoch', 'train_loss_total', 'val_loss_total', 'learning_rate',
                    'train_recon_loss', 'train_kl_loss', 'train_adv_loss', 
                    'train_fm_loss', 'train_logdet_loss', 'train_unclamped_total_loss'
                    # Add val individual losses if/when validate() returns them
                ]
                writer.writerow(headers)
    
    def _log_epoch_to_csv(self, epoch_data: Dict):
        """Appends a row of epoch data to the CSV log file."""
        # Ensure all headers are present in epoch_data, fill with None if not
        # This makes it robust if some keys are missing in some calls
        headers = [
            'epoch', 'train_loss_total', 'val_loss_total', 'learning_rate',
            'train_recon_loss', 'train_kl_loss', 'train_adv_loss', 
            'train_fm_loss', 'train_logdet_loss', 'train_unclamped_total_loss'
        ]
        row = [epoch_data.get(h) for h in headers]
        with open(self.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
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
            
            train_total_loss, train_loss_breakdown = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                logger.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            
            self.writer.add_scalar('epoch/train_loss_total', train_total_loss, epoch)
            self.writer.add_scalar('epoch/val_loss_total', val_loss, epoch)
            self.writer.add_scalar('epoch/learning_rate', new_lr, epoch)
            for key, value in train_loss_breakdown.items(): # Log train loss breakdown to TensorBoard
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            logger.info(
                f"Epoch {epoch} - Overall Summary - Train Total Loss: {train_total_loss:.4f}, "
                f"Val Total Loss: {val_loss:.4f}, LR: {new_lr:.6f}"
            )
            
            # Log to CSV
            epoch_csv_data = {
                'epoch': epoch,
                'train_loss_total': train_total_loss,
                'val_loss_total': val_loss,
                'learning_rate': new_lr,
                'train_recon_loss': train_loss_breakdown.get('recon_loss'),
                'train_kl_loss': train_loss_breakdown.get('kl_loss'),
                'train_adv_loss': train_loss_breakdown.get('adv_loss'),
                'train_fm_loss': train_loss_breakdown.get('fm_loss'),
                'train_logdet_loss': train_loss_breakdown.get('logdet_loss'),
                'train_unclamped_total_loss': train_loss_breakdown.get('unclamped_total_loss')
            }
            self._log_epoch_to_csv(epoch_csv_data)
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % self.config['training']['save_interval'] == 0 or epoch == self.config['training']['epochs'] - 1:
                self.save_checkpoint(is_best)
            
            self.run_synthesis_test(epoch)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        self.writer.close()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        epoch_total_loss = 0
        epoch_metrics = {
            'recon_loss': 0.0, 'kl_loss': 0.0, 'adv_loss': 0.0,
            'fm_loss': 0.0, 'logdet_loss': 0.0, 'unclamped_total_loss': 0.0
        }
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        # Initialize gradient accumulation
        accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            # Determine device type for autocast
            autocast_device_type = self.device.type # 'cuda', 'mps', 'cpu'
            # Enable autocast only if fp16_run is true AND device is cuda or mps
            autocast_enabled = self.config['training'].get('fp16_run', False) and autocast_device_type in ['cuda', 'mps']

            # Forward pass with mixed precision
            with autocast(device_type=autocast_device_type, dtype=torch.float16, enabled=autocast_enabled):
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
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('grad_clip_thresh', 1.0)
                )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update learning rate
                self.scheduler.step()
            
            epoch_total_loss += loss.item() * accumulation_steps
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key] += loss_dict[key]
            self.steps += 1
            
            # Log training progress
            if self.steps % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item() * accumulation_steps, self.steps)
                self.writer.add_scalar('train/learning_rate', 
                                     self.scheduler.get_last_lr()[0],
                                     self.steps)
                self.writer.add_scalar('train/recon_loss', loss_dict['recon_loss'], self.steps)
                self.writer.add_scalar('train/kl_loss', loss_dict['kl_loss'], self.steps)
                self.writer.add_scalar('train/adv_loss', loss_dict['adv_loss'], self.steps)
                self.writer.add_scalar('train/fm_loss', loss_dict['fm_loss'], self.steps)
                self.writer.add_scalar('train/logdet_loss', loss_dict['logdet_loss'], self.steps)
                if 'unclamped_total_loss' in loss_dict:
                    self.writer.add_scalar('train/unclamped_total_loss', loss_dict['unclamped_total_loss'], self.steps)
                
                # Log model parameters
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'params/{name}', param.data, self.steps)
                        self.writer.add_histogram(f'grads/{name}', param.grad.data, self.steps)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}",
                'adv': f"{loss_dict['adv_loss']:.4f}",
                'fm': f"{loss_dict['fm_loss']:.4f}",
                'logdet': f"{loss_dict['logdet_loss']:.4f}",
                'unclamped': f"{loss_dict.get('unclamped_total_loss', float('nan')):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_epoch_total_loss = epoch_total_loss / num_batches
        avg_epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        logger.info(
            f"Epoch {epoch} - Train Summary - Total Loss: {avg_epoch_total_loss:.4f}, " +
            f"Recon: {avg_epoch_metrics['recon_loss']:.4f}, KL: {avg_epoch_metrics['kl_loss']:.4f}, " +
            f"Adv: {avg_epoch_metrics['adv_loss']:.4f}, FM: {avg_epoch_metrics['fm_loss']:.4f}, " +
            f"Logdet: {avg_epoch_metrics['logdet_loss']:.4f}, Unclamped: {avg_epoch_metrics['unclamped_total_loss']:.4f}, " +
            f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
        )
        # Return all average training losses for this epoch
        return avg_epoch_total_loss, avg_epoch_metrics
    
    def validate(self, val_loader: DataLoader):
        """Validate model with mixed precision"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda', dtype=torch.float16, enabled=self.config['training'].get('fp16_run', True)):
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
    
    def load_checkpoint(self, path_to_checkpoint_file: Optional[str] = None):
        """Load model checkpoint from a specific path or the default 'latest.pt'"""
        if path_to_checkpoint_file:
            checkpoint_path = Path(path_to_checkpoint_file)
            logger.info(f"Attempting to load checkpoint from specified path: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / 'latest.pt'
            logger.info(f"Attempting to load latest checkpoint from: {checkpoint_path}")

        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.current_epoch = checkpoint['current_epoch'] + 1 # Start next epoch
                self.best_loss = checkpoint['best_loss']
                self.steps = checkpoint['steps']
                # Ensure config compatibility if needed, though VITS config is part of checkpoint
                # For example, you might want to log if loaded config differs from current run config
                # self.config = checkpoint['config'] # Or merge/validate configs
                logger.info(f"Successfully loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.current_epoch}, step {self.steps}.")
            except Exception as e:
                logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
                logger.error("Starting from scratch.")
        else:
            logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    
    def synthesize(self, text: str, output_filename_prefix: str):
        """Generate mel-spectrogram from text and save it."""
        self.model.eval()
        
        text_tensor = self.text_processor.text_to_sequence(text)
        text_length = torch.tensor([len(text_tensor)], device=self.device)
        text_tensor = text_tensor.unsqueeze(0).to(self.device)
        
        output_mel_path = self.sample_dir / f"{output_filename_prefix}.mel.pt"

        try:
            with torch.no_grad():
                # Use autocast for consistency if used in training, but ensure model is on correct device
                autocast_device_type = self.device.type
                autocast_enabled = self.config['training'].get('fp16_run', False) and autocast_device_type in ['cuda', 'mps']
                with autocast(device_type=autocast_device_type, dtype=torch.float16, enabled=autocast_enabled):
                    model_output = self.model(text_tensor, text_length) # y is None, so inference path
                
                if 'y_hat' not in model_output or model_output['y_hat'] is None:
                    logger.error(f"Could not find 'y_hat' in model output during synthesis for text: {text}")
                    return

                mel_spec = model_output['y_hat'].squeeze(0).cpu() # Remove batch dim, move to CPU
            
            torch.save(mel_spec, output_mel_path)
            logger.info(f"Saved synthesized mel-spectrogram to: {output_mel_path}")

            # Optional: Plot and save mel-spectrogram as an image
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower', interpolation='none')
            # plt.colorbar()
            # plt.title(f"Synthesized Mel: {text[:50]}...")
            # plt.xlabel("Frames")
            # plt.ylabel("Mel Channels")
            # plt.tight_layout()
            # plt.savefig(self.sample_dir / f"{output_filename_prefix}.mel.png")
            # plt.close()

        except Exception as e:
            logger.error(f"Error during synthesis for text '{text}': {e}")
        finally:
            self.model.train() # Ensure model is back in train mode if it was set to eval

    def run_synthesis_test(self, epoch: int):
        """Runs synthesis for a few test sentences at the end of an epoch."""
        logger.info(f"Running synthesis test for epoch {epoch}...")
        test_sentences = self.config.get('synthesis_test_sentences', [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog."
        ])
        if not test_sentences:
            logger.info("No test sentences provided for synthesis.")
            return

        for i, text in enumerate(test_sentences):
            filename_prefix = f"epoch_{epoch:04d}_sample_{i:02d}"
            self.synthesize(text, filename_prefix) 
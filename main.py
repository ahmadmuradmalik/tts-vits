import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
import logging
import argparse
from typing import Optional

from src.models.vits import VITS
from src.data.dataset import TTSDataset
from src.training.trainer import TTSTrainer
from src.utils.config import get_default_config, load_config, save_config
from src.data.processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VITS TTS model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    return parser.parse_args()

def setup_directories(output_dir: str) -> None:
    """Create necessary directories"""
    dirs = ['checkpoints', 'logs', 'samples']
    for dir_name in dirs:
        (Path(output_dir) / dir_name).mkdir(parents=True, exist_ok=True)

def main():
    """Main training script"""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
        save_config(config, os.path.join(args.output_dir, 'config.yaml'))
    
    # Update config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = VITS(
        vocab_size=config['data']['vocab_size'],
        hidden_channels=config['model']['hidden_channels'],
        filter_channels=config['model']['filter_channels'],
        filter_kernel_size=config['model']['filter_kernel_size'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        kernel_size=config['model']['kernel_size'],
        dilation_rate=config['model']['dilation_rate'],
        n_flows=config['model']['n_flows'],
        n_layers_flow=config['model']['n_layers_flow']
    ).to(device)
    
    # Create datasets
    train_dataset = TTSDataset(
        args.data_dir,
        config,
        split='train'
    )
    val_dataset = TTSDataset(
        os.path.join(os.path.dirname(args.data_dir), 'val'),
        config,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TTSDataset.collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TTSDataset.collate_fn,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = TTSTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()

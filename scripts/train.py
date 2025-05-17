import argparse
import logging
import os
import torch
from pathlib import Path
import yaml

from src.models.vits import VITS
from src.training.trainer import TTSTrainer
from src.data.dataset import TTSDataset
from src.utils.config import get_default_config, save_config
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VITS model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, output_dir / 'config.yaml')
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
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
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = TTSTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=str(output_dir)
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 
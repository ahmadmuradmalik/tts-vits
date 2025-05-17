import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.models.vits import VITS
from src.data.dataset import TTSDataset
from src.training.trainer import TTSTrainer
from src.utils.config import get_default_config, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Test VITS training with small dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/test', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get default config and modify for testing
    config = get_default_config()
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['log_interval'] = 10
    config['training']['eval_interval'] = 50
    config['training']['save_interval'] = 100
    
    # Save test config
    save_config(config, output_dir / 'config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Create datasets with limited samples
    train_dataset = TTSDataset(
        args.data_dir,
        config,
        split='train',
        max_samples=args.max_samples
    )
    val_dataset = TTSDataset(
        args.data_dir,
        config,
        split='val',
        max_samples=args.max_samples // 5
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=TTSDataset.collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=TTSDataset.collate_fn,
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
    logger.info("Starting test training...")
    trainer.train(train_loader, val_loader)
    logger.info("Test training completed!")

if __name__ == '__main__':
    main() 
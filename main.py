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
from src.utils.logging_utils import setup_logging
from src.utils.plotting import generate_plots

# Remove basicConfig, setup_logging will handle configuration.
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__) # Get logger after setup_logging is called in main

def parse_args():
    parser = argparse.ArgumentParser(description='Train VITS TTS model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to output directory')
    # parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file') # This was replaced by resume_checkpoint_path
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to resume checkpoint file')
    return parser.parse_args()

def setup_directories(output_dir: str):
    """Create necessary directories"""
    # Convert to Path object for consistency
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    (output_dir_path / 'checkpoints').mkdir(exist_ok=True)
    (output_dir_path / 'logs').mkdir(exist_ok=True)
    (output_dir_path / 'samples').mkdir(exist_ok=True)

def main():
    args = parse_args()

    # Setup directories first, so log file can be placed correctly
    setup_directories(args.output_dir)

    # Setup logging to file and console
    # The log file will be in the output_dir/logs, e.g., outputs/logs/training.log
    log_file_path = Path(args.output_dir) / 'logs' / 'training.log'
    setup_logging(log_file_path)
    # Now that logging is set up, we can get the logger for this module
    # global logger # Not needed if logger is obtained after setup
    # logger = logging.getLogger(__name__) # Already defined at module level, will use new config

    logger.info("Application started.")
    logger.info(f"Arguments: {args}")
    
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = get_default_config()
        # Ensure output_dir for config saving is a Path object
        save_config(config, Path(args.output_dir) / 'config.yaml')
        logger.info(f"Using default configuration, saved to {Path(args.output_dir) / 'config.yaml'}")
    
    logger.info(f"Effective configuration: {config}")

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
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
    logger.info("Model initialized.")
    
    train_dataset = TTSDataset(
        args.data_dir,
        config,
        split='train'
    )
    val_dataset_path = Path(args.data_dir).parent / 'val' # Construct val path robustly
    val_dataset = TTSDataset(
        str(val_dataset_path), # TTSDataset expects string path
        config,
        split='val'
    )
    logger.info("Datasets created.")
    
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TTSDataset.collate_fn,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TTSDataset.collate_fn,
        pin_memory=pin_memory
    )
    logger.info("Dataloaders created.")
    
    trainer = TTSTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    logger.info("Trainer initialized.")
    
    if args.resume_checkpoint_path:
        trainer.load_checkpoint(args.resume_checkpoint_path)
    else:
        trainer.load_checkpoint()
    
    try:
        logger.info("Starting training process...")
        trainer.train(train_loader, val_loader)
        logger.info("Training process completed.")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state as a safety checkpoint...")
        trainer.save_checkpoint(is_best=False)
        logger.info("Safety checkpoint saved. Exiting.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during training: {e}")
    finally:
        logger.info("Attempting to generate plots from training logs...")
        generate_plots(str(Path(args.output_dir) / 'logs'))
        logger.info("Plot generation process finished.")

if __name__ == '__main__':
    main()

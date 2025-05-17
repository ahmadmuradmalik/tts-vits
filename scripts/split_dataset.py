import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_dataset(metadata_path, output_dir, train_ratio=0.9, val_ratio=0.05):
    """Split dataset into train/val/test sets."""
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split dataset
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split metadata
    train_df.to_csv(Path(output_dir) / 'train' / 'metadata.csv', index=False)
    val_df.to_csv(Path(output_dir) / 'val' / 'metadata.csv', index=False)
    test_df.to_csv(Path(output_dir) / 'test' / 'metadata.csv', index=False)
    
    # Log statistics
    logger.info(f"Dataset split statistics:")
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--metadata', type=str, required=True,
                      help='Path to metadata.csv file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for split datasets')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split dataset
    split_dataset(args.metadata, args.output_dir)
    
    logger.info("Dataset splitting completed successfully!")

if __name__ == '__main__':
    main() 
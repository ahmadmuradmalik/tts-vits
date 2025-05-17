import pandas as pd
import os
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_metadata_paths(metadata_path: str, output_path: str, data_dir: str):
    """
    Fix the audio paths in the metadata file.
    
    Args:
        metadata_path: Path to the input metadata file
        output_path: Path to save the fixed metadata file
        data_dir: Base directory containing the audio files
    """
    logger.info(f"Reading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Get the base directory for audio files
    data_dir = Path(data_dir)
    
    # Fix audio paths
    if 'audio_path' in df.columns:
        # Remove any existing data/ljspeech prefix if present
        df['audio_path'] = df['audio_path'].apply(
            lambda x: str(Path(x).relative_to('data/ljspeech') if str(x).startswith('data/ljspeech') else x)
        )
        
        # Ensure paths are relative to the data directory
        df['audio_path'] = df['audio_path'].apply(
            lambda x: str(Path(x).relative_to(data_dir) if str(x).startswith(str(data_dir)) else x)
        )
    
    # Add split column if not present
    if 'split' not in df.columns:
        logger.info("Adding train/val/test split")
        total_samples = len(df)
        train_size = int(0.9 * total_samples)
        val_size = int(0.05 * total_samples)
        
        splits = ['train'] * train_size + ['val'] * val_size + ['test'] * (total_samples - train_size - val_size)
        df['split'] = splits
    
    # Save fixed metadata
    logger.info(f"Saving fixed metadata to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Fixed {len(df)} entries in metadata file")

def main():
    parser = argparse.ArgumentParser(description='Fix metadata file paths')
    parser.add_argument('--input', type=str, required=True, help='Path to input metadata file')
    parser.add_argument('--output', type=str, required=True, help='Path to save fixed metadata file')
    parser.add_argument('--data-dir', type=str, required=True, help='Base directory containing audio files')
    
    args = parser.parse_args()
    
    fix_metadata_paths(args.input, args.output, args.data_dir)

if __name__ == '__main__':
    main() 
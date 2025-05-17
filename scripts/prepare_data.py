import pandas as pd
import shutil
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_ljspeech_dataset():
    """Prepare LJSpeech dataset for training"""
    processed_dir = Path("data/ljspeech_processed")
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    
    # Clean up old directories if they exist
    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "wavs").mkdir(exist_ok=True)
    
    # Read original metadata
    metadata_path = Path("data/ljspeech/LJSpeech-1.1/metadata.csv")
    df = pd.read_csv(metadata_path, sep="|", header=None, 
                    names=["ID", "transcript", "normalized_transcript"])
    
    # Always set audio_path to 'wavs/LJXXXX-XXXX.wav'
    df["audio_path"] = df["ID"].apply(lambda x: f"wavs/{x}.wav")
    
    # Split data into train/val/test (90/5/5)
    n_samples = len(df)
    train_size = int(0.9 * n_samples)
    val_size = int(0.05 * n_samples)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    # Add split column for clarity
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Save metadata for each split
    train_df.to_csv(train_dir / "metadata.csv", index=False)
    val_df.to_csv(val_dir / "metadata.csv", index=False)
    test_df.to_csv(test_dir / "metadata.csv", index=False)
    
    # Copy audio files
    source_wavs = Path("data/ljspeech/LJSpeech-1.1/wavs")
    
    for split_df, split_dir in [(train_df, train_dir), (val_df, val_dir), (test_df, test_dir)]:
        for audio_path in split_df["audio_path"]:
            source_path = source_wavs / Path(audio_path).name
            target_path = split_dir / audio_path
            if source_path.exists():
                shutil.copy2(source_path, target_path)
            else:
                logger.warning(f"Audio file not found: {source_path}")
    
    logger.info(f"Saved {len(train_df)} samples for train split")
    logger.info(f"Saved {len(val_df)} samples for val split")
    logger.info(f"Saved {len(test_df)} samples for test split")
    logger.info("Dataset preparation completed successfully!")

if __name__ == "__main__":
    prepare_ljspeech_dataset() 
import pandas as pd
import shutil
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restructure_ljspeech():
    """Restructure LJSpeech dataset into clean train/val/test folders with wavs and metadata.csv"""
    # Original LJSpeech paths
    original_root = Path("data/ljspeech/LJSpeech-1.1")
    original_wavs = original_root / "wavs"
    metadata_path = original_root / "metadata.csv"

    # New structure
    base_dir = Path("data")
    splits = ["train", "val", "test"]
    split_ratios = [0.9, 0.05, 0.05]

    # Read metadata
    df = pd.read_csv(metadata_path, sep="|", header=None, names=["ID", "transcript", "normalized_transcript"])
    df["audio_path"] = df["ID"].apply(lambda x: f"wavs/{x}.wav")

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_samples = len(df)
    train_end = int(split_ratios[0] * n_samples)
    val_end = train_end + int(split_ratios[1] * n_samples)

    split_dfs = {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }

    for split, split_df in split_dfs.items():
        split_dir = base_dir / split
        wavs_dir = split_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        # Write metadata.csv
        split_df[["ID", "transcript", "normalized_transcript", "audio_path"]].to_csv(split_dir / "metadata.csv", index=False)
        logger.info(f"Wrote metadata for {split} split: {len(split_df)} samples")
        # Copy audio files
        for audio_path in split_df["audio_path"]:
            src = original_wavs / Path(audio_path).name
            dst = wavs_dir / Path(audio_path).name
            if not dst.exists():
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    logger.warning(f"Missing source audio: {src}")
        logger.info(f"Copied audio files for {split} split to {wavs_dir}")

    logger.info("Restructuring completed successfully!")

if __name__ == "__main__":
    restructure_ljspeech() 
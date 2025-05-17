import os
import argparse
import pandas as pd
import torchaudio
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_metadata(input_dir, output_path, max_samples=None):
    """Create metadata file for LJSpeech dataset."""
    metadata_path = os.path.join(input_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    # Read the original metadata
    df = pd.read_csv(metadata_path, sep='|', header=None,
                    names=['file_id', 'transcription', 'normalized_transcription'])
    
    # Filter and process files
    processed_data = []
    wav_dir = os.path.join(input_dir, 'wavs')
    
    logger.info(f"Processing {len(df)} files...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing metadata"):
        if max_samples and len(processed_data) >= max_samples:
            break
            
        file_id = row['file_id']
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        if not os.path.exists(wav_path):
            logger.warning(f"Audio file not found: {wav_path}")
            continue
            
        try:
            # Load audio to verify it's valid
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Get duration
            duration = waveform.shape[1] / sample_rate
            
            processed_data.append({
                'file_id': file_id,
                'audio_path': wav_path,
                'text': row['normalized_transcription'],
                'duration': duration
            })
            
        except Exception as e:
            logger.warning(f"Error processing {wav_path}: {str(e)}")
            continue
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed metadata
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Saved metadata for {len(processed_data)} files to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare LJSpeech dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Path to LJSpeech dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    create_metadata(args.input_dir, metadata_path, args.max_samples)
    
    logger.info("Dataset preparation completed successfully!")

if __name__ == '__main__':
    main() 
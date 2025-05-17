import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing and mel-spectrogram generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config['data']['sample_rate']
        self.n_mels = config['data']['n_mels']
        self.hop_length = config['data']['hop_length']
        self.win_length = config['data']['win_length']
        self.n_fft = config['data']['n_fft']
        self.mel_fmin = config['data']['mel_fmin']
        self.mel_fmax = config['data']['mel_fmax']
        self.max_wav_value = config['data']['max_wav_value']
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            power=1.0,
            normalized=True
        )
        
        # Initialize amplitude to DB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        )
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and normalize audio file"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            waveform = waveform / self.max_wav_value
            return waveform
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            raise
    
    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generate mel spectrogram from waveform"""
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec
    
    def process_audio(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio file and return waveform and mel spectrogram"""
        waveform = self.load_audio(audio_path)
        mel_spec = self.get_mel_spectrogram(waveform)
        return waveform, mel_spec

class TextProcessor:
    """Handles text processing and tokenization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocab_size = config['data']['vocab_size']
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3
        
        # Initialize character set
        self.char_to_id = {chr(i): i + 4 for i in range(128)}  # ASCII characters
        self.id_to_char = {i + 4: chr(i) for i in range(128)}
        
        # Add special tokens
        for token, idx in [
            ('<pad>', self.pad_token),
            ('<unk>', self.unk_token),
            ('<bos>', self.bos_token),
            ('<eos>', self.eos_token)
        ]:
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
    
    def text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text to sequence of token IDs"""
        sequence = [self.bos_token]
        for char in text:
            sequence.append(self.char_to_id.get(char, self.unk_token))
        sequence.append(self.eos_token)
        return torch.tensor(sequence, dtype=torch.long)
    
    def sequence_to_text(self, sequence: torch.Tensor) -> str:
        """Convert sequence of token IDs to text"""
        text = []
        for token_id in sequence:
            if token_id == self.eos_token:
                break
            if token_id not in [self.pad_token, self.bos_token]:
                text.append(self.id_to_char.get(token_id.item(), '<unk>'))
        return ''.join(text)
    
    def process_text(self, text: str) -> torch.Tensor:
        """Process text and return token sequence"""
        return self.text_to_sequence(text)

class DataProcessor:
    """Main data processing class that coordinates audio and text processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        self.segment_size = config['data']['segment_size']
    
    def process_file(self, audio_path: str, text: str) -> Dict[str, torch.Tensor]:
        """Process a single audio-text pair"""
        try:
            # Process audio
            waveform, mel_spec = self.audio_processor.process_audio(audio_path)
            
            # Process text
            text_seq = self.text_processor.process_text(text)
            
            # Create segments if needed
            if waveform.size(1) > self.segment_size:
                start = torch.randint(0, waveform.size(1) - self.segment_size, (1,))
                waveform = waveform[:, start:start + self.segment_size]
                mel_spec = mel_spec[:, :, start//self.config['data']['hop_length']:
                                  (start + self.segment_size)//self.config['data']['hop_length']]
            
            return {
                'waveform': waveform,
                'mel_spec': mel_spec,
                'text': text_seq
            }
        except Exception as e:
            logger.error(f"Error processing file {audio_path}: {str(e)}")
            raise
    
    def process_dataset(self, data_dir: str, output_dir: str, num_workers: int = 4) -> None:
        """Process entire dataset in parallel"""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata from CSV
        metadata_path = data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Read CSV file
        df = pd.read_csv(metadata_path)
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                audio_path = Path(row['audio_path'])
                text = row['text']
                output_path = output_dir / f"{audio_path.stem}.pt"
                
                if not output_path.exists():
                    futures.append(
                        executor.submit(self._process_and_save, audio_path, text, output_path)
                    )
            
            # Wait for all processing to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel processing: {str(e)}")
    
    def _process_and_save(self, audio_path: Path, text: str, output_path: Path) -> None:
        """Process and save a single file"""
        try:
            processed_data = self.process_file(str(audio_path), text)
            torch.save(processed_data, output_path)
            logger.info(f"Processed and saved: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            raise 
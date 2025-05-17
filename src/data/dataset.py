import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import numpy as np
import librosa
import torchaudio
import torchaudio.transforms as T
import logging
import random
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence

from src.data.processor import DataProcessor

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio preprocessing and feature extraction."""
    def __init__(self, sample_rate: int = 22050, n_mels: int = 80, 
                 hop_length: int = 256, win_length: int = 1024,
                 mel_fmin: float = 0, mel_fmax: float = 8000):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
        # Initialize mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax
        )
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform. Always returns (batch, n_mels, T)."""
        mel = self.mel_transform(waveform)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # (1, n_mels, T)
        return mel

class TextProcessor:
    """Handles text preprocessing and tokenization."""
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab = self._load_vocab(vocab_path) if vocab_path else self._create_default_vocab()
        self.vocab_size = len(self.vocab)
    
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary from file."""
        with open(vocab_path, 'r') as f:
            return json.load(f)
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """Create default vocabulary with basic characters."""
        chars = list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-')
        return {char: i for i, char in enumerate(chars)}
    
    def text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text to sequence of token IDs."""
        sequence = [self.vocab.get(char, self.vocab[' ']) for char in text]
        return torch.tensor(sequence, dtype=torch.long)

class TTSDataset(Dataset):
    """Dataset class for TTS training"""
    
    def __init__(self, split_dir, config, split='train'):
        self.split_dir = split_dir
        self.split = split
        self.config = config
        self.metadata_path = os.path.join(split_dir, 'metadata.csv')
        self.metadata = pd.read_csv(self.metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.split_dir, row['audio_path'])
        text = str(row['normalized_transcript'])
        try:
            waveform, sr = torchaudio.load(audio_path)
            return {
                'waveform': waveform,
                'text': text,
                'audio_path': audio_path,
                'config': self.config
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {audio_path}: {e}")
            raise

    @staticmethod
    def collate_fn(batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from src.data.dataset import TextProcessor, AudioProcessor

        # Initialize processors with config from the first batch item
        config = batch[0]['config'] if 'config' in batch[0] else {
            'data': {
                'sample_rate': 22050,
                'n_mels': 80,
                'hop_length': 256,
                'win_length': 1024,
                'n_fft': 1024,
                'mel_fmin': 0,
                'mel_fmax': 8000
            }
        }
        text_processor = TextProcessor()
        audio_processor = AudioProcessor(
            sample_rate=config['data']['sample_rate'],
            n_mels=config['data']['n_mels'],
            hop_length=config['data']['hop_length'],
            win_length=config['data']['win_length'],
            mel_fmin=config['data']['mel_fmin'],
            mel_fmax=config['data']['mel_fmax']
        )

        # Process text
        text_sequences = [text_processor.text_to_sequence(item['text']) for item in batch]
        text_sequences_padded = pad_sequence(text_sequences, batch_first=True)  # (batch, max_text_len)
        text_lengths = torch.tensor([len(seq) for seq in text_sequences], dtype=torch.long)

        # Process audio
        waveforms = [item['waveform'].squeeze() for item in batch]
        waveforms_padded = pad_sequence(waveforms, batch_first=True)  # (batch, max_T)
        waveform_lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
        waveforms_padded = waveforms_padded.unsqueeze(1)  # (batch, 1, max_T)
        
        # Compute mel spectrograms for the entire batch at once
        mel_specs = audio_processor.compute_mel_spectrogram(waveforms_padded)  # (batch, 1, max_T) -> (batch, n_mels, T)
        if mel_specs.dim() == 4:
            mel_specs = mel_specs.squeeze(1)  # Remove channel dim if present
        assert mel_specs.dim() == 3, f"mel_specs should be 3D, got {mel_specs.shape}"
        
        # Compute mel lengths based on waveform lengths
        mel_lengths = (waveform_lengths / config['data']['hop_length']).long()
        assert mel_lengths.dim() == 1, f"mel_lengths should be 1D, got {mel_lengths.shape}"
        
        return {
            'audio': waveforms_padded,  # (batch, 1, max_audio_len)
            'audio_lengths': waveform_lengths,
            'mel_spec': mel_specs,  # (batch, n_mels, max_mel_len)
            'mel_lengths': mel_lengths,
            'text': text_sequences_padded,
            'text_lengths': text_lengths
        }
    
    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Get a sample with additional metadata"""
        item = self.metadata.iloc[idx]
        data = self.__getitem__(idx)
        data['metadata'] = item.to_dict()
        return data 
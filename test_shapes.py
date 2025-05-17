import torch
import logging
import pandas as pd
import os
from src.models.vits import VITS
from src.utils.loss import compute_loss
import random
import numpy as np
from src.utils.text import text_to_sequence
import torchaudio
from src.data.dataset import AudioProcessor

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_tensor_stats(tensor, name):
    """Check tensor statistics and print if NaN or Inf is present"""
    if torch.isnan(tensor).any():
        logger.error(f"NaN found in {name}")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"Inf found in {name}")
        return False
    
    # Handle integer tensors
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        logger.info(f"{name} stats - min: {tensor.min().item()}, max: {tensor.max().item()}")
    else:
        logger.info(f"{name} stats - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
    return True

def load_audio_file(file_path):
    """Load and preprocess audio file"""
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

# Load metadata
metadata_path = "data/ljspeech_processed/train/metadata.csv"
metadata = pd.read_csv(metadata_path)

# Initialize model
model = VITS(
    vocab_size=100,
    hidden_channels=192,
    filter_channels=768,
    filter_kernel_size=3,
    n_heads=2,
    n_layers=6,
    kernel_size=5,
    dilation_rate=1,
    n_flows=4,
    n_layers_flow=4
)

# Initialize AudioProcessor with typical config
audio_processor = AudioProcessor(
    sample_rate=22050,
    n_mels=80,
    hop_length=256,
    win_length=1024,
    mel_fmin=0,
    mel_fmax=8000
)

# Run 5 batches with real data
batch_size = 2
for i in range(5):
    print(f"\nBatch {i+1}:")
    print("-" * 50)
    
    # Sample random batch from metadata
    batch_samples = metadata.sample(n=batch_size)
    
    # Process text
    texts = []
    text_lengths = []
    for text in batch_samples['normalized_transcript']:
        text_seq = text_to_sequence(text)
        texts.append(text_seq)
        text_lengths.append(len(text_seq))
    
    # Pad texts to max length in batch
    max_text_len = max(text_lengths)
    text_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    for j, text in enumerate(texts):
        text_tensor[j, :len(text)] = torch.tensor(text)
    text_lengths = torch.tensor(text_lengths)
    
    # Load and process audio
    waveforms = []
    audio_lengths = []
    for audio_path in batch_samples['audio_path']:
        full_path = os.path.join("data/ljspeech_processed/train", audio_path)
        waveform = load_audio_file(full_path)
        waveforms.append(waveform.squeeze(0))  # Remove channel dim for padding
        audio_lengths.append(waveform.shape[1])
    
    # Pad waveforms to max length in batch
    max_audio_len = max(audio_lengths)
    audio_tensor = torch.zeros(batch_size, max_audio_len)
    for j, waveform in enumerate(waveforms):
        audio_tensor[j, :waveform.shape[0]] = waveform
    audio_tensor = audio_tensor.unsqueeze(1)  # (batch, 1, max_audio_len)
    
    # Compute real mel spectrogram for the batch
    mel_spec = audio_processor.compute_mel_spectrogram(audio_tensor)
    if mel_spec.dim() == 4:
        mel_spec = mel_spec.squeeze(1)  # Remove channel dim if present
    assert mel_spec.dim() == 3, f"mel_spec should be 3D, got {mel_spec.shape}"
    audio_lengths = torch.tensor(audio_lengths)
    
    # Check input tensors
    logger.info("Checking input tensors...")
    check_tensor_stats(text_tensor, "text")
    check_tensor_stats(mel_spec, "mel_spec")
    check_tensor_stats(audio_tensor, "audio")
    
    # Model forward pass
    y = {
        'mel_spec': mel_spec,
        'audio_lengths': audio_lengths,
        'audio': audio_tensor
    }
    
    output = model(text_tensor, text_lengths, y, audio_lengths)
    
    # Check model outputs
    logger.info("\nChecking model outputs...")
    for key in ['y_hat', 'm_q', 'logs_q', 'm_p', 'logs_p', 'z', 'logdet', 'd_hat']:
        if key in output:
            if not check_tensor_stats(output[key], f"output['{key}']"):
                logger.error(f"Problem found in {key}")
                # Print detailed stats for problematic tensor
                tensor = output[key]
                logger.error(f"Shape: {tensor.shape}")
                logger.error(f"NaN count: {torch.isnan(tensor).sum().item()}")
                logger.error(f"Inf count: {torch.isinf(tensor).sum().item()}")
                if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                    logger.error(f"Min: {tensor.min().item()}")
                    logger.error(f"Max: {tensor.max().item()}")
                else:
                    logger.error(f"Min: {tensor.min().item():.4f}")
                    logger.error(f"Max: {tensor.max().item():.4f}")
                    logger.error(f"Mean: {tensor.mean().item():.4f}")
    
    # Compute loss
    loss, loss_dict = compute_loss(output, mel_spec, audio_lengths)
    
    # Check loss components
    logger.info("\nChecking loss components...")
    for name, value in loss_dict.items():
        if np.isnan(value) or np.isinf(value):
            logger.error(f"Problem found in {name}: {value}")
        else:
            logger.info(f"{name}: {value:.4f}")
    
    # Print shapes of key tensors
    print("\nTensor Shapes:")
    for key in ['y_hat', 'm_q', 'logs_q', 'm_p', 'logs_p', 'z', 'logdet', 'd_real', 'd_hat']:
        if key in output:
            print(f"output['{key}']: {output[key].shape}")
    if 'f' in output:
        print(f"output['f']: {[f.shape for f in output['f']]}")
    if 'f_hat' in output:
        print(f"output['f_hat']: {[f.shape for f in output['f_hat']]}")
    
    # Print values for KL debugging
    if 'logs_q' in output:
        print(f"\nlogs_q values: {output['logs_q']}")
    if 'm_q' in output:
        print(f"m_q values: {output['m_q']}")
    if 'm_p' in output:
        print(f"m_p values: {output['m_p']}")
    if 'logs_p' in output:
        print(f"logs_p values: {output['logs_p']}")
    
    # Check gradients
    loss.backward()
    print("\nChecking gradients...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient found in {name}")
            elif torch.isinf(param.grad).any():
                logger.error(f"Inf gradient found in {name}")
            else:
                logger.info(f"{name} grad stats - min: {param.grad.min().item():.4f}, max: {param.grad.max().item():.4f}, mean: {param.grad.mean().item():.4f}")
    
    # Zero gradients for next iteration
    model.zero_grad()

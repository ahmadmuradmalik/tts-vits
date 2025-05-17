import torch
import logging
from src.models.vits import VITS
from src.utils.loss import compute_loss
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Create dummy inputs for a small batch
batch_size = 2
text_len = 50
mel_len = 100
n_mels = 80

# Create dummy text input (batch_size, text_len)
text = torch.randint(0, 100, (batch_size, text_len))
text_lengths = torch.tensor([text_len] * batch_size)

# Create dummy mel spectrogram (batch_size, n_mels, mel_len)
mel_spec = torch.randn(batch_size, n_mels, mel_len)
audio_lengths = torch.tensor([mel_len] * batch_size)

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

# Run 5 batches
for i in range(5):
    print(f"\nBatch {i+1}:")
    print("-" * 50)
    
    # Forward pass
    y = {
        'mel_spec': mel_spec,
        'audio_lengths': audio_lengths,
        'audio': torch.randn(batch_size, 1, mel_len * 256)
    }
    output = model(text, text_lengths, y, audio_lengths)
    
    # Compute loss
    loss, loss_dict = compute_loss(output, mel_spec, audio_lengths)
    
    # Print loss components
    print(f"Total Loss: {loss.item():.4f}")
    print("Loss Components:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.4f}")
    
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
        print(f"logs_q values: {output['logs_q']}")
    if 'm_q' in output:
        print(f"m_q values: {output['m_q']}")
    if 'm_p' in output:
        print(f"m_p values: {output['m_p']}")
    if 'logs_p' in output:
        print(f"logs_p values: {output['logs_p']}")

if __name__ == "__main__":
    test_shapes() 
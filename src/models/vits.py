import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

class TextEncoder(nn.Module):
    """
    Encodes input text into a latent representation.
    Uses a combination of embedding layer and convolutional layers with residual connections.
    """
    def __init__(self, vocab_size: int, hidden_channels: int, filter_channels: int, 
                 filter_kernel_size: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        # Embedding layer converts token IDs to dense vectors
        self.emb = nn.Embedding(vocab_size, hidden_channels)
        # Stack of convolutional layers with residual connections
        self.encoder = nn.ModuleList([
            nn.Sequential(
                # First conv layer expands the channel dimension
                nn.Conv1d(hidden_channels, filter_channels, filter_kernel_size, padding=(filter_kernel_size-1)//2),
                nn.ReLU(),
                # Second conv layer projects back to hidden_channels
                nn.Conv1d(filter_channels, hidden_channels, 1),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        # LayerNorm is applied after all conv layers
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert token IDs to embeddings
        x = self.emb(x)  # (batch, time, hidden_channels)
        
        # Transpose for conv layers
        x = x.transpose(1, 2)  # (batch, hidden_channels, time)
        
        # Apply encoder layers with residual connections
        for i, layer in enumerate(self.encoder):
            x = x + layer(x)
        
        # Transpose back for LayerNorm
        x = x.transpose(1, 2)  # (batch, time, hidden_channels)
        
        # Apply LayerNorm
        x = self.layer_norm(x)
        
        # Transpose back for output
        x = x.transpose(1, 2)  # (batch, hidden_channels, time)
        
        return x, x_lengths

class PosteriorEncoder(nn.Module):
    """
    Encodes audio features into a latent space using dilated convolutions.
    Outputs mean and log variance for the variational posterior.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, 
                 dilation_rate: int, n_layers: int, gin_channels: int = 0):
        super().__init__()
        # Initial projection layer
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # Projection layer to convert mel spectrograms to hidden_channels
        self.mel_projection = nn.Conv1d(80, hidden_channels, 1)
        # Stack of dilated convolutional layers
        self.enc = nn.ModuleList([
            nn.Sequential(
                # Dilated conv with increasing dilation rate
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                         dilation=dilation_rate**i, padding=(kernel_size-1)//2 * dilation_rate**i),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, 1),
                nn.BatchNorm1d(hidden_channels)
            ) for i in range(n_layers)
        ])
        # Final projection to get mean and log variance
        self.post = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)
        
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial projection
        x = self.mel_projection(x)  # Project mel spectrograms to hidden_channels
        
        x = self.pre(x)
        
        # Apply dilated conv layers
        for i, layer in enumerate(self.enc):
            x = x + layer(x)
        
        # Get mean and log variance
        stats = self.post(x)
        
        m, logs = torch.split(stats, [stats.size(1)//2]*2, dim=1)
        
        return m, logs, x_lengths

class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    Uses multiple layers of 1D convolutions with spectral normalization.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(in_channels, hidden_channels, 15, stride=1, padding=7)),
            nn.LeakyReLU(0.1)
        ))
        
        # Hidden layers
        for i in range(n_layers):
            in_ch = hidden_channels * (2 ** i)
            out_ch = hidden_channels * (2 ** (i + 1))
            self.layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(in_ch, out_ch, 41, stride=4, padding=20)),
                nn.LeakyReLU(0.1)
            ))
        
        # Output layer
        self.layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(out_ch, 1, 41, stride=1, padding=20))
        ))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features

class Flow(nn.Module):
    """
    Normalizing flow layer for the decoder.
    Uses affine coupling layers for invertible transformations.
    """
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 5, dilation_rate: int = 1):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        # Affine coupling layer
        self.pre = nn.Conv1d(channels//2, hidden_channels, 1)
        self.enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                         dilation=dilation_rate**i, padding=dilation_rate**i * (kernel_size - 1) // 2, stride=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, 1, padding=0, stride=1),
                nn.BatchNorm1d(hidden_channels)
            ) for i in range(3)
        ])
        self.post = nn.Conv1d(hidden_channels, channels//2 * 2, 1)
        
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split input
        x1, x2 = torch.split(x, [self.channels//2]*2, dim=1)
        
        # Compute scale and shift
        h = self.pre(x1)
        for i, layer in enumerate(self.enc):
            h = h + layer(h)
        stats = self.post(h)
        m, logs = torch.split(stats, [self.channels//2]*2, dim=1)
        
        # Apply affine transformation
        x2 = (x2 * torch.exp(logs) + m) * x_mask
        x = torch.cat([x1, x2], dim=1)
        
        # Compute log determinant
        logdet = torch.sum(logs * x_mask, [1, 2])
        
        return x, logdet

class Decoder(nn.Module):
    """
    Decoder network that converts latent representations to mel-spectrograms.
    Uses normalizing flows for high-quality audio generation.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 192, kernel_size: int = 5,
                 dilation_rate: int = 1, n_flows: int = 4, n_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_flows = n_flows
        
        # Initial projection
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # Flow layers
        self.flows = nn.ModuleList([
            Flow(hidden_channels, hidden_channels, kernel_size, dilation_rate)
            for _ in range(n_flows)
        ])
        
        # Final projection to mel-spectrogram
        self.post = nn.Conv1d(hidden_channels, 80, 1)  # 80 mel bands
        
    def forward(self, z: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial projection
        x = self.pre(z)
        
        # Apply flow layers
        logdet = 0
        for i, flow in enumerate(self.flows):
            x, logdet_flow = flow(x, x_mask)
            logdet = logdet + logdet_flow
        
        # Final projection to mel-spectrogram
        y_hat = self.post(x)
        
        return y_hat, logdet

class VITS(nn.Module):
    """
    Variational Inference with adversarial learning for end-to-end Text-to-Speech.
    Combines text encoder and posterior encoder with normalizing flows.
    """
    def __init__(self, vocab_size: int, hidden_channels: int = 192, filter_channels: int = 768,
                 filter_kernel_size: int = 3, n_heads: int = 2, n_layers: int = 6,
                 kernel_size: int = 5, dilation_rate: int = 1, n_flows: int = 4,
                 n_layers_flow: int = 4, use_spectral_norm: bool = False):
        super().__init__()
        # Text encoder for processing input text
        self.text_encoder = TextEncoder(vocab_size, hidden_channels, filter_channels,
                                      filter_kernel_size, n_heads, n_layers)
        # Posterior encoder for audio features
        self.posterior_encoder = PosteriorEncoder(hidden_channels, hidden_channels,
                                                kernel_size, dilation_rate, n_layers)
        # Decoder for generating mel-spectrograms
        self.decoder = Decoder(hidden_channels, hidden_channels, kernel_size,
                             dilation_rate, n_flows, n_layers_flow)
        # Discriminator for adversarial training
        self.discriminator = Discriminator(80)  # 80 mel bands
        
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, 
                y: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None) -> dict:
        # Create mask for variable length text sequences
        x_mask = torch.unsqueeze(torch.arange(x.size(1), device=x.device) < x_lengths.unsqueeze(1), 1).float()
        
        # Encode text
        x, m_p = self.text_encoder(x, x_lengths)
        
        if y is not None:
            # Create mask for variable length audio sequences
            y_mask = torch.unsqueeze(torch.arange(y['mel_spec'].size(2), device=y['mel_spec'].device) < y['audio_lengths'].unsqueeze(1), 1).float()
            
            # During training: encode audio features
            z, m_q, logs_q = self.posterior_encoder(y['mel_spec'], y['audio_lengths'])
            
            # Generate mel-spectrogram
            y_hat, logdet = self.decoder(z, y_mask)
            
            # Get discriminator outputs for real and generated mel spectrograms
            d_real, f_real = self.discriminator(y['mel_spec'])  # Use real mel spectrogram
            
            d_hat, f_hat = self.discriminator(y_hat)  # Use generated mel spectrogram
            
            return {
                'x': x,
                'm_p': m_p,
                'z': z,
                'm_q': m_q,
                'logs_q': logs_q,
                'y_hat': y_hat,
                'logdet': logdet,
                'd_hat': d_hat,
                'f_hat': f_hat,
                'f': f_real
            }
        # During inference: only return text encoding
        return {
            'x': x,
            'm_p': m_p
        } 
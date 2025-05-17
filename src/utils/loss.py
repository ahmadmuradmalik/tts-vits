import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_loss(output: Dict, y: torch.Tensor, y_lengths: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Compute the VITS loss components:
    1. Reconstruction loss (L1 loss between mel spectrograms)
    2. KL divergence loss for variational inference
    3. Adversarial loss for the discriminator
    4. Feature matching loss for the discriminator
    5. Log determinant loss from normalizing flows
    """
    # Create mask for variable length sequences
    # Use the time dimension (last dimension) for the mask
    y_mask = torch.unsqueeze(torch.arange(y.size(2), device=y.device) < y_lengths.unsqueeze(1), 1).float()
    
    # Reconstruction loss (L1 loss between mel spectrograms)
    # y_hat is the generated mel spectrogram from the decoder
    # y is the target mel spectrogram
    recon_loss = F.l1_loss(output['y_hat'] * y_mask, y * y_mask)
    
    # KL divergence loss
    # Expand logs_q to match m_q shape
    logs_q = output['logs_q'].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
    logs_q = logs_q.expand_as(output['m_q'])  # [batch_size, channels, time]
    
    # Compute KL divergence with numerical stability
    # KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where sigma^2 = exp(logs)
    kl_loss = torch.mean(-0.5 * torch.sum(
        1 + logs_q - output['m_q'].pow(2) - torch.clamp(logs_q.exp(), min=1e-6),
        dim=[1,2]
    ))
    
    # Adversarial loss
    adv_loss = torch.mean((1 - output['d_hat']) ** 2)
    
    # Feature matching loss
    # Handle each feature map separately
    fm_loss = 0.0
    for f_hat, f in zip(output['f_hat'], output['f']):
        fm_loss += torch.mean(torch.abs(f_hat - f))
    fm_loss = fm_loss / len(output['f_hat'])  # Average over all feature maps
    
    # Log determinant loss (from normalizing flows)
    logdet_loss = -torch.mean(output['logdet'])
    
    # Total loss
    total_loss = recon_loss + kl_loss + adv_loss + fm_loss + logdet_loss
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'adv_loss': adv_loss.item(),
        'fm_loss': fm_loss.item(),
        'logdet_loss': logdet_loss.item(),
        'total_loss': total_loss.item()
    } 
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_loss(output: Dict, y: torch.Tensor, y_lengths: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    y_mask = torch.unsqueeze(torch.arange(y.size(2), device=y.device) < y_lengths.unsqueeze(1), 1).float()
    epsilon = 1e-8
    
    y_hat = torch.clamp(output['y_hat'], min=-1e6, max=1e6)
    y = torch.clamp(y, min=-1e6, max=1e6)
    recon_loss = F.l1_loss(y_hat * y_mask + epsilon, y * y_mask + epsilon)
    
    logs_q = output['logs_q'] 
    m_q = output['m_q'] 
    
    kl_loss = torch.mean(-0.5 * (
        1.0 + 
        logs_q - 
        torch.clamp(m_q.pow(2), max=100.0) - 
        torch.clamp(torch.exp(logs_q), min=epsilon, max=1e6) 
    ))
    
    kl_loss = torch.clamp(kl_loss, min=0.0)

    d_hat = torch.clamp(output['d_hat'], min=-1e6, max=1e6)
    adv_loss = torch.mean(torch.clamp((1 - d_hat) ** 2, max=100.0))
    
    fm_loss = 0.0
    if 'f_hat' in output and 'f' in output and output['f_hat'] is not None and output['f'] is not None:
        for f_hat, f_val in zip(output['f_hat'], output['f']):
            if f_hat is not None and f_val is not None:
                f_hat = torch.clamp(f_hat, min=-1e6, max=1e6)
                f_val = torch.clamp(f_val, min=-1e6, max=1e6)
                fm_loss += torch.mean(torch.clamp(torch.abs(f_hat - f_val), max=100.0))
        if output['f_hat']: 
             fm_loss = fm_loss / len(output['f_hat'])
    else: 
        fm_loss = torch.tensor(0.0, device=y.device)

    logdet = torch.clamp(output['logdet'], min=-100.0, max=100.0)
    logdet_loss = -torch.mean(logdet)
    
    unclamped_total_loss = (
        recon_loss +
        kl_loss * 0.5 + 
        adv_loss * 0.1 + 
        fm_loss * 0.1 + 
        logdet_loss * 0.5
    )

    total_loss = torch.clamp(
        unclamped_total_loss, 
        max=100.0
    )
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'adv_loss': adv_loss.item(),
        'fm_loss': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss, 
        'logdet_loss': logdet_loss.item(),
        'unclamped_total_loss': unclamped_total_loss.item(),
        'total_loss': total_loss.item()
    } 
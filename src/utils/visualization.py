import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

def plot_flow_transformation(
    z: torch.Tensor,
    log_det: torch.Tensor,
    save_path: str = 'flow_visualization.png'
) -> None:
    """
    Visualize the transformation of data through a normalizing flow layer.
    
    Args:
        z: Transformed data tensor
        log_det: Log determinant of the transformation
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 4))
    
    # Plot transformed data
    plt.subplot(1, 2, 1)
    plt.imshow(z[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Transformed Data')
    
    # Plot log determinant
    plt.subplot(1, 2, 2)
    plt.plot(log_det.cpu().numpy())
    plt.title('Log Determinant')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_components(
    losses: Dict[str, List[float]],
    save_path: str = 'loss_components.png'
) -> None:
    """
    Plot the different loss components over training.
    
    Args:
        losses: Dictionary of loss components and their values
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in losses.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_mel_spectrograms(
    real_mel: torch.Tensor,
    generated_mel: torch.Tensor,
    save_path: str = 'mel_comparison.png'
) -> None:
    """
    Compare real and generated mel-spectrograms.
    
    Args:
        real_mel: Real mel-spectrogram tensor
        generated_mel: Generated mel-spectrogram tensor
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot real mel-spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(real_mel[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Real Mel-Spectrogram')
    
    # Plot generated mel-spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(generated_mel[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Generated Mel-Spectrogram')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_training_step(
    model_output: Dict,
    real_mel: torch.Tensor,
    step: int,
    output_dir: str = 'visualizations'
) -> None:
    """
    Create visualizations for a single training step.
    
    Args:
        model_output: Dictionary containing model outputs
        real_mel: Real mel-spectrogram tensor
        step: Current training step
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot flow transformation
    plot_flow_transformation(
        model_output['z'],
        model_output['logdet'],
        os.path.join(output_dir, f'flow_step_{step}.png')
    )
    
    # Plot mel-spectrogram comparison
    plot_mel_spectrograms(
        real_mel,
        model_output['y_hat'],
        os.path.join(output_dir, f'mel_step_{step}.png')
    ) 
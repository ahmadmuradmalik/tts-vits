from typing import Dict, Any
import yaml
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    return config

def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {path}")

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for VITS model and training"""
    return {
        'data': {
            'sample_rate': 22050,
            'n_mels': 80,
            'hop_length': 256,
            'win_length': 1024,
            'n_fft': 1024,
            'mel_fmin': 0,
            'mel_fmax': 8000,
            'vocab_size': 256,  # ASCII character set
            'max_wav_value': 32768.0,
            'segment_size': 8192,
        },
        'model': {
            'hidden_channels': 192,
            'filter_channels': 768,
            'filter_kernel_size': 3,
            'n_heads': 2,
            'n_layers': 6,
            'kernel_size': 3,
            'dilation_rate': 1,
            'n_flows': 4,
            'n_layers_flow': 4,
            'use_spectral_norm': False,
            'hidden_channels_dp': 192,
            'kernel_size_dp': 3,
            'dropout': 0.1,
        },
        'training': {
            'epochs': 1000,
            'batch_size': 16,
            'learning_rate': 2e-4,
            'fp16_run': True,
            'log_interval': 200,
            'eval_interval': 1000,
            'save_interval': 1000,
            'warmup_epochs': 0,
            'grad_clip_thresh': 1.0,
            'accumulation_steps': 4,  # Gradient accumulation steps
            'steps_per_epoch': 1000,  # For OneCycleLR scheduler
            'weight_decay': 0.01,  # L2 regularization
            'beta1': 0.8,  # Adam optimizer beta1
            'beta2': 0.99,  # Adam optimizer beta2
            'output_dir': 'outputs',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
        },
        'loss': {
            'lambda_kl': 0.5,  # Reduced KL loss weight
            'lambda_fm': 0.1,  # Reduced feature matching loss weight
            'lambda_mel': 45.0,
            'lambda_dur': 1.0,
            'lambda_adv': 0.1,  # Reduced adversarial loss weight
        },
        'inference': {
            'max_inference_len': 1000,
            'temperature': 0.667,
            'length_scale': 1.0,
            'noise_scale': 0.667,
            'noise_scale_w': 0.8,
        }
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary"""
    required_sections = ['data', 'model', 'training', 'loss', 'inference']
    required_data = ['sample_rate', 'n_mels', 'hop_length', 'win_length', 'n_fft']
    required_model = ['hidden_channels', 'filter_channels', 'n_heads', 'n_layers']
    required_training = ['epochs', 'batch_size', 'learning_rate']
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required section: {section}")
            return False
    
    # Check required data parameters
    for param in required_data:
        if param not in config['data']:
            logger.error(f"Missing required data parameter: {param}")
            return False
    
    # Check required model parameters
    for param in required_model:
        if param not in config['model']:
            logger.error(f"Missing required model parameter: {param}")
            return False
    
    # Check required training parameters
    for param in required_training:
        if param not in config['training']:
            logger.error(f"Missing required training parameter: {param}")
            return False
    
    return True

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values"""
    def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_config = deep_update(config.copy(), updates)
    if not validate_config(updated_config):
        raise ValueError("Invalid configuration after update")
    
    return updated_config 
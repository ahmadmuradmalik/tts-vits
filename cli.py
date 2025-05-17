import click
import torch
import yaml
from pathlib import Path
import logging
from typing import Optional
import textwrap

# Lazy load rich components
def get_rich_components():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    return Console, Panel, Table, Markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for help text
_help_text = None

def get_help_text():
    """Get cached help text"""
    global _help_text
    if _help_text is None:
        _help_text = """
# VITS Text-to-Speech CLI

## Quick Start

1. Generate config:
   ```bash
   python cli.py config generate-config -o config.yaml
   ```

2. Preprocess data:
   ```bash
   python cli.py data preprocess -d /path/to/raw/data -o /path/to/processed/data
   ```

3. Train model:
   ```bash
   python cli.py train train-model -d /path/to/data -o outputs --gpu 0
   ```

4. Generate speech:
   ```bash
   python cli.py infer synthesize -c model.pt -t "Hello world" -o output.wav
   ```

## Commands

### Training
- `train train-model`: Train VITS model
  - `--config`: Config file path
  - `--data-dir`: Data directory
  - `--output-dir`: Output directory
  - `--checkpoint`: Checkpoint for resuming
  - `--num-epochs`: Number of epochs
  - `--batch-size`: Batch size
  - `--learning-rate`: Learning rate
  - `--gpu`: GPU device ID (-1 for CPU)

### Inference
- `infer synthesize`: Generate speech
  - `--checkpoint`: Model checkpoint
  - `--text`: Text to synthesize
  - `--output`: Output audio file
  - `--gpu`: GPU device ID

### Data
- `data preprocess`: Process audio data
  - `--data-dir`: Data directory
  - `--output-dir`: Output directory
  - `--sample-rate`: Sample rate
  - `--n-mels`: Mel bands

### Config
- `config generate-config`: Generate config
  - `--output`: Config file path

## Tips

1. Data:
   - Use WAV format
   - Consistent sample rates
   - Use validation split

2. Training:
   - Start with small dataset
   - Monitor loss components
   - Use checkpoints

3. Inference:
   - Keep text concise
   - Try different checkpoints
   - Use GPU for speed

## Troubleshooting

1. CUDA OOM:
   - Reduce batch size
   - Use gradient accumulation
   - Smaller model config

2. Poor Quality:
   - Check preprocessing
   - Verify sample rate
   - More training data

3. Instability:
   - Adjust learning rate
   - Check loss weights
   - Verify normalization
"""
    return _help_text

def print_help():
    """Print help information"""
    try:
        Console, Panel, Table, Markdown = get_rich_components()
        console = Console()
        
        # Print help text
        console.print(Panel.fit(
            Markdown(get_help_text()),
            title="VITS CLI Help",
            border_style="blue"
        ))
        
        # Print examples
        table = Table(title="Examples", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan")
        table.add_column("Example", style="yellow")
        
        examples = [
            ("Generate Config", "python cli.py config generate-config -o config.yaml"),
            ("Preprocess Data", "python cli.py data preprocess -d data/raw -o data/processed"),
            ("Train Model", "python cli.py train train-model -d data/processed -o outputs --gpu 0"),
            ("Synthesize", "python cli.py infer synthesize -c model.pt -t 'Hello' -o speech.wav")
        ]
        
        for cmd, ex in examples:
            table.add_row(cmd, ex)
        
        console.print("\n")
        console.print(table)
        
    except ImportError:
        # Fallback to simple text if rich is not available
        print(get_help_text())

@click.group()
def cli():
    """VITS Text-to-Speech CLI"""
    pass

@cli.command()
def help():
    """Show comprehensive help information"""
    print_help()

@cli.group()
def train():
    """Training commands"""
    pass

@cli.group()
def infer():
    """Inference commands"""
    pass

@cli.group()
def data():
    """Data management commands"""
    pass

@cli.group()
def config():
    """Configuration management commands"""
    pass

@train.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--data-dir', '-d', type=click.Path(exists=True), required=True, help='Path to data directory')
@click.option('--output-dir', '-o', type=click.Path(), default='outputs', help='Output directory')
@click.option('--checkpoint', type=click.Path(), help='Path to checkpoint file for resuming training')
@click.option('--num-epochs', type=int, help='Number of epochs to train')
@click.option('--batch-size', type=int, help='Batch size for training')
@click.option('--learning-rate', type=float, help='Learning rate')
@click.option('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
def train_model(config: Optional[str], data_dir: str, output_dir: str, checkpoint: Optional[str],
                num_epochs: Optional[int], batch_size: Optional[int], learning_rate: Optional[float], gpu: int):
    """Train the VITS model"""
    try:
        from src.models.vits import VITS
        from src.training.trainer import TTSTrainer
        from src.utils.config import get_default_config, save_config
        from src.data.dataset import TTSDataset
        from torch.utils.data import DataLoader

        # Load or create config
        if config:
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = get_default_config()
        
        # Update config with command line arguments
        if num_epochs:
            cfg['training']['epochs'] = num_epochs
        if batch_size:
            cfg['training']['batch_size'] = batch_size
        if learning_rate:
            cfg['training']['learning_rate'] = learning_rate
        
        # Set device
        if gpu >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu}')
        else:
            device = torch.device('cpu')
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        save_config(cfg, output_path / 'config.yaml')
        
        # Initialize model
        model = VITS(
            vocab_size=cfg['data']['vocab_size'],
            hidden_channels=cfg['model']['hidden_channels'],
            filter_channels=cfg['model']['filter_channels'],
            filter_kernel_size=cfg['model']['filter_kernel_size'],
            n_heads=cfg['model']['n_heads'],
            n_layers=cfg['model']['n_layers'],
            kernel_size=cfg['model']['kernel_size'],
            dilation_rate=cfg['model']['dilation_rate'],
            n_flows=cfg['model']['n_flows'],
            n_layers_flow=cfg['model']['n_layers_flow']
        ).to(device)
        
        # Load checkpoint if provided
        if checkpoint:
            logger.info(f"Loading checkpoint from {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        
        # Setup dataset and dataloader
        dataset = TTSDataset(data_dir)
        train_loader = DataLoader(
            dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        # Initialize trainer
        trainer = TTSTrainer(cfg)
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_loader, cfg['training']['epochs'])
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise click.ClickException(str(e))

@infer.command()
@click.option('--checkpoint', '-c', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--text', '-t', required=True, help='Text to synthesize')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output audio file path')
@click.option('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
def synthesize(checkpoint: str, text: str, output: str, gpu: int):
    """Synthesize speech from text"""
    try:
        from src.models.vits import VITS
        from src.training.trainer import TTSTrainer
        from src.utils.config import get_default_config

        # Set device
        if gpu >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu}')
        else:
            device = torch.device('cpu')
        
        # Load model
        cfg = get_default_config()
        model = VITS(
            vocab_size=cfg['data']['vocab_size'],
            hidden_channels=cfg['model']['hidden_channels'],
            filter_channels=cfg['model']['filter_channels'],
            filter_kernel_size=cfg['model']['filter_kernel_size'],
            n_heads=cfg['model']['n_heads'],
            n_layers=cfg['model']['n_layers'],
            kernel_size=cfg['model']['kernel_size'],
            dilation_rate=cfg['model']['dilation_rate'],
            n_flows=cfg['model']['n_flows'],
            n_layers_flow=cfg['model']['n_layers_flow']
        ).to(device)
        
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        
        # Initialize trainer for synthesis
        trainer = TTSTrainer(cfg)
        
        # Synthesize speech
        logger.info(f"Synthesizing speech for text: {text}")
        trainer.synthesize(text, output)
        
        logger.info(f"Speech saved to {output}")
        
    except Exception as e:
        logger.error(f"Error during synthesis: {str(e)}")
        raise click.ClickException(str(e))

@data.command()
@click.option('--data-dir', '-d', type=click.Path(exists=True), required=True, help='Path to data directory')
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Output directory for processed data')
@click.option('--sample-rate', type=int, default=22050, help='Target sample rate')
@click.option('--n-mels', type=int, default=80, help='Number of mel bands')
def preprocess(data_dir: str, output_dir: str, sample_rate: int, n_mels: int):
    """Preprocess audio data"""
    try:
        from src.data.dataset import AudioProcessor
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # Process data
        logger.info(f"Processing data from {data_dir}")
        processor.process_directory(data_dir, output_dir)
        
        logger.info(f"Data processing complete. Output saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.option('--output', '-o', type=click.Path(), required=True, help='Output config file path')
def generate_config(output: str):
    """Generate default configuration file"""
    try:
        from src.utils.config import get_default_config, save_config
        
        # Generate and save config
        config = get_default_config()
        save_config(config, output)
        logger.info(f"Default configuration saved to {output}")
        
    except Exception as e:
        logger.error(f"Error generating config: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli() 
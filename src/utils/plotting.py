import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_plots(log_dir: str, csv_filename: str = "epoch_metrics.csv"):
    """Reads epoch metrics from a CSV file and generates loss plots."""
    log_dir_path = Path(log_dir)
    csv_file_path = log_dir_path / csv_filename

    if not csv_file_path.exists():
        logger.warning(f"CSV log file not found at {csv_file_path}. Skipping plot generation.")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {e}. Skipping plot generation.")
        return

    if df.empty:
        logger.info("CSV log file is empty. Skipping plot generation.")
        return

    plots_saved = []

    # Plot 1: Total Training and Validation Loss vs. Epoch
    if 'epoch' in df.columns and 'train_loss_total' in df.columns and 'val_loss_total' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['epoch'], df['train_loss_total'], label='Total Train Loss', marker='o')
        plt.plot(df['epoch'], df['val_loss_total'], label='Total Validation Loss', marker='o')
        plt.title('Total Train & Validation Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = log_dir_path / "total_losses_vs_epoch.png"
        plt.savefig(plot_path)
        plt.close()
        plots_saved.append(plot_path)
        logger.info(f"Saved total losses plot to {plot_path}")
    else:
        logger.warning("Skipping total losses plot due to missing columns in CSV.")

    # Plot 2: Individual Training Loss Components vs. Epoch
    loss_components = [
        'train_recon_loss', 'train_kl_loss', 'train_adv_loss', 
        'train_fm_loss', 'train_logdet_loss', 'train_unclamped_total_loss'
    ]
    plot_individual_losses = False
    for lc in loss_components:
        if lc in df.columns and 'epoch' in df.columns:
            plot_individual_losses = True # At least one component can be plotted
            break
    
    if plot_individual_losses:
        plt.figure(figsize=(14, 8))
        for lc in loss_components:
            if lc in df.columns and 'epoch' in df.columns and not df[lc].isnull().all():
                plt.plot(df['epoch'], df[lc], label=lc.replace("train_", "").replace("_", " ").title(), marker='.')
        plt.title('Training Loss Components vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component Value')
        plt.legend()
        plt.grid(True)
        plot_path = log_dir_path / "training_loss_components_vs_epoch.png"
        plt.savefig(plot_path)
        plt.close()
        plots_saved.append(plot_path)
        logger.info(f"Saved training loss components plot to {plot_path}")
    else:
        logger.warning("Skipping training loss components plot due to missing columns or all NaN values in CSV.")

    # Plot 3: Learning Rate vs. Epoch
    if 'epoch' in df.columns and 'learning_rate' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='o', color='green')
        plt.title('Learning Rate vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plot_path = log_dir_path / "learning_rate_vs_epoch.png"
        plt.savefig(plot_path)
        plt.close()
        plots_saved.append(plot_path)
        logger.info(f"Saved learning rate plot to {plot_path}")
    else:
        logger.warning("Skipping learning rate plot due to missing columns in CSV.")

    if not plots_saved:
        logger.info("No plots were generated.")

# Example usage (you would call this from main.py):
# if __name__ == '__main__':
#     # Configure basic logging for testing this script directly
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     # Assuming your logs are in 'outputs/logs' relative to where you run from
#     # Create a dummy CSV for testing if needed
#     dummy_log_dir = Path(__file__).parent.parent.parent / 'outputs' / 'logs' # Adjust path as needed
#     dummy_log_dir.mkdir(parents=True, exist_ok=True)
#     dummy_csv_path = dummy_log_dir / "epoch_metrics.csv"
#     if not dummy_csv_path.exists():
#         header = ['epoch','train_loss_total','val_loss_total','learning_rate','train_recon_loss','train_kl_loss','train_adv_loss','train_fm_loss','train_logdet_loss','train_unclamped_total_loss']
#         rows = [
#             [0,10.5,12.1,0.0004,2.0,1.0,0.5,0.5,0.5,10.0],
#             [1,9.5,11.1,0.00038,1.8,0.9,0.4,0.4,0.4,9.0],
#             [2,8.5,10.1,0.00035,1.5,0.8,0.3,0.3,0.3,8.0]
#         ]
#         with open(dummy_csv_path, 'w', newline='') as f:
#             writer_obj = csv.writer(f)
#             writer_obj.writerow(header)
#             writer_obj.writerows(rows)
#         logger.info(f"Created dummy CSV for testing at {dummy_csv_path}")
#     generate_plots(str(dummy_log_dir)) 
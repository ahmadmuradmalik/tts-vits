import torch
import torch.nn as nn # Added for nn.Module typing if needed for grad check
from torch.utils.data import DataLoader
import os
import sys
import json # Added for model config
import torch.optim as optim # Added for optimizer

# Add src to path to import TTSDataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import TTSDataset, TextProcessor # Assuming this is the correct path
from models.vits import VITS # Added
from utils.loss import compute_loss # Added

def check_for_nans_infs(tensor_dict, prefix=""):
    nan_found = False
    for name, tensor_item in tensor_dict.items(): # Renamed tensor to tensor_item
        if isinstance(tensor_item, list):
            for i, t_sub_item in enumerate(tensor_item):
                if isinstance(t_sub_item, torch.Tensor):
                    if torch.isnan(t_sub_item).any():
                        print(f"{prefix}{name}[{i}] has NaNs!")
                        nan_found = True
                    if torch.isinf(t_sub_item).any():
                        print(f"{prefix}{name}[{i}] has Infs!")
                        nan_found = True
        elif isinstance(tensor_item, torch.Tensor):
            if torch.isnan(tensor_item).any():
                print(f"{prefix}{name} has NaNs!")
                nan_found = True
            if torch.isinf(tensor_item).any():
                print(f"{prefix}{name} has Infs!")
                nan_found = True
        elif tensor_item is None:
            print(f"{prefix}{name} is None.") # Handle None case if a key might be missing
            pass # Or treat as an issue depending on context

    # Removed the "No NaNs" print from here, will be printed by calling context if needed
    return nan_found

def check_gradients(model: nn.Module, prefix="Grad Check: "):
    nan_grad_found = False
    inf_grad_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"{prefix}NaN found in gradient of {name}")
                nan_grad_found = True
            if torch.isinf(param.grad).any():
                print(f"{prefix}Inf found in gradient of {name}")
                inf_grad_found = True
        elif param.requires_grad:
            # This case might be okay if no loss depends on this param for this batch
            # print(f"{prefix}Warning: Gradient is None for trainable parameter {name}")
            pass 
    if not nan_grad_found and not inf_grad_found:
        print(f"{prefix}No NaNs or Infs found in gradients.")
    return nan_grad_found or inf_grad_found

if __name__ == "__main__":
    # Basic configuration (adjust as needed, especially paths and audio params)
    # This config should ideally match your actual training config
    # Forcing fp16_run to False for this debugging script
    config_str = """
    {
        "data": {
            "split_dir_root": "data/ljspeech_processed",
            "sample_rate": 22050,
            "n_mels": 80,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "batch_size": 4
        },
        "text": {},
        "model": {
            "hidden_channels": 192,
            "filter_channels": 768,
            "filter_kernel_size": 3,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 5,
            "dilation_rate": 1,
            "n_flows": 4,
            "n_layers_flow": 4,
            "use_spectral_norm": false
        },
        "training": {
            "fp16_run": false,
            "learning_rate": 1e-4,
            "num_test_batches": 5
        }
    }
    """
    config = json.loads(config_str)
    
    # Determine vocab_size
    # temp_text_processor = TextProcessor(config.get('text', {}).get('vocab_path'))
    # vocab_size = temp_text_processor.vocab_size
    # Hardcoding vocab_size for now based on default TextProcessor, adjust if you use a custom vocab
    # Default chars: ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'
    vocab_size = 70 # len(list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-')) + 1 for pad/blank if used, or check TextProcessor


    train_split_dir = os.path.join(config['data']['split_dir_root'], 'train')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Attempting to load dataset from: {train_split_dir}")
    if not os.path.exists(train_split_dir):
        print(f"Error: Training data directory not found: {train_split_dir}")
        exit()
    
    if not os.path.exists(os.path.join(train_split_dir, 'metadata.csv')):
        print(f"Error: metadata.csv not found in {train_split_dir}")
        exit()

    try:
        dataset_config_for_loader = {
            'data': config['data'],
            'text': config.get('text', {}) 
        }
        train_dataset = TTSDataset(split_dir=train_split_dir, config=dataset_config_for_loader, split='train')
        
        if len(train_dataset) == 0:
            print("Error: Dataset is empty.")
            exit()
            
        print(f"Dataset loaded. Number of samples: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=False, # No shuffle for reproducible test
            collate_fn=TTSDataset.collate_fn,
            num_workers=0
        )

        print("Fetching one batch...")
        try:
            first_batch = next(iter(train_loader))
            print("Batch fetched successfully.")

            # Move batch to device
            first_batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in first_batch.items()}

            # Check for NaNs/Infs in batch from data loader (already done, but good to have)
            print("\\n--- Checking batch from DataLoader ---")
            if check_for_nans_infs({
                "audio_input": first_batch['audio'],
                "mel_spec_input": first_batch['mel_spec']
            }, prefix="Batch Input: "):
                print("NaNs/Infs DETECTED in the data loader output. Stopping.")
                exit()

            # Initialize Model
            print("\\n--- Initializing Model and Optimizer ---")
            model_config_params = config['model']
            temp_text_processor = TextProcessor(vocab_path=config.get('text', {}).get('vocab_path'))
            actual_vocab_size = temp_text_processor.vocab_size
            print(f"Using vocab_size: {actual_vocab_size} from TextProcessor")

            model = VITS(
                vocab_size=actual_vocab_size, 
                **model_config_params
            ).to(device)
            model.eval() # Set to eval mode for single pass, no dropout/batchnorm updates if any were training-specific

            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['training']['learning_rate'], 
                weight_decay=0.01, 
                betas=(0.8, 0.99)
            )

            num_batches_to_test = config['training']['num_test_batches']
            print(f"\n--- Starting Mini Training Loop for {num_batches_to_test} Batches ---")

            for batch_idx, first_batch in enumerate(train_loader):
                if batch_idx >= num_batches_to_test:
                    break
                
                print(f"\n--- Batch {batch_idx + 1}/{num_batches_to_test} ---")
                model.train() # Ensure model is in training mode
                optimizer.zero_grad()

                first_batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in first_batch.items()}

                print("Checking batch from DataLoader...")
                batch_check_dict = {"audio_input": first_batch['audio'], "mel_spec_input": first_batch['mel_spec']}
                if check_for_nans_infs(batch_check_dict, prefix="Batch Input: "):
                    print("NaNs/Infs DETECTED in DataLoader output. Stopping.")
                    exit()
                print("Batch input is clean.")

                print("Performing Model Forward Pass...")
                y_dict_for_model = {
                    'mel_spec': first_batch['mel_spec'],
                    'audio_lengths': first_batch['mel_lengths']
                }
                model_output = model(
                    x=first_batch['text'], x_lengths=first_batch['text_lengths'],
                    y=y_dict_for_model, y_lengths=first_batch['mel_lengths']
                )
                
                print("Checking Model Output...")
                if check_for_nans_infs(model_output, prefix="Model Output: "):
                    print("NaNs/Infs DETECTED in model output. Stopping.")
                    exit()
                print("Model output is clean.")

                print("Computing Loss...")
                loss, loss_dict = compute_loss(
                    output=model_output, y=first_batch['mel_spec'], y_lengths=first_batch['mel_lengths']
                )
                
                print("Checking Loss Output...")
                loss_check_dict = {"total_loss_tensor": loss, **loss_dict}
                if check_for_nans_infs(loss_check_dict, prefix="Loss: "):
                     print(f"Loss is NaN/Inf: {loss.item() if isinstance(loss, torch.Tensor) else loss}. Stopping.")
                     exit()
                print(f"Losses are clean. Total Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}")

                print("Performing Backward Pass...")
                loss.backward()
                
                print("Checking Gradients...")
                if check_gradients(model):
                    print("NaNs/Infs DETECTED in gradients. Stopping.")
                    exit()
                print("Gradients are clean.")

                print("Optimizer Step...")
                optimizer.step()
                print(f"Batch {batch_idx + 1} completed successfully.")

            print(f"\n--- Mini Training Loop Completed for {num_batches_to_test} Batches ---")
            print("No NaNs/Infs encountered during the training test.")

        except StopIteration:
            print("Error: DataLoader is empty.")
        except Exception as e:
            print(f"Error during script execution: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error initializing script components: {e}")
        import traceback
        traceback.print_exc() 
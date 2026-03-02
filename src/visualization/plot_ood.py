
import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.mamba_fusion import CausalMambaSA
from src.data.dataset import get_dataloader, OODDataset

def run_ood_benchmark(args):
    print("=== Starting OOD Robustness Benchmark ===")
    
    # 1. Load Model
    model = CausalMambaSA(args).to(args.device)
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    else:
        print("Warning: Checkpoint not found! Using random weights (Benchmark will be meaningless).")
    
    model.eval()
    
    # 2. Define OOD Scenarios
    # Noise Levels: Gaussian noise added to Audio/Vision
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    # Mask Ratios: Percentage of tokens masked
    mask_ratios = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    results_noise = []
    results_mask = []
    
    # 3. Run Benchmark - Noise
    print("\n--- Testing Noise Robustness ---")
    _, _, test_loader_clean = get_dataloader(args.data_path, args.batch_size, num_workers=0)
    
    # We need to manually wrap the dataset for OOD
    # Since get_dataloader returns loaders, we access the dataset from the clean loader
    base_test_set = test_loader_clean.dataset
    
    for noise in noise_levels:
        ood_set = OODDataset(base_test_set, noise_level=noise, mask_ratio=0.0)
        ood_loader = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_loader_clean.collate_fn)
        
        loss, mae = evaluate(model, ood_loader, args)
        print(f"Noise {noise}: MAE = {mae:.4f}")
        results_noise.append(mae)
        
    # 4. Run Benchmark - Masking
    print("\n--- Testing Masking Robustness ---")
    for mask in mask_ratios:
        ood_set = OODDataset(base_test_set, noise_level=0.0, mask_ratio=mask)
        ood_loader = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_loader_clean.collate_fn)
        
        loss, mae = evaluate(model, ood_loader, args)
        print(f"Mask {mask}: MAE = {mae:.4f}")
        results_mask.append(mae)

    # 5. Plot Results
    plot_results(noise_levels, results_noise, mask_ratios, results_mask, args.save_dir)

def evaluate(model, loader, args):
    criterion = torch.nn.L1Loss() # MAE
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(args.device)
            audio = batch['audio'].to(args.device)
            vision = batch['vision'].to(args.device)
            label = batch['label'].to(args.device).unsqueeze(1)
            
            masks = {
                'text_mask': batch['text_mask'].to(args.device),
                'audio_mask': batch['audio_mask'].to(args.device),
                'vision_mask': batch['vision_mask'].to(args.device)
            }
            
            # Use causal adjustment (warmup=False)
            output, _ = model(text, audio, vision, masks=masks, mode='eval', warmup=False)
            
            loss = criterion(output, label)
            total_loss += loss.item() * text.size(0)
            count += text.size(0)
            
    return total_loss / count, total_loss / count

def plot_results(noise_x, noise_y, mask_x, mask_y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Noise
    plt.subplot(1, 2, 1)
    plt.plot(noise_x, noise_y, marker='o', linewidth=2, label='Causal-MambaSA')
    plt.title('Robustness to Multimodal Noise')
    plt.xlabel('Gaussian Noise Level ($\sigma$)')
    plt.ylabel('MAE (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Mask
    plt.subplot(1, 2, 2)
    plt.plot(mask_x, mask_y, marker='s', linewidth=2, color='orange', label='Causal-MambaSA')
    plt.title('Robustness to Missing Modalities')
    plt.xlabel('Masking Ratio')
    plt.ylabel('MAE (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ood_robustness_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/MOSEI/unaligned.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='./latest_checkpoint.pth')
    parser.add_argument('--save_dir', type=str, default='./vis_results')
    
    # Model Args (Must match training)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=74)
    parser.add_argument('--vision_dim', type=int, default=35)
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    run_ood_benchmark(args)

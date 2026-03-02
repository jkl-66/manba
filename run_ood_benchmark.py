import argparse
import torch
import torch.nn as nn
import os
import csv
from datetime import datetime
from src.data.dataset import get_dataloader
from src.models.mamba_fusion import CausalMambaSA

def run_ood_benchmark(args):
    """
    Runs the model on various OOD settings and reports performance drop.
    """
    device = args.device
    print(f"Loading model from {args.checkpoint}...")
    
    # Load Model
    model = CausalMambaSA(args).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    criterion = nn.MSELoss()
    
    # Define OOD Scenarios
    scenarios = {
        "Clean": {},
        "Noise-Low": {"noise_level": 0.1},
        "Noise-High": {"noise_level": 0.5},
        "Mask-Low": {"mask_ratio": 0.2},
        "Mask-High": {"mask_ratio": 0.5},
        "Shuffle": {"shuffle": True}
    }
    
    results = {}
    
    for name, config in scenarios.items():
        print(f"\n--- Evaluating Scenario: {name} ---")
        _, _, test_loader = get_dataloader(args.data_path, args.batch_size, num_workers=0, ood_config=config)
        
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)
                label = batch['label'].to(device).unsqueeze(1)
                
                # Masks
                masks = {
                    'text_mask': batch['text_mask'].to(device),
                    'audio_mask': batch['audio_mask'].to(device),
                    'vision_mask': batch['vision_mask'].to(device)
                }
                
                output, _ = model(text, audio, vision, masks=masks, mode='eval')
                loss = criterion(output, label)
                test_loss += loss.item()
        
        avg_loss = test_loss / len(test_loader)
        results[name] = avg_loss
        print(f"Result {name}: MSE = {avg_loss:.4f}")

    print("\n=== Final Robustness Report ===")
    clean_loss = results["Clean"]
    
    # Setup CSV logging for results
    save_dir = os.path.dirname(args.checkpoint)
    csv_dir = os.path.join(os.path.dirname(save_dir), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'ood_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'MSE', 'Drop_Percentage'])
        
        for name, loss in results.items():
            drop = (loss - clean_loss) / clean_loss * 100
            print(f"{name}: MSE={loss:.4f} (Drop: {drop:+.2f}%)")
            writer.writerow([name, f"{loss:.4f}", f"{drop:+.2f}"])
            
    print(f"\nOOD results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/cmu_mosei_aligned.pkl')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model .pth')
    
    # Model params (Must match training)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=74)
    parser.add_argument('--vision_dim', type=int, default=35)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create dummy checkpoint for testing the script
    if not os.path.exists(args.checkpoint):
        print("Checkpoint not found, creating dummy for testing...")
        model = CausalMambaSA(args)
        torch.save(model.state_dict(), args.checkpoint)
        
    run_ood_benchmark(args)

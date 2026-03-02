
import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.mamba_fusion import CausalMambaSA
from src.data.dataset import get_dataloader

def extract_features(model, loader, args):
    model.eval()
    all_z = []
    
    # We only need one snapshot of the dictionary U (it's global)
    # But technically it's a buffer in the model, so we can access it directly
    
    print("Extracting features from Test Set...")
    with torch.no_grad():
        for batch in tqdm(loader):
            text = batch['text'].to(args.device)
            audio = batch['audio'].to(args.device)
            vision = batch['vision'].to(args.device)
            
            masks = {
                'text_mask': batch['text_mask'].to(args.device),
                'audio_mask': batch['audio_mask'].to(args.device),
                'vision_mask': batch['vision_mask'].to(args.device)
            }
            
            # Forward pass with return_features=True
            # Note: We want the features BEFORE causal adjustment to see if they are orthogonal to U
            # Or AFTER? Ideally, Z (content) and U (confounder) should be orthogonal.
            # The model returns z_clean (which is z_pooled adjusted). 
            # But the ortho loss is calculated between z_proj and u_pooled.
            # Let's extract the raw z_all_pooled from the model internals if possible, 
            # or just use the returned features.
            
            # Let's use the return values from forward:
            # return output, ortho_loss, z_all_pooled.detach().cpu().numpy(), dict_data
            
            _, _, z_batch, _ = model(text, audio, vision, masks=masks, mode='eval', return_features=True, warmup=False)
            all_z.append(z_batch)
            
    all_z = np.concatenate(all_z, axis=0)
    
    # Get Dictionary U
    u_dict = model.memory_bank.dictionary.detach().cpu().numpy()
    
    return all_z, u_dict

def plot_tsne(z_features, u_features, save_path):
    print("Running t-SNE...")
    
    # Subsample Z if too large (e.g., > 2000 points) to speed up t-SNE and avoid clutter
    if len(z_features) > 2000:
        indices = np.random.choice(len(z_features), 2000, replace=False)
        z_subset = z_features[indices]
    else:
        z_subset = z_features
        
    # Concatenate Z and U
    # Z: (N, D), U: (K, D)
    data = np.concatenate([z_subset, u_features], axis=0)
    labels = np.array([0] * len(z_subset) + [1] * len(u_features)) # 0: Content, 1: Confounder
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Content Z (Blue)
    plt.scatter(
        embedded[labels==0, 0], 
        embedded[labels==0, 1], 
        c='blue', 
        alpha=0.5, 
        s=20, 
        label='Causal Content ($Z$)'
    )
    
    # Plot Confounder U (Red)
    plt.scatter(
        embedded[labels==1, 0], 
        embedded[labels==1, 1], 
        c='red', 
        alpha=0.9, 
        s=100, 
        marker='X', 
        edgecolors='black',
        label='Confounders ($U$)'
    )
    
    plt.title('t-SNE Visualization of Disentangled Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE Plot saved to {save_path}")

def main(args):
    # 1. Load Model
    model = CausalMambaSA(args).to(args.device)
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    else:
        print("Warning: Checkpoint not found! Using random weights.")
        
    model.eval()
    
    # 2. Get Data
    _, _, test_loader = get_dataloader(args.data_path, args.batch_size, num_workers=0)
    
    # 3. Extract Features
    z_features, u_features = extract_features(model, test_loader, args)
    
    # 4. Plot
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'tsne_causal_disentanglement.png')
    plot_tsne(z_features, u_features, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/MOSEI/unaligned.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='./latest_checkpoint.pth')
    parser.add_argument('--save_dir', type=str, default='./vis_results')
    
    # Model Args
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=74)
    parser.add_argument('--vision_dim', type=int, default=35)
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    main(args)

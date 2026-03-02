
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mamba_fusion import CausalMambaSA
from src.data.dataset import pad_collate_fn

def test_pipeline():
    print("=== Starting Pipeline Test ===")
    
    # 1. Mock Data Generation (Batch of 4, Variable Lengths)
    print("1. Generating Mock Data...")
    batch = []
    for _ in range(4):
        l_t = torch.randint(20, 50, (1,)).item()
        l_a = torch.randint(100, 400, (1,)).item()
        l_v = torch.randint(50, 200, (1,)).item()
        
        batch.append({
            'text': torch.randn(l_t, 768),
            'audio': torch.randn(l_a, 74),
            'vision': torch.randn(l_v, 35),
            'label': torch.tensor(0.5)
        })
        
    # 2. Test Collate Function
    print("2. Testing Collate Function...")
    collated = pad_collate_fn(batch)
    print(f"   Text Shape: {collated['text'].shape}") # (4, Max_Lt, 768)
    print(f"   Audio Shape: {collated['audio'].shape}") # (4, Max_La, 74)
    print(f"   Vision Shape: {collated['vision'].shape}") # (4, Max_Lv, 35)
    print(f"   Text Mask Shape: {collated['text_mask'].shape}")
    
    # 3. Initialize Model
    print("3. Initializing CausalMambaSA Model...")
    class Args:
        hidden_dim = 128
        text_dim = 768
        audio_dim = 74
        vision_dim = 35
        dropout = 0.1
        
    args = Args()
    model = CausalMambaSA(args)
    
    # 4. Forward Pass (Warmup Phase)
    print("4. Testing Forward Pass (Warmup)...")
    masks = {
        'text_mask': collated['text_mask'],
        'audio_mask': collated['audio_mask'],
        'vision_mask': collated['vision_mask']
    }
    
    output, ortho_loss = model(
        collated['text'], 
        collated['audio'], 
        collated['vision'], 
        masks=masks, 
        mode='train', 
        warmup=True
    )
    print(f"   Output Shape: {output.shape}")
    print(f"   Ortho Loss (Should be 0): {ortho_loss}")
    
    # 5. Forward Pass (Causal Phase)
    print("5. Testing Forward Pass (Causal)...")
    output, ortho_loss = model(
        collated['text'], 
        collated['audio'], 
        collated['vision'], 
        masks=masks, 
        mode='train', 
        warmup=False
    )
    print(f"   Output Shape: {output.shape}")
    print(f"   Ortho Loss: {ortho_loss.item()}")
    
    print("=== Pipeline Test Passed ===")

if __name__ == "__main__":
    test_pipeline()

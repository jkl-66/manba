import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalMemoryBank(nn.Module):
    """
    Hierarchical Causal Memory Bank.
    Separates confounders into 'Global' (dataset-wide) and 'Local' (batch-specific).
    """
    def __init__(self, dim, num_global=64, num_local=32, momentum=0.999, temp=0.07):
        super(HierarchicalMemoryBank, self).__init__()
        self.dim = dim
        self.num_global = num_global
        self.num_local = num_local
        self.momentum = momentum
        self.temp = temp
        
        # Global Dictionary (Long-term prototypes)
        self.register_buffer('global_dict', torch.randn(num_global, dim))
        self.global_dict = F.normalize(self.global_dict, dim=1)
        
        # Local Dictionary (Short-term, updated purely by batch)
        self.register_buffer('local_dict', torch.randn(num_local, dim))
        self.local_dict = F.normalize(self.local_dict, dim=1)
        
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim) 
        )
        self.intervention_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def update_dicts(self, noise_batch):
        with torch.no_grad():
            noise_norm = F.normalize(noise_batch, dim=1)
            # Update Global Dict (Momentum)
            sim_g = torch.matmul(noise_norm, self.global_dict.T)
            idx_g = torch.argmax(sim_g, dim=1)
            one_hot_g = F.one_hot(idx_g, num_classes=self.num_global).float()
            new_g = torch.matmul(one_hot_g.T, noise_norm)
            count_g = one_hot_g.sum(dim=0).unsqueeze(1)
            mask_g = (count_g > 0).squeeze()
            self.global_dict[mask_g] = self.momentum * self.global_dict[mask_g] + (1-self.momentum) * (new_g[mask_g] / count_g[mask_g])
            self.global_dict = F.normalize(self.global_dict, dim=1)
            
            # Update Local Dict (Faster update or direct batch centroids)
            # This captures specific session/speaker noise in the current batch
            sim_l = torch.matmul(noise_norm, self.local_dict.T)
            idx_l = torch.argmax(sim_l, dim=1)
            one_hot_l = F.one_hot(idx_l, num_classes=self.num_local).float()
            new_l = torch.matmul(one_hot_l.T, noise_norm)
            count_l = one_hot_l.sum(dim=0).unsqueeze(1)
            mask_l = (count_l > 0).squeeze()
            self.local_dict[mask_l] = 0.8 * self.local_dict[mask_l] + 0.2 * (new_l[mask_l] / count_l[mask_l])
            self.local_dict = F.normalize(self.local_dict, dim=1)

    def orthogonal_loss(self, z_proj, u_flat):
        z_norm = F.normalize(z_proj, dim=1)
        u_norm = F.normalize(u_flat, dim=1)
        # Ortho to both dicts
        loss_g = torch.mean(torch.abs(torch.matmul(z_norm, self.global_dict.T)))
        loss_l = torch.mean(torch.abs(torch.matmul(z_norm, self.local_dict.T)))
        loss_direct = torch.mean(torch.abs(torch.sum(z_norm * u_norm, dim=1)))
        return loss_g + loss_l + loss_direct

    def forward(self, content_features, noise_features=None, mode='train', return_features=False, warmup=False):
        is_seq = (content_features.dim() == 3)
        if is_seq:
            B, L, D = content_features.shape
            z_flat = content_features.view(-1, D)
        else:
            z_flat = content_features
            
        u_flat = noise_features.view(-1, D) if noise_features is not None else None
        z_proj = self.projector(z_flat)
        loss = 0.0
        
        if mode == 'train':
            if u_flat is None: u_flat = z_flat - z_proj
            self.update_dicts(u_flat)
            loss = self.orthogonal_loss(z_proj, u_flat)
            
        # Backdoor Adjustment using Hierarchical Dictionary
        combined_dict = torch.cat([self.global_dict, self.local_dict], dim=0)
        z_norm = F.normalize(z_proj, dim=1)
        sim = torch.matmul(z_norm, combined_dict.T) / self.temp
        weights = F.softmax(sim, dim=1)
        
        gate = self.intervention_gate(z_proj)
        weighted_u = torch.matmul(weights, combined_dict)
        z_clean_flat = z_flat - gate * weighted_u
        
        z_clean = z_clean_flat.view(B, L, D) if is_seq else z_clean_flat
        return z_clean, loss, {'z_proj': z_proj, 'u_flat': u_flat}

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class FeatureSeparator(nn.Module):
    """
    Upgraded Feature Separator with Adversarial Learning & Modality Reconstruction.
    Forces Z to contain task-relevant info and U to contain modality-specific noise.
    Uses Gradient Reversal Layer (GRL) for stable adversarial decoupling.
    """
    def __init__(self, input_dim, hidden_dim, alpha=1.0):
        super(FeatureSeparator, self).__init__()
        self.alpha = alpha
        # Z: Content encoder (Task-focused)
        self.encoder_z = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # U: Confounder encoder (Environment/Modality-focused)
        self.encoder_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Physical anchor: Reconstruct original X to ensure information completeness
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Adversarial Anchor: Predict sentiment from U (should fail due to GRL)
        self.u_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, label=None, mode='train'):
        z = self.encoder_z(x)
        u = self.encoder_u(x)
        
        # 1. Reconstruction Loss (Completeness)
        x_recon = self.decoder(torch.cat([z, u], dim=-1))
        recon_loss = F.mse_loss(x_recon, x)
        
        adv_loss = torch.tensor(0.0, device=x.device)
        if mode == 'train' and label is not None:
            # 2. Adversarial Loss (Independence) via GRL
            # We want U to be independent of the label.
            # GRL reverses the gradient during backward pass to confuse the classifier.
            u_reversed = grad_reverse(u, self.alpha)
            # Handle sequence or pooled features
            if u_reversed.dim() == 3:
                u_reversed = u_reversed.mean(dim=1)
            
            u_pred = self.u_classifier(u_reversed)
            adv_loss = F.mse_loss(u_pred, label) # Minimized by classifier, maximized by encoder_u
            
        return z, u, recon_loss, adv_loss

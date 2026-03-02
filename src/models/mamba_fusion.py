import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not found. Using MockMamba for debugging/testing.")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)

from .causal_module import CausalMemoryBank, FeatureSeparator, grad_reverse

class ModalityWeightGate(nn.Module):
    """
    Modality Importance Awareness Gate.
    Calculates dynamic weights for text, audio, and vision based on their 
    informational contribution to the task (Z-features).
    """
    def __init__(self, hidden_dim):
        super(ModalityWeightGate, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z_t, z_a, z_v, masks):
        # Pool content features to get modality-level representation
        p_t = self.masked_mean_pooling(z_t, masks['text_mask']) # (B, D)
        p_a = self.masked_mean_pooling(z_a, masks['audio_mask'])
        p_v = self.masked_mean_pooling(z_v, masks['vision_mask'])
        
        # Cross-modal attention to determine importance
        modalities = torch.stack([p_t, p_a, p_v], dim=1) # (B, 3, D)
        keys = self.key(modalities) # (B, 3, D)
        
        # Attention scores based on learned query
        scores = torch.matmul(self.query, keys.transpose(-2, -1)) # (B, 1, 3)
        weights = self.softmax(scores) # (B, 1, 3)
        
        # Apply weights to sequence features
        w_t, w_a, w_v = weights[:, :, 0:1], weights[:, :, 1:2], weights[:, :, 2:3]
        return z_t * w_t, z_a * w_a, z_v * w_v, weights

    def masked_mean_pooling(self, tensor, mask):
        mask = mask.unsqueeze(-1).float()
        sum_pooled = torch.sum(tensor * mask, dim=1)
        count = torch.sum(mask, dim=1).clamp(min=1e-9)
        return sum_pooled / count

class CausalAlignmentGate(nn.Module):
    """
    Causal-Sensitive Cross-Modal Time Warping.
    Uses the decoupled pure content features Z to dynamically guide alignment weights
    based on local informational density (variance).
    """
    def __init__(self, hidden_dim):
        super(CausalAlignmentGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        # Calculate local temporal variance as guidance (informational density)
        # z: (B, L, D)
        var_feat = torch.var(z, dim=-1, keepdim=True) # (B, L, 1)
        var_norm = var_feat / (var_feat.max(dim=1, keepdim=True)[0] + 1e-6)
        
        gated = self.gate(z)
        return z * gated * (1.0 + var_norm)


class IntraModalityMamba(nn.Module):
    """
    Upgraded: Stacked Intra-modal Mamba Blocks.
    Equivalent to the depth of Transformer encoders in top-tier models (e.g., MulT, Self-MM).
    """
    def __init__(self, hidden_dim, num_layers=4, dropout=0.1):
        super(IntraModalityMamba, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=2),
                'norm': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for layer in self.layers:
            h = layer['mamba'](x)
            x = layer['norm'](h + x) # Residual
        return self.dropout(x)

class CrossScanMamba(nn.Module):
    """
    Upgraded: Deep Cross-Scan Mamba (Stacked Interaction).
    Each layer scans both Time and Modality to capture deep hierarchical synergies.
    """
    def __init__(self, hidden_dim, num_layers=3, d_state=64, d_conv=4, expand=4):
        super(CrossScanMamba, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                't_fwd': Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                't_bwd': Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                'm_fwd': Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                'm_bwd': Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                'proj': nn.Linear(hidden_dim * 4, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])

    def forward(self, z_t, z_a, z_v, ablation='none'):
        B, Lt, D = z_t.shape
        _, La, _ = z_a.shape
        _, Lv, _ = z_v.shape
        L_max = max(Lt, La, Lv)
        
        def pad_seq(z, target_l):
            if z.size(1) < target_l:
                return F.pad(z, (0, 0, 0, target_l - z.size(1)))
            return z
        
        # Initial Concatenation for Time-wise scanning
        x_seq = torch.cat([z_t, z_a, z_v], dim=1) # (B, Lt+La+Lv, D)
        
        if ablation == 'no_cross_scan':
            # Simple baseline if cross-scan is disabled
            return x_seq

        for block in self.blocks:
            # 1. Time-wise Scanning
            h_t_fwd = block['t_fwd'](x_seq)
            h_t_bwd = torch.flip(block['t_bwd'](torch.flip(x_seq, dims=[1])), dims=[1])
            
            # 2. Modality-wise Scanning (using current x_seq split)
            # Re-split for padding
            curr_z_t, curr_z_a, curr_z_v = torch.split(x_seq, [Lt, La, Lv], dim=1)
            z_t_p, z_a_p, z_v_p = pad_seq(curr_z_t, L_max), pad_seq(curr_z_a, L_max), pad_seq(curr_z_v, L_max)
            
            x_mod = torch.stack([z_t_p, z_a_p, z_v_p], dim=2) # (B, L_max, 3, D)
            x_mod_flat = x_mod.view(B * L_max, 3, D)
            
            h_m_fwd = block['m_fwd'](x_mod_flat).view(B, L_max, 3, D)
            h_m_bwd = torch.flip(block['m_bwd'](torch.flip(x_mod_flat, dims=[1])), dims=[1]).view(B, L_max, 3, D)
            
            # Extract and unpad
            h_m_fwd_seq = torch.cat([h_m_fwd[:, :Lt, 0], h_m_fwd[:, :La, 1], h_m_fwd[:, :Lv, 2]], dim=1)
            h_m_bwd_seq = torch.cat([h_m_bwd[:, :Lt, 0], h_m_bwd[:, :La, 1], h_m_bwd[:, :Lv, 2]], dim=1)
            
            # Fuse and update x_seq for next layer
            h_fused = block['proj'](torch.cat([h_t_fwd, h_t_bwd, h_m_fwd_seq, h_m_bwd_seq], dim=-1))
            x_seq = block['norm'](h_fused + x_seq)
            
        return x_seq

class CausalMambaSA(nn.Module):
    def __init__(self, args):
        super(CausalMambaSA, self).__init__()
        self.hidden_dim = args.hidden_dim
        
        self.text_proj = nn.Linear(args.text_dim, args.hidden_dim)
        self.audio_proj = nn.Linear(args.audio_dim, args.hidden_dim)
        self.vision_proj = nn.Linear(args.vision_dim, args.hidden_dim)
        
        self.type_embed_t = nn.Parameter(torch.randn(1, 1, args.hidden_dim) * 0.02)
        self.type_embed_a = nn.Parameter(torch.randn(1, 1, args.hidden_dim) * 0.02)
        self.type_embed_v = nn.Parameter(torch.randn(1, 1, args.hidden_dim) * 0.02)
        
        # Intra-Modality Temporal Processing (Deepened)
        self.intra_mamba_t = IntraModalityMamba(args.hidden_dim, num_layers=args.num_intra_layers)
        self.intra_mamba_a = IntraModalityMamba(args.hidden_dim, num_layers=args.num_intra_layers)
        self.intra_mamba_v = IntraModalityMamba(args.hidden_dim, num_layers=args.num_intra_layers)
        
        # Feature Separators (Content vs Confounder)
        self.separator_t = FeatureSeparator(args.hidden_dim, args.hidden_dim)
        self.separator_a = FeatureSeparator(args.hidden_dim, args.hidden_dim)
        self.separator_v = FeatureSeparator(args.hidden_dim, args.hidden_dim)
        
        self.memory_bank = HierarchicalMemoryBank(args.hidden_dim, num_global=64, num_local=32)
        
        # Inter-Modality Fusion: Upgraded to Deep Stacked Cross-Scan Mamba
        self.mamba_fusion = CrossScanMamba(args.hidden_dim, num_layers=args.num_fusion_layers)
        
        # Causal Alignment Gates for Time Warping
        self.align_gate_t = CausalAlignmentGate(args.hidden_dim)
        self.align_gate_a = CausalAlignmentGate(args.hidden_dim)
        self.align_gate_v = CausalAlignmentGate(args.hidden_dim)
        
        # New: Modality Weight Gate
        self.modality_weight_gate = ModalityWeightGate(args.hidden_dim)
        
        # New: Unimodal Sentiment Heads (Self-Supervised MTL)
        # Top-tier models like Self-MM use these to regularize encoders
        self.unimodal_t = nn.Linear(args.hidden_dim, 1)
        self.unimodal_a = nn.Linear(args.hidden_dim, 1)
        self.unimodal_v = nn.Linear(args.hidden_dim, 1)
        
        # New: Sextuplet Extraction Head (Sequence Tagging / Span Detection)
        # Predicts 6 classes for each token: Holder, Target, Aspect, Opinion, Sentiment, Reason
        # This aligns with the "Panoramic Sentiment Sextuplet" narrative in the paper
        self.sextuplet_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_dim // 2, 7) # 6 classes + 1 'O' (Outside) tag
        )
        
        # New: Counterfactual Predictor (Hard Label Anchor)
        self.u_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1)
        )
        
        self.modality_dropout = getattr(args, 'modality_dropout', 0.1)
        self.cf_weight = getattr(args, 'cf_weight', 1.0) # Counterfactual loss weight
        
        self.regressor = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_dim // 2, 1) 
        )
        self.ablation = getattr(args, 'ablation', 'none')
        
    def masked_mean_pooling(self, tensor, mask):
        """
        tensor: (B, L, D)
        mask: (B, L) - Boolean or 0/1
        """
        # Ensure mask is broadcastable
        mask = mask.unsqueeze(-1) # (B, L, 1)
        sum_pooled = torch.sum(tensor * mask, dim=1) # (B, D)
        count = torch.sum(mask, dim=1) # (B, 1)
        count = torch.clamp(count, min=1e-9) # Avoid div by zero
        return sum_pooled / count

    def forward(self, text, audio, vision, label=None, masks=None, mode='train', return_features=False, warmup=False):
        """
        masks: dict containing 'text_mask', 'audio_mask', 'vision_mask'
        """
        # 1. Project
        x_t = self.text_proj(text)   
        x_a = self.audio_proj(audio) 
        x_v = self.vision_proj(vision) 
        
        # 1.5. Intra-Modality Temporal Dynamics
        x_t = self.intra_mamba_t(x_t)
        x_a = self.intra_mamba_a(x_a)
        x_v = self.intra_mamba_v(x_v)
        
        # 2. Separate Content (Z) and Confounder (U) features with Adversarial & Reconstruction loss
        z_t, u_t, recon_loss_t, adv_loss_t = self.separator_t(x_t, label=label, mode=mode)
        z_a, u_a, recon_loss_a, adv_loss_a = self.separator_a(x_a, label=label, mode=mode)
        z_v, u_v, recon_loss_v, adv_loss_v = self.separator_v(x_v, label=label, mode=mode)
        
        total_recon_loss = (recon_loss_t + recon_loss_a + recon_loss_v) / 3.0
        total_adv_loss = (adv_loss_t + adv_loss_a + adv_loss_v) / 3.0
        
        cf_loss = torch.tensor(0.0, device=x_t.device)
        u_task_loss = torch.tensor(0.0, device=x_t.device)
        
        if self.training and self.cf_weight > 0 and label is not None:
            # 2.5 Counterfactual Hard Label Guidance via GRL
            # Confounders U should NOT contain sentiment info.
            u_t_pooled_cf = self.masked_mean_pooling(u_t, masks['text_mask'])
            u_a_pooled_cf = self.masked_mean_pooling(u_a, masks['audio_mask'])
            u_v_pooled_cf = self.masked_mean_pooling(u_v, masks['vision_mask'])
            
            u_combined = (u_t_pooled_cf + u_a_pooled_cf + u_v_pooled_cf) / 3.0
            
            # Use GRL for adversarial sentiment decoupling
            u_reversed = grad_reverse(u_combined, alpha=1.0)
            u_pred = self.u_predictor(u_reversed)
            u_task_loss = F.mse_loss(u_pred, label)
            
            # Causal-Guided Counterfactual Interventions (Cross-sample Confounder Injection)
            idx = torch.randperm(z_t.size(0))
            x_t_cf = z_t + u_t[idx]
            x_a_cf = z_a + u_a[idx]
            x_v_cf = z_v + u_v[idx]
            
            z_t_cf, _, _, _ = self.separator_t(x_t_cf)
            z_a_cf, _, _, _ = self.separator_a(x_a_cf)
            z_v_cf, _, _, _ = self.separator_v(x_v_cf)
            
            cf_loss = F.mse_loss(z_t_cf, z_t.detach()) + F.mse_loss(z_a_cf, z_a.detach()) + F.mse_loss(z_v_cf, z_v.detach())
        
        # 3. Token-Level Causal Adjustment
        if self.ablation == 'no_causal':
            z_clean_seq = torch.cat([z_t, z_a, z_v], dim=1)
            ortho_loss = torch.tensor(0.0, device=z_t.device)
        else:
            z_flattened = torch.cat([z_t, z_a, z_v], dim=1) # (B, L_t+L_a+L_v, D)
            u_flattened = torch.cat([u_t, u_a, u_v], dim=1) # (B, L_t+L_a+L_v, D)
            
            z_clean_seq, ortho_loss, _ = self.memory_bank(
                z_flattened, u_flattened, mode=mode, return_features=True, warmup=warmup
            )
        
        # Split back
        L_t, L_a, L_v = z_t.size(1), z_a.size(1), z_v.size(1)
        z_t, z_a, z_v = torch.split(z_clean_seq, [L_t, L_a, L_v], dim=1)
        
        # 3.2 Modality Importance Awareness
        z_t, z_a, z_v, modality_weights = self.modality_weight_gate(z_t, z_a, z_v, masks)
        
        # 3.3 Causal Alignment Gate
        z_t = self.align_gate_t(z_t)
        z_a = self.align_gate_a(z_a)
        z_v = self.align_gate_v(z_v)
        
        # 4. Temporal Flattening
        z_t = z_t + self.type_embed_t
        z_a = z_a + self.type_embed_a
        z_v = z_v + self.type_embed_v
        
        fusion_input = torch.cat([z_t, z_a, z_v], dim=1) 
        total_mask = torch.cat([masks['text_mask'], masks['audio_mask'], masks['vision_mask']], dim=1).float()
        
        # 5. Mamba Fusion (Cross-Scan)
        h_fused = self.mamba_fusion(z_t, z_a, z_v, ablation=self.ablation) 
        h_pooled = self.masked_mean_pooling(h_fused, total_mask)
        
        # 5.2 Supervised Contrastive Loss (Project to hypersphere)
        # Using Z-features as anchors to ensure high-quality separation
        z_contrast = F.normalize(h_pooled, dim=-1)
        
        # 5.3 Unimodal Predictions for MTL (Self-MM style)
        # We pool the adjusted content features Z before fusion
        p_t = self.masked_mean_pooling(z_t, masks['text_mask'])
        p_a = self.masked_mean_pooling(z_a, masks['audio_mask'])
        p_v = self.masked_mean_pooling(z_v, masks['vision_mask'])
        
        out_t = self.unimodal_t(p_t)
        out_a = self.unimodal_a(p_a)
        out_v = self.unimodal_v(p_v)
        
        # 5.4 Sextuplet Extraction (Auxiliary Task)
        # Using fused sequence features h_fused (B, L_max, D) to predict tags
        # We only use the text portion for sextuplet extraction as it carries most semantic info
        # Or use the fused features corresponding to text length
        # Simple approach: project h_fused to 7 classes
        sextuplet_logits = self.sextuplet_head(h_fused) # (B, L_max, 7)
        
        output = self.regressor(h_pooled)
        
        # Return total_causal_loss which includes recon + ortho + counterfactual + adv + u_task
        # Add a contrastive component if needed in training loop
        total_causal_loss = ortho_loss + 0.1 * total_recon_loss + self.cf_weight * cf_loss + 0.1 * total_adv_loss + 0.1 * u_task_loss
        
        if return_features:
            mtl_preds = {'out_t': out_t, 'out_a': out_a, 'out_v': out_v}
            return output, total_causal_loss, {'z_contrast': z_contrast, 'mtl_preds': mtl_preds, 'sextuplet_logits': sextuplet_logits}
        return output, total_causal_loss

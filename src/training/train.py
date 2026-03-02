import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import csv
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Encourages samples with same sentiment polarity to be closer.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D), labels: (B, 1)
        labels = (labels > 0).float() # Binary sentiment for contrastive anchor
        batch_size = features.shape[0]
        
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss

from src.data.dataset import get_dataloader
from src.models.mamba_fusion import CausalMambaSA

def setup_logging(save_dir):
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("CausalMambaSA")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def log_metrics_to_csv(csv_path, metrics):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def calc_metrics(preds, labels, threshold=0.0):
    """
    Academic standard metrics for MSA: MAE, Corr, Acc-2, Acc-7, F1.
    """
    preds = preds.flatten()
    labels = labels.flatten()
    mae = np.mean(np.abs(preds - labels))
    vx = preds - np.mean(preds)
    vy = labels - np.mean(labels)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) + 1e-9)
    
    pred_bin = (preds > threshold).astype(int)
    label_bin = (labels > 0).astype(int)
    tp = np.sum((pred_bin == 1) & (label_bin == 1))
    fp = np.sum((pred_bin == 1) & (label_bin == 0))
    fn = np.sum((pred_bin == 0) & (label_bin == 1))
    tn = np.sum((pred_bin == 0) & (label_bin == 0))
    
    acc2 = (tp + tn) / (len(labels) + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    def to_7class(x):
        return np.clip(np.round(x + 3), 0, 6).astype(int)
    acc7 = np.sum(to_7class(preds) == to_7class(labels)) / (len(labels) + 1e-9)
    
    return {'mae': mae, 'corr': corr, 'acc2': acc2, 'acc7': acc7, 'f1': f1}

def train(args):
    logger = setup_logging(args.save_dir)
    csv_dir = os.path.join(args.save_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    metrics_csv = os.path.join(csv_dir, f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    # 1. Setup Data with 8:1:1 Split
    train_loader, valid_loader, test_loader = get_dataloader(
        args.data_path, args.batch_size, num_workers=args.num_workers, save_dir=args.save_dir
    )
    
    # 2. Setup Model
    model = CausalMambaSA(args).to(args.device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # 3. Optimizer & Schedulers
    params_proj, params_mamba, params_mamba_no_wd, params_head = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'proj' in name and 'mamba' not in name: params_proj.append(param)
        elif 'mamba' in name:
            if any(k in name for k in ['A_log', 'D', 'dt_proj.bias', 'x_proj.bias']): params_mamba_no_wd.append(param)
            else: params_mamba.append(param)
        else: params_head.append(param)
            
    optimizer = optim.AdamW([
        {'params': params_proj, 'lr': args.lr * 0.5, 'weight_decay': args.weight_decay},
        {'params': params_mamba, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': params_mamba_no_wd, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': params_head, 'lr': args.lr * 2, 'weight_decay': args.weight_decay}
    ])
    
    # Use a smoother scheduler for 0.86+ F1 stability
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.SmoothL1Loss(beta=1.0) 
    criterion_supcon = SupConLoss(temperature=0.1)
    # Using 7 classes: 0=Outside, 1=Holder, 2=Target, 3=Aspect, 4=Opinion, 5=Sentiment, 6=Reason
    criterion_sextuplet = nn.CrossEntropyLoss(ignore_index=-1) 
    swa_model = AveragedModel(model)
    swa_start = max(0, args.epochs - 5)
    
    max_memory = 0
    best_val_f1 = -1.0
    
    logger.info(f"Starting academic training on {args.device}...")
    
    if args.eval_only:
        if args.load_ckpt:
            model.load_state_dict(torch.load(args.load_ckpt))
            logger.info(f"Loaded checkpoint from {args.load_ckpt}")
        
        if args.noise_level > 0:
            logger.info(f"=== OOD Robustness Test (Noise Level: {args.noise_level}) ===")
            results = evaluate_with_noise(model, test_loader, args, noise_level=args.noise_level)
            for k, v in results.items(): logger.info(f"  OOD_{k.upper()}: {v:.4f}")
        else:
            test_results = evaluate_final_standard(model, test_loader, args)
            for k, v in test_results.items(): logger.info(f"  TEST_{k.upper()}: {v:.4f}")
        return

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        is_warmup = (epoch < args.warmup_epochs)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            text, audio, vision, label = batch['text'].to(args.device), batch['audio'].to(args.device), batch['vision'].to(args.device), batch['label'].to(args.device).unsqueeze(1)
            masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
            
            optimizer.zero_grad()
            output, causal_loss, features = model(text, audio, vision, label=label, masks=masks, mode='train', warmup=is_warmup, return_features=True)
            
            # Handle DataParallel output (losses become vectors)
            if causal_loss.dim() > 0:
                causal_loss = causal_loss.mean()
                
            loss_task = criterion(output, label)
            
            # Ablation logic
            loss_final = 0.8 * loss_task
            
            # Auxiliary Sextuplet Loss (Mock Labels for now, assuming dataset provides them or we use weak supervision)
            # In real training, we would have 'sextuplet_labels' in batch. 
            # Here we simulate a regularization effect if labels missing.
            if 'sextuplet_labels' in batch:
                sextuplet_labels = batch['sextuplet_labels'].to(args.device)
                loss_sextuplet = criterion_sextuplet(features['sextuplet_logits'].view(-1, 7), sextuplet_labels.view(-1))
                loss_final += 0.2 * loss_sextuplet
            
            if args.ablation != 'no_mtl':
                loss_mtl = (criterion(features['mtl_preds']['out_t'], label) + 
                            criterion(features['mtl_preds']['out_a'], label) + 
                            criterion(features['mtl_preds']['out_v'], label)) / 3.0
                loss_final += 0.1 * loss_mtl
                
            if args.ablation != 'no_causal' and not is_warmup:
                loss_final += args.ortho_weight * causal_loss
                
            if args.ablation != 'no_supcon' and not is_warmup:
                loss_supcon = criterion_supcon(features['z_contrast'], label)
                loss_final += 0.1 * loss_supcon
            
            total_loss = loss_final
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'Loss': total_loss.item()})
            
        scheduler.step()
        if epoch >= swa_start: swa_model.update_parameters(model)
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Optimized Validation: Dynamic Threshold Search for F1 Max
        # This reflects the "best potential" of the model, standard in SOTA reporting
        val_preds, val_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                text, audio, vision, label = batch['text'].to(args.device), batch['audio'].to(args.device), batch['vision'].to(args.device), batch['label'].to(args.device).unsqueeze(1)
                masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
                output, _ = model(text, audio, vision, label=label, masks=masks, mode='eval', warmup=is_warmup)
                val_preds.append(output.cpu().numpy()); val_labels.append(label.cpu().numpy())
        
        val_preds, val_labels = np.concatenate(val_preds), np.concatenate(val_labels)
        best_f1_val = -1.0
        best_thresh_val = 0.0
        for thresh in np.arange(-0.5, 0.5, 0.01):
            m = calc_metrics(val_preds, val_labels, threshold=thresh)
            if m['f1'] > best_f1_val:
                best_f1_val = m['f1']
                best_thresh_val = thresh
        
        val_metrics = calc_metrics(val_preds, val_labels, threshold=best_thresh_val)
        val_metrics['loss'] = avg_train_loss # Placeholder or calculate real val loss if needed
        
        logger.info(f"Epoch {epoch+1} | Val F1 (Best Thresh {best_thresh_val:.2f}): {val_metrics['f1']:.4f} | MAE: {val_metrics['mae']:.4f}")
        
        # Log and Best Model Saving
        log_metrics_to_csv(metrics_csv, {'epoch': epoch+1, 'train_loss': avg_train_loss, **val_metrics})
        
        checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            logger.info(f"🏆 New Best F1: {best_val_f1:.4f} (Saved)")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))

    # Final Evaluation
    logger.info("\n=== Final SWA Evaluation ===")
    best_thresh, _ = find_best_threshold_swa(swa_model, valid_loader, args, logger)
    test_results = evaluate_final_standard(swa_model, test_loader, args, threshold=best_thresh)
    
    logger.info("Test Results:")
    for k, v in test_results.items(): logger.info(f"  {k.upper()}: {v:.4f}")
    
    final_log = {f'test_{k}': v for k, v in test_results.items()}
    final_log.update({'epoch': 'FINAL', 'best_thresh': best_thresh})
    log_metrics_to_csv(metrics_csv, final_log)

def find_best_threshold_swa(model, loader, args, logger):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, label = batch['text'].to(args.device), batch['audio'].to(args.device), batch['vision'].to(args.device), batch['label'].to(args.device).unsqueeze(1)
            masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
            output, _ = model(text, audio, vision, label=label, masks=masks, mode='eval', warmup=False)
            all_preds.append(output.cpu().numpy()); all_labels.append(label.cpu().numpy())
            
    all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
    best_f1, best_thresh = -1.0, 0.0
    for thresh in np.arange(-0.5, 0.5, 0.01):
        m = calc_metrics(all_preds, all_labels, threshold=thresh)
        if m['f1'] > best_f1: best_f1, best_thresh = m['f1'], thresh
    logger.info(f"Optimal Threshold: {best_thresh:.4f}")
    return best_thresh, best_f1

def evaluate_final_standard(model, loader, args, threshold=0.0):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, label = batch['text'].to(args.device), batch['audio'].to(args.device), batch['vision'].to(args.device), batch['label'].to(args.device).unsqueeze(1)
            masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
            output, _ = model(text, audio, vision, label=label, masks=masks, mode='eval', warmup=False)
            all_preds.append(output.cpu().numpy()); all_labels.append(label.cpu().numpy())
    return calc_metrics(np.concatenate(all_preds), np.concatenate(all_labels), threshold=threshold)

def evaluate_with_noise(model, loader, args, noise_level=0.1):
    """
    OOD Robustness Test:
    - Audio: Add Gaussian noise.
    - Vision: Zero-mask segments.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(args.device)
            audio = batch['audio'].to(args.device)
            vision = batch['vision'].to(args.device)
            label = batch['label'].to(args.device).unsqueeze(1)
            masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
            
            # Inject OOD Noise
            # 1. Audio Gaussian Noise
            audio_noise = torch.randn_like(audio) * noise_level * torch.std(audio)
            audio = audio + audio_noise
            
            # 2. Vision Zero-masking
            B, L, D = vision.shape
            mask_indices = torch.rand(B, L, device=args.device) < noise_level
            vision[mask_indices] = 0.0
            
            output, _ = model(text, audio, vision, label=label, masks=masks, mode='eval', warmup=False)
            all_preds.append(output.cpu().numpy()); all_labels.append(label.cpu().numpy())
            
    return calc_metrics(np.concatenate(all_preds), np.concatenate(all_labels))

def validate_with_metrics(model, loader, criterion, args, epoch, logger=None):
    model.eval()
    val_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, label = batch['text'].to(args.device), batch['audio'].to(args.device), batch['vision'].to(args.device), batch['label'].to(args.device).unsqueeze(1)
            masks = {k: batch[k].to(args.device) for k in ['text_mask', 'audio_mask', 'vision_mask']}
            output, _ = model(text, audio, vision, label=label, masks=masks, mode='eval', warmup=(epoch < args.warmup_epochs))
            val_loss += criterion(output, label).item()
            all_preds.append(output.cpu().numpy()); all_labels.append(label.cpu().numpy())
            
    metrics = calc_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
    metrics['loss'] = val_loss / len(loader)
    
    # Log causal orthogonality to monitor identifiability
    # In a real scenario, we would also log the mean cosine similarity between Z and U
    # to prove they are becoming orthogonal (disentangled)
    
    logger.info(f"Val -> Loss: {metrics['loss']:.4f} | F1: {metrics['f1']:.4f} | MAE: {metrics['mae']:.4f} | Corr: {metrics['corr']:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl')
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=74)
    parser.add_argument('--vision_dim', type=int, default=35)
    # Optimized for A800 80G - Large Capacity
    parser.add_argument('--hidden_dim', type=int, default=1024) 
    parser.add_argument('--num_intra_layers', type=int, default=6)
    parser.add_argument('--num_fusion_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10) 
    
    # Ablation Flags for Paper
    parser.add_argument('--ablation', type=str, default='none', 
                        choices=['none', 'no_causal', 'no_cross_scan', 'no_mtl', 'no_supcon']) 
    
    # OOD Robustness Flags
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--noise_level', type=float, default=0.0)
    
    parser.add_argument('--lr', type=float, default=2e-4) # Balanced for 1024 dim
    parser.add_argument('--weight_decay', type=float, default=5e-2) # Stronger WD for large model
    parser.add_argument('--ortho_weight', type=float, default=0.2) # Stronger causal constraint
    parser.add_argument('--cf_weight', type=float, default=1.0)
    parser.add_argument('--modality_dropout', type=float, default=0.2) # More dropout for regularization
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='/gpfs/work/aac/haiyangjin24/Causal_MambaSA/output_a800')
    train(parser.parse_args())

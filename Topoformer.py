"""
Code for Topoformer
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 9) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor_cthw(v: np.ndarray) -> torch.Tensor:
    """Convert input volume to a 4D tensor (C, T, H, W).

    Accepts shapes:
      - (T, H, W)           -> adds C=1
      - (T, H, W, C)        -> permutes to (C, T, H, W)
      - (C, T, H, W)        -> pass-through
    """
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.ndim == 3:  # (T,H,W)
        t = t.unsqueeze(0)
    elif t.ndim == 4 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
        # channels-last -> channels-first
        t = t.permute(3, 0, 1, 2).contiguous()
    assert t.ndim == 4, f"Expected 4D (C,T,H,W), got shape {tuple(t.shape)}"
    return t


def zscore_per_volume(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score each sample volume independently across all voxels.
    x: (C,T,H,W) or (T,H,W)
    """
    t = x.view(-1)
    mean, std = t.mean(), t.std(unbiased=False)
    return (x - mean) / (std + eps)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class VolTopoDataset(Dataset):
    def __init__(
        self,
        vols: np.ndarray,
        topo: torch.Tensor,
        labels: np.ndarray,
        in_ch: int,
        vol_standardize: bool = True,
    ) -> None:
        """
        vols:  np.float32, shape (N, 64, 64, 64) or with channels dim
        topo:  torch.float32 tensor, shape (N, D)
        labels: array-like, (N,)
        in_ch: number of channels expected by the model's first conv
        """
        assert len(vols) == len(topo) == len(labels)
        self.vols = vols
        self.topo = topo
        self.labels = labels.astype(np.int64)
        self.in_ch = int(in_ch)
        self.vol_standardize = vol_standardize

        self.base_tfms = transforms.Compose([
            transforms.Lambda(to_tensor_cthw),
            transforms.Lambda(zscore_per_volume) if vol_standardize else transforms.Lambda(lambda x: x),
            # tile the single channel, if expecting more channels 
            transforms.Lambda(lambda x: x if x.shape[0] == self.in_ch else x.repeat(self.in_ch // x.shape[0], 1, 1, 1)
                              if x.shape[0] in (1, 3) and self.in_ch % x.shape[0] == 0 else x),
        ])

    def __len__(self) -> int:
        return len(self.vols)

    def __getitem__(self, idx: int):
        vol = self.base_tfms(self.vols[idx])  
        topo = self.topo[idx]                 
        y = int(self.labels[idx])
        return vol, topo, y


# -----------------------------------------------------------------------------
# Data Loading Helpers
# -----------------------------------------------------------------------------

def load_npz(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    f = np.load(npz_path, allow_pickle=True)

    img_keys = [
        "image",
        "images",
        "imgs",
        "x",
        "X",
        "data",
        "vols",
        "volume",
        "volumes",
    ]
    lbl_keys = ["label", "labels", "y", "Y", "targets"]
    img_key = next((k for k in img_keys if k in f), None)
    lbl_key = next((k for k in lbl_keys if k in f), None)
    if img_key is None or lbl_key is None:
        raise KeyError(f"{npz_path}: couldn't find image/label keys. Found: {list(f.keys())}")

    X = f[img_key]
    y = f[lbl_key]

    if isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype == object):
        X = np.stack([np.asarray(x) for x in X], axis=0)

    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 4:
        pass  # (N,T,H,W)
    elif X.ndim == 5:
        if X.shape[1] in (1, 3, 4) or X.shape[-1] in (1, 3, 4):
            pass
        else:
            raise ValueError(f"{npz_path}: unexpected 5D shape {X.shape}; expected C in {1,3,4}")
    else:
        raise ValueError(
            f"{npz_path}: expected (N,64,64,64) or 5D with C in {{1,3,4}}, got {X.shape}"
        )

    return X, y.astype(np.int64)


def load_topo_csv(csv_path: str | Path, expected_dim: int) -> torch.Tensor:
    df = pd.read_csv(csv_path, header=None)
    drop_cols = {
        "id",
        "label",
        "ID",
        "Label",
        "Modality",
    }
    df = df.loc[:, ~df.columns.isin(drop_cols)].select_dtypes(exclude="object")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isnull().values.any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Non-numeric values in columns {bad} of {csv_path}")
    if df.shape[1] != expected_dim:
        raise ValueError(f"{csv_path}: expected {expected_dim} cols, got {df.shape[1]}")
    return torch.tensor(df.values, dtype=torch.float32)


def compute_block_stats(x: torch.Tensor, block: int = 150) -> Tuple[torch.Tensor, torch.Tensor]:
    N, D = x.shape
    assert D % block == 0, f"D={D} must be divisible by block={block}"
    xb = x.reshape(N, D // block, block)
    mu = xb.mean(dim=0, keepdim=True)
    sd = xb.std(dim=0, keepdim=True, unbiased=False)
    return mu, sd


def apply_block_norm(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    N, D = x.shape
    xb = x.reshape(N, mu.shape[1], mu.shape[2])
    xb = (xb - mu) / (sd + eps)
    return xb.reshape(N, D)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class Topoformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        topo_dim: int,
        in_ch: int = 1,
        proj_dim: int = 64,
        mlp_hidden: int = 128,
        p_drop: float = 0.3,
        use_rgb_mean: bool = True,
        pretrained_weights: R3D_18_Weights | None = R3D_18_Weights.KINETICS400_V1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = r3d_18(weights=pretrained_weights)

        # adapt first conv if channel count changes or needed
        old_conv: nn.Conv3d = self.backbone.stem[0]
        need_new = (in_ch != old_conv.in_channels) or use_rgb_mean
        if need_new:
            new_conv = nn.Conv3d(
                in_ch,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                w_rgb = old_conv.weight 
                w_mean = w_rgb.mean(dim=1, keepdim=True)  
                new_w = w_mean.expand(-1, in_ch, -1, -1, -1)
                new_conv.weight.copy_(new_w)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.backbone.stem[0] = new_conv

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()

        # topo branch 
        self.topo_mlp = nn.Sequential(
            nn.Linear(topo_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, embed_dim),
        )

        # fusion + classifier
        fused_dim = 2 * embed_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        # projection head for SupCon
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden, bias=False),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, proj_dim, bias=True),
        )

    def image_embed(self, x: torch.Tensor) -> torch.Tensor: 
        return self.backbone(x)

    def topo_embed(self, v: torch.Tensor) -> torch.Tensor:  
        return self.topo_mlp(v)

    def forward(self, vol: torch.Tensor, topo: torch.Tensor, contrastive: bool = False):
        feats_img = self.image_embed(vol)  
        feats_top = self.topo_embed(topo)   
        fused = torch.cat([feats_img, feats_top], dim=1)  
        fused = self.fusion(fused)                         
        logits = self.classifier(fused)
        if contrastive:
            z_img = F.normalize(self.proj_head(feats_img), dim=1)
            z_top = F.normalize(self.proj_head(feats_top), dim=1)
            feats = torch.cat([z_img, z_top], dim=0)  
            return logits, feats
        return logits


class SupConLoss(nn.Module):
    def __init__(self, temp: float = 0.07):
        super().__init__()
        self.T = temp

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = feats.device
        labels = labels.repeat_interleave(2)  
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)

        logits = feats @ feats.T / self.T
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        eye = torch.eye(mask.size(0), device=device)
        mask_self = 1 - eye
        mask = mask * mask_self

        exp_logits = torch.exp(logits) * mask_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return loss.mean()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # Paths
    img_prefix: str = "ODELIA_"
    topo_prefix: str = "ODELIA_"
    tda_type: str = "M20_"  # e.g., "M20_" (width 20 sliding band filtration)

    # Topology
    topo_dim: int = 450
    topo_block: int = 150

    # Model
    in_ch: int = 1  
    freeze_backbone: bool = False
    proj_dim: int = 64
    mlp_hidden: int = 128
    p_drop: float = 0.3

    # Training
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    supcon_lambda: float = 0.1
    supcon_temp: float = 0.07
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    grad_clip_norm: float = 1.0

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    run_name: str = "topoformer"


# -----------------------------------------------------------------------------
# Training / Evaluation
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: Config, in_ch: int):
    # NPZ splits
    train_images, train_labels = load_npz(Path(cfg.img_prefix) / "train.npz")
    val_images, val_labels = load_npz(Path(cfg.img_prefix) / "val.npz")
    test_images, test_labels = load_npz(Path(cfg.img_prefix) / "test.npz")

    # Topological CSVs
    train_topo = load_topo_csv(Path(cfg.topo_prefix) / f"{cfg.tda_type}train.csv", cfg.topo_dim)
    val_topo = load_topo_csv(Path(cfg.topo_prefix) / f"{cfg.tda_type}val.csv", cfg.topo_dim)
    test_topo = load_topo_csv(Path(cfg.topo_prefix) / f"{cfg.tda_type}test.csv", cfg.topo_dim)

    # Normalize topo using train stats only
    mu_b, sd_b = compute_block_stats(train_topo, block=cfg.topo_block)
    train_topo = apply_block_norm(train_topo, mu_b, sd_b)
    val_topo = apply_block_norm(val_topo, mu_b, sd_b)
    test_topo = apply_block_norm(test_topo, mu_b, sd_b)

    train_set = VolTopoDataset(train_images.astype(np.float32), train_topo, train_labels, in_ch)
    val_set = VolTopoDataset(val_images.astype(np.float32), val_topo, val_labels, in_ch)
    test_set = VolTopoDataset(test_images.astype(np.float32), test_topo, test_labels, in_ch)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    num_classes = int(len(np.unique(train_labels)))
    return train_loader, val_loader, test_loader, num_classes


def evaluate(model: nn.Module, data_loader: DataLoader, num_classes: int, device: torch.device):
    model.eval()
    all_probs, all_labels = [], []
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for vol, topo, labels in data_loader:
            vol, topo, labels = vol.to(device), topo.to(device), labels.to(device)
            logits = model(vol, topo, contrastive=False)
            loss = ce_loss(logits, labels)
            total_loss += loss.item() * vol.size(0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    total_loss /= len(data_loader.dataset)
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = probs.argmax(axis=1)

    if num_classes == 2:
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        auc = roc_auc_score(y_true, probs, multi_class="ovr")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    spec_per_class = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_per_class.append(spec)
    specificity = float(np.mean(spec_per_class))

    metrics = {
        "loss": total_loss,
        "auc": float(auc),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "specificity_macro": float(specificity),
    }
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg, cfg.in_ch)

    # Model & Optim
    model = Topoformer(
        num_classes=num_classes,
        topo_dim=cfg.topo_dim,
        in_ch=cfg.in_ch,
        proj_dim=cfg.proj_dim,
        mlp_hidden=cfg.mlp_hidden,
        p_drop=cfg.p_drop,
        pretrained_weights=R3D_18_Weights.KINETICS400_V1,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss(temp=cfg.supcon_temp)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Checkpoints
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -float("inf")
    best_state = None

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for vol, topo, labels in pbar:
            vol, topo, labels = vol.to(device), topo.to(device), labels.to(device)
            bsz = vol.size(0)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits, feats_2B = model(vol, topo, contrastive=True)
                loss_ce = criterion(logits, labels)
                loss_supc = supcon_loss(feats_2B, labels)
                loss = loss_ce + cfg.supcon_lambda * loss_supc

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * bsz
            n_train += bsz

        scheduler.step()
        train_loss = running_loss / max(1, n_train)

        # Validation
        val_metrics = evaluate(model, val_loader, num_classes, device)
        val_auc = val_metrics["auc"]

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | Train {train_loss:.4f} | "
            f"Val {val_metrics['loss']:.4f} | Val AUC {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "cfg": asdict(cfg),
                    "epoch": epoch + 1,
                    "model_state": best_state,
                    "val_auc": best_auc,
                },
                ckpt_dir / f"{cfg.run_name}_best.pt",
            )
            print(f"new best checkpoint saved (AUC={best_auc:.4f})")

    # Test with best weights
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, num_classes, device)
    print("\nTest Set Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"\nTotal runtime: {time.perf_counter() - t0:.2f} s")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fusion + SupCon Training")
    parser.add_argument("--img_prefix", type=str, default="ODELIA_", help="Path prefix to NPZ files")
    parser.add_argument("--topo_prefix", type=str, default="ODELIA_", help="Path prefix to CSV files")
    parser.add_argument("--tda_type", type=str, default="M20_", help="Prefix for topo CSVs (e.g., M20_) ")
    parser.add_argument("--topo_dim", type=int, default=450)
    parser.add_argument("--topo_block", type=int, default=150)
    parser.add_argument("--in_ch", type=int, default=1, help="Model input channels; 1 for grayscale volumes")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--supcon_lambda", type=float, default=0.1)
    parser.add_argument("--supcon_temp", type=float, default=0.07)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed-precision")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="fusion_supcon_r3d18")

    args = parser.parse_args()
    cfg = Config(
        img_prefix=args.img_prefix,
        topo_prefix=args.topo_prefix,
        tda_type=args.tda_type,
        topo_dim=args.topo_dim,
        topo_block=args.topo_block,
        in_ch=args.in_ch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        supcon_lambda=args.supcon_lambda,
        supcon_temp=args.supcon_temp,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        freeze_backbone=args.freeze_backbone,
        grad_clip_norm=args.grad_clip_norm,
        ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

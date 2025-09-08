"""
TopoGate: Image guided width selection for sliding band filtration
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def set_seed(seed: int = 33) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------------

class HyperGate(nn.Module):
    """Attention-style gating over topology widths using image token context.

    Args:
        img_dim: embedding dim of image tokens (e.g., 512)
        n_h: number of topology widths (gates)
        n_heads_attn: heads for internal MHA when forming context
    """

    def __init__(self, img_dim: int, n_h: int, n_heads_attn: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(img_dim, img_dim)
        # learned key/value prototypes per width
        self.k = nn.Parameter(torch.randn(n_h, img_dim))
        self.v = nn.Parameter(torch.randn(n_h, img_dim))
        self.attn = nn.MultiheadAttention(img_dim, num_heads=n_heads_attn, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(img_dim, img_dim // 2),
            nn.GELU(),
            nn.Linear(img_dim // 2, n_h),
        )

    def forward(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """Compute gate weights.
        img_tokens: [B, P, D]
        returns:    [B, n_h] (softmax)
        """
        # mean pool tokens -> query
        q = self.q_proj(img_tokens.mean(dim=1)).unsqueeze(1)  # [B,1,D]
        # expand learned keys/values for this batch
        k = self.k.unsqueeze(0).expand(img_tokens.size(0), -1, -1)  # [B,n_h,D]
        v = self.v.unsqueeze(0).expand_as(k)                         # [B,n_h,D]
        ctx, _ = self.attn(q, k, v)                                  # [B,1,D]
        logits = self.head(ctx.squeeze(1))                           # [B,n_h]
        return logits.softmax(dim=-1)


class TopoTransformerHead(nn.Module):
    """Token-wise Transformer head over topology vector.

    Interprets a length-`topo_dim` vector as a sequence of tokens (scalars), embeds
    each to `d_model`, adds sinusoidal PE, encodes with Transformer, and outputs
    per-token logits which are averaged.
    """

    def __init__(
        self,
        topo_dim: int = 100,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.topo_dim = topo_dim
        self.embed = nn.Linear(1, d_model)

        # Fixed sinusoidal positional encoding [1, topo_dim, d_model]
        pe = torch.zeros(topo_dim, d_model)
        pos = torch.arange(topo_dim).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, topo_vec: torch.Tensor) -> torch.Tensor:
        """topo_vec: [B, topo_dim] -> logits [B, topo_dim, num_classes]"""
        x = topo_vec.unsqueeze(-1)               
        x = self.embed(x) + self.pos_enc[:, : self.topo_dim]  
        x = self.encoder(x)                        
        return self.classifier(x)                 


class R3D18Encoder(nn.Module):
    """ResNet-3D (r3d_18) with first conv adapted to 1 input channel.

    Exposes forward_features() that returns feature map.
    """

    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if use_pretrained else None
        m = r3d_18(weights=weights)
        old = m.stem[0]  # Conv3d(3,64,...)
        new = nn.Conv3d(
            in_channels=1,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True)) 
        m.stem[0] = new
        self.net = m
        self.num_features = m.fc.in_features  

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # replicate forward until layer4 (keep T), then return feature map
        x = self.net.stem(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  
        return x


class TopoGate3D(nn.Module):
    """
    3D ResNet-18 backbone + HyperGate + topology Transformer head.

    Inputs
    ------
    volumes: [B, 64, 64, 64] (grayscale)
    topo_feats: [B, n_widths, topo_dim]
    """

    def __init__(self, topo_dim: int = 100, num_classes: int = 3, n_widths: int = 2, use_pretrained: bool = True):
        super().__init__()
        self.backbone = R3D18Encoder(use_pretrained=use_pretrained)
        emb_dim = self.backbone.num_features

        self.gate = HyperGate(emb_dim, n_h=n_widths)
        self.classifier = TopoTransformerHead(
            topo_dim=topo_dim, d_model=256, nhead=4, nlayers=3, num_classes=num_classes
        )

        # Kinetics-400 grayscale normalization
        mean_kinetics = (0.43216 + 0.394666 + 0.37645) / 3.0
        std_kinetics = (0.22803 + 0.22145 + 0.216989) / 3.0
        self.register_buffer("mean3d", torch.tensor([mean_kinetics]).view(1, 1, 1, 1, 1))
        self.register_buffer("std3d", torch.tensor([std_kinetics]).view(1, 1, 1, 1, 1))

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)  
        B, C, T, H, W = feat.shape
        return feat.view(B, C, T * H * W).permute(0, 2, 1)  

    def forward(self, volumes: torch.Tensor, topo_feats: torch.Tensor) -> torch.Tensor:
        x = volumes.unsqueeze(1)                 
        x = (x - self.mean3d) / self.std3d
        img_emb = self._encode(x)              
        w = self.gate(img_emb)                 

        # weighted sum over widths -> [B, topo_dim]
        weighted_topo = torch.sum(w.unsqueeze(-1) * topo_feats, dim=1)
        tok_logits = self.classifier(weighted_topo)  
        return tok_logits.mean(dim=1)                


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class TopoVolumeDataset(Dataset):
    """Loads volumes and per-width topology CSVs, with train dataset normalization.

    NPZ is expected to contain keys like 'train_images', 'train_labels', 'val_images', etc.
    Images are [N,64,64,64] (or [N,64,64,64,1]) grayscale volumes.
    Each CSV must be [N, topo_dim] for the corresponding split.
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str,  # 'train' | 'val' | 'test'
        csv_paths: Sequence[str | Path],
        mu: torch.Tensor | None = None,
        sd: torch.Tensor | None = None,
        topo_dim: int = 100,
    ) -> None:
        self.split = split
        data = np.load(npz_path, allow_pickle=True)
        imgs = data[f"{split}_images"]
        if imgs.ndim == 5 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        assert imgs.ndim == 4, "images must be [N,64,64,64] (grayscale)"

        self.images = imgs.astype(np.float32)
        self.labels = data[f"{split}_labels"].astype(np.int64)

        topo_per_width = []
        for p in csv_paths:
            df = pd.read_csv(p, header=None)
            df = df.select_dtypes(exclude="object")
            df = df.apply(pd.to_numeric, errors="coerce")
            if df.isnull().values.any():
                raise ValueError(f"Non-numeric values found in {p}")
            if df.shape[1] != topo_dim:
                raise ValueError(f"{p}: expected {topo_dim} columns, got {df.shape[1]}")
            topo_per_width.append(df.values.astype(np.float32))

        topo_arr = np.stack(topo_per_width, axis=1)  # [N, n_widths, topo_dim]
        if topo_arr.shape[0] != len(self.images):
            raise ValueError("CSV rows count must match number of images")

        self.topo_feats = torch.from_numpy(topo_arr)  # float32
        self.mu = mu
        self.sd = sd

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self.images[idx])       # [64,64,64]
        topo = self.topo_feats[idx].clone()            # [n_widths, topo_dim]
        if self.mu is not None and self.sd is not None:
            topo = (topo - self.mu) / self.sd
        return {"image": vol, "topo_feats": topo, "label": int(self.labels[idx])}


# ----------------------------------------------------------------------------
# Config and utilities
# ----------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    train_npz: str
    val_npz: str
    train_csvs: List[str]
    val_csvs: List[str]

    # Data
    topo_dim: int = 100
    batch_size: int = 16
    num_workers: int = 4

    # Model
    num_classes: int = 3
    use_pretrained: bool = True

    # Training
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    amp: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 42

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    run_name: str = "topogate"


def build_dataloaders(cfg: Config):
    # compute train-set μ/σ over all widths and samples, per token dim
    raw_train = TopoVolumeDataset(cfg.train_npz, "train", cfg.train_csvs, topo_dim=cfg.topo_dim)
    with torch.no_grad():
        mu = raw_train.topo_feats.mean(dim=(0, 1))            # [topo_dim]
        sd = raw_train.topo_feats.std(dim=(0, 1)).clamp_min(1e-6)

    train_ds = TopoVolumeDataset(cfg.train_npz, "train", cfg.train_csvs, mu, sd, topo_dim=cfg.topo_dim)
    val_ds = TopoVolumeDataset(cfg.val_npz, "val", cfg.val_csvs, mu, sd, topo_dim=cfg.topo_dim)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    n_widths = raw_train.topo_feats.shape[1]
    return train_loader, val_loader, n_widths, mu, sd


def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device):
    model.eval()
    y_true, y_prob = [], []
    ce = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            volumes = batch["image"].to(device)
            topo = batch["topo_feats"].to(device)
            labels = batch["label"].to(device)

            logits = model(volumes, topo)
            loss = ce(logits, labels)

            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            y_true.append(labels.cpu().numpy())
            y_prob.append(F.softmax(logits, dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "auc": float(auc),
    }


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, n_widths, mu, sd = build_dataloaders(cfg)

    model = TopoGate3D(
        topo_dim=cfg.topo_dim,
        num_classes=cfg.num_classes,
        n_widths=n_widths,
        use_pretrained=cfg.use_pretrained,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -float("inf")
    best_state = None

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            volumes = batch["image"].to(device)
            topo = batch["topo_feats"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(volumes, topo)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)

        # Validate
        val_metrics = evaluate(model, val_loader, cfg.num_classes, device)
        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | Train {train_loss:.4f} | Val {val_metrics['loss']:.4f} | "
            f"Val AUC {val_metrics['auc']:.4f} | Val F1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "cfg": asdict(cfg),
                    "epoch": epoch + 1,
                    "model_state": best_state,
                    "mu": mu,
                    "sd": sd,
                    "val_auc": best_auc,
                },
                ckpt_dir / f"{cfg.run_name}_best.pt",
            )
            print(f"  ↳ New best checkpoint saved (AUC={best_auc:.4f})")

    # Analyze gating on validation with best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    gate_weights = []
    with torch.no_grad():
        for batch in val_loader:
            volumes = batch["image"].to(device)
            x = volumes.unsqueeze(1)
            x = (x - model.mean3d) / model.std3d
            w = model.gate(model._encode(x))  
            gate_weights.append(w.cpu())
    avg_w = torch.cat(gate_weights).mean(dim=0)  
    best_width_idx = int(avg_w.argmax().item())

    print("\nAverage gate weights (val):", avg_w.tolist())
    print("Selected width index:", best_width_idx)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"\nTotal runtime: {time.perf_counter() - t0:.2f} s")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="TopoGate")

    # Paths
    p.add_argument("--train_npz", type=str, required=True, help="NPZ file containing train_* keys")
    p.add_argument("--val_npz", type=str, required=True, help="NPZ file containing val_* keys")
    p.add_argument("--train_csvs", type=str, nargs="+", required=True, help="List of CSVs for train widths")
    p.add_argument("--val_csvs", type=str, nargs="+", required=True, help="List of CSVs for val widths")

    # Data & model
    p.add_argument("--topo_dim", type=int, default=100)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_pretrained", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default="adaptive_topo_hypergate")

    a = p.parse_args()
    return Config(
        train_npz=a.train_npz,
        val_npz=a.val_npz,
        train_csvs=a.train_csvs,
        val_csvs=a.val_csvs,
        topo_dim=a.topo_dim,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        num_classes=a.num_classes,
        use_pretrained=a.use_pretrained,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        amp=not a.no_amp,
        grad_clip_norm=a.grad_clip_norm,
        seed=a.seed,
        ckpt_dir=a.ckpt_dir,
        run_name=a.run_name,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

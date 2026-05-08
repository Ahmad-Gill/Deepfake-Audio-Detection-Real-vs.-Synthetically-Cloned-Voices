# Reference 02: RawNet2 Implementation Guide
## Days 2–4 — Baseline Reproduction

**Target:** EER ≤ 5.3% on ASVspoof 2019 LA eval set (published: 4.66%)

---

## Architecture Summary

```
Input: Raw waveform (64,000 samples, 16kHz)
   ↓
Sinc Filter Bank (20 filters, fixed Mel-scale, kernel=1024)
   ↓
6 × Residual Blocks with FMS attention
   (blocks 1-2: 128 channels, blocks 3-6: 512 channels)
   ↓
GRU (1024 hidden units, bidirectional=False)
   ↓
FC layer → 2 logits (bonafide, spoof)
   ↓
Output: detection score
```

---

## Step 1: Implement the Model

Save as `src/models/rawnet2.py`:

```python
"""src/models/rawnet2.py"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Sinc-parameterized convolutional filters (fixed, non-learnable).
    Filters are initialized on Mel scale and kept frozen.
    This is intentional: prevents overfitting to 6 known training attacks.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        # Initialize cutoff frequencies on Mel scale (fixed)
        low_hz  = 30.0
        high_hz = sample_rate / 2 - 100
        mel_low  = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)

        self.register_buffer('f1', torch.from_numpy(
            hz_points[:-2]).float().view(-1, 1) / sample_rate)
        self.register_buffer('f2', torch.from_numpy(
            (hz_points[2:] - hz_points[:-2])).float().view(-1, 1) / sample_rate)

        # Hamming window (fixed)
        n = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / kernel_size)
        self.register_buffer('window', window.view(1, -1))

    def forward(self, x):
        # x: (batch, 1, time)
        half_K = (self.kernel_size - 1) / 2
        n = torch.linspace(-half_K, half_K, self.kernel_size,
                           device=x.device).unsqueeze(0)
        n = n + 1e-9  # avoid division by zero

        low  = torch.clamp(self.f1, min=1e-6)
        band = torch.clamp(self.f2, min=1e-6)
        high = torch.clamp(low + band, max=0.5 - 1e-6)

        # Compute sinc filters
        low_pass1 = 2 * low  * torch.sinc(2 * low  * n)
        low_pass2 = 2 * high * torch.sinc(2 * high * n)
        band_pass = (low_pass2 - low_pass1) * self.window

        # Normalize
        band_pass = band_pass / (2 * band_pass.abs().sum(dim=-1, keepdim=True) + 1e-9)

        return F.conv1d(x, band_pass.unsqueeze(1),
                        stride=1, padding=self.kernel_size // 2)


class FMS(nn.Module):
    """Feature Map Scaling — channel attention for residual blocks."""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Linear(channels, channels)
        self.shift = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (batch, channels, time)
        s = x.mean(dim=-1)                        # (batch, channels)
        scale = torch.sigmoid(self.scale(s))      # (batch, channels)
        shift = self.shift(s)                     # (batch, channels)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * scale + shift


class ResBlock(nn.Module):
    """Residual block with FMS attention."""
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.fms   = FMS(out_channels)
        self.pool  = nn.MaxPool1d(3)
        self.relu  = nn.LeakyReLU(0.3)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            residual = self.shortcut(x)
        out = self.relu(out + residual)
        out = self.fms(out)
        return self.pool(out)


class RawNet2(nn.Module):
    """
    RawNet2 for ASVspoof anti-spoofing.
    Reference: Tak et al., ICASSP 2021 (arXiv:2011.01108)
    """
    def __init__(self, sinc_out=20, block_channels=(128, 128, 256, 256, 512, 512),
                 gru_hidden=1024, num_classes=2):
        super().__init__()

        self.sinc = SincConv(sinc_out, kernel_size=1024)
        self.bn0  = nn.BatchNorm1d(sinc_out)
        self.relu = nn.LeakyReLU(0.3)
        self.pool = nn.MaxPool1d(3)

        # 6 residual blocks
        channels = [sinc_out] + list(block_channels)
        self.res_blocks = nn.ModuleList([
            ResBlock(channels[i], channels[i+1])
            for i in range(len(block_channels))
        ])

        self.bn_before_gru = nn.BatchNorm1d(block_channels[-1])
        self.gru = nn.GRU(input_size=block_channels[-1],
                          hidden_size=gru_hidden,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Linear(gru_hidden, num_classes)

    def forward(self, x):
        # x: (batch, time) → add channel dim
        x = x.unsqueeze(1)                         # (batch, 1, time)
        x = self.sinc(x)                           # (batch, sinc_out, time)
        x = self.pool(self.relu(self.bn0(x)))

        for block in self.res_blocks:
            x = block(x)                           # (batch, channels, time')

        x = self.bn_before_gru(x)
        x = x.permute(0, 2, 1)                    # (batch, time', channels)
        _, h = self.gru(x)                         # h: (1, batch, gru_hidden)
        x = h.squeeze(0)                           # (batch, gru_hidden)
        return self.fc(x)                          # (batch, 2)

    def get_score(self, x):
        """Returns spoof score (higher = more likely spoof)."""
        logits = self.forward(x)
        return logits[:, 1] - logits[:, 0]         # log-likelihood ratio
```

---

## Step 2: Training Script

Save as `train_rawnet2.py`:

```python
"""train_rawnet2.py — Train RawNet2 on ASVspoof 2019 LA"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.rawnet2 import RawNet2
from src.data.dataset import ASVspoof2019LA
from src.utils.metrics import compute_eer
from src.utils.logger import ResultsLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for wavs, labels, _ in loader:
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(wavs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(wavs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(wavs)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_attacks = [], [], []
    for wavs, labels, attacks in loader:
        wavs = wavs.to(device)
        scores = model.get_score(wavs).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())
        all_attacks.extend(attacks)
    eer, threshold = compute_eer(np.array(all_scores),
                                  np.array(all_labels))
    return eer, threshold, all_scores, all_labels, all_attacks


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    # Datasets
    augmentor = None
    if args.augment:
        from src.augmentation.rawboost import RawBoostAugmentor
        augmentor = RawBoostAugmentor(modes=[1, 2])

    train_ds = ASVspoof2019LA(args.data_dir, "train", augmentor=augmentor)
    dev_ds   = ASVspoof2019LA(args.data_dir, "dev")
    eval_ds  = ASVspoof2019LA(args.data_dir, "eval")

    # Class-weighted loss
    weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = RawNet2().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    ckpt_dir = Path(f"checkpoints/rawnet2/seed{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dev_eer = float('inf')
    patience_count = 0
    logger = ResultsLogger(f"results/rawnet2/seed{args.seed}_log.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        dev_eer, dev_thresh, *_ = evaluate(model, dev_loader, device)
        scheduler.step(dev_eer)

        print(f"Epoch {epoch:3d} | Loss {train_loss:.4f} | "
              f"Acc {train_acc:.3f} | Dev EER {dev_eer:.2f}%")

        logger.log(epoch, train_loss, train_acc, dev_eer)

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "dev_eer": dev_eer,
                "seed": args.seed,
            }, ckpt_dir / "best.pt")
            print(f"  ✅ New best dev EER: {dev_eer:.2f}%")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on eval set
    print("\n--- FINAL EVALUATION ON EVAL SET ---")
    ckpt = torch.load(ckpt_dir / "best.pt")
    model.load_state_dict(ckpt["model_state"])
    eval_eer, eval_thresh, scores, labels, attacks = evaluate(
        model, eval_loader, device)
    print(f"EVAL EER: {eval_eer:.2f}%  (best dev epoch: {ckpt['epoch']})")

    # Save scores for t-DCF computation and per-attack analysis
    import pandas as pd
    pd.DataFrame({
        "score": scores, "label": labels, "attack": attacks
    }).to_csv(ckpt_dir / "eval_scores.csv", index=False)

    return eval_eer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/LA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--augment", action="store_true",
                        help="Apply RawBoost augmentation (modes 1+2)")
    args = parser.parse_args()
    main(args)
```

---

## Step 3: Training Commands

```bash
# Configuration C1: RawNet2, no augmentation, 3 seeds
for seed in 42 123 456; do
    python train_rawnet2.py \
        --data_dir data/raw/LA \
        --seed $seed \
        --batch_size 32 \
        --lr 1e-4 \
        --epochs 100 \
        --patience 5
done

# Configuration C2: RawNet2 + RawBoost
for seed in 42 123 456; do
    python train_rawnet2.py \
        --data_dir data/raw/LA \
        --seed $seed \
        --augment
done
```

---

## Step 4: Verify Convergence

Expected training progression:
```
Epoch 1:  Dev EER ~45-50%  (random-ish, class-weighted)
Epoch 5:  Dev EER ~25-35%
Epoch 15: Dev EER ~15-20%
Epoch 30: Dev EER ~10-15%
Epoch 50: Dev EER ~7-10%
Epoch 70: Dev EER ~5-8%
Best:     Dev EER ~4-6%
```

**If EER is not moving below 20% by epoch 30 → go to debugging guide.**

---

## Checklist

```
□ src/models/rawnet2.py created
□ train_rawnet2.py created
□ C1 (no augment): 3 seeds complete, EER logged
□ C2 (RawBoost): 3 seeds complete, EER logged
□ eval_scores.csv saved for each run
□ Git commit: "C1 and C2: RawNet2 baseline complete"
```

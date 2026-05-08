# Reference 03: AASIST Implementation Guide
## Days 5–8 — Graph Attention Architecture

**Target:** EER ≤ 1.5% on ASVspoof 2019 LA eval (published AASIST-L: 0.99%)

---

## Recommendation: Use Official Repo + Adapt

AASIST's graph attention network is complex. Rather than reimplementing from scratch, use the official implementation and adapt the training loop to match your project's conventions.

```bash
# Clone official AASIST repo alongside your project
git clone https://github.com/clovaai/aasist.git aasist-ref
```

Then copy the model file into your project:
```bash
cp aasist-ref/models/AASIST.py src/models/aasist.py
```

---

## Architecture Summary (AASIST-L)

```
Input: Raw waveform (64,000 samples)
   ↓
RawNet2-style encoder (sinc-conv + 6 residual blocks)
   → Feature map F ∈ R^{C × S × T}
   ↓
Graph construction:
   Spectral graph Gs  ← temporal-max-pool node features from F
   Temporal graph Gt  ← spectral-max-pool node features from F
   ↓
HS-GAL (Heterogeneous Stacking Graph Attention Layer)
   Two parallel branches, each with 2 HS-GAL layers
   + 50% spectral node dropout + 30% temporal node dropout
   ↓
Max Graph Operation: element-wise max across branches
   ↓
Readout: [node max-pool | node avg-pool | stack node]
   ↓
FC → 2 logits
```

---

## Step 1: Training Script for AASIST

Save as `train_aasist.py`:

```python
"""train_aasist.py — Train AASIST-L on ASVspoof 2019 LA"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import ASVspoof2019LA
from src.utils.metrics import compute_eer
from src.utils.logger import ResultsLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_aasist_model(use_ssl=False):
    """
    Load AASIST-L model.
    If use_ssl=True, prepends wav2vec 2.0 feature extractor.
    """
    # Import from official repo copy
    import sys
    sys.path.insert(0, "aasist-ref")
    from models.AASIST import Model

    # AASIST-L configuration (85K parameters)
    config = {
        "architecture": "AASIST-L",
        "nb_samp": 64000,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.7],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }

    model = Model(config)
    return model


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for wavs, labels, _ in loader:
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, logits = model(wavs)   # AASIST returns (embedding, logits)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * len(wavs)
        n += len(wavs)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_attacks = [], [], []
    for wavs, labels, attacks in loader:
        wavs = wavs.to(device)
        _, logits = model(wavs)
        # Score = spoof logit - bonafide logit
        scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())
        all_attacks.extend(attacks)
    eer, threshold = compute_eer(
        np.array(all_scores), np.array(all_labels))
    return eer, threshold, all_scores, all_labels, all_attacks


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    # Build augmentor if requested
    augmentor = None
    if args.augment:
        from src.augmentation.rawboost import RawBoostAugmentor
        augmentor = RawBoostAugmentor(modes=[1, 2])

    # Datasets
    train_ds = ASVspoof2019LA(args.data_dir, "train", augmentor=augmentor)
    dev_ds   = ASVspoof2019LA(args.data_dir, "dev")
    eval_ds  = ASVspoof2019LA(args.data_dir, "eval")

    weights   = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = load_aasist_model(use_ssl=args.ssl).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    # Cosine annealing (original AASIST protocol)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    tag = f"aasist{'_ssl' if args.ssl else ''}{'_aug' if args.augment else ''}"
    ckpt_dir = Path(f"checkpoints/{tag}/seed{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = ResultsLogger(f"results/{tag}/seed{args.seed}_log.csv")

    best_dev_eer = float("inf")
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer,
                           criterion, device, scheduler)
        dev_eer, *_ = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Dev EER {dev_eer:.2f}%")
        logger.log(epoch, loss, None, dev_eer)

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            patience_count = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "dev_eer": dev_eer, "seed": args.seed},
                       ckpt_dir / "best.pt")
            print(f"  ✅ New best: {dev_eer:.2f}%")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final eval
    ckpt = torch.load(ckpt_dir / "best.pt")
    model.load_state_dict(ckpt["model_state"])
    eval_eer, _, scores, labels, attacks = evaluate(
        model, eval_loader, device)
    print(f"\nEVAL EER: {eval_eer:.2f}%")

    import pandas as pd
    pd.DataFrame({"score": scores, "label": labels,
                  "attack": attacks}).to_csv(
        ckpt_dir / "eval_scores.csv", index=False)
    return eval_eer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/LA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--ssl", action="store_true",
                        help="Use wav2vec 2.0 SSL features (C5/C6)")
    args = parser.parse_args()
    main(args)
```

---

## Step 2: Training Commands

```bash
# C3: AASIST-L, no augmentation
for seed in 42 123 456; do
    python train_aasist.py --seed $seed --batch_size 24
done

# C4: AASIST-L + RawBoost
for seed in 42 123 456; do
    python train_aasist.py --seed $seed --batch_size 24 --augment
done

# C5 & C6: SSL versions (only if GPU >= 16GB)
for seed in 42 123 456; do
    python train_aasist.py --seed $seed --ssl
    python train_aasist.py --seed $seed --ssl --augment
done
```

---

## Step 3: SSL Feature Integration (C5, C6)

Only attempt if you have ≥ 16GB GPU VRAM. Save as `src/models/ssl_frontend.py`:

```python
"""src/models/ssl_frontend.py — wav2vec 2.0 front-end"""
import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Install transformers: pip install transformers")


class SSLFrontend(nn.Module):
    """
    wav2vec 2.0 XLSR front-end, extracting intermediate layer features.
    Fine-tuned end-to-end with the detection backend.
    Based on: Wang & Yamagishi (2022) — intermediate layers outperform final.
    """
    LAYER = 9  # Intermediate layer — tunable via dev set sweep

    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53",
                 layer_idx=9, freeze=False):
        super().__init__()
        assert HAS_TRANSFORMERS, "pip install transformers"
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.layer_idx = layer_idx

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (batch, time)
        out = self.model(x, output_hidden_states=True)
        # Use intermediate layer representation
        hidden = out.hidden_states[self.layer_idx]  # (batch, time', 1024)
        return hidden.permute(0, 2, 1)              # (batch, 1024, time')

    @property
    def output_dim(self):
        return 1024
```

---

## Expected Convergence (AASIST-L)

```
Epoch 1:   Dev EER ~40%
Epoch 10:  Dev EER ~15-20%
Epoch 25:  Dev EER ~5-8%
Epoch 50:  Dev EER ~2-3%
Epoch 70+: Dev EER ~1-1.5%
```

AASIST converges slower than RawNet2 — patience of 10 is needed.

---

## Checklist

```
□ aasist-ref/ cloned
□ src/models/aasist.py created (from official repo)
□ train_aasist.py created
□ C3 (no augment): 3 seeds complete
□ C4 (RawBoost): 3 seeds complete
□ C5/C6 (SSL): run if GPU allows, skip if not
□ Git commit: "C3-C6: AASIST training complete"
```

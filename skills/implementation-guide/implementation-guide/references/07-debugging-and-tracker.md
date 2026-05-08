# Reference 07: Debugging Guide — Common Issues & Fixes

---

## Issue 1: EER Not Decreasing (Stuck Above 20% After 30 Epochs)

**Diagnosis checklist — check in this order:**

```python
# Test 1: Is class weighting applied?
weights = train_ds.get_class_weights()
print(weights)  # Should be ~[5.5, 0.56] — bonafide heavily upweighted
# If [1.0, 1.0] → class weighting not applied → fix criterion

# Test 2: Is the score direction correct?
# Higher score = more likely SPOOF
# If your model outputs logits[:,0]=bonafide, logits[:,1]=spoof:
score = logits[:, 1] - logits[:, 0]  # correct
# NOT: score = logits[:, 0] - logits[:, 1]  # wrong direction

# Test 3: Is EER computed on eval (not dev)?
# Dev EER of 5% ≠ Eval EER of 5%
# Always check: "Which loader did I pass to evaluate()?"

# Test 4: Is the data loader shuffling training data?
train_loader = DataLoader(train_ds, shuffle=True, ...)  # must be True
```

**Most common fix:** Class weights not applied. Add this:
```python
weights = train_ds.get_class_weights().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## Issue 2: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Fixes in order of preference:**

```python
# Fix 1: Reduce batch size
--batch_size 16  # (from 32)
--batch_size 8   # (if still OOM)

# Fix 2: Use gradient checkpointing (AASIST)
model = load_aasist_model()
model.gradient_checkpointing_enable()

# Fix 3: Mixed precision training (saves ~40% VRAM)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    _, logits = model(wavs)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Fix 4: Clear cache between eval and train
torch.cuda.empty_cache()
```

---

## Issue 3: Audio Loading Errors

```
RuntimeError: Error opening '/path/to/file.flac'
```

**Fix:**
```python
# Verify the path convention
# Protocol files use: LA_T_1000137 (without .flac)
# Your loader should append .flac:
audio_path = audio_dir / f"{entry['file']}.flac"  # correct
```

```bash
# Check a specific file exists:
ls data/raw/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac
```

---

## Issue 4: Loss is NaN

**Causes and fixes:**

```python
# Cause 1: Learning rate too high
--lr 1e-5  # reduce from 1e-4

# Cause 2: No gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # add this

# Cause 3: Silent audio (zero waveform after RawBoost)
# RawBoost can produce near-zero output on very short files
# Fix: add epsilon to normalization in rawboost.py
x = x / (np.abs(x).max() + 1e-9)  # already there, but check

# Cause 4: Mixed precision overflow
# Use: scaler = GradScaler(init_scale=2**8)  (lower initial scale)
```

---

## Issue 5: Dev EER Good But Eval EER Much Worse

**Example:** Dev EER = 3%, Eval EER = 25%

**This is expected and important.** It means your model generalizes poorly to unseen attacks.

```
DO NOT be alarmed — this gap IS the problem your project is studying.
DO report both numbers.
DO NOT tune on eval set to close this gap artificially.
```

To improve generalization:
1. Add RawBoost augmentation (should help 10-30% relative)
2. Use SSL features (should help significantly)
3. Train longer with lower LR

---

## Issue 6: AASIST Gives "Model has no attribute X" Error

The official AASIST repo has slightly different API across versions. Check:

```python
# Some versions return: output, embedding
# Some versions return: embedding, output
# Check what your version does:
out = model(wavs)
print(type(out), len(out))

# If tuple of 2: (embedding, logits) — use out[1] for logits
# If just tensor: direct logits — use out
```

---

## Issue 7: Reproducibility — Different EER with Same Seed

```python
# Make sure ALL these are set BEFORE any torch operations:
def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # THIS LINE IS CRITICAL

# Also: num_workers=0 for fully deterministic DataLoader
# (but this is much slower — use 4 workers and accept slight variance)
```

---

## Issue 8: Training Too Slow (No GPU / Slow GPU)

```bash
# Check if GPU is being used:
watch -n 1 nvidia-smi

# If GPU utilization is low (<50%):
# Increase num_workers in DataLoader
DataLoader(train_ds, num_workers=8, pin_memory=True, prefetch_factor=2)

# If no GPU at all, reduce batch and expect 10-20x slower training
# RawNet2: ~2-3 hours/epoch on CPU → not feasible
# Solution: Use Google Colab (free T4 GPU) or Kaggle (free P100)
```

**Google Colab quick setup:**
```python
# Mount your Google Drive where data is stored
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install soundfile torchaudio scipy

# Then run your training script
!python train_rawnet2.py --data_dir /content/drive/MyDrive/LA
```

---

# Reference 08: Results Tracker

## How to Track and Organize All Results

---

## Automated Results Collection Script

Save as `scripts/collect_results.py`:

```python
"""scripts/collect_results.py
Collect all training results and generate the final summary table.
Run this after all configurations are complete.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.metrics import compute_eer

CONFIGS = [
    ("C1", "RawNet2",          "checkpoints/rawnet2",        False, False),
    ("C2", "RawNet2+RawBoost", "checkpoints/rawnet2_aug",    False, True),
    ("C3", "AASIST-L",         "checkpoints/aasist",         False, False),
    ("C4", "AASIST-L+RawBoost","checkpoints/aasist_aug",     False, True),
    ("C5", "AASIST-L+SSL",     "checkpoints/aasist_ssl",     True,  False),
    ("C6", "AASIST-L+SSL+Aug", "checkpoints/aasist_ssl_aug", True,  True),
]
SEEDS = [42, 123, 456]

# Published baselines for comparison
BASELINES = [
    ("B1", "CQCC+GMM",        "2019", 9.57, 0.2366),
    ("B2", "LFCC+GMM",        "2019", 8.09, 0.2116),
    ("B3", "RawNet2",         "2021", 4.66, 0.1294),
    ("B4", "AASIST",          "2022", 0.83, 0.0275),
    ("B5", "AASIST-L",        "2022", 0.99, 0.0309),
    ("B6", "Resolution-Aware","2026", 0.16, None),
]

def collect_seed_results(config_dir):
    """Return list of (eer, t_dcf) for each available seed."""
    results = []
    for seed in SEEDS:
        csv_path = Path(config_dir) / f"seed{seed}" / "eval_scores.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        eer, _ = compute_eer(df["score"].values, df["label"].values)
        results.append(eer)
    return results

def main():
    rows = []

    # Add baselines
    for bid, name, year, eer, tdcf in BASELINES:
        tdcf_str = f"{tdcf:.4f}" if tdcf else "—"
        rows.append({
            "ID": bid, "System": name, "Year": year,
            "EER (%)": f"{eer:.2f}", "min t-DCF": tdcf_str,
            "Std": "—", "Type": "Baseline"
        })

    # Add your results
    for cid, name, ckpt_dir, is_ssl, is_aug in CONFIGS:
        seed_eers = collect_seed_results(ckpt_dir)
        if not seed_eers:
            rows.append({
                "ID": cid, "System": f"Ours: {name}", "Year": "2026",
                "EER (%)": "—", "min t-DCF": "—",
                "Std": "—", "Type": "Ours (pending)"
            })
            continue

        mean_eer = np.mean(seed_eers)
        std_eer  = np.std(seed_eers)
        rows.append({
            "ID": cid,
            "System": f"Ours: {name}",
            "Year": "2026",
            "EER (%)": f"{mean_eer:.2f}",
            "min t-DCF": "—",  # compute separately with official toolkit
            "Std": f"±{std_eer:.2f}",
            "Type": "Ours"
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/final_table.csv", index=False)
    print("="*60)
    print("FINAL RESULTS TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nSaved to: results/final_table.csv")

    # Print what to paste into your final report
    print("\n--- For your final report (markdown table) ---")
    print("| System | Year | EER (%) | Std | min t-DCF |")
    print("|--------|------|---------|-----|-----------|")
    for _, row in df.iterrows():
        print(f"| {row['System']} | {row['Year']} | "
              f"{row['EER (%)']} | {row['Std']} | {row['min t-DCF']} |")


if __name__ == "__main__":
    main()
```

---

## Daily Progress Tracker

Keep `results/progress.md` updated daily:

```markdown
# Experiment Progress Tracker

## Status Overview

| Config | Status | Best Dev EER | Eval EER (mean±std) |
|--------|--------|-------------|---------------------|
| C1: RawNet2 | ✅ Done | 4.1% | 4.8% ± 0.3% |
| C2: RawNet2+Aug | 🔄 Running | — | — |
| C3: AASIST-L | ⏳ Queued | — | — |
| C4: AASIST-L+Aug | ⏳ Queued | — | — |
| C5: AASIST+SSL | ⏳ Queued | — | — |
| C6: AASIST+SSL+Aug | ⏳ Queued | — | — |

## Log

### May 9
- Environment set up, dataset verified ✅
- verify_dataset.py: ALL CHECKS PASSED

### May 10
- C1 seed 42: training started
- ...
```

---

## Minimum Results for Final Report

Even if not all configurations complete, you need at minimum:

```
Minimum viable results set:
✅ C1 (RawNet2 baseline)        — proves reproducibility
✅ C2 (RawNet2 + RawBoost)      — proves augmentation helps
✅ C3 or C4 (AASIST-L)          — proves architecture improvement
⭐ C6 (AASIST+SSL+Aug)           — your best result (if possible)

With just C1+C2+C3: you have a complete ablation showing
augmentation effect AND architecture effect independently.
```

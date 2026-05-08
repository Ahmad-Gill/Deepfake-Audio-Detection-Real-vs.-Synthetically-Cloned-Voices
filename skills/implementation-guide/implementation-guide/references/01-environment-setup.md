# Reference 01: Environment Setup & Dataset Verification
## Day 1 — Complete Before Any Training

---

## Step 1: Create Conda Environment

```bash
conda create -n deepfake python=3.8 -y
conda activate deepfake

# Core ML stack
pip install torch==1.13.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# If no CUDA (CPU only — slower but works)
pip install torch==1.13.1 torchaudio==0.13.1

# Supporting libraries
pip install numpy pandas scikit-learn scipy \
    librosa soundfile tqdm matplotlib seaborn \
    tensorboard pyyaml

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 2: Set Up Your Project Repository

```bash
cd your-project-repo

# Create source structure
mkdir -p src/{models,data,augmentation,utils}
mkdir -p results/{rawnet2,aasist,ssl}
mkdir -p checkpoints/{rawnet2,aasist,ssl}
mkdir -p logs

# Create __init__.py files
touch src/__init__.py src/models/__init__.py \
      src/data/__init__.py src/augmentation/__init__.py \
      src/utils/__init__.py

# Save dependencies
pip freeze > requirements.txt
```

---

## Step 3: Download and Organize ASVspoof 2019 LA

```bash
# Dataset lives at: https://datashare.ed.ac.uk/handle/10283/3336
# You need: LA.zip (the Logical Access track)
# After downloading, extract to data/raw/

unzip LA.zip -d data/raw/

# Expected structure after extraction:
data/raw/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/
│   └── flac/  ← 25,380 .flac files
├── ASVspoof2019_LA_dev/
│   └── flac/  ← 24,844 .flac files
└── ASVspoof2019_LA_eval/
    └── flac/  ← 71,237 .flac files
```

---

## Step 4: Run Dataset Verification Script

Save as `scripts/verify_dataset.py` and run it. It will catch any missing files or corrupted audio before you waste training time.

```python
"""verify_dataset.py — Run before ANY training. Must pass all checks."""
import os
import sys
from pathlib import Path
import soundfile as sf

BASE = Path("data/raw/LA")
PROTOCOLS = BASE / "ASVspoof2019_LA_cm_protocols"
SPLITS = {
    "train": (BASE / "ASVspoof2019_LA_train/flac",
              PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt",
              2580, 22800),
    "dev":   (BASE / "ASVspoof2019_LA_dev/flac",
              PROTOCOLS / "ASVspoof2019.LA.cm.dev.trl.txt",
              2548, 22296),
    "eval":  (BASE / "ASVspoof2019_LA_eval/flac",
              PROTOCOLS / "ASVspoof2019.LA.cm.eval.trl.txt",
              7355, 63882),
}

all_passed = True

for split, (audio_dir, proto_file, exp_bona, exp_spoof) in SPLITS.items():
    print(f"\n{'='*50}")
    print(f"Checking {split} split...")

    # Check protocol exists
    if not proto_file.exists():
        print(f"  ❌ Protocol file missing: {proto_file}")
        all_passed = False
        continue

    # Parse protocol
    entries = []
    with open(proto_file) as f:
        for line in f:
            parts = line.strip().split()
            # Format: SPEAKER FILE - ATTACK LABEL
            entries.append({
                "file": parts[1],
                "attack": parts[3],
                "label": parts[4]  # bonafide or spoof
            })

    bona = sum(1 for e in entries if e["label"] == "bonafide")
    spoof = sum(1 for e in entries if e["label"] == "spoof")

    print(f"  Protocol: {len(entries)} entries")
    print(f"  Bonafide: {bona} (expected {exp_bona})")
    print(f"  Spoofed:  {spoof} (expected {exp_spoof})")

    if bona != exp_bona or spoof != exp_spoof:
        print(f"  ❌ Count mismatch!")
        all_passed = False
    else:
        print(f"  ✅ Counts correct")

    # Check audio files (sample 20 random files)
    import random
    sample = random.sample(entries, min(20, len(entries)))
    missing = 0
    corrupt = 0
    for e in sample:
        fpath = audio_dir / f"{e['file']}.flac"
        if not fpath.exists():
            missing += 1
        else:
            try:
                data, sr = sf.read(fpath)
                assert sr == 16000, f"Sample rate {sr} != 16000"
            except Exception:
                corrupt += 1

    if missing > 0:
        print(f"  ❌ {missing}/20 sampled files missing")
        all_passed = False
    elif corrupt > 0:
        print(f"  ❌ {corrupt}/20 sampled files corrupt")
        all_passed = False
    else:
        print(f"  ✅ Audio files OK (sample of 20)")

    # Show attack distribution for eval
    if split == "eval":
        from collections import Counter
        attacks = Counter(e["attack"] for e in entries)
        print(f"\n  Eval attack distribution:")
        for atk, cnt in sorted(attacks.items()):
            print(f"    {atk}: {cnt}")

print(f"\n{'='*50}")
if all_passed:
    print("✅ ALL CHECKS PASSED — Dataset is ready for training")
else:
    print("❌ SOME CHECKS FAILED — Fix issues before training")
    sys.exit(1)
```

**Expected output when everything is correct:**
```
✅ Counts correct  (for all 3 splits)
✅ Audio files OK  (for all 3 splits)
✅ ALL CHECKS PASSED — Dataset is ready for training
```

---

## Step 5: Create the Shared Dataset Loader

Save as `src/data/dataset.py`. Both RawNet2 and AASIST use this same loader.

```python
"""src/data/dataset.py — Shared ASVspoof 2019 LA dataset loader"""
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ASVspoof2019LA(Dataset):
    """
    ASVspoof 2019 Logical Access dataset.
    Returns fixed-length waveform tensors + binary labels (0=bonafide, 1=spoof).
    """
    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(self, base_dir, split, max_len=64000, augmentor=None):
        """
        Args:
            base_dir: Path to data/raw/LA/
            split: 'train', 'dev', or 'eval'
            max_len: Fixed waveform length (64000 = 4 sec at 16kHz)
            augmentor: Optional RawBoost augmentor object
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.max_len = max_len
        self.augmentor = augmentor

        split_dirs = {
            "train": "ASVspoof2019_LA_train",
            "dev":   "ASVspoof2019_LA_dev",
            "eval":  "ASVspoof2019_LA_eval",
        }
        proto_files = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
        }

        self.audio_dir = self.base_dir / split_dirs[split] / "flac"
        proto_path = (self.base_dir / "ASVspoof2019_LA_cm_protocols"
                      / proto_files[split])

        self.entries = []
        with open(proto_path) as f:
            for line in f:
                parts = line.strip().split()
                self.entries.append({
                    "file":   parts[1],
                    "attack": parts[3],
                    "label":  self.LABEL_MAP[parts[4]],
                })

    def __len__(self):
        return len(self.entries)

    def _pad_or_crop(self, waveform):
        """Pad with zeros or randomly crop to max_len."""
        length = len(waveform)
        if length < self.max_len:
            # Pad: repeat waveform until long enough
            repeats = (self.max_len // length) + 1
            waveform = np.tile(waveform, repeats)
        # Random crop during training, center crop during eval
        if self.split == "train":
            start = random.randint(0, len(waveform) - self.max_len)
        else:
            start = (len(waveform) - self.max_len) // 2
        return waveform[start: start + self.max_len]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = self.audio_dir / f"{entry['file']}.flac"

        waveform, sr = sf.read(audio_path)
        assert sr == 16000, f"Expected 16kHz, got {sr}"

        # Convert stereo to mono if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)
        waveform = self._pad_or_crop(waveform)

        # Apply augmentation (training only)
        if self.augmentor is not None and self.split == "train":
            waveform = self.augmentor(waveform)

        return (torch.FloatTensor(waveform),
                entry["label"],
                entry["attack"])   # attack ID useful for per-attack eval

    def get_class_weights(self):
        """Returns [w_bonafide, w_spoof] for weighted cross-entropy."""
        labels = [e["label"] for e in self.entries]
        n_total = len(labels)
        n_bona  = sum(1 for l in labels if l == 0)
        n_spoof = sum(1 for l in labels if l == 1)
        # Weight inversely proportional to class frequency
        w_bona  = n_total / (2 * n_bona)
        w_spoof = n_total / (2 * n_spoof)
        return torch.FloatTensor([w_bona, w_spoof])
```

---

## Checklist Before Moving to RawNet2

```
□ conda environment created and activated
□ torch.cuda.is_available() returns True
□ Dataset downloaded and extracted to data/raw/LA/
□ verify_dataset.py runs and shows ✅ ALL CHECKS PASSED
□ src/data/dataset.py created
□ Git commit: "Day 1: environment setup and dataset verified"
```

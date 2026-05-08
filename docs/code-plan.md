# CodePlan: Deepfake Audio Detection — AASIST-L + RawBoost

## Project Goal
Build a binary classifier (genuine vs. synthetic speech) on the **ASVspoof 2019 Logical Access (LA)** track dataset. Primary emphasis is **generalization** to unseen attacks, not just minimizing in-distribution EER. Evaluate with **EER (%)** and **min t-DCF** across 3 random seeds.

---

## Architecture Decision (Final, Do Not Change)

| Component | Choice | Reason |
|---|---|---|
| Input | Raw waveform (64,000 samples @ 16 kHz) | No hand-crafted features — preserves phase info |
| Model | **AASIST-L** (85K params) | SOTA-competitive single system, fits free Colab T4 |
| Augmentation | **RawBoost series ①+②** | Closes clean-train / noisy-test gap without external data |
| Loss | **Weighted Cross-Entropy** | Handles 9:1 spoof/bona-fide imbalance |
| Optimizer | **Adam**, lr=1e-4 | Per original AASIST paper |
| LR Schedule | **Cosine annealing** over 100 epochs | Per original AASIST paper |
| Batch size | 24 | Fits in ~6–8 GB VRAM (free Colab T4) |
| Seeds | 3 (report mean ± std) | Single-seed results are statistically unreliable |
| Evaluation | Official ASVspoof scoring toolkit | EER + min t-DCF per-attack and pooled |

**Reference implementations:**
- AASIST: https://github.com/clovaai/aasist
- RawBoost: https://github.com/TakHemlata/RawBoost-antispoofing
- ASVspoof scoring: https://www.asvspoof.org/resources

---

## Project File Structure to Create

```
project/
├── data/
│   └── ASVspoof2019/
│       └── LA/
│           ├── ASVspoof2019_LA_train/flac/
│           ├── ASVspoof2019_LA_dev/flac/
│           ├── ASVspoof2019_LA_eval/flac/
│           └── ASVspoof2019_LA_cm_protocols/
│               ├── ASVspoof2019.LA.cm.train.trn.txt
│               ├── ASVspoof2019.LA.cm.dev.trl.txt
│               └── ASVspoof2019.LA.cm.eval.trl.txt
├── models/
│   └── aasist_l/           # AASIST-L model definition (from reference repo)
├── rawboost/
│   └── rawboost.py         # RawBoost augmentation (from reference repo)
├── scoring/
│   └── evaluate_tDCF_asvspoof19.py   # Official ASVspoof scoring script
├── train.py                # Main training script
├── evaluate.py             # Evaluation script (EER + min t-DCF)
├── dataset.py              # PyTorch Dataset class for ASVspoof 2019 LA
├── config.py               # All hyperparameters in one place
├── utils.py                # Seed setting, checkpoint save/load, logging
└── results/
    └── seed_{N}/           # Checkpoints and score files per seed
```

---

## Step-by-Step Implementation Plan

### Step 1 — Environment Setup

Install the following packages in Colab or local environment:

```bash
pip install torch torchaudio numpy scipy scikit-learn pandas matplotlib
```

PyTorch version: >= 1.12.0
Python version: >= 3.8

Mount Google Drive in Colab (critical — prevents data loss on session disconnect):
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Step 2 — Dataset Download and Organisation

Download ASVspoof 2019 LA from the official Edinburgh DataShare:
- URL: https://datashare.ed.ac.uk/handle/10283/3336
- Files needed: `LA.zip` only (not PA)
- Size: ~14 GB
- Extract to `data/ASVspoof2019/LA/`

Protocol files (label files) are inside `ASVspoof2019_LA_cm_protocols/`. Format:
```
LA_0001  - LA_T_1000137  alaw  A04  spoof
LA_0001  - LA_T_1000482  -     -    bonafide
```
Columns: speaker_id, audio_filename, environment, attack_id, label

---

### Step 3 — Dataset Class (`dataset.py`)

Implement a PyTorch `Dataset` that:
1. Reads the protocol `.txt` file into a list of `(filepath, label)` pairs
   - label: `1` for bonafide, `0` for spoof
2. Loads audio with `torchaudio.load()`, resamples to 16 kHz if needed
3. Crops or zero-pads to exactly **64,000 samples** (4 seconds)
   - Training: random crop start position
   - Dev/Eval: crop from the beginning (deterministic)
4. During training only: applies **RawBoost series ①+②** augmentation on the raw waveform
5. Returns `(waveform, label, attack_id)` — attack_id needed for per-attack eval

Key detail: compute **class weights** from training label counts for weighted cross-entropy:
```python
n_bonafide = # count from train protocol
n_spoof    = # count from train protocol
weight = torch.tensor([n_spoof / n_bonafide, 1.0])  # upweight bonafide
```

---

### Step 4 — RawBoost Augmentation (`rawboost/rawboost.py`)

Copy `process_Rawboost_feature` from https://github.com/TakHemlata/RawBoost-antispoofing

Use **algorithm 1** (series ①+②): convolutive + impulsive noise only.

Parameters (from paper, do not change):
```python
# Component ①: convolutive
nBands   = 5          # number of notch filters
minF     = 20         # min center frequency (Hz)
maxF     = 8000       # max center frequency (Hz)
minBW    = 100        # min bandwidth (Hz)
maxBW    = 1000       # max bandwidth (Hz)
minCoeff = 10         # min FIR filter coefficients
maxCoeff = 100        # max FIR filter coefficients
minG     = 0          # min linear gain (dB)
maxG     = 0          # max linear gain (dB)
minBiasLinNonLin = 5  # min non-linear gain (dB)
maxBiasLinNonLin = 20 # max non-linear gain (dB)

# Component ②: impulsive
N_f      = 5          # number of harmonics
P        = 10         # max % of samples perturbed
g_sd     = 2          # fixed gain for impulsive noise
```

Apply **only during training**, not during dev or eval.

---

### Step 5 — AASIST-L Model (`models/aasist_l/`)

Copy the AASIST-L model definition from https://github.com/clovaai/aasist

Key files from that repo:
- `models/AASIST.py` — full model class
- Use the **AASIST-L config** (not full AASIST):

```json
{
  "architecture": "AASIST",
  "nb_samp": 64000,
  "first_conv": 128,
  "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
  "gat_dims": [64, 32],
  "pool_ratios": [0.5, 0.7, 0.5, 0.7],
  "temperatures": [2.0, 2.0, 100.0, 100.0]
}
```

Do not modify the architecture — use it exactly as provided in the reference repo.

---

### Step 6 — Config (`config.py`)

Centralise all hyperparameters:

```python
SEED_LIST       = [1, 2, 3]
BATCH_SIZE      = 24
NUM_EPOCHS      = 100
LEARNING_RATE   = 1e-4
ADAM_BETAS      = (0.9, 0.999)
NUM_WORKERS     = 2
SAMPLE_RATE     = 16000
NUM_SAMPLES     = 64000          # 4 seconds
PATIENCE        = 10             # early stopping on dev EER
CHECKPOINT_DIR  = "results/"
DATA_ROOT       = "data/ASVspoof2019/LA/"
USE_RAWBOOST    = True
RAWBOOST_ALGO   = 1              # series ①+② (algorithm index in RawBoost repo)
```

---

### Step 7 — Training Script (`train.py`)

Structure of the training loop:

```
for seed in [1, 2, 3]:
    set_seed(seed)
    build model (AASIST-L)
    build optimizer (Adam, lr=1e-4)
    build scheduler (CosineAnnealingLR, T_max=NUM_EPOCHS)
    build loss (weighted CrossEntropyLoss)
    build train DataLoader (with RawBoost, shuffle=True)
    build dev DataLoader   (no RawBoost, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train one epoch
        evaluate on dev set → compute EER
        scheduler.step()
        if dev EER improved:
            save checkpoint to results/seed_{seed}/best_model.pt
        if no improvement for PATIENCE epochs:
            break

    load best checkpoint
    evaluate on eval set → compute EER + min t-DCF
    save score file to results/seed_{seed}/eval_scores.txt
```

`set_seed(seed)` must set: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic = True`.

---

### Step 8 — Evaluation Script (`evaluate.py`)

Two-stage evaluation:

**Stage 1 — EER (can compute ourselves):**
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
```

**Stage 2 — min t-DCF (use official script):**
- Download `evaluate_tDCF_asvspoof19.py` from https://www.asvspoof.org/resources
- It requires a score file in the format: `filename score label`
- Run: `python evaluate_tDCF_asvspoof19.py --cm_score_file results/seed_1/eval_scores.txt`

**Per-attack reporting:**
Parse the eval protocol to split scores by `attack_id` (A07–A19). Report EER for each attack separately. Pay special attention to **A17** (VAE-VC) — historically the hardest attack.

---

### Step 9 — Results Aggregation

After all 3 seeds complete, compute:
```
pooled EER  = mean([eer_seed1, eer_seed2, eer_seed3]) ± std
pooled tDCF = mean([tdcf_seed1, tdcf_seed2, tdcf_seed3]) ± std
```

Fill into the results table in the LaTeX report (`reports/LatexCode.txt`).

Baselines to compare against (from ASVspoof 2019 paper):
| Model | EER (%) | min t-DCF |
|---|---|---|
| B01 CQCC-GMM (official baseline) | 9.57 | 0.2366 |
| B02 LFCC-GMM (official baseline) | 8.09 | 0.2116 |
| AASIST (full, no aug) | 0.83 | 0.0275 |
| AASIST-L (no aug, reported) | 0.99 | 0.0309 |
| **Our target: AASIST-L + RawBoost** | TBD | TBD |

---

## Colab-Specific Instructions

1. Always mount Google Drive at the start of every session
2. Store data, checkpoints, and results on Drive — not in `/content/` (ephemeral)
3. Run one seed per session (~3–4 hours each on T4)
4. At start of a new session, resume from the last saved checkpoint:
   ```python
   model.load_state_dict(torch.load('drive/MyDrive/.../best_model.pt'))
   ```
5. If GPU is not available, go to Runtime → Change runtime type → GPU → T4

---

## Performance Targets

| Milestone | EER (%) | min t-DCF |
|---|---|---|
| Must beat (B01) | < 9.57 | < 0.2366 |
| Must beat (B02) | < 8.09 | < 0.2116 |
| Competitive (AASIST-L reported) | < 0.99 | < 0.0309 |
| With RawBoost improvement | < 0.99 | < 0.0309 |

Expected: our AASIST-L + RawBoost should approximately match or slightly exceed reported AASIST-L numbers. Any result between 1–3% EER is a strong student project result.

---

## Files Already in This Repository

| File | Location | Purpose |
|---|---|---|
| Paper summaries (all 10) | `paper-summaries/` | Background reading |
| Consolidated summary | `paper-summaries/ALL_PAPER_SUMMARY.md` | Quick reference |
| LaTeX report | `reports/LatexCode.txt` | Paste into Overleaf |

---

## Key References for This Implementation

1. AASIST paper: Jung et al., ICASSP 2022. arXiv:2110.01200
2. RawBoost paper: Tak et al., ICASSP 2022. arXiv:2111.04433
3. ASVspoof 2019 dataset: Wang et al., CSL 2020. DOI:10.1016/j.csl.2020.101114
4. RawNet2 paper: Tak et al., ICASSP 2021. arXiv:2011.01108
5. Comparative study (P2SGrad, multi-seed): Wang & Yamagishi, Interspeech 2021. arXiv:2103.11326

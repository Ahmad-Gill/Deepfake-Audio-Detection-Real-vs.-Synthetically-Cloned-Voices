---
name: implementation-guide
description: "Use this skill for any implementation, coding, debugging, or experimental work on the AI-600 Deepfake Audio Detection project. Triggers include: 'help me implement RawNet2', 'set up my environment', 'run my experiments', 'debug training', 'compute EER', 'run ablations', 'per-attack analysis', 'my training is not converging', 'how do I evaluate my model', 'write training code', 'help me with AASIST', 'integrate RawBoost', 'track my results', or any request involving code, training, evaluation, or debugging for this project. This skill covers the complete May 9–20 implementation sprint between the midterm and final report."
---

# Implementation Guide — AI-600 Deepfake Audio Detection
## May 9–20 Sprint: From Baseline to Full Ablation Study

---

## How to Use This Skill

Before responding, identify what phase the user is in and load the relevant reference:

```
WHERE ARE YOU?
│
├── "Just starting / setting up"
│   └── READ: references/01-environment-setup.md
│
├── "Implementing / training RawNet2"
│   └── READ: references/02-rawnet2-guide.md
│
├── "Implementing / training AASIST"
│   └── READ: references/03-aasist-guide.md
│
├── "Integrating RawBoost augmentation"
│   └── READ: references/04-rawboost-guide.md
│
├── "Running experiments / ablations"
│   └── READ: references/05-experiments-guide.md
│
├── "Computing EER / evaluating model"
│   └── READ: references/06-evaluation-guide.md
│
├── "Per-attack analysis"
│   └── READ: references/06-evaluation-guide.md (Section 3)
│
├── "Something is broken / not converging"
│   └── READ: references/07-debugging-guide.md
│
└── "Tracking / logging results"
    └── READ: references/08-results-tracker.md
```

If the user's situation spans multiple files, load all relevant ones before responding.

---

## Project Context

**Goal:** Train binary classifiers on ASVspoof 2019 LA to distinguish bonafide from spoofed speech. Achieve generalization to unseen attacks (A07–A19 in eval set).

**Stack:**
- Python 3.8, PyTorch 1.13+, torchaudio 0.13+
- Dataset: ASVspoof 2019 LA (Edinburgh DataShare)
- Repo structure: `src/models/`, `src/data/`, `src/augmentation/`, `results/`

**6 Configurations to Run:**
| ID | Features | Architecture | Augmentation |
|----|----------|-------------|-------------|
| C1 | Raw sinc | RawNet2 | None |
| C2 | Raw sinc | RawNet2 | RawBoost ①+② |
| C3 | Raw sinc | AASIST-L | None |
| C4 | Raw sinc | AASIST-L | RawBoost ①+② |
| C5 | SSL (wav2vec 2.0) | AASIST-L | None |
| C6 | SSL (wav2vec 2.0) | AASIST-L | RawBoost ①+② |

Each run × 3 random seeds → report mean ± std EER and min t-DCF.

**Target Numbers (to beat):**
- C1 (RawNet2): ~4.66% EER (published)
- C3/C4 (AASIST-L): ~0.99% EER (published)
- Stretch: <0.83% EER

---

## Daily Sprint Plan (Quick Reference)

| Days | Focus | Reference File | Done When |
|------|-------|---------------|-----------|
| 1 | Environment + dataset verified | 01-environment-setup.md | `python verify_dataset.py` passes |
| 2–4 | RawNet2 baseline running | 02-rawnet2-guide.md | EER ≤ 5.3% on eval set |
| 4 | RawBoost integrated | 04-rawboost-guide.md | C2 training starts |
| 5–8 | AASIST-L training | 03-aasist-guide.md | EER ≤ 1.5% on eval set |
| 9–11 | All 6 configurations | 05-experiments-guide.md | All C1–C6 results logged |
| 12 | Per-attack analysis | 06-evaluation-guide.md | Per-attack table generated |
| 12–13 | Results table finalized | 08-results-tracker.md | CSV ready for final report |

---

## Core Principles for This Project

**On reproducibility:** Run every configuration with seeds 42, 123, 456. Report mean ± std. Never report a single run as your result — this was a specific commitment in your midterm report.

**On the eval vs dev split:** NEVER tune hyperparameters on the eval set. Use dev set for early stopping and hyperparameter decisions. Eval set is touched exactly once per configuration.

**On results interpretation:** EER of X% on eval set is meaningful. EER on dev set is only useful for stopping. Don't confuse the two.

**On debugging:** If EER is not moving below 15% after 30 epochs, something is wrong with data loading or class weighting — check debugging guide before running more epochs.

**On time management:** C1 (RawNet2) is the critical path. If C1 is not running by Day 3, focus all effort there. C5 and C6 (SSL) are stretch goals — skip if time is short.

---

## When You're Stuck

Always check in this order:
1. Is the dataset loaded correctly? Run `verify_dataset.py`
2. Is the class imbalance being handled? Check loss weighting
3. Is the EER metric computed on eval (not dev)?
4. Check `references/07-debugging-guide.md` for your specific error

Paste your error message and Claude will diagnose it.

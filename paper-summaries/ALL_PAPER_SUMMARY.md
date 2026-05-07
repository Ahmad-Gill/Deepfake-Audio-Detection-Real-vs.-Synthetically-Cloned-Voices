# Mini-Report: 10 Paper Summaries (ASVspoof 2019 LA Literature Review)

> **Context:** Literature review for a deepfake audio detection project on the ASVspoof 2019 Logical Access (LA) track. Emphasis on **generalization** over high training accuracy.
>
> **Convention:** Items relevant to generalization, ASVspoof 2019 LA evaluation, or EER/t-DCF reporting are flagged ⚑.
>
> **Note:** Papers 1, 2, 3, and 7 are surveys/dataset/SSL pre-training papers, not single-model papers. For these, "Finalized Model" reports the SOTA recipe or baseline being characterized.

---

## 1. [Survey on Speech Deepfake Detection — Li et al., 2024](01-FPD-survey-speech-deepfake-detection.md)

- **Finalized Model:** Survey paper — no single model. SOTA recipe identified: **wav2vec 2.0 (XLSR) front-end + AASIST back-end + OC-Softmax + RawBoost/codec augmentation**.
- **Dataset Used:** Reviews ASVspoof 2019 LA, 2021 LA/DF, ITW, PartialSpoof, CodecFake, MLAAD across 200+ papers.
- **Accuracy Achieved (on ASVspoof 2019 LA ⚑):**
  - Best single system: **0.22% EER** (LFCC + SDC + codec aug + Bi-LSTM + SE-ResNeXT, ICASSP'23)
  - wav2vec2-XLSR + MLP: 0.31% EER
  - AASIST + RawBoost: 0.82% EER
  - Cross-dataset (trained 2019-LA → ITW): collapses to 7–30% EER
- **Architecture Details (recipe-level findings ⚑):**
  - Loss: **OC-Softmax** > AM-Softmax > CE for generalization (separate margins for bona fide/fake)
  - Augmentation: **Codec aug (−22% EER on 2021-LA)**, **RawBoost** (noise+RIR), masking, SpecMix
  - Regularization: **Knowledge Distillation** is best technique for cross-dataset generalization ⚑
  - Activations: PReLU, AReLU
  - Other: Sub-band restriction (0–4 kHz) improves robustness ⚑

---

## 2. [ASVspoof 2019 Database — Wang et al., 2020](02-FPD-asvspoof2019-database.md)

- **Finalized Model:** Dataset paper. Provides **two baselines: B1 (CQCC-GMM, 512 components)** and **B2 (LFCC-GMM, 512 components)**.
- **Dataset Used (this is THE target dataset ⚑):**
  - Built on VCTK (107 speakers, 16 kHz). Speaker-disjoint splits.
  - LA Train: 2,580 bona fide + 22,800 spoofed (A01–A06)
  - LA Dev: 2,548 bona fide + 22,296 spoofed (A01–A06)
  - LA Eval: 7,355 bona fide + 63,882 spoofed (A07–A19, mostly unknown attacks ⚑)
- **Accuracy Achieved (LA Eval ⚑):**
  - B1 (CQCC-GMM): **9.57% EER, 0.2366 min-tDCF**
  - B2 (LFCC-GMM): **8.09% EER, 0.2116 min-tDCF**
  - Hardest attacks: A10 (Tacotron2+WaveRNN), A13, A17 (VAE VC)
- **Architecture Details:**
  - Optimizer/LR: N/A (GMM, EM training)
  - Loss: GMM log-likelihood ratio
  - Augmentation: None
  - Other: Introduces **min-tDCF** as primary metric ⚑ (combines CM + ASV cost). Train/dev contain known attacks only; eval mostly unknown — built-in generalization stress test ⚑.

---

## 3. [ASVspoof 2019 Challenge Results — Todisco et al., Interspeech 2019](03-FPD-asvspoof2019-challenge-results.md)

- **Finalized Model:** Challenge results paper. Top LA system: **T05 — neural-network ensemble (anonymized)**.
- **Dataset Used:** ASVspoof 2019 LA + PA, 63 participating teams (48 LA submissions, 50 PA).
- **Accuracy Achieved (LA Eval ⚑):**
  - **T05 (winner): 0.22% EER, 0.0069 min-tDCF** — >30× better than baselines
  - T45: 1.86% EER, 0.0510 min-tDCF
  - 27 of 48 teams beat B02 baseline
  - PA: T28 winner — 0.39% EER, 0.0096 min-tDCF
- **Architecture Details (aggregate trends ⚑):**
  - Top-7 LA systems all used **neural networks**; top-9 used **ensemble fusion** ⚑
  - β weight per attack varies (β≈26 for A17) — t-DCF and EER can rank attacks oppositely
  - Hardest attacks: A17 (VAE VC), A10 (Tacotron2+WaveRNN), A13
  - No specific optimizer/loss reported (anonymized submissions)

---

## 4. [End-to-End Anti-Spoofing with RawNet2 — Tak et al., ICASSP 2021](04-CA-rawnet2-end-to-end-antispoofing.md)

- **Finalized Model:** **RawNet2** — fixed sinc filters (Mel/inv-Mel/linear) → 6 residual blocks with FMS attention → GRU(1024) → FC → softmax(2). Best single: **S3 (linear-scale)**; best fusion: **L+S1**.
- **Dataset Used:** ASVspoof 2019 LA only. Train+90% of Dev for training, 10% Dev held out as validation. Eval = A07–A19.
- **Accuracy Achieved (LA Eval ⚑):**
  - RawNet2 S3 single: **4.66% EER, 0.1294 min-tDCF**
  - RawNet2 S1 (Mel) single: 5.64% EER, 0.1301 min-tDCF
  - **L+S1 fusion (LFCC-GMM + RawNet2): 1.12% EER, 0.0330 min-tDCF**
  - L+S1+S2+S3 fusion on **A17: 0.0808 min-tDCF** (best published A17 at the time)
- **Architecture Details:**
  - **Optimizer: Adam, lr=1e-4** ⚑
  - **Epochs: 100, batch size: 32**
  - **Loss: Cross-Entropy** (softmax over 2 classes)
  - Input: ~4 sec raw waveform (64,000 samples @ 16 kHz)
  - Augmentation: **None**
  - Notable: **Sinc filters fixed (not learned)** to prevent overfitting given only 6 known attacks ⚑ — explicit generalization-motivated design choice. **FMS** filter-wise attention. Score-level **SVM fusion** with LFCC-GMM.

---

## 5. [AASIST — Jung et al., ICASSP 2022](05-CA-aasist-spectrotemporal-graph-attention.md)

- **Finalized Model:** **AASIST** — RawNet2 encoder → spectral graph Gs + temporal graph Gt → **HS-GAL** (heterogeneous stacking graph attention) → **MGO** (max graph operation, 2 parallel branches) → readout (max+avg+stack node) → FC. Lightweight variant: **AASIST-L (85K params)**.
- **Dataset Used:** ASVspoof 2019 LA only.
- **Accuracy Achieved (LA Eval ⚑):**
  - **AASIST: 0.83% EER, 0.0275 min-tDCF** (SOTA single system at publication)
  - **AASIST-L (85K params): 0.99% EER, 0.0309 min-tDCF**
  - Beats RawGAT-ST (1.06% EER, 0.0335)
- **Architecture Details:**
  - **Optimizer: Adam, lr=1e-4** ⚑
  - **LR Schedule: Cosine annealing** ⚑
  - **Loss: Cross-Entropy** (weighted CE)
  - Input: 64,000 raw samples (~4 sec)
  - Augmentation: **None reported**
  - Notable: Graph pooling (50% spectral, 30% temporal node removal — acts as regularization ⚑); **stack node = [CLS]-style aggregator**; results averaged over **3 random seeds** ⚑; AASIST-L tuned via **population-based training**.

---

## 6. [Comparative Study on Neural CMs — Wang & Yamagishi, Interspeech 2021](06-CA-comparative-study-neural-spoofing-countermeasures.md)

- **Finalized Model:** **LCNN-LSTM-sum** (LCNN + 2 Bi-LSTM layers + skip connection + average pooling + FC) with **LFCC** front-end and new **P2SGrad** loss. Best fusion: front-end diversity (LFCC + LFB + Spectrogram).
- **Dataset Used:** ASVspoof 2019 LA only. Each configuration trained 6 times (different seeds).
- **Accuracy Achieved (LA Eval ⚑):**
  - **LFCC + LCNN-LSTM-sum + P2SGrad: 1.92% EER (best), 0.057 min-tDCF**
  - Range across 6 seeds: 1.92–3.10% EER (intra-model variance) ⚑
  - Front-end fusion (LFCC+LFB+Spec): **1.074% EER**
  - LFCC + LCNN-LSTM-sum + Sigmoid: 3.92% EER
- **Architecture Details:**
  - **Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8)** ⚑
  - **LR: 3e-4, halved every 10 epochs** ⚑
  - Batch size: 8 or 64
  - **Loss: P2SGrad (new) — MSE on cosine distances, zero hyperparameters** ⚑; also tested AM-Softmax, OC-Softmax, Sigmoid
  - Augmentation: **None**
  - Notable: **Multi-seed (6 runs) + Holm-Bonferroni significance testing** ⚑ — methodological gold standard for reproducibility. Variable-length input via Bi-LSTM + average pooling beats fixed trim/pad.

---

## 7. [wav2vec 2.0 — Baevski et al., NeurIPS 2020](07-FS-wav2vec2-self-supervised-speech-representations.md)

- **Finalized Model:** **wav2vec 2.0** — 7-block CNN feature encoder + Transformer context network (BASE: 12 blocks/95M params; LARGE: 24 blocks/317M params) + product quantization (G=2, V=320). **Not a deepfake detector** — used as upstream SSL feature extractor.
- **Dataset Used:** Pre-train on LibriSpeech (960h) or Libri-Light (60k h, EN). Fine-tune on TIMIT/LibriSpeech (10 min – 960h).
- **Accuracy Achieved:** ASR metric (WER), not anti-spoofing — best 1.8/3.3 WER on Librispeech test-clean/other; not directly comparable to ASVspoof EER.
- **Architecture Details (relevant for downstream use ⚑):**
  - **Optimizer: Adam, polynomial LR decay**; SpecAugment-style masking during fine-tuning ⚑
  - **Loss: Contrastive (κ=0.1, K=100 distractors) + diversity loss (α=0.1)** — pre-training only
  - Augmentation: **Span masking — p=0.065, M=10 steps, ~49% of timesteps masked** ⚑ (regularization built into pre-training objective)
  - Notable: Pre-trained checkpoints (`wav2vec2-base/large/XLS-R`) are the foundation for SOTA ASVspoof 2019 LA detectors (see paper 8). Encoder freezes for first 10k fine-tuning updates.

---

## 8. [SSL Front-Ends for Spoofing CMs — Wang & Yamagishi, Odyssey 2022](08-FS-ssl-frontends-spoofing-countermeasures.md)

- **Finalized Model:** **W2V-XLSR (317M, multilingual) + LLGF (LCNN→BiLSTM→GAP→FC), fully fine-tuned end-to-end**.
- **Dataset Used:** Train on ASVspoof 2019 LA train. Eval on **2019 LA, 2015 LA, 2021 LA eval, 2021 DF** (4-way cross-condition test) ⚑.
- **Accuracy Achieved (LA Eval ⚑):**
  - **W2V-XLSR + LLGF (fine-tuned): 0.11% EER on 2019 LA, 0.25% on 2015 LA, ~7.6% on 2021 LA** ⚑
  - Fixed (frozen) W2V-XLSR + LLGF: 1.47% EER on 2019, 3.97% on 2015
  - LFCC-LLGF baseline: 2.98% on 2019, **collapses to 29.42%** on 2015 ⚑
  - Caveat: SSL min-tDCF (0.120) is **worse than LFCC (0.098)** — likely failing on A17
- **Architecture Details:**
  - **Optimizer: Adam with LR warm-up** (essential for fine-tuning) ⚑
  - **Loss: P2SGrad** (zero hyperparameters)
  - Augmentation: **None**
  - Notable: **Fine-tuning is essential** ⚑ (10–20× better than frozen). **Multilingual pre-training > monolingual at same size** ⚑ (data diversity drives generalization). When fine-tuned, simple GAP+FC back-end suffices. Sub-band analysis: SSL relies on 0.1–2.4 kHz (linguistic content) vs LFCC on 5.6–8 kHz (vocoder artifacts) — explains generalization ⚑.

---

## 9. [RawBoost — Tak et al., ICASSP 2022](09-AG-rawboost-data-augmentation-antispoofing.md)

- **Finalized Model:** **RawNet2 + RawBoost augmentation (series ①+②: convolutive non-linear + impulsive signal-dependent noise)**. Augmentation method, not a new architecture.
- **Dataset Used:** Train on ASVspoof 2019 LA train+dev; eval on **ASVspoof 2021 LA** (7 telephony conditions C1–C7).
- **Accuracy Achieved (2021 LA pooled ⚑):**
  - **RawBoost ①+②: 5.31% EER, 0.3099 min-tDCF** (27% relative min-tDCF reduction, 44% EER reduction over baseline)
  - RawNet2 baseline (no aug): 9.50% EER, 0.4257 min-tDCF
  - Outperforms WavAugment, SpecAugment, codec augmentation
  - WavAugment pitch+reverb: **15.66% EER** (worse than baseline — wrong augmentation can hurt)
- **Architecture Details:**
  - **Optimizer: Adam, lr=1e-4** ⚑
  - **Batch=128, 100 epochs**
  - **Loss: Cross-Entropy**
  - **Augmentation (the contribution) ⚑:** Three on-the-fly raw-waveform components:
    - ① Linear+non-linear convolutive (Hammerstein systems, 5 notch filters, fc∈[20,8k]Hz)
    - ② Impulsive signal-dependent noise (Prel∈[0,10]%, gain=2)
    - ③ Stationary additive noise (SNR∈[10,40] dB, FIR-coloured)
    - Best combination: **series ①+②**; ③ only helps alone
  - Notable: **No external data needed** ⚑ — pure DSP, model-agnostic plug-in. Direct generalization-targeted design.

---

## 10. [Does Audio Deepfake Detection Generalize? — Müller et al., Interspeech 2022](10-AG-does-audio-deepfake-detection-generalize.md)

- **Finalized Model:** No single model — **uniform re-evaluation of 12 architectures** (LSTM, LCNN, LCNN-Attention, LCNN-LSTM, MesoNet, MesoInception, ResNet18, Transformer, CRNNSpoof, RawNet2, RawPC, RawGAT-ST). Best on ASVspoof: RawGAT-ST. Best in-the-wild: RawNet2 / MesoInception.
- **Dataset Used:** Train: ASVspoof 2019 LA train+dev. Eval: ASVspoof 2019 LA eval **AND new In-The-Wild dataset (37.9h, 58 celebrities, sourced from social media)** ⚑.
- **Accuracy Achieved ⚑:**
  - **RawGAT-ST (raw, full): 1.23% EER on ASVspoof → 37.15% EER on In-The-Wild**
  - RawNet2 (raw, 4s): 4.35% on ASVspoof → **33.94% (best ITW)**
  - MesoInception (logspec, full): 10.02% → 37.41% ITW
  - LCNN (logspec, 4s): → 91.11% ITW (near-random)
  - **Headline: 200–1000% degradation from ASVspoof to in-the-wild** ⚑
  - Adding eval data to training does NOT improve ITW (33.9 → 33.1%) — distributional, not data-quantity, problem ⚑
- **Architecture Details (uniform protocol applied across all 12) ⚑:**
  - **Optimizer: Adam, lr=1e-4 with LR scheduler** ⚑
  - **Epochs: 100, early stopping (patience=5)** ⚑
  - **Loss: Cross-Entropy with log-softmax**
  - Augmentation: **None** (deliberate — controlled comparison)
  - **Each experiment repeated 3× with random seeds, mean ± std reported** ⚑
  - Notable engineering findings ⚑:
    - **cqtspec > logspec >> melspec** (~37% relative EER improvement)
    - **Full variable-length input > fixed 4s truncation** (~50% EER reduction)
    - Field may have over-fit to ASVspoof/VCTK distribution

---

## Cross-Paper Generalization-Relevant Synthesis (for the literature review)

| Generalization Lever | Best Source | Effect |
|---|---|---|
| Multilingual SSL front-end | Paper 8 (W2V-XLSR fine-tuned) | 2015 LA: 29.42% → 0.25% EER |
| Loss function | Papers 1, 6 (P2SGrad / OC-Softmax) | Stable, hyperparameter-free, SOTA-competitive |
| Augmentation | Paper 9 (RawBoost ①+②) | Cross-channel: 27% min-tDCF reduction |
| Knowledge distillation | Paper 1 | Best for cross-dataset (2019 → ITW) |
| Multi-seed evaluation | Papers 5, 6, 10 | Required — single-run results unreliable |
| Feature choice (spectral) | Paper 10 (cqtspec, not melspec) | ~37% relative EER improvement |
| Variable-length input | Paper 10 | ~50% EER improvement vs 4s trunc |
| Reality check | Paper 10 (In-The-Wild) | 200–1000% degradation — the real benchmark of generalization |

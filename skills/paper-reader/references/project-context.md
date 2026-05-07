# Project Context — Deepfake Audio Detection

## Course & Deliverable
- **Course:** AI-600 Deep Learning (Spring 2026)
- **Project:** Deepfake Audio Detection: Real vs. Synthetically Cloned Voices
- **Type:** Supervised binary classification on audio
- **Mid-report deadline:** 08/05/26 (25% weight)
- **Final report deadline:** 22/05/26 (50% weight)
- **Report format:** ICML-style, 4–5 pages

## Mid-Report Requirements
The mid-report must include:
1. Brief literature review
2. Proposed methodology / architecture
3. Progress so far
4. Preliminary results (mandatory)

## Mid-Report Sections (ICML Format)
When mapping paper content to the mid-report, use these sections:
- **Abstract** (~150 words)
- **Introduction** (problem statement, motivation, contribution summary)
- **Literature Review** (organized by: traditional features → CNN-based → end-to-end → SSL-based → generalization studies)
- **Proposed Methodology** (architecture, features, augmentation, training setup)
- **Preliminary Results** (baseline reproduction, initial experiments)
- **Conclusion / Next Steps**

## Dataset
- **Primary:** ASVspoof 2019 — Logical Access (LA) track
- 121,000+ utterances, 107 speakers
- 19 spoofing systems (A01–A19): TTS and voice conversion
- Train/dev: attacks A01–A06 (known); Eval: attacks A07–A19 (unknown)
- **Key challenge:** Generalization to unknown attacks in the eval set

## Evaluation Metrics
- **EER** (Equal Error Rate) — primary metric
- **t-DCF** (tandem Detection Cost Function) — secondary metric
- Lower is better for both

## Core Research Challenge
The project emphasizes **generalization**: models must perform well on spoofing attacks they have never seen during training. This means:
- Cross-attack generalization (train on A01–A06, eval on A07–A19)
- Data augmentation strategies matter
- Feature representations that capture synthesis artifacts (not speaker identity) are preferred

## Key Baselines to Know
When reading any paper, compare its results against these known baselines on ASVspoof 2019 LA eval:
- CQCC + GMM: EER ~9.57% (official challenge baseline)
- LFCC + GMM: EER ~8.09% (official challenge baseline)
- RawNet2: EER ~5.13%
- AASIST: EER ~0.83%
- AASIST + SSL (wav2vec 2.0): varies by configuration

## 10 Papers in Our Reading List
These are the 10 papers selected for the mid-report. When processing any paper, note how it relates to these:

1. **A Survey on Speech Deepfake Detection** (2024) — lit review backbone
2. **ASVspoof 2019 Database Paper** (2019) — dataset specification
3. **ASVspoof 2019 Challenge Overview** (2019) — evaluation protocol
4. **RawNet2** (2020) — end-to-end baseline
5. **AASIST** (2021) — SOTA architecture baseline
6. **Comparative Study on Neural Countermeasures** (2021) — benchmark numbers
7. **wav2vec 2.0** (2020) — SSL feature foundation
8. **Self-supervised Front Ends for Spoofing** (2021) — SSL layer selection
9. **RawBoost** (2021) — data augmentation method
10. **Does Audio Deepfake Detection Generalize?** (2022) — generalization problem

## Literature Review Categories
When categorizing a paper, assign it to one of these:
- **Survey / Review**
- **Benchmark / Dataset / Challenge**
- **Feature Extraction — Handcrafted** (MFCC, LFCC, CQT, CQCC)
- **Feature Extraction — Self-Supervised** (wav2vec 2.0, HuBERT, WavLM)
- **Model Architecture — CNN/ResNet**
- **Model Architecture — End-to-End** (raw waveform input)
- **Model Architecture — Graph/Attention**
- **Model Architecture — Other**
- **Data Augmentation**
- **Generalization / Cross-Dataset**
- **Robustness (Noise / Adversarial / Channel)**
- **Real-time / Deployment**

## How to Assess Relevance
Rate each paper's relevance to the project as:
- **Essential** — directly used in methodology, dataset, or evaluation
- **High** — provides important context, baselines, or techniques
- **Medium** — useful background but not central
- **Low** — tangentially related, skip for mid-report

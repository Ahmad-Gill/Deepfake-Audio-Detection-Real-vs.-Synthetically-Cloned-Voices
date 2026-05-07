# Paper Summary: End-to-End Anti-Spoofing with RawNet2

**Authors:** Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, Nicholas Evans, Anthony Larcher
**Affiliation:** EURECOM, Sophia Antipolis, France; LIUM – Université du Maine, France
**Venue:** ICASSP 2021
**arXiv:** 2011.01108v3 (December 2021)
**Code:** https://github.com/eurecom-asp/rawnet2-antispoofing

---

## 1. What Problem Does This Paper Solve?

Most anti-spoofing countermeasures rely on **handcrafted spectral features** (CQCC, LFCC, MFCC) fed into GMM or shallow classifiers. While these work well for many attacks, **A17** — a neural network voice conversion (VC) system using VAE-based acoustic modelling with direct waveform filtering — consistently evades detection across nearly all submitted systems in the ASVspoof 2019 challenge.

The paper asks: **can an end-to-end deep neural network operating directly on raw waveforms detect attacks that handcrafted features cannot?**

Specifically, it is the **first application of RawNet2** (originally designed for speaker verification) to anti-spoofing countermeasures. The paper:
- Adapts RawNet2 for the spoofing detection task
- Tests whether raw-waveform models learn complementary artefact cues to spectral-feature classifiers
- Focuses explicitly on the worst-case A17 attack as the primary stress test

---

## 2. Why Does This Problem Matter?

- **A17 is the hardest attack in ASVspoof 2019 LA**: It achieves only 3.92% ASV EER (nearly matching real speech) and has the highest min-tDCF of all attacks because of the high penalty weight β≈26
- **Handcrafted features are attack-specific**: They capture artefacts of known attacks but cannot generalise to unforeseen ones. A17 artefacts reside at the sub-band and phase level — regions where CQCC/LFCC provide limited discriminative power
- **Generalisation to unknown attacks is the core challenge**: In real-world deployments, the nature of future spoofing attacks cannot be predicted. A more general, learned representation is preferable
- **Once a vulnerability is exposed, patches require training data**: If A17 can only be detected after adding A17 examples to training, the system is reactive rather than proactive
- **Raw waveform models avoid feature engineering bias**: They learn task-specific representations directly from signal, potentially capturing artefacts invisible to fixed filterbanks

---

## 3. What Has Been Tried Before?

### Handcrafted Feature + GMM Approaches
- **CQCC-GMM (B1)**: Constant-Q cepstral coefficients; good for vocoder artefacts in frequency; pooled EER 9.57%, A17 t-DCF 0.5859
- **LFCC-GMM (B2)**: Linear frequency cepstral coefficients; pooled EER 8.09%, A17 t-DCF 0.2042
- **High-spectral-resolution LFCC (L)**: 70 linearly-spaced filters + GMM; pooled min-tDCF 0.0904, A17 t-DCF 0.3524 — outperforms all but 3 of 48 ASVspoof 2019 submissions

### One-Class / Anomaly Detection Approaches
- Attempted using classifiers trained only on bona fide data (no spoofed training examples)
- Have shown success in anomaly/novelty detection but fail on ASVspoof 2019 A17 specifically — artefacts are too subtle

### End-to-End Approaches (Prior to This Paper)
- **RawNet (Jung et al., 2019)**: Raw waveform CNN for speaker verification with LSTM aggregation
- **RawNet2 (Jung et al., 2020)**: Improved version with SincNet-inspired first layer, residual blocks, FMS attention, GRU aggregation — showed 11–20% relative EER improvement over i-vector/x-vector on VoxCeleb
- Neither RawNet variant had been applied to anti-spoofing before this paper

---

## 4. What Does This Paper Propose?

### RawNet2 for Anti-Spoofing

RawNet2 is adapted from speaker verification to spoofing detection. The full architecture processes ~4 seconds of raw audio (64,000 samples at 16 kHz):

| Layer | Configuration | Output Shape |
|-------|--------------|-------------|
| Input | 64,000 raw samples | — |
| **Fixed Sinc filters** (modified) | Conv(129, 128), MaxPool(3), BN+LeakyReLU | (21,290, 128) |
| Residual block × 2 | Conv(3,1,128) × 2, MaxPool(3), **FMS** | (2,365, 128) |
| Residual block × 4 | Conv(3,1,512) × 2, MaxPool(3), **FMS** | (29, 512) |
| GRU | 1024 hidden units | (1,024) |
| FC | 1,024 units | (1,024) |
| Output | Softmax | 2 (bona fide / spoof) |

**Key components:**
- **Fixed Sinc filters**: First convolutional layer is a bank of band-pass sinc filters. Unlike the original RawNet2, filter bandwidth and spectral position are **fixed** (not learned), because the small number of distinct spoofing attack algorithms in training (only 6) would cause overfitting if learned freely
- **Filter-wise Feature Map Scaling (FMS)**: Sigmoid-gated attention applied to each residual block's output. Acts as a channel attention mechanism, emphasising the most discriminative filter outputs
- **GRU temporal aggregation**: Captures temporal patterns across the utterance — critical for detecting transient clicking artefacts in A17

### Three Filter Bank Configurations Tested

| Config | Sinc Filter Scaling | Motivation |
|--------|-------------------|-----------|
| **S1** | Fixed Mel-scaled | Standard Mel filterbank; matches perceptual importance |
| **S2** | Fixed inverse Mel-scaled | Emphasises high-frequency sub-bands where spoofing artefacts concentrate |
| **S3** | Fixed linear-scale | Uniform frequency coverage; best for detecting broadband artefacts |

### Key Modifications from Original RawNet2

1. **No layer normalisation on input**: Hurts performance in this setting (likely due to sparse attack diversity)
2. **Fixed sinc filters** (vs learned): Prevents overfitting to only 6 known attack types in training
3. **Shorter filter length**: 129 samples (vs 251) — better suited for cue durations relevant to spoofing artefacts
4. **Larger second residual block**: 512 filters (vs 128) — more capacity for discriminative feature extraction
5. **Cosine similarity output** replaced by **softmax over 2 classes**: Better calibrated for binary spoof/bona fide decision

### Training Details
- Optimiser: Adam, lr = 0.0001
- Epochs: 100, batch size: 32
- Data re-partitioning: 90% of development set merged with training; 10% held out for validation
- Loss: Cross-entropy

---

## 5. Experiments and Results

### Dataset
ASVspoof 2019 LA — three partitions:
- **Training**: 2,580 bona fide + 22,800 spoofed (A01–A06, known attacks)
- **Development**: 2,548 bona fide + 22,296 spoofed (A01–A06)
- **Evaluation**: 7,355 bona fide + 63,882 spoofed (A07–A19, unknown attacks)

### Per-Attack Results on Evaluation Set (Table 2)

Selected attacks — min-tDCF (lower is better):

| Attack | L (LFCC-GMM) | S1 (Mel) | S2 (Inv-Mel) | S3 (Linear) |
|--------|-------------|---------|------------|------------|
| A07 | 0.0011 | 0.0277 | 0.0131 | 0.0382 |
| A10 (Tacotron2) | 0.1536 | 0.0373 | 0.0300 | 0.0493 |
| A13 | 0.0798 | 0.0192 | 0.0100 | 0.0505 |
| **A17 (VAE VC)** | **0.3524** | **0.2620** | **0.2626** | **0.1810** |
| A17 (best single) | 0.3524 | — | — | **0.1810** (S3) |
| **Pooled** | **0.0904** | **0.1301** | **0.1175** | **0.1294** |
| Pooled EER (%) | **3.50** | 5.64 | 5.13 | 4.66 |

**Observation**: All three RawNet2 variants are **worse than the LFCC baseline on pooled metrics** but **substantially better than the baseline on A17**. S3 (linear-scale) achieves the best A17 result (0.181 vs baseline's 0.352 — a ~49% reduction).

### Fusion Results (Table 3) — Evaluation Set

| System | min-tDCF | EER (%) | A17 min-tDCF |
|--------|---------|---------|-------------|
| **T05** (best 2019 challenge) | **0.0069** | **0.22** | — |
| **L+S1** | **0.0330** | **1.12** | **0.1161** |
| L+S1+S2+S3 | 0.0347 | 1.14 | **0.0808** (best A17) |
| L+S3 | 0.0370 | 1.14 | 0.0965 |
| L+S2 | 0.0443 | 1.35 | 0.1339 |
| T45 | 0.0510 | 1.86 | 0.2208 |
| T60 | 0.0755 | 2.64 | 0.3254 |
| L alone (LFCC-GMM) | 0.0904 | 3.50 | 0.3524 |
| LFCC-B2 (official baseline) | 0.2116 | 8.09 | 0.2042 |
| CQCC-B1 (official baseline) | 0.2366 | 9.57 | 0.5859 |

**Key results:**
- **L+S1 fusion achieves 2nd-best published results** (min-tDCF 0.0330) at the time of writing — behind only T05 (0.0069)
- **L+S1+S2+S3 achieves best published A17 result** (0.0808) — RawNet2 is uniquely effective against this worst-case attack
- Fusion improvements confirm RawNet2 learns **complementary cues** that LFCC-GMM misses
- L+S1 outperforms all other reported fusion baselines and is reproducible with open-source code

### Why Does RawNet2 Detect A17?

The paper's hypothesis (validated by ablation logic, not yet fully confirmed):
- A17 attacks produce **phase-related artefacts** — occasional, punctual clicking noises
- The **linear-phase sinc filters** in RawNet2's first layer produce phase-aligned waveforms passed to residual blocks
- The **GRU's temporal attention** detects the intermittent, transient nature of these clicks across time
- LFCC/CQCC use power spectral features — **phase information is discarded** during feature extraction, making A17 invisible to them

---

## 6. Limitations and Weaknesses

1. **Pooled performance is inferior to baseline**: All three RawNet2 single systems (S1, S2, S3) underperform the LFCC-GMM baseline (L) on pooled min-tDCF and EER. RawNet2 is not a standalone replacement

2. **Requires fusion to be competitive**: Competitive results depend on SVM-based fusion of L+RawNet2; a single system alone is insufficient

3. **Still far behind T05**: Even the best fusion (L+S1, 0.0330) is ~5× worse than the top 2019 challenge system (T05, 0.0069). The gap is substantial

4. **A17 min-tDCF remains very high**: After fusion, A17 min-tDCF is 0.0808–0.1161 vs pooled 0.0330–0.0347 — the A17 problem is significantly mitigated but not solved

5. **Fixed filters limit adaptability**: Fixing sinc filter parameters solves the overfitting problem but limits the model's ability to discover optimal frequency regions for unseen attacks

6. **Architecture not purpose-designed for anti-spoofing**: RawNet2 was designed for speaker verification; the adaptation is pragmatic but not principled. Artefact cues may require different temporal/spectral scales than speaker identity cues

7. **Fusion method uses SVM**: An SVM trained on score-level outputs — not end-to-end. A joint training approach might yield better results

8. **No analysis of generalisation beyond A17**: While the paper focuses on A17, it doesn't fully explore which attacks benefit most/least from the end-to-end representation vs the spectral baseline

9. **Single corpus**: English-only VCTK-based ASVspoof 2019. Generalisation to other languages, recording conditions, or out-of-domain TTS is not evaluated

---

## 7. Key Takeaways

1. **Raw waveform models learn genuinely different artefact cues than spectral features**: The fact that fusion consistently improves over either system alone — especially on A17 — proves the representations are complementary, not redundant

2. **Phase information is critical for detecting certain VC attacks**: LFCC/CQCC discard phase; RawNet2's sinc filters preserve it. A17's waveform filtering approach leaves phase-domain artefacts invisible to conventional features

3. **Temporal modelling matters for transient artefacts**: The GRU layer captures clicking/artefact patterns that occur intermittently across the utterance — something frame-level GMM classifiers fundamentally cannot do

4. **Fixing learned parameters is sometimes right**: In low-diversity training settings (only 6 attack types), fixing sinc filter positions prevents overfitting and maintains generalisation

5. **Fusion of complementary systems is currently the most reliable path to strong performance**: No single architecture captures all attack types; diverse feature + model ensembles remain necessary

6. **A17 requires special treatment**: It is uniquely difficult — high β weight, phase-domain artefacts, VAE-based acoustic model. Any serious anti-spoofing system should be specifically tested on A17

7. **End-to-end approaches are not a silver bullet**: Despite RawNet2's success on A17, its overall performance is worse than the carefully engineered LFCC baseline. The lesson is complementarity, not replacement

---

## 8. Relevance to My Deepfake Detection Project (ASVspoof 2019 LA)

### Direct Architectural Influence

| Component | What to Take from RawNet2 |
|-----------|--------------------------|
| Input layer | Raw waveform (64,000 samples / ~4 sec) is a viable input modality alongside LFCC |
| Sinc filter bank | Consider fixed Mel or inverse-Mel filters to avoid overfitting with limited attack diversity in training |
| Residual blocks | Use residual connections for deeper networks — prevents gradient vanishing |
| FMS (attention) | Filter-wise attention improves discriminative representation — use channel attention in your architecture |
| GRU aggregation | Temporal modelling of frame-level features is important; GRU or self-attention over time |
| Fusion | Plan a fusion strategy combining your primary model with a complementary one |

### Performance Context for Your Project

| Milestone | Target |
|-----------|--------|
| Beat CQCC-GMM baseline | EER < 9.57%, min-tDCF < 0.2366 |
| Beat LFCC-GMM baseline | EER < 8.09%, min-tDCF < 0.2116 |
| Match high-res LFCC (L) | EER < 3.50%, min-tDCF < 0.0904 |
| Match L+S1 fusion | EER < 1.12%, min-tDCF < 0.0330 |
| Match T05 (SOTA 2019) | EER < 0.22%, min-tDCF < 0.0069 |

### A17 as a Stress Test

- Always report A17 min-tDCF separately — it is the acid test for generalisation
- If your model achieves good pooled EER but fails on A17, acknowledge this explicitly
- The paper shows A17 t-DCF can be reduced from 0.35 (LFCC) to 0.08 (full fusion) — cite this as the upper bound of what's achievable

### Why This Paper Justifies a Raw Waveform Branch in Your Architecture

If your model uses LFCC or CQCC features, adding a raw waveform branch (even a simple one) that fuses at score level can substantially improve A17 performance. This is low-cost, reproducible (open-source code available), and directly targeted at the hardest attack in your benchmark.

---

## 9. What to Use in My Mid-Report

### Use Directly

- **Table 2** — cite as evidence that raw waveform models excel on A17 while underperforming overall (motivates fusion)
- **Table 3** — cite L+S1 (0.0330 min-tDCF, 1.12% EER) as a strong published baseline; frame your model's performance relative to this
- **"Complementary representations" argument** — use in your methods section to justify any multi-feature or fusion approach
- **RawNet2 architecture (Table 1)** — reference if you adopt sinc filters, residual blocks, or FMS in your model
- **A17 phase hypothesis** — cite as the rationale for why temporal + raw waveform models are needed beyond LFCC

### Narrative Framing

Use this paper to establish:
1. *Why spectral features alone are insufficient*: A17 has phase-domain artefacts invisible to LFCC/CQCC — motivates going beyond handcrafted features
2. *Complementarity principle*: Different model families capture different artefact types — this justifies ensemble/fusion approaches
3. *End-to-end learning benefit*: Raw waveform models learn representations not constrained by frequency-domain feature engineering

### What NOT to Over-Claim

- Don't frame RawNet2 as superior overall — it's inferior on pooled metrics; it's only superior on A17
- Make clear that competitive results require fusion, not a standalone raw waveform model

### Citation

```
H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher,
"End-to-end anti-spoofing with RawNet2," in Proc. ICASSP, 2021.
arXiv: 2011.01108
```

---

*Summary generated: 2026-05-07*

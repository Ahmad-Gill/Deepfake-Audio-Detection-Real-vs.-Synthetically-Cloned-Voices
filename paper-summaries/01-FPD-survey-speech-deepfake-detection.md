# Paper Summary: A Survey on Speech Deepfake Detection

**Citation:** Li, M., Ahmadiadli, Y., & Zhang, X.-P. (2024). *A Survey on Speech Deepfake Detection*. arXiv:2404.13914v2.  
**Authors:** Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang — Toronto Metropolitan University & Tsinghua University  
**Scope:** Reviews 200+ papers published up to March 2024  
**Type:** Comprehensive survey / literature review

---

## 1. What Problem Does This Paper Solve?

Speech deepfakes — synthetically generated or voice-converted audio — are increasingly realistic and pose serious threats: political misinformation, phone fraud, and bypassing voice-based biometric authentication (ASV systems). The field has grown so fast that no single survey covered the **full detection pipeline** end-to-end.

This paper fills that gap by systematically reviewing every component of building a speech deepfake detector:
- Available datasets and evaluation metrics
- Feature extraction (front-end)
- Classifier architectures (back-end)
- Training optimization (data augmentation, loss functions, activation functions)
- Advanced robustness techniques (adversarial defense, cross-dataset generalization)
- Emerging sub-tasks: **partially fake speech detection** and **spoofing-aware speaker verification (SASV)**

It is also the **first survey** to:
- Include partially Deepfake speech (not just fully fake utterances)
- Systematically evaluate training optimization techniques
- Provide open-source availability information for SOTA models and datasets

---

## 2. Why Does This Problem Matter?

| Threat | Example |
|--------|---------|
| Political manipulation | Deepfake speeches mimicking Canadian politicians (2019) influenced public opinion |
| Phone scams | Synthesized voices impersonating family members or executives to authorize transactions |
| Voice authentication bypass | Fake voices tricking ASV (speaker verification) biometric systems |
| Partial manipulation | Altering only names, dates, or locations within otherwise genuine speech to spread targeted misinformation |

Detection is distinct from image/video deepfake detection because:
- Audio operates in a different perceptual and signal domain
- Replay and synthesis attacks require different countermeasures
- Voice conversion (VC) is fundamentally harder to detect than TTS
- Neural codec-based generation (LLMs + audio codecs) produces near-perfect fakes that fool most existing detectors

---

## 3. What Has Been Tried Before?

### Prior Surveys
Previous surveys (e.g., Khanjani et al. 2021; Nawaz et al. 2023) either:
- Focus on image/video deepfakes rather than audio
- Review model architectures only, skipping training optimization
- Evaluate only fully fake utterances, ignoring partial fakes
- Lack systematic open-source information

### Prior Detection Approaches (chronological trend)

**Classical ML era:**
- GMM (Gaussian Mixture Models), SVM, Random Forests
- Hand-crafted features: MFCC, LFCC, CQCC
- Served as baselines in early ASVspoof challenges (2015, 2019)
- Good interpretability; poor generalization to unseen attacks

**CNN/ResNet era:**
- LCNN (Light CNN with Max-Feature-Map activation)
- ResNet with squeeze-and-excitation (SE-Net)
- Res2Net, DenseNet, Res2Net variants
- Features: Mel-spectrogram, CQT-spectrogram treated as 2D images

**Graph Neural Network era:**
- RawGAT: Graphs over spectral/temporal nodes
- AASIST: Heterogeneous graph combining spectral + temporal subgraphs with attention
- Strong results but high complexity

**Self-supervised learning (SSL) era (current):**
- wav2vec 2.0, WavLM, HuBERT used as feature extractors
- Pre-trained on massive speech corpora; extract high-level representations
- Significantly outperform hand-crafted features on cross-dataset tests
- Multilingual SSL models (XLS-R, Whisper, MMS) generalize better across languages

**End-to-End (E2E) approaches:**
- RawNet2: SincNet (learnable filters) + GRU layers; official ASVspoof baseline
- RW-ResNet: 1D CNN directly on raw waveform
- Note: many claimed E2E models still use fixed-frequency SincNet filters

---

## 4. What Does This Paper Propose?

This is a **survey paper** — it does not propose a single new model. Instead it proposes:

### A Comprehensive Review Framework

```
Speech Deepfake Detection Pipeline
├── Datasets & Evaluation Metrics
│   ├── Fully Deepfake datasets (ASVspoof2019-LA, ASVspoof2021-LA/DF, ITW, CodecFake, MLAAD, ...)
│   ├── Partially Deepfake datasets (PartialSpoof, HAD, ADD2022-PF, Psynd, ...)
│   └── Metrics: EER, t-DCF, a-DCF, Range-based EER
├── Feature Engineering (Front-End)
│   ├── Hand-crafted: MFCC, LFCC, CQCC, LVQ, phase, bispectrum, spectrogram
│   ├── DL features: SincNet, wav2vec 2.0, WavLM, ResNet embeddings
│   └── Analysis-oriented: prosody, silence, sub-band (0–4kHz)
├── Classifier Architecture (Back-End)
│   ├── Traditional ML: GMM, SVM, RF
│   ├── CNN/LCNN, ResNet/SE-Net/Res2Net/DenseNet
│   ├── GNN: RawGAT, AASIST
│   ├── Transformer: CCT, OCT, Rawformer
│   ├── TDNN: ECAPA-TDNN
│   └── DART: differentiable architecture search
├── Training Optimization
│   ├── Data augmentation: masking, mix-up (SpecMix), noise+RIR (RawBoost), codec augmentation
│   ├── Loss functions: CE, AM-Softmax, OC-Softmax, Focal, Center, MSE
│   └── Activation functions: PReLU, AReLU
├── Robustness & Advanced Training
│   ├── Siamese networks, LoRA, Knowledge Distillation
│   ├── Adversarial defense: adversarial training (PGD, FGSM)
│   ├── Cross-dataset generalization: domain adaptation, channel-effect correction
│   └── Explainability: SHAP, Grad-CAM, LRP
├── SASV: ASV + Deepfake CM Integration
│   ├── Cascaded, score-fusion, embedding-fusion, integrated E2E
└── Partial Deepfake Detection
    ├── Frame-level, multi-task, boundary detection
```

### Key Contributions of the Survey
1. First to evaluate **optimization techniques** (augmentation, loss, activation) across SOTA models
2. First to systematically compare **partially fake detection** approaches
3. Provides **open-source links** to reproducible models and datasets
4. Establishes **strong baselines** for future benchmark experiments
5. Identifies gaps: reproducibility, multilingual detection, codec-based deepfakes

---

## 5. Experiments and Results

### SOTA Single-System Performance (ASVspoof series, EER %)

| Model | Feature | Classifier | Loss | EER (2019-LA) | EER (2021-LA) | EER (2021-DF) |
|-------|---------|-----------|------|--------------|--------------|--------------|
| INTERSPEECH'21 | Mel-Spec | SE-ResNet-18 | AM-Softmax | 1.14 | — | — |
| SPL'21 | CNN→RawNet2 | CNN→MLP | CE | 1.64 | — | — |
| INTERSPEECH'22 | wav2vec2.0-XLSR | MLP | CE | **0.31** | — | — |
| ODYSSEY'22 | wav2vec2.0-XLSR | Bi-LSTM→MLP | CE | 1.28 | 6.53 | 4.75 |
| ODYSSEY'22 | RawBoost | AASIST | CE | **0.82** | **2.85** | — |
| ICASSP'23 | RawNet2 | Rawformer | CE | 0.59 | 4.98 | 4.53 |
| ALGORITHM'23 | wav2vec 2.0 | Transformer | CE | — | 1.18 | 4.72 |
| ICASSP'23 | FIR filter, codec, shift | SDC+Bi-LSTM | Auto-encoder→SE-ResNeXT | **0.22** | 3.50 | **3.41** |

**Key finding:** wav2vec 2.0 + MLP achieves 0.31% EER on 2019-LA — the best reported single-system result on that benchmark.

### Cross-Dataset Performance (trained on ASVspoof2019-LA, tested on ITW)

| Model | EER (2019-LA) | EER (ITW) |
|-------|--------------|----------|
| INTERSPEECH'23 | 0.63 | 24.50 |
| SPL'24 | 1.79 | 29.66 |
| ICASSP'24 (KD) | **0.39** | **7.68** |
| ICASSP'24 (KD+RawBoost) | 0.13 | 12.50 |

**Key finding:** Performance collapses dramatically on out-of-domain data. Knowledge distillation (KD) is the most effective technique for improving cross-dataset generalization.

### Training Optimization Ablation (EER reduction with vs. without technique)

| Technique | Dataset | EER without | EER with | Improvement |
|-----------|---------|------------|---------|-------------|
| Masking | ASVspoof2019-LA | 6.51 | 5.14 | −1.37 |
| Codec augmentation | ASVspoof2021-LA | 30.17 | 7.96 | −22.21 |
| Noise addition (RawBoost) | ASVspoof2021-LA | 9.50 | 5.31 | −4.19 |
| AM-Softmax | ASVspoof2019-LA | 1.41 | 1.29 | −0.12 |
| OC-Softmax | ASVspoof2021-LA | — | — | significant |
| Center loss | ASVspoof2019-LA | 0.68 | 0.52 | −0.16 |

**Key finding:** **Codec augmentation** and **noise addition** provide the largest gains, especially for 2021 datasets with real-world conditions.

### SASV (Spoofing-Aware Speaker Verification) SOTA

| Category | Best Model | SV-EER | SPF-EER | SASV-EER |
|----------|-----------|--------|---------|---------|
| Score Fusion | [4] ResNet-48 + ResNet-48 | **0.19** | **0.25** | **0.22** |
| Embedding Fusion | [24] ECAPA-TDNN + Res2Net | 0.28 | 0.28 | 0.28 |
| Integrated E2E | MFA-Conformer [115] | 1.83 | 0.58 | **1.19** |

**Finding:** Simple score-fusion still outperforms complex integrated E2E systems in most metrics.

### Partially Deepfake Detection SOTA

| Method | Category | Feature | Classifier | PartialSpoof (EER) |
|--------|---------|---------|-----------|-------------------|
| [208] INTERSPEECH'21 | Multi-task | LFCC | SE-LCNN, LSTM | 5.90 |
| [206] TASLP'22 | Multi-task | wav2vec2.0-large | Gated-MLP | **0.49** |
| [104] ICASSP'22 | Boundary | wav2vec2.0-XLSR | MLP, Transformer | — (ADD: **4.80**) |
| [13] DADA'23 | Boundary | WavLM, ResNet | Transformer, Bi-LST | — (ADD2023: **67.13** F1) |

---

## 6. Limitations and Weaknesses

### Of the Field (identified by the survey):

**Reproducibility crisis:**
- Only ~10% of papers release source code
- Missing details on loss functions and hyperparameter configurations prevent replication
- Makes it impossible to fairly compare methods

**Cross-dataset generalization:**
- Models trained on clean ASVspoof2019-LA collapse to near-random performance on real-world data (ITW dataset)
- Domain shift from channel effects, background noise, and codec diversity is the primary cause
- Transfer learning and domain adaptation remain underdeveloped

**Neural codec-based deepfakes:**
- Detectors trained on vocoder-based speech (ASVspoof) fail on codec-generated speech (SpeechX, EnCodec)
- New datasets (CodecFake, Codecfake) exist but training data diversity is still insufficient

**Multilingual limitations:**
- Most research is English-only
- Only 1 dataset (MLAAD) covers more than 2 languages across 23 languages
- Accent variation is only studied in English

**Adversarial attack vulnerability:**
- Nearly all detectors are vulnerable to FGSM and PGD adversarial attacks
- Adversarial training requires substantial compute
- Defenses based on one attack type don't generalize

**Partial deepfake detection:**
- Frame-level methods struggle with short manipulated segments (shorter than a phoneme)
- Current test sets lack segments from genuinely different speakers, making speaker verification integration hard to evaluate

**Real-time and privacy:**
- Almost no work on streaming/real-time detection
- On-device deployment raises biometric privacy concerns (no privacy-preserving solutions exist yet)

### Of the Survey Itself:
- Does not include physical replay attacks (intentionally excluded but limits scope)
- Does not quantitatively meta-analyze all 200+ papers (qualitative synthesis)
- Coverage limited to papers before March 2024 (misses very recent codec model advances)

---

## 7. Key Takeaways

### What Works Best (Current SOTA Recipe)

```
Best general-purpose pipeline:
  Front-end:  wav2vec 2.0 (XLSR large) — pre-trained SSL features
  Back-end:   AASIST (heterogeneous graph attention network)
  Loss:       OC-Softmax (separate margins for bona fide / fake)
  Data aug:   RawBoost (noise + RIR) + Codec augmentation
  Result:     ~0.3–0.8% EER on ASVspoof2019-LA
```

### Feature Engineering Insights
- SSL features (wav2vec 2.0, WavLM) > learnable filters (SincNet) > hand-crafted (LFCC, CQCC) for cross-dataset generalization
- Sub-band (0–4kHz) is more robust than full-band under noise and codec conditions
- Phase and bispectral features complement magnitude features but are not competitive alone
- Silence features are useful: TTS algorithms struggle to model natural pause duration patterns

### Architecture Insights
- AASIST is the strongest single classifier; it models spectro-temporal artifact relationships via graphs
- RawNet2 (SincNet + GRU) is the reproducible official baseline
- Transformer + ResNet hybrids (Rawformer) provide a good tradeoff
- Ensemble/fusion of multiple models consistently outperforms single models

### Training Insights
- Codec augmentation is the single most impactful data augmentation for real-world robustness
- OC-Softmax outperforms standard CE because it models bona fide and fake with separate compactness margins
- Knowledge distillation is the best technique for improving cross-dataset generalization without retraining

### Generalization Insights
- Cross-dataset performance is the field's biggest unsolved problem
- Domain mismatch (channel effects, sampling rates) — not model capacity — is the primary cause of failure
- KD + Gradient Reversal Layer (GRL) for channel-effect discrimination is a promising direction

---

## 8. Relevance to My Deepfake Detection Project (ASVspoof 2019 LA)

### Directly Applicable Findings

**Dataset choice (ASVspoof2019-LA):**
- This is the most widely used benchmark in the literature — all SOTA comparisons use it
- It is clean (no noise, no codec distortion) — models trained on it do not generalize to noisy/real-world conditions out of the box
- The LA (Logical Access) subset contains 19 TTS/VC attack types; 6 unseen in training

**Features to use:**
- Start with **LFCC** or **CQCC** as a reproducible baseline (available in the ASVspoof toolkit)
- For best results: **wav2vec 2.0** features (XLSR-53 or large variant)
- Sub-band restriction to 0–4kHz can improve robustness

**Classifiers to use:**
- **LCNN** or **ResNet** as lightweight baseline
- **AASIST** as the strongest single classifier (has open-source code)
- **RawNet2** as the official reproducible baseline

**Training setup:**
- Loss function: **OC-Softmax** over standard CE if you want better generalization
- Data augmentation: **SpecAugment** (masking) is fast and easy to implement as a first step
- For stronger results: **RawBoost** (noise + RIR augmentation)

**Evaluation metrics:**
- Primary: **EER** (standard for ASVspoof2019-LA)
- Secondary: **t-DCF** (if integrating with a speaker verification component)

### Things to Be Aware Of
- ASVspoof2019-LA is a "clean lab" dataset — do not overstate real-world applicability
- If testing on real-world audio, expect EER to jump to 20–30% even for SOTA models
- Neural codec-generated deepfakes (not in ASVspoof2019) will likely fool your trained model
- Reproducibility matters: log all hyperparameters and seed values

### Architecture Recommendation for Your Project

| Goal | Recommended Approach |
|------|---------------------|
| Baseline / fast to implement | LFCC + LCNN or CQCC + GMM |
| Best EER on ASVspoof2019-LA | wav2vec 2.0 (XLSR) + AASIST + OC-Softmax |
| Interpretable model | LFCC/CQCC + ResNet + SHAP for explanation |
| Robust to real-world noise | RawBoost augmentation + codec augmentation |
| Lightweight / edge deployment | Knowledge distillation from AASIST to LCNN |

---

## 9. What to Use in My Mid-Report

### Section: Related Work / Literature Review

Use this paper as your **primary survey reference**. Cite it as:
> Li et al. (2024) provide a comprehensive survey of speech deepfake detection covering 200+ papers through March 2024, reviewing feature extraction, classifier architectures, training optimization, and emerging challenges such as cross-dataset generalization and partial deepfake detection.

**Key points to mention:**
- The field evolved from GMM + MFCC baselines → SSL + graph networks
- ASVspoof challenges (2015, 2019, 2021) drove benchmark progress
- The best single-system result on ASVspoof2019-LA is 0.22% EER (ICASSP 2023)
- Cross-dataset generalization remains the central unsolved challenge

### Section: Dataset Description

Use Table 1 from the paper to justify your dataset choice:
- ASVspoof2019-LA: 19 attack types, 10,256 real utterances, 90,192 fake utterances, clean conditions, 16kHz
- Standard benchmark used by virtually all papers — enables direct comparison

### Section: Methodology

Justify your feature and model choices by referencing Table 2 (SOTA model performance) and Table 3 (feature categorization):
- If using LFCC: "LFCC has been shown to outperform MFCC for deepfake detection due to its linear frequency scale capturing higher-frequency artifacts [Sahidullah et al., 2015]"
- If using wav2vec 2.0: "Self-supervised pre-trained features from wav2vec 2.0 significantly outperform hand-crafted features, achieving 0.31% EER on ASVspoof2019-LA [INTERSPEECH 2022]"
- If using AASIST: "AASIST models spectro-temporal artifact relationships using a heterogeneous graph attention network, representing the current SOTA classifier for speech deepfake detection"

### Section: Evaluation

Cite EER and t-DCF from this paper to contextualize your results against SOTA benchmarks (Table 2).

### Section: Limitations / Future Work

Draw directly from Section 7 of the paper:
- Reproducibility gap (~10% of papers release code)
- Cross-dataset generalization remains unsolved
- Neural codec-based deepfakes challenge existing detectors
- Multilingual detection is underdeveloped

### Figures/Tables to Reproduce or Reference
- **Table 1** (Dataset statistics) — cite in dataset section
- **Table 2** (SOTA model performance) — cite in related work or results comparison
- **Table 3** (Feature categorization) — cite when justifying feature choice
- **Table 5** (Training optimization ablation) — cite when justifying augmentation strategy

---

## Quick Reference Card

| Aspect | Best Choice | EER on 2019-LA |
|--------|------------|----------------|
| Feature | wav2vec 2.0 XLSR | 0.31% |
| Classifier | AASIST | 0.82% |
| Loss | OC-Softmax | Improves generalization |
| Augmentation | Codec aug + RawBoost | −22% EER on 2021-LA |
| Combined SOTA | LFCC + SDC + codec augmentation + Bi-LSTM | **0.22%** |

**GitHub links from paper:**
- RawGAT-ST: `https://github.com/eurecom-asp/RawGAT-ST-antispoofing`
- AASIST: `https://github.com/clovaai/aasist`
- SSL anti-spoofing: `https://github.com/TakHemlata/SSL_Anti-spoofing`
- Rawformer: `https://github.com/rat0070/Rawformer-implementation-anti-spoofing`
- SASV: `https://github.com/sasv-challenge/ASVspoof5-SASVBaseline`

---

*Summary generated from: Li, M., Ahmadiadli, Y., & Zhang, X.-P. (2024). A Survey on Speech Deepfake Detection. arXiv:2404.13914v2. 38 pages.*

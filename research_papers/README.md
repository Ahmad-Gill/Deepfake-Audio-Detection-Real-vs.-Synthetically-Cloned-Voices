# Deepfake Audio Detection: 10 Essential Papers (Mid-Report Selection)

## 10 PAPERS TO KEEP

### Foundation & Problem Definition
1. **A Survey on Speech Deepfake Detection** (2024) [2404.13914v2]
   - **Role:** Lit review backbone and taxonomy
   - **Use in:** Structure your literature review sections around this paper's categorization

2. **ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech** (2019) [1911.01601v4]
   - **Role:** Dataset specification
   - **Use in:** Dataset section; extract attack types A01–A19, speaker splits, protocols

3. **ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection** (2019) [1904.05441v2]
   - **Role:** Evaluation metrics and challenge protocol
   - **Use in:** Evaluation section; define EER, t-DCF; cite CQCC+GMM baseline numbers

### Core Architectures (Baselines & Proposed Method)
4. **End-to-End Anti-Spoofing with RawNet2** (2020) [2011.01108v3]
   - **Role:** Primary baseline (end-to-end paradigm)
   - **Use in:** Methodology section; implement/reproduce as main baseline

5. **AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention** (2021) [2110.01200v1]
   - **Role:** SOTA architecture (0.83% EER on ASVspoof 2019 LA)
   - **Use in:** Methodology section; either use as proposed method or strongest baseline

6. **A Comparative Study on Recent Neural Spoofing Countermeasures for ASVspoof 2019** (2021) [2103.11326v2]
   - **Role:** Baseline benchmark numbers across multiple models
   - **Use in:** Results/baseline comparison table; justifies which baselines you selected

### Feature Strategy (What Goes Into Your Model)
7. **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** (2020) [2006.11477v3]
   - **Role:** Foundation for SSL-based feature extraction
   - **Use in:** Background section if using SSL features; explain contextualized representations

8. **Investigating Self-Supervised Front Ends for Speech Spoofing Countermeasures** (2021) [2111.07725v3]
   - **Role:** Which SSL layer/model to use for anti-spoofing
   - **Use in:** Methodology section; justify your SSL feature extraction choices

### Augmentation & Generalization (Project Core Challenge)
9. **RawBoost: A Raw Data Boosting and Augmentation Method for Anti-Spoofing** (2021) [2111.04433v2]
   - **Role:** Data augmentation strategy implementation
   - **Use in:** Methodology/experiments; implement and ablate RawBoost in your pipeline

10. **Does Audio Deepfake Detection Generalize?** (2022) [2203.16263v5]
    - **Role:** Problem motivation (cross-dataset generalization failure)
    - **Use in:** Introduction and motivation; justifies entire project direction

---

## 20 PAPERS TO SKIP

### Papers Covered by Your 10 Keeps (Redundant)
1. **ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection** (2021) [2109.00537v1]
   - *Why skip:* You're using ASVspoof 2019, not 2021. Challenge overview (#3) covers your protocol.

2. **Automatic Speaker Verification Spoofing and Deepfake Detection** (2022) [2202.12233v2]
   - *Why skip:* Survey paper superseded by 2024 survey (#1).

3. **Where Are We in Audio Deepfake Detection? A Systematic Review** (2024) [3736765.pdf]
   - *Why skip:* Third survey; redundant with 2024 survey (#1).

4. **Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead** (2025) [sensors-25-01989.pdf]
   - *Why skip:* Fourth survey; one survey is sufficient for mid-report.

5. **Audio Deepfake Detection Using Deep Learning** (2025) [Engineering Reports - 2025 - Shaaban - Audio Deepfake Detection Using Deep Learning.pdf]
   - *Why skip:* Engineering overview adds nothing beyond 2024 survey.

6. **STC Antispoofing Systems for the ASVspoof 2019 Challenge** (2019) [1904.05576v1]
   - *Why skip:* Challenge submission; its numbers appear in comparative study (#6).

7. **Deep Residual Neural Networks for Audio Spoofing Detection** (N/A) [Deep-Residual-Neural-Networks-for-Audio-Spoofing-Detection.pdf]
   - *Why skip:* ResNet baseline already benchmarked in comparative study (#6).

8. **Res2Net and LCNN Variant** (2020) [2010.13995v2]
   - *Why skip:* CNN variant already benchmarked in comparative study (#6).

### Too Advanced or Niche for Mid-Report
9. **Audio Deepfake Detection with Self-Supervised WavLM and Multi-Fusion** (2023) [2312.08089v2]
   - *Why skip:* Advanced SSL fusion; paper (#8) gives sufficient SSL guidance.

10. **Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection** (2025) [2502.03559v2]
    - *Why skip:* Deep SSL analysis; save for final report if implementing specific SSL layer selection.

11. **Toward Improving Synthetic Audio Spoofing Detection Robustness via Meta-Learning and Adversarial Examples** (N/A) [Toward_Improving_Synthetic_Audio_Spoofing_Detection_Robustness_via_Meta-Learning_and_Disentangled_Training_With_Adversarial_Examples.pdf]
    - *Why skip:* Meta-learning is advanced; unlikely to be in your mid-report methodology.

12. **Beyond Identity: A Generalizable Approach for Deepfake Audio Detection** (2025) [2505.06766v1]
    - *Why skip:* Good generalization paper but paper (#10) already defines the problem. Save for final report.

13. **Generalizable Detection of Audio Deepfakes** (2025) [2507.01750v1]
    - *Why skip:* Redundant with paper (#10) and Beyond Identity theme. Save for final report.

14. **Toward Noise-Aware Audio Deepfake Detection** (2025) [2512.13744v1]
    - *Why skip:* Noise robustness is secondary concern. Address in final report ablations if time permits.

### Too Basic or Out of Scope
15. **Deepfake Audio Detection via MFCC Features** (N/A) [Deepfake_Audio_Detection_via_MFCC_Featur.pdf]
    - *Why skip:* MFCC is baseline feature; one sentence via survey (#1) is sufficient.

16. **IEEE T-BIOM: Anti-Spoofing with Spectral Features** (2021) [2102.05889v1]
    - *Why skip:* Spectral feature analysis covered in survey (#1) taxonomy.

17. **Robust DeepFake Audio Detection via an Improved NeXt-TDNN** (2025) [applsci-15-09685.pdf]
    - *Why skip:* Niche TDNN variant; not a standard baseline.

18. **IEEE/ACM Trans. Spectro-Temporal Detection Model** (2022) [2210.02437v3]
    - *Why skip:* AASIST (#5) already represents spectro-temporal modeling and is more impactful.

19. **Anti-Spoofing Detection Model (IEEE Journal)** (2023) [2308.14970v1]
    - *Why skip:* Unclear added value beyond comparative study (#6) and AASIST (#5).

20. **Real-Time Detection of AI-Generated Speech** (2023) [2308.12734v1]
    - *Why skip:* Deployment focus; out of scope for academic project.

---

## Reading Strategy for 10 Papers

**Week 1 — Foundation (4 papers)**
- Read papers #1, #2, #3, #10 in order
- Output: Lit review outline, dataset spec, problem motivation

**Week 2 — Architectures (3 papers)**
- Read papers #4, #5, #6
- Output: Baseline descriptions, benchmark comparison table

**Week 3 — Features & Augmentation (3 papers)**
- Read papers #7, #8, #9
- Output: Feature extraction methodology, augmentation strategy

**Total Reading Load:** ~15–18 hours across 3 weeks

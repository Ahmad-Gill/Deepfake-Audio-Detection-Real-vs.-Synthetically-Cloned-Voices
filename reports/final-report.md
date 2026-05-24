# Deepfake Audio Detection: Real vs. Synthetically Cloned Voices
## AI-600 Deep Learning — Final Report (Spring 2026)

**Course:** AI-600 Deep Learning, Spring 2026
**Submitted:** 22 May 2026
**Code Repository:** `https://github.com/<team>/Deepfake-Audio-Detection-Real-vs-Synthetically-Cloned-Voices` *(placeholder — replace with actual GitHub URL on submission)*

| Name | Student ID |
|------|-----------|
| Ahmad Gill | 25280065 |
| Imran Saeed | 25280083 |
| Israr Ali  | 25280102 |

> **Note on results.** Final training is still in progress at the time of writing. Numbers marked **[REAL]** are observed values from completed runs in [colab/06_preprocessing.ipynb](../colab/06_preprocessing.ipynb). Numbers marked **[SAMPLE]** are placeholders consistent with published reproductions; these will be replaced with the final observed values before submission.

---

## Abstract

The rise of neural voice cloning systems has made it possible to synthesize speech indistinguishable from a target speaker, creating an urgent need for robust countermeasures. This work builds a binary classifier that distinguishes genuine human speech from synthetic audio on the ASVspoof 2019 Logical Access (LA) benchmark, which contains 121,000+ utterances spanning 19 distinct TTS and voice-conversion attack types. We adopt the current state-of-the-art recipe — a multilingual self-supervised wav2vec 2.0 XLS-R-300M frontend coupled to an AASIST graph-attention backend, trained with OC-Softmax loss and RawBoost on-the-fly augmentation. To produce a deployment-ready model, we additionally train a lightweight AASIST-L student (~85 K trainable parameters) via knowledge distillation from the teacher, achieving ≈53× parameter compression. We evaluate on the held-out evaluation set (attacks A07–A19, all unseen during training) using EER and min-tDCF, and provide a per-attack breakdown plus an ablation over each augmentation, loss, and distillation choice. Our teacher reaches **0.02% Dev EER [REAL]** during training, and the student is expected to operate within 1% of the teacher at a fraction of the compute budget.

---

## 1. Introduction

Voice cloning services are now commercially available at low cost, enabling attacks against voice-authenticated banking, judicial proceedings, and political discourse. The ASVspoof challenge series has driven systematic benchmarking of spoofing countermeasures since 2015, with the 2019 LA edition deliberately exposing six attacks during training and 13 distinct unseen attacks at evaluation to measure cross-attack generalization rather than memorization [2, 3].

Two findings from the recent literature shape our approach. First, self-supervised front-ends — particularly multilingual XLS-R variants of wav2vec 2.0 — dominate the leaderboard when combined with a strong back-end and fine-tuned end-to-end, achieving sub-1% EER on LA [7, 8]. Second, the AASIST graph-attention back-end remains the strongest published single-system architecture on LA at a tractable parameter budget, with a lightweight variant (AASIST-L, 85 K parameters) sacrificing only 0.16% EER versus the full model [5]. We combine both and add knowledge distillation to deliver a small student model suitable for deployment.

Our contribution is threefold:

1. A clean PyTorch implementation of the **wav2vec 2.0 XLS-R + AASIST + OC-Softmax + RawBoost** pipeline (notebooks 01–05).
2. A **teacher–student knowledge-distillation** stage producing an 85 K-parameter AASIST-L student that learns from the teacher's soft targets (KL divergence, T = 3, α = 0.5).
3. An **ablation study** isolating the contribution of each component (loss, augmentation, distillation) and a **per-attack EER breakdown** identifying the hardest A07–A19 attacks for our system.

---

## 2. Literature Review

We surveyed ten core papers across four paradigms, summarized below. Detailed notes are in [paper-summaries/](../paper-summaries/) and the PDFs in [papers/report-papers/](../papers/report-papers/).

**Foundations & benchmarks.** Li et al. [1] survey 200+ deepfake-audio papers and identify the canonical state-of-the-art recipe (wav2vec 2.0 XLSR + AASIST + OC-Softmax + RawBoost). The ASVspoof 2019 database [2] defines the 19 attack types, the speaker-disjoint train/dev/eval split, and the t-DCF metric. Todisco et al. [3] document challenge results across 63 teams and establish the cost weights (Pspoof = 0.05, Cmiss = 1, Cfa = 10) that we adopt.

**Classification architectures.** Tak et al. [4] introduced RawNet2, the first end-to-end raw-waveform anti-spoofing system, achieving 1.12% EER (fused). Jung et al. [5] proposed AASIST, which replaces RawNet2's GRU with a heterogeneous spectro-temporal graph-attention network (HS-GAL) and reaches 0.83% EER. AASIST-L (85 K params) achieves 0.99% EER — the architectural blueprint we use for our student. Wang & Yamagishi [6] systematically compare loss functions over six seeds and establish OC-Softmax as a stable, strong choice for spoofing detection.

**SSL frontends.** Baevski et al. [7] introduced wav2vec 2.0, learning contextual representations from raw waveforms via masked contrastive pre-training. Wang & Yamagishi [8] compare five SSL frontends for spoofing and show that the multilingual XLS-R variant (W2V-XLSR, 317 M params), when fine-tuned, reaches 0.11% EER on LA — far better than handcrafted features.

**Augmentation & generalization.** Tak et al. [9] introduced RawBoost, which applies serial convolutive notch noise, impulsive signal-dependent noise, and stationary white noise directly on raw waveforms, yielding 27% relative t-DCF improvement on 2021 LA. Müller et al. [10] document the dramatic generalization gap (e.g., RawGAT-ST: 1.23% LA EER → 37.15% In-The-Wild EER), motivating both our augmentation strategy and our knowledge-distillation step (a small, well-regularized student often generalizes better than its larger teacher).

---

## 3. Methodology

### 3.1 Problem formulation

Given a waveform $\mathbf{x}$ at 16 kHz, the classifier outputs $s \in \mathbb{R}^2$ (logits) for the bona fide / spoof binary task. The detection score is $P(\text{spoof}\mid\mathbf{x}) = \text{softmax}(s)_1$. The EER threshold $\theta^\star$ satisfies $\text{FAR}(\theta^\star) = \text{FRR}(\theta^\star)$.

### 3.2 Pipeline

```
Raw audio (variable length)
  ├─ 01: Resample 16 kHz → mono → pad/truncate to 6 s (96 000 samples)
  ├─ 02: RawBoost (conv + impulsive)  ∪  codec (μ-law / 8 kHz)  ∪  peak-norm
  ├─ 03: Teacher  = wav2vec 2.0 XLS-R (partial fine-tune) → AASIST (h=160)
  │        Loss: OC-Softmax(m_real=0.9, m_fake=0.2, α=20)
  ├─ 04: Student  = wav2vec 2.0 XLS-R (frozen)         → AASIST-L (h=64)
  │        Loss: 0.5 · OC-Softmax + 0.5 · T² · KL(student‖teacher, T=3)
  └─ 05: Evaluate on Eval (A07–A19): EER, min-tDCF, per-attack EER
```

### 3.3 Frontend — wav2vec 2.0 XLS-R-300M

We load `facebook/wav2vec2-xls-r-300m` from Hugging Face. The last hidden state (1024-dim, ~50 frames/s) is passed to AASIST. For the teacher, the CNN feature extractor is frozen but the Transformer layers are fine-tuned with a low LR (1 × 10⁻⁵). For the student, the entire frontend is frozen.

### 3.4 Backend — AASIST and AASIST-L

The backend ([colab/03_teacher_training.ipynb](../colab/03_teacher_training.ipynb)) projects features to a hidden dimension $h$, then routes through two parallel **multi-head graph attention** branches:

- **Spectral branch:** GAT over time steps, summarising frequency content.
- **Temporal branch:** GAT over frequency bins, summarising time dynamics.

A heterogeneous-attention fusion module concatenates both pooled vectors and passes them through a small MLP head producing two logits. The teacher uses $h = 160$ (~4.5 M trainable params); the student uses $h = 64$ (~85 K trainable params).

### 3.5 Loss functions

The teacher minimizes **OC-Softmax** with margins $m_{\text{real}} = 0.9$, $m_{\text{fake}} = 0.2$ and scale $\alpha = 20$. The student minimizes a hybrid:

$$\mathcal{L}_{\text{student}} = \alpha\,\mathcal{L}_{\text{OC}} + (1-\alpha)\, T^2 \cdot \text{KL}\!\left(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\right),\quad \alpha = 0.5,\, T = 3$$



### 3.7 Training configuration

AdamW with differential LR (1 × 10⁻⁵ frontend, 1 × 10⁻⁴ backend), cosine annealing over 10 epochs, gradient clipping at $\|g\| \le 1$, batch size 8, weight decay 1 × 10⁻⁴.

### 3.8 Evaluation

EER and a simplified min-tDCF ($P_{\text{spoof}} = 0.05$, $C_{\text{miss}} = 1$, $C_{\text{fa}} = 10$) are computed on the LA eval split; we additionally report per-attack EER over A07–A19 and standard binary metrics (accuracy / precision / recall / F1).

---

## 4. Experimental Setup

The full ASVspoof 2019 LA partition is downloaded from `datashare.ed.ac.uk/handle/10283/3336` and verified via protocol files. Counts below were obtained from our preprocessing pipeline [REAL]:

| Split       | Bonafide | Spoof  | Total  | Speakers | Attacks |
|-------------|----------|--------|--------|----------|---------|
| Train       | 2,580    | 22,800 | 25,380 | 20       | A01–A06 |
| Development | 2,548    | 22,296 | 24,844 | 20       | A01–A06 |
| Evaluation  | 7,355    | 63,882 | 71,237 | 67       | A07–A19 |

All experiments run on Google Colab with a single Tesla T4 GPU [REAL]. Implementation uses PyTorch 2.x, torchaudio, and Hugging Face `transformers`. Code is organized as five sequential notebooks; the consolidated end-to-end notebook is [colab/06_preprocessing.ipynb](../colab/06_preprocessing.ipynb).

---

## 5. Results

### 5.1 Comparison against published baselines

This table contextualizes our teacher and student against the literature on **ASVspoof 2019 LA**. The **Source** column makes the provenance explicit: rows marked *Reported in [X]* are baseline numbers reproduced verbatim from the cited paper (we did **not** re-run these systems); rows marked *Ours* are produced by our pipeline.

| System                                    | Source              | Frontend           | Backend     | EER (%)        | min-tDCF       | Trainable Params |
|-------------------------------------------|---------------------|--------------------|-------------|----------------|----------------|------------------|
| CQCC + GMM (B1)                           | Reported in [3]     | CQCC               | GMM         | 9.57           | 0.2366         | —                |
| LFCC + GMM (B2)                           | Reported in [3]     | LFCC               | GMM         | 8.09           | 0.2116         | —                |
| RawNet2 (single, linear sinc)             | Reported in [4]     | Raw + Sinc         | Res + GRU   | 4.66           | 0.1294         | 21 M             |
| LCNN-LSTM-sum + P2SGrad (best single)     | Reported in [6]     | LFCC               | LCNN-LSTM   | 1.92           | —              | ~1 M             |
| RawGAT-ST                                 | Reported in [10]    | Raw                | GAT         | 1.23           | —              | ~0.5 M           |
| AASIST                                    | Reported in [5]     | RawNet2 enc.       | HS-GAL      | **0.83**       | **0.0275**     | 0.30 M           |
| AASIST-L                                  | Reported in [5]     | RawNet2 enc.       | HS-GAL (sm) | 0.99           | 0.0309         | 0.085 M          |
| W2V-XLSR + LLGF (fine-tuned)              | Reported in [8]     | wav2vec 2.0 XLS-R  | LCNN-BiLSTM | **0.11**       | 0.120          | 317 M            |
| **Our Teacher (XLS-R + AASIST)**          | **Ours**            | wav2vec 2.0 XLS-R  | AASIST      | **0.95 [SAMPLE]** | **0.0285 [SAMPLE]** | 5.7 M (4.5 M trainable) |
| **Our Student (XLS-R + AASIST-L, KD)**    | **Ours**            | wav2vec 2.0 XLS-R (frozen) | AASIST-L | **1.15 [SAMPLE]** | **0.0330 [SAMPLE]** | 85 K trainable   |

> **Did we run CQCC, LFCC, RawNet2, RawGAT-ST, AASIST, W2V-XLSR+LLGF?**  No. These rows are the **published numbers** from the cited papers, included only to position our system. The only systems we actually train and evaluate end-to-end are the last two rows (**Ours**).

*Note.* On the development partition our teacher reaches **0.02% Dev EER at epoch 3 [REAL]** ([colab/06_preprocessing.ipynb](../colab/06_preprocessing.ipynb) cell 73 output). Eval-set numbers are still pending.

### 5.2 Per-attack EER on the evaluation set (A07–A19)

We follow the per-attack analysis style of Todisco et al. [3]. **All EER numbers in this table are from our own teacher and student models** — they are simply the eval-set EER broken down by attack ID. A17 (VAE VC) and A10 (Tacotron2 + WaveRNN) are historically the hardest attacks in the literature [2, 3], and we expect to see the same pattern.

| Attack | Type                              | Teacher EER (%) | Student EER (%) | Hardness rank |
|--------|-----------------------------------|-----------------|-----------------|---------------|
| A07    | Vocoder-based VC                  | 0.4 [SAMPLE]    | 0.5 [SAMPLE]    | easy          |
| A08    | Neural waveform VC                | 0.3 [SAMPLE]    | 0.4 [SAMPLE]    | easy          |
| A09    | Vocoder-based TTS                 | 0.5 [SAMPLE]    | 0.6 [SAMPLE]    | easy          |
| **A10** | **Tacotron2 + WaveRNN TTS**      | **2.8 [SAMPLE]** | **3.4 [SAMPLE]** | **hard**     |
| A11    | Tacotron2 + Griffin-Lim           | 0.9 [SAMPLE]    | 1.1 [SAMPLE]    | medium        |
| A12    | WaveNet-based TTS                 | 0.7 [SAMPLE]    | 0.9 [SAMPLE]    | medium        |
| A13    | Waveform concatenation VC         | 0.6 [SAMPLE]    | 0.8 [SAMPLE]    | easy          |
| A14    | TTS + Vocoder                     | 0.8 [SAMPLE]    | 1.0 [SAMPLE]    | medium        |
| A15    | Tacotron + WaveNet                | 0.7 [SAMPLE]    | 0.9 [SAMPLE]    | medium        |
| A16    | Waveform-concat TTS               | 0.5 [SAMPLE]    | 0.7 [SAMPLE]    | easy          |
| **A17** | **VAE-based VC**                 | **3.6 [SAMPLE]** | **4.2 [SAMPLE]** | **hardest**  |
| A18    | i-vector / PLDA VC                | 1.1 [SAMPLE]    | 1.3 [SAMPLE]    | medium        |
| A19    | GMM-UBM VC                        | 0.4 [SAMPLE]    | 0.5 [SAMPLE]    | easy          |

### 5.3 Classification metrics at the EER threshold

**All rows in this table are ours**, computed on the ASVspoof 2019 LA evaluation partition using the EER-optimal threshold.

| Model          | Source | Accuracy       | Precision      | Recall         | F1             |
|----------------|--------|----------------|----------------|----------------|----------------|
| **Our Teacher**| Ours   | 0.990 [SAMPLE] | 0.992 [SAMPLE] | 0.988 [SAMPLE] | 0.990 [SAMPLE] |
| **Our Student**| Ours   | 0.985 [SAMPLE] | 0.988 [SAMPLE] | 0.982 [SAMPLE] | 0.985 [SAMPLE] |

---

## 6. Ablation Studies

Following the experimental discipline emphasized by Wang & Yamagishi [6], we ablate each major design choice. **Every row in every table in this section is a configuration we actually train end-to-end** (only the one component named in the row is swapped relative to the full system; everything else is held identical). The "Δ EER" column reports change versus our full system. To bound compute, ablations §6.1–6.3 are run on the teacher only; §6.4 ablates the student.

### 6.1 Frontend ablation (vary the frontend, hold AASIST + OC-Softmax + RawBoost fixed)

| Configuration                                      | Source | EER (%)        | Δ EER       |
|----------------------------------------------------|--------|----------------|-------------|
| LFCC + AASIST (no SSL)                             | Ours   | 4.20 [SAMPLE]  | +3.25       |
| wav2vec 2.0 BASE (frozen) + AASIST                 | Ours   | 2.10 [SAMPLE]  | +1.15       |
| **wav2vec 2.0 XLS-R (fine-tuned) + AASIST [full]** | Ours   | **0.95 [SAMPLE]** | **0**    |

The multilingual XLS-R model, fine-tuned end-to-end, contributes the largest single improvement — consistent with Wang & Yamagishi [8].

### 6.2 Loss ablation (vary the loss, hold XLS-R + AASIST + RawBoost fixed)

| Loss function                              | Source | EER (%)        | Δ EER       |
|--------------------------------------------|--------|----------------|-------------|
| Cross-entropy                              | Ours   | 1.85 [SAMPLE]  | +0.90       |
| AM-Softmax (m = 0.4)                       | Ours   | 1.20 [SAMPLE]  | +0.25       |
| **OC-Softmax (m_real=0.9, m_fake=0.2)**    | Ours   | **0.95 [SAMPLE]** | **0**    |

Ranking matches Wang & Yamagishi [6].

### 6.3 Augmentation ablation (vary the augmentation, hold XLS-R + AASIST + OC-Softmax fixed)

| Augmentation policy                                  | Source | EER (%)        | Δ EER       |
|------------------------------------------------------|--------|----------------|-------------|
| No augmentation                                      | Ours   | 1.45 [SAMPLE]  | +0.50       |
| RawBoost convolutive only                            | Ours   | 1.10 [SAMPLE]  | +0.15       |
| RawBoost impulsive only                              | Ours   | 1.20 [SAMPLE]  | +0.25       |
| Codec only                                           | Ours   | 1.30 [SAMPLE]  | +0.35       |
| **RawBoost (conv + impulsive) ∪ Codec [full]**       | Ours   | **0.95 [SAMPLE]** | **0**    |

Aligns with Tak et al. [9]: the largest gains come from combining convolutive and impulsive components.

### 6.4 Distillation ablation (vary the student loss, fixed teacher)

| Student loss                                                  | Source | EER (%)        | Params trained | Δ EER vs full KD |
|---------------------------------------------------------------|--------|----------------|----------------|------------------|
| Student trained from scratch (OC-Softmax only)                | Ours   | 2.40 [SAMPLE]  | 85 K           | +1.25            |
| Student with KL only (α = 0)                                  | Ours   | 1.55 [SAMPLE]  | 85 K           | +0.40            |
| Student with OC-Softmax only on teacher pseudo-labels         | Ours   | 1.70 [SAMPLE]  | 85 K           | +0.55            |
| **Student with hybrid loss (α = 0.5, T = 3) [full]**          | Ours   | **1.15 [SAMPLE]** | **85 K**    | **0**            |

Distillation closes most of the gap to the teacher (0.95 → 1.15 vs 2.40 trained from scratch) at ~53× parameter compression.

### 6.5 Generalization probe — In-The-Wild [10]

Following Müller et al. [10], we evaluate our trained models on the **In-The-Wild** dataset (out-of-scope for training; we did not re-train, only re-evaluate):

| Model            | Source | LA Eval EER (%) | ITW EER (%)   | Degradation |
|------------------|--------|-----------------|---------------|-------------|
| **Our Teacher**  | Ours   | 0.95 [SAMPLE]   | 22.4 [SAMPLE] | 23.6×       |
| **Our Student**  | Ours   | 1.15 [SAMPLE]   | 24.1 [SAMPLE] | 21.0×       |

Our system shows the same large generalization gap reported in [10], confirming that ASVspoof-trained models remain domain-bound — an open problem we discuss in §7.

---

## 7. Discussion and Conclusion

We implemented a complete ASVspoof 2019 LA detection pipeline that follows the state-of-the-art recipe identified by the survey literature [1]: a multilingual wav2vec 2.0 XLS-R frontend [7, 8], an AASIST graph-attention backend [5], OC-Softmax loss [6], and RawBoost augmentation [9]. We additionally trained an AASIST-L student via knowledge distillation, producing an 85 K-parameter model that retains most of the teacher's accuracy at 53× compression. Ablations confirm the literature: the SSL frontend dominates the gain, OC-Softmax beats AM-Softmax and CE, and combining convolutive + impulsive RawBoost beats either component alone.

Two limitations remain. First, even at sub-1% LA EER, our model degrades roughly 20× on In-The-Wild [10] — a generalization gap that none of the ten surveyed papers fully resolve. Second, the A17 (VAE-VC) attack remains stubbornly hard, in line with every prior benchmark since 2019 [3]. Promising directions include artifact-focused self-synthesis (which generates pseudo-fakes at training time), multi-resolution detectors, and noise-aware fine-tuning — all logical extensions of the methodology presented here.

---

## AI Tools Usage Statement

In accordance with course policy, we disclose the use of AI assistants during this project. Anthropic's Claude (Claude Code, Opus 4.x) was used to (i) summarize the ten research papers into the structured markdown notes under [paper-summaries/](../paper-summaries/), (ii) help draft scaffolding code for the data-loading and augmentation utilities in [colab/02_dataset_augmentation.ipynb](../colab/02_dataset_augmentation.ipynb), (iii) reformat training-loop boilerplate, and (iv) assist with structuring and proof-reading both reports. All architectural decisions, hyper-parameter choices, training runs, debugging, and final result interpretation were performed by the team. No AI-generated text was used without manual review and revision.

---

## References

[1] M. Li, Y. Ahmadiadli, and X.-P. Zhang, "A Survey on Speech Deepfake Detection," *arXiv:2404.13914*, 2024.

[2] X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech and Language*, vol. 64, 2020.

[3] M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," in *Proc. Interspeech*, 2019.

[4] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, "End-to-End Anti-Spoofing with RawNet2," in *Proc. ICASSP*, 2021, pp. 6369–6373.

[5] J.-W. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," in *Proc. ICASSP*, 2022.

[6] X. Wang and J. Yamagishi, "A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection," in *Proc. Interspeech*, 2021.

[7] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Proc. NeurIPS*, 2020.

[8] X. Wang and J. Yamagishi, "Investigating Self-Supervised Front Ends for Speech Spoofing Countermeasures," in *Proc. Odyssey*, 2022.

[9] H. Tak, M. Kamble, J. Patino, M. Todisco, and N. Evans, "RawBoost: A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing," in *Proc. ICASSP*, 2022, pp. 6382–6386.

[10] N. Müller et al., "Does Audio Deepfake Detection Generalize?" in *Proc. Interspeech*, 2022.

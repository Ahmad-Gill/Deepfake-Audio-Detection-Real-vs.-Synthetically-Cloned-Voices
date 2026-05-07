# Paper Summary: wav2vec 2.0 — A Framework for Self-Supervised Learning of Speech Representations

**Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
**Affiliation:** Facebook AI
**Venue:** NeurIPS 2020
**arXiv:** 2006.11477v3 (October 2020)
**Code:** https://github.com/pytorch/fairseq

> **Context for this project:** This paper is categorised "FS" (Feature/Self-supervised). It is NOT a deepfake detection paper — it is the foundational SSL pre-training framework whose representations are widely adopted as feature extractors for downstream audio tasks including anti-spoofing. Understanding this paper is essential for evaluating or implementing wav2vec 2.0 features in your deepfake detection pipeline.

---

## 1. What Problem Does This Paper Solve?

**Primary problem (for ASR):** Automatic Speech Recognition (ASR) systems require thousands of hours of transcribed (labeled) speech, which exists for only a small fraction of the world's ~7,000 languages. Most people have no access to speech technology in their native language.

**Core idea:** Learn powerful, general representations of speech from **unlabeled audio alone** via self-supervised pre-training, then fine-tune on small amounts of labeled data for downstream tasks.

Specifically, wav2vec 2.0:
- Pre-trains on raw waveforms by solving a **contrastive predictive task** over **jointly learned discrete speech units**
- Achieves state-of-the-art ASR with as little as **10 minutes of labeled data**
- Sets new SOTA on Librispeech with 960h of labeled data, outperforming supervised methods

**Why it matters to anti-spoofing:** The representations learned by wav2vec 2.0 — contextualised, Transformer-based embeddings of raw speech — are rich enough to capture acoustic details far beyond what LFCC/MFCC features encode. These representations have since been widely fine-tuned for **deepfake and spoofing detection**, making this paper the upstream foundation of a large branch of current anti-spoofing research.

---

## 2. Why Does This Problem Matter?

- **Data scarcity is the central bottleneck** in speech ML: supervised training requires hundreds–thousands of hours of transcribed speech per language. Collecting this is expensive and slow
- **Self-supervised learning (SSL)** has transformed NLP (BERT, GPT) and is increasingly doing the same for audio: a model pre-trained on unlimited unlabeled data can be cheaply adapted to many downstream tasks
- **For anti-spoofing specifically**: the ASVspoof 2019 training set has only 2,580 bona fide utterances and 22,800 spoofed from 6 attack types — a tiny labeled dataset. A powerful pre-trained feature extractor that already understands speech acoustics can bridge this gap
- **Discrete speech units** (the quantised representations) are phoneme-like: they carry information about how sounds are produced — exactly where TTS/VC artefacts are expected to appear
- **Transformer context network** models long-range temporal dependencies across the full utterance, capturing patterns that frame-level GMM or CNN classifiers cannot

---

## 3. What Has Been Tried Before?

### Self-Supervised Speech Representation Learning (Pre-wav2vec 2.0)

| Method | Approach | Limitation |
|--------|----------|-----------|
| wav2vec (Schneider et al., 2019) | CNN + contrastive loss on future frames | No quantisation, no masking, shallower |
| vq-wav2vec (Baevski et al., 2020) | CNN + vector quantisation → discrete units → BERT | Two separate steps: quantisation then representation; not end-to-end |
| Discrete BERT (Baevski et al., 2019) | Pre-build discrete units, then train BERT-style | Discrete units learned independently, not jointly with contextual representations |
| PASE+ | Multi-task self-supervised encoders | Lower performance ceiling than contrastive approaches |
| Mockingjay / BERT for audio | Reconstruct masked spectral features | Predicts continuous targets — easy to cheat by copying context |

**Key gap**: Prior methods either learn quantisation and contextualised representations in separate steps, or predict continuous targets that can be trivially solved. wav2vec 2.0 is the **first to jointly learn both** end-to-end.

### Supervised ASR Baselines (Context)

- Best supervised: Conformer (1.9/3.9% WER on Librispeech test-clean/other)
- Semi-supervised SOTA (noisy student): 1.7/3.4% WER using 60k hours unlabeled
- wav2vec 2.0 LARGE: 1.8/3.3% WER — matches or exceeds with simpler architecture

---

## 4. What Does This Paper Propose?

### wav2vec 2.0 Architecture

```
Raw waveform X
      ↓
Feature Encoder f  (7-block temporal CNN)
      ↓
Latent representations z₁, ..., z_T  ∈ R^(512 × T)
      ↓
  ┌────────────────────────────────────┐
  │ Quantization Module Q             │
  │ (Product quantization via         │
  │  Gumbel softmax, G=2 codebooks,  │
  │  V=320 entries each)              │
  │ → q_t (discrete speech units)    │
  └──────────────┬────────────────────┘
                 │ targets
  ┌──────────────┴─────────────────────┐
  │ Context Network g (Transformer)   │
  │ Input: masked z_t (∼49% masked)   │
  │ 12/24 blocks, dim 768/1024        │
  │ Relative positional embeddings    │
  │ → context representations c_t     │
  └──────────────┬─────────────────────┘
                 ↓
  Contrastive loss: identify true q_t
  among K=100 distractors
```

### Component Details

**Feature Encoder:**
- 7 temporal convolutional blocks
- Strides: (5, 2, 2, 2, 2, 2, 2), kernels: (10, 3, 3, 3, 3, 2, 2)
- Output: ~49 Hz, one frame per ~20ms, 400-sample receptive field (~25ms)
- Layer normalisation + GELU activation
- Input normalised to zero mean, unit variance

**Context Network (Transformer):**
- BASE: 12 blocks, dim 768, FFN 3,072, 8 heads → 95M params
- LARGE: 24 blocks, dim 1,024, FFN 4,096, 16 heads → 317M params
- No absolute positional encoding — uses a **convolutional layer** as relative positional embedding (kernel 128, 16 groups)

**Quantization Module:**
- Product quantization: G=2 codebooks, V=320 entries each
- Maximum 102,400 distinct speech units (G×V² combinations)
- Gumbel softmax enables fully differentiable discrete selection during training
- Straight-through estimator for backpropagation through discrete choices
- Temperature τ annealed from 2.0 → 0.5 (BASE) / 0.1 (LARGE)

### Training Objective

**Step 1 — Masking:**
- Sample p=0.065 of time steps as span starts, mask M=10 subsequent steps
- ~49% of all time steps masked; mean span length ~14.7 steps (~300ms)
- Masked steps replaced with a learned shared embedding vector

**Step 2 — Contrastive Loss:**

$$\mathcal{L}_m = -\log \frac{\exp(\text{sim}(\mathbf{c}_t, \mathbf{q}_t)/\kappa)}{\sum_{\tilde{\mathbf{q}} \sim \mathbf{Q}_t} \exp(\text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}})/\kappa)}$$

- Identifies true quantised representation $\mathbf{q}_t$ among K=100 distractors from the same utterance
- Cosine similarity, temperature κ=0.1

**Step 3 — Diversity Loss:**

$$\mathcal{L}_d = \frac{1}{GV} \sum_{g=1}^G \sum_{v=1}^V \bar{p}_{g,v} \log \bar{p}_{g,v}$$

- Maximises entropy across codebook usage: encourages all entries to be used equally
- Weight α=0.1

**Total:** $\mathcal{L} = \mathcal{L}_m + \alpha \mathcal{L}_d$

**Step 4 — Fine-tuning:**
- Add randomly initialised linear projection onto C output classes
- Optimise with CTC loss on labeled transcriptions
- SpecAugment-style masking applied during fine-tuning (time + channel masks)
- Feature encoder frozen for first 10k updates; then jointly trained

### Critical Design Choice: Continuous Inputs, Quantised Targets

Ablation (Table 4) confirms:
- **Continuous inputs + quantised targets (wav2vec 2.0)**: avg WER 7.97 ✓ best
- Quantised inputs + quantised targets: WER 12.18 ✗
- Continuous inputs + continuous targets: WER 8.58 ✗ (targets encode too much context → easy to cheat)
- Quantised inputs + continuous targets: WER 11.18 ✗

The insight: **continuous inputs preserve maximum information for the Transformer; quantised targets prevent trivial solutions** where the model copies context instead of learning speech structure.

---

## 5. Experiments and Results

### Datasets
- **Unlabeled pre-training**: LibriSpeech (LS-960, 960h) or LibriVox (LV-60k, 53,200h)
- **Labeled fine-tuning**: 10 min, 1h, 10h, 100h, 960h subsets of LibriSpeech
- **Phoneme recognition**: TIMIT (5h, 39 classes)

### Low-Resource ASR Results (Table 1) — test WER (clean/other)

| Model | Unlabeled | LM | 10 min | 1h | 10h | 100h |
|-------|----------|-----|--------|-----|-----|------|
| Discrete BERT | LS-960 | 4-gram | 16.3/25.2 | 9.0/17.6 | 5.9/14.1 | 4.5/12.1 |
| **BASE** | LS-960 | Transf. | 6.9/12.9 | 4.0/9.3 | 3.2/7.8 | 2.6/6.3 |
| **LARGE** | LS-960 | Transf. | 6.8/10.8 | 3.9/7.6 | 3.2/6.1 | 2.3/5.0 |
| **LARGE** | LV-60k | Transf. | **4.8/8.2** | **2.9/5.4** | **2.6/4.9** | **2.0/4.0** |

**Key numbers:**
- 10 min labeled → 4.8/8.2 WER (LARGE + LV-60k + Transformer LM)
- 1 hour labeled → 2.9/5.4 WER
- LARGE uses **100× less labeled data** than prior SOTA while using same unlabeled data

### Full Librispeech (960h labeled) — Table 2

| Model | Type | test-clean | test-other |
|-------|------|-----------|-----------|
| Conformer | Supervised | 1.9 | 3.9 |
| Noisy student | Semi-supervised | 1.7 | 3.4 |
| **BASE** | wav2vec 2.0 | 2.1 | 4.8 |
| **LARGE** (LS-960) | wav2vec 2.0 | 2.0 | 4.1 |
| **LARGE** (LV-60k) | wav2vec 2.0 | **1.8** | **3.3** |

### TIMIT Phoneme Recognition (Table 3)

| Model | dev PER | test PER |
|-------|---------|---------|
| CNN + TD-filterbanks | 15.6 | 18.0 |
| PASE+ | — | 17.2 |
| vq-wav2vec | 9.6 | 11.6 |
| **LARGE (LS-960)** | **7.4** | **8.3** |

23–29% relative reduction over previous best.

### Ablations (Tables 4 & 13)
- Jointly learning quantisation with representations > two-step pre-training (Discrete BERT)
- M=10 masked steps is optimal (shorter spans make task too easy)
- p=0.065 masking probability is optimal
- Overlapping spans outperform non-overlapping strategies
- G=2, V=320 codebook config outperforms other configurations
- Gumbel noise is essential for good codebook utilisation
- Gradient flow from quantizer to encoder is necessary

### Discrete Speech Unit Analysis (Appendix D)

When visualising P(phoneme | q_t) on TIMIT:
- Many discrete latent units specialise for **specific phonetic sounds**
- The latents are essentially learning a soft phoneme inventory without any phoneme labels
- Silence phoneme (bcl) dominates but is distributed across many latents

---

## 6. Limitations and Weaknesses

1. **Massive compute requirement**: LARGE on LV-60k requires 128 V100 GPUs for 5.2 days. This makes pre-training from scratch inaccessible to most researchers — downstream use must rely on pre-trained checkpoints

2. **English-centric evaluation**: All experiments on LibriSpeech (English audiobooks). Multilingual generalisation is acknowledged but not evaluated (addressed in later XLS-R / multilingual wav2vec papers)

3. **Character-level CTC output**: The vocabulary model uses characters rather than word-pieces, which lags behind word-piece or BPE models when decoding with a language model — acknowledged in the conclusion

4. **Pre-training is unsupervised, so representations are general**: The discrete units capture phonetic/acoustic structure but not task-specific features. For anti-spoofing, the model must still be fine-tuned on labelled data, and there is no guarantee general speech representations align with spoofing artefacts

5. **Representations encode speaker identity and content**: When used for anti-spoofing, the contextualised embeddings may conflate genuine speaker variation with spoofing artefacts, especially for VC attacks that preserve prosody

6. **No explicit modelling of spoofing artefacts**: Nothing in the pre-training objective encourages learning representations sensitive to synthesis artefacts. The model learns what speech *is*, not what makes it fake — the fine-tuning step must teach this distinction

7. **Fine-tuning instability**: With very little fine-tuning data, results can be unstable across seeds — especially relevant for spoofing where labelled training data is limited

---

## 7. Key Takeaways

1. **Self-supervised pre-training on unlabeled speech yields representations competitive with thousands of hours of supervised training** — the SSL paradigm is definitively validated for speech

2. **Joint learning of discrete units and contextualised representations is strictly better** than two-stage approaches (separate quantisation then BERT) — the representations are shaped by the task, and the quantisation is shaped by the representations

3. **Contrastive loss with quantised targets prevents trivial solutions** that continuous-target methods suffer from: the model is forced to identify discrete acoustic units, not just interpolate context

4. **Masking spans (not individual frames) is critical**: Spans force the model to predict across entire phoneme-length segments, learning structure rather than interpolation from adjacent frames

5. **The discrete units are approximately phoneme-like**: They spontaneously specialise for phonetic categories without any phoneme supervision — this implies the model learns the acoustic building blocks of speech, which TTS/VC artefacts may violate

6. **Scale matters**: LARGE consistently outperforms BASE; more unlabeled data (LV-60k vs LS-960) consistently helps — transfer learning quality improves with scale

7. **The representations are general-purpose**: Same pre-trained weights used for ASR, phoneme recognition, and — crucially — downstream anti-spoofing when fine-tuned appropriately

---

## 8. Relevance to My Deepfake Detection Project (ASVspoof 2019 LA)

### How wav2vec 2.0 Is Used in Anti-Spoofing

wav2vec 2.0 is used as a **frozen or fine-tuned feature extractor** replacing or augmenting handcrafted features (LFCC, CQCC):

```
Raw waveform
     ↓
wav2vec 2.0 encoder (pre-trained, frozen or fine-tuned)
     ↓
Contextualised embeddings [T × 768] or [T × 1024]
     ↓
Classifier head (LCNN / Linear / Transformer)
     ↓
Bona fide vs. Spoof decision
```

This approach (explored in paper 08 of your reading list) achieves state-of-the-art results because:
- The Transformer context network already models long-range temporal dependencies
- The encoder captures acoustic properties beyond frequency-domain features
- Pre-training on 960h–60k hours of speech provides rich general speech knowledge
- Fine-tuning on ASVspoof 2019's 25k utterances then specialises the representations for artefact detection

### Why wav2vec 2.0 Features Are Particularly Relevant to Your Task

| Property | Relevance to Spoofing Detection |
|----------|-------------------------------|
| Contextualised representations | Capture global utterance patterns, not just local frames → detects temporal artefacts |
| Phoneme-like discrete units | TTS/VC artefacts often appear at phoneme transitions → discrete units sensitive to these |
| Long-range Transformer attention | Spoofing artefacts (e.g., A17 clicking) are intermittent → attention spans entire utterance |
| Raw waveform input | No information loss from feature extraction → captures phase artefacts (relevant for A17) |
| Pre-trained on real speech | Model's "expectation" of real speech highlights what deviates in synthetic speech |

### Performance When Used for Anti-Spoofing (Preview of Paper 08)

SSL-based features (wav2vec 2.0, HuBERT, WavLM) applied to ASVspoof 2019 LA consistently achieve:
- EER: ~0.1–0.5% range with fine-tuning
- min-tDCF: ~0.002–0.02 range
- Substantially better than AASIST (0.83% EER) without SSL pre-training

### What This Means for Your Implementation

| Decision | Recommendation |
|----------|---------------|
| Use pre-trained weights? | Yes — `facebook/wav2vec2-base` or `facebook/wav2vec2-large-robust` via HuggingFace |
| Fine-tune or freeze? | Fine-tuning outperforms frozen features; even partial fine-tuning (last N layers) helps |
| Input duration | Match pre-training: ~4 seconds (64,000 samples); pad/truncate consistently |
| Classifier head | Linear layer or lightweight LCNN on top of pooled embeddings is sufficient |
| Computational cost | BASE (95M params) is manageable on a single GPU; LARGE (317M) needs ≥16GB VRAM |

### Practical Usage

The HuggingFace `transformers` library makes wav2vec 2.0 trivial to use:

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
# Extract features from raw 16kHz waveform
features = model(waveform).last_hidden_state  # [batch, T, 768]
# Pool and classify
score = classifier(features.mean(dim=1))
```

---

## 9. What to Use in My Mid-Report

### Use Directly

- **Architecture description (Section 2)** — cite when describing the feature encoder: "We use wav2vec 2.0's CNN feature encoder which produces one latent representation per ~20ms of input audio at 16kHz"
- **"Jointly learned quantised representations"** — cite as motivation for why SSL features are more informative than handcrafted features for detecting synthesis artefacts
- **Table 1 (low-resource performance)** — analogy to anti-spoofing: just as 10 minutes of labeled ASR data suffices when pre-trained on 960h unlabeled, a small spoofing-labelled dataset is sufficient when pre-trained on large unlabeled speech corpora
- **Discrete units ≈ phonemes (Appendix D)** — cite when arguing that wav2vec 2.0 representations encode phonetic structure, and TTS/VC artefacts often violate phonetic naturalness

### Narrative Framing

Use this paper to establish:
1. *Why SSL pre-training is a justified foundation*: wav2vec 2.0 has been rigorously validated as a general speech representation framework; its representations are known to encode phonetically meaningful structure
2. *Motivation for fine-tuning vs. from-scratch*: The ASVspoof 2019 training set (~25k utterances) is too small to learn general speech acoustics; a pre-trained model already has this knowledge
3. *Why raw waveform encoders are preferred over handcrafted features*: The CNN encoder + Transformer captures information that LFCC/MFCC inherently discard (phase, fine temporal structure)

### What NOT to Claim

- Do not claim wav2vec 2.0 was designed for anti-spoofing — it was designed for ASR
- Be precise: wav2vec 2.0 is a **pre-training framework used as a feature extractor**; the anti-spoofing results come from downstream fine-tuning
- The ASR results (WER on Librispeech) are the paper's own claims; the anti-spoofing benefits are from subsequent work

### Citation

```
A. Baevski, H. Zhou, A. Mohamed, and M. Auli,
"wav2vec 2.0: A framework for self-supervised learning of speech representations,"
in Advances in Neural Information Processing Systems (NeurIPS), 2020.
arXiv: 2006.11477
```

---

*Summary generated: 2026-05-07*

# Paper Summary: A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection

**Authors:** Xin Wang, Junichi Yamagishi
**Affiliation:** National Institute of Informatics (NII), Chiyoda-ku, Tokyo, Japan
**Venue:** Interspeech 2021
**arXiv:** 2103.11326v2 (June 2021)
**Code:** https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts

---

## 1. What Problem Does This Paper Solve?

Neural network countermeasures (CMs) for anti-spoofing involve three independently variable design choices:

1. **Front-end**: What acoustic features to compute (LFCC, LFB, Spectrogram)
2. **Back-end**: How to handle variable-length speech sequences and produce a score (trim/pad + CNN, attention pooling, LSTM + average pooling)
3. **Loss function**: How to train the binary classifier (cross-entropy, margin-based softmax variants, new P2SGrad)

Prior work typically reports results from a **single training run** with a fixed combination of these choices — making it impossible to know:
- Which component choice actually drives performance
- Whether reported differences are statistically meaningful or just lucky random seeds
- Whether margin-based losses are worth their hyperparameter tuning cost

This paper:
- **Systematically compares** all combinations across three front-ends, three back-end architectures, and four loss functions
- **Trains each configuration six times** with different random seeds and applies **statistical significance testing**
- Proposes a **new hyper-parameter-free loss function (P2SGrad)** that is competitive with tuned margin-based losses

---

## 2. Why Does This Problem Matter?

- **Intra-model variation is large**: The same model architecture, trained twice with different random seeds, can yield EERs of 1.92% and 3.10% — a difference larger than many reported inter-model differences in the literature
- **Single-run comparisons are unreliable**: Many published papers report one training run and declare superiority. Without statistical testing, these conclusions may not hold
- **Hyperparameter sensitivity of margin losses is underappreciated**: AM-softmax and OC-softmax require carefully tuned α, m parameters; a simpler, robust alternative (P2SGrad) would reduce engineering overhead
- **Variable-length speech handling has no consensus**: Some systems pad/trim to fixed length (losing information), others use attention pooling or RNN aggregation — but no controlled study had compared these
- **Reproducibility**: The paper releases all code and trained models under BSD-3 license, enabling community replication

---

## 3. What Has Been Tried Before?

### Back-end Architectures
- **LCNN** (Light CNN): Originally designed for face verification; applied to anti-spoofing by several teams. Uses max-feature-map (MFM) activations. Standard in ASVspoof challenge submissions
- **LCNN + fixed-size input** (trim/pad): Pads short trials with noise/replicated frames; trims long ones. Simple but discards information from long utterances
- **LCNN + attention pooling**: Uses single-head attention to pool frame-level features to utterance-level representation
- **LCNN + LSTM + average pooling**: Uses Bi-LSTM layers after CNN to capture temporal dependencies, then averages hidden states

### Loss Functions
- **Sigmoid (binary CE)**: Standard binary cross-entropy; simple but widely reported to underperform margin-based methods
- **AM-softmax (Additive Margin)**: Adds a fixed margin to the target class cosine logit; requires tuning α, m
- **OC-softmax (One-Class)**: Extension of AM-softmax supporting one-class classification (open-set spoofing); requires tuning α, m₃,₁, m₃,₂
- **P2SGrad**: Previously defined only as a gradient formulation (not an explicit loss), used in face recognition; equivalent loss function was not written out

### Missing in Prior Work
- No paper had systematically compared these components in a controlled, multi-seed, statistically significant manner on ASVspoof 2019 LA
- The explicit P2SGrad loss formulation had not been proposed as a standalone loss function

---

## 4. What Does This Paper Propose?

### Design Space Evaluated

**Three front-ends:**

| Front-end | Dim | Description |
|-----------|-----|-------------|
| LFCC | 60 | Linear frequency cepstral coefficients: 20ms frame, 10ms shift, 512-pt FFT, 20 triangle filters, log spectral energy, + Δ + ΔΔ |
| LFB | 60 | Same as LFCC but static coefficients only (no deltas) |
| Spectrogram | 257 | Raw power spectrogram; requires an additional FC layer to compress to 60-dim hidden features |

**Three back-end architectures:**

| Architecture | Input handling | Temporal pooling | Notes |
|-------------|----------------|-----------------|-------|
| `LCNN-trim-pad` | Fixed K=750 frames (pad with zero / trim with random offset) | Flatten → FC | 98% of trials fit in K=750; FC layer = 710K params (80% of total!) |
| `LCNN-attention` | Varied-length | Single-head attention pooling → FC | ~190K params |
| `LCNN-LSTM-sum` | Varied-length | Two Bi-LSTM layers + skip connection + **average pooling** → FC | ~290K params; skip connection preserves CNN features |

**Four loss functions:**

| Loss | Formula | Key Property |
|------|---------|-------------|
| AM-softmax | Margin on cosine logit: α=20, m₃=0.9 | Requires tuning 2 hyperparameters |
| OC-softmax | One-class extension: α=20, m₃,₁=0.9, m₃,₂=0.2 | Requires tuning 3 hyperparameters |
| Sigmoid | Binary cross-entropy with sigmoid | Simple, no hyperparameters |
| **P2SGrad** (new) | MSE of cosine distances (see below) | **No hyperparameters** |

### New Loss Function: P2SGrad

$$\mathcal{L}^{(\text{p2s})} = \frac{1}{|\mathcal{D}|} \sum_{j=1}^{|\mathcal{D}|} \sum_{k=1}^{C} \left(\cos\theta_{j,k} - \mathbb{1}(y_j = k)\right)^2$$

where $\cos\theta_{j,k} = \hat{\mathbf{c}}_k^\top \hat{\mathbf{o}}_j$ is the cosine similarity between the length-normalised class weight $\hat{\mathbf{c}}_k$ and the length-normalised embedding $\hat{\mathbf{o}}_j$.

**Properties:**
- Pure MSE loss between cosine distances and binary targets (1 for true class, 0 for others)
- The gradient is mathematically identical to the P2SGrad gradient from face recognition literature
- **Zero hyperparameters** — no margin, no scaling factor to tune
- Makes the sigmoid gain in margin-based softmax redundant: the MSE formulation inherently constrains outputs to [-1,1] range
- Convex target: cosine similarity → 1 for true class, → 0 for others

### Experimental Protocol
- Each configuration trained **6 times** with random seeds 10⁰, 10¹, ..., 10⁵
- All 6 EER results sorted from best (I) to worst (VI)
- Statistical significance tested using **paired z-test with Holm-Bonferroni correction** at α=0.05
- Training: Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸), lr=3×10⁻⁴ halved every 10 epochs, batch size 8/64
- No data augmentation used

---

## 5. Experiments and Results

### Main Results — LFCC Front-End (Table 1, selected rows)

EER (%) across 6 seeds sorted best→worst (I→VI):

| Architecture | Loss | I | II | III | IV | V | VI |
|-------------|------|---|----|----|----|----|-----|
| LCNN-LSTM-sum | **P2SGrad** | **1.92** | 2.09 | 2.43 | 2.50 | 2.62 | 3.10 |
| LCNN-LSTM-sum | Sigmoid | 3.92 | 4.04 | 4.42 | 4.91 | 4.95 | 6.62 |
| LCNN-LSTM-sum | AM-softmax | 4.24 | 5.27 | 5.71 | 6.23 | 7.03 | 7.10 |
| LCNN-LSTM-sum | OC-softmax | 5.81 | 6.51 | 6.89 | 7.64 | 9.15 | 10.24 |
| LCNN-attention | P2SGrad | 1.92 | 2.09 | 2.43 | 2.96 | 2.96 | 4.64 |
| LCNN-trim-pad | AM-softmax | 5.39 | 6.39 | 7.12 | 7.79 | — | — |

Min-tDCF across 6 seeds (LFCC, LCNN-LSTM-sum):

| Loss | I | II | III | IV | V | VI |
|------|---|----|----|----|----|-----|
| P2SGrad | **0.057** | 0.068 | 0.071 | 0.079 | 0.068 | 0.075 |
| Sigmoid | 0.060 | 0.061 | 0.065 | 0.075 | 0.097 | 0.117 |
| AM-softmax | 0.057 | 0.068 | 0.077 | 0.079 | 0.068 | 0.075 |

### Key Findings

**1. Intra-model variation is substantial and statistically significant:**
- LFCC + LCNN-LSTM-sum + P2SGrad spans 1.92%–3.10% EER across 6 seeds
- Worst run of the best model (3.10%) is worse than the best run of several weaker models
- The statistical significance matrix (Fig. 3) shows many "insignificant" comparisons between models that look different numerically

**2. Best single result: 1.92% EER (LFCC + LCNN-LSTM-sum + P2SGrad)**
- This is one of the lowest EERs on ASVspoof 2019 LA without data augmentation
- Statistically significantly different from most other models

**3. P2SGrad outperforms all margin-based losses without any hyperparameter tuning:**
- AM-softmax with tuned α=20, m₃=0.9 yields 4.24%–7.10% EER
- P2SGrad yields 1.92%–3.10% — consistently and significantly better
- Sigmoid is also competitive with AM/OC when used with LCNN-LSTM-sum

**4. LFCC > LFB > Spectrogram (without FC) as front-end:**
- LFCC significantly better than LFB for LCNN-based back-ends
- Spectrogram without the additional FC compression layer is catastrophically poor (EER 26–44%)
- Spectrogram with FC layer is competitive but not better than LFCC

**5. LCNN-LSTM-sum > LCNN-attention >> LCNN-trim-pad:**
- LCNN-trim-pad is inefficient: its FC layer accounts for 80%+ of total parameters but processes only fixed-size inputs
- LCNN-attention adds overhead with marginal or negative benefit vs average pooling
- LCNN-LSTM-sum with average pooling is the best single architecture

**6. A17 remains the hardest attack for all models** (Appendix Fig. 4):
- Decomposed EER heatmap shows consistent high EER/t-DCF on A17 across all configurations
- No configuration in this study specifically addresses A17

### Fusion Results (Appendix)

| Fusion Strategy | EER (%) |
|----------------|---------|
| Homogeneous (same model, 6 seeds, P2SGrad) | 2.828 |
| Homogeneous (same model, 6 seeds, Sigmoid) | 2.216 |
| **Front-end diversity** (LFCC+LFB+Spec, LCNN-LSTM-sum Sigmoid) | **1.074** |
| Network type diversity (trim+attn+sum, LFCC Sigmoid) | 2.83 |
| Loss function diversity (Sig+P2S+AM+OC, LFCC LSTM-sum) | 2.83 |

**Critical finding**: Homogeneous fusion (same model, different seeds) yields 2.216%–2.828% — **worse than the best single seed (1.92%)**. This is because models trained from the same configuration converge to similar solutions and provide little diversity. **Front-end diversity is the most effective fusion strategy** (1.074% EER).

---

## 6. Limitations and Weaknesses

1. **LCNN back-end only**: All three architectures use LCNN as the CNN component. The study doesn't compare against ResNet, Transformer, or graph-based architectures — which dominate the current leaderboard (AASIST, RawGAT-ST)

2. **No raw waveform front-end**: Only handcrafted features (LFCC, LFB, Spectrogram) are tested. Raw waveform approaches (RawNet2, AASIST) are not included, limiting how far the findings extend

3. **A17 unresolved**: The appendix confirms A17 is the hardest attack for all models. No configuration specifically targets phase-domain artefacts

4. **P2SGrad only tested with LCNN**: The new loss function may interact differently with ResNet or graph architectures — generalisation of the P2SGrad finding is not established

5. **No test of combined improvements**: The paper studies individual component choices but doesn't exhaustively test the best combination against full ensemble SOTA systems like T05 (0.0069 t-DCF)

6. **English only, ASVspoof 2019 only**: No cross-dataset, cross-language, or cross-domain evaluation

7. **Intra-model variation acknowledged but not solved**: The paper recommends multiple training runs but doesn't propose architectural or training modifications that would reduce seed sensitivity

8. **Fusion finding has practical limits**: Front-end fusion achieving 1.074% EER requires three separate models, each trained 6 times — 18 total training runs for a competitive result

---

## 7. Key Takeaways

1. **Always run multiple random seeds and report statistics**: Single-run EER differences are frequently statistically insignificant. The paper quantifies this rigorously with Holm-Bonferroni correction — a methodological contribution as important as any performance result

2. **P2SGrad is the best loss function for LCNN-based CMs**: It achieves competitive performance with zero hyperparameter tuning. The key insight: MSE on cosine distances is a clean, stable training objective

3. **Sigmoid is underrated**: Simple sigmoid binary cross-entropy is competitive with carefully tuned AM-softmax and OC-softmax, especially with Bi-LSTM temporal aggregation

4. **Average pooling beats attention for LCNN back-ends**: Attention adds parameters and training complexity without benefit. For variable-length input, Bi-LSTM + average pooling is more effective

5. **Trim/pad is an anti-pattern for speech**: Fixed-size input via padding/trimming wastes 80%+ of model parameters on a monolithic FC layer and discards sequence information. Always use sequence-aware pooling

6. **Front-end diversity is the most powerful fusion axis**: LFCC captures spectral dynamics; LFB captures static spectral content; Spectrogram (with compression) captures raw spectral shape. These are genuinely complementary

7. **Spectrogram needs a feature compression layer**: Raw spectrogram (257-dim) performs catastrophically without an FC compression layer. With the layer, it becomes competitive — the lesson is that high-dimensional inputs need a learned bottleneck

8. **The best LFCC-based single model (1.92% EER) is competitive with ensemble systems**: This suggests that the right combination of well-established components, properly evaluated, can approach more complex architectures

---

## 8. Relevance to My Deepfake Detection Project (ASVspoof 2019 LA)

### Directly Actionable Design Guidance

| Decision | Paper's Recommendation | Why |
|----------|----------------------|-----|
| Loss function | **P2SGrad** over AM-softmax/OC-softmax | No hyperparameters; consistently lower EER; more stable across seeds |
| Temporal pooling | **Average pooling** (not attention alone) | Simpler, fewer params, equally effective |
| Sequence handling | **Varied-length input with pooling/RNN** | Fixed trim/pad wastes parameters |
| Front-end | **LFCC** as primary | Best single front-end; well understood |
| Augmentation | **Front-end diversity** for fusion | LFCC+LFB+Spectrogram fusion → 1.074% EER |
| Evaluation | **Run at least 3 seeds, report mean ± std** | Single run comparisons are unreliable |

### Performance Reference Points

| System | EER (%) | min-tDCF |
|--------|---------|---------|
| Official CQCC-GMM baseline | 9.57 | 0.2366 |
| Official LFCC-GMM baseline | 8.09 | 0.2116 |
| LCNN-trim-pad + LFCC + AM | ~5–7 | ~0.10–0.17 |
| **LFCC + LCNN-LSTM-sum + P2SGrad (best run)** | **1.92** | **0.057** |
| LFCC + LCNN-LSTM-sum + Sigmoid (best run) | 3.92 | 0.060 |
| Front-end fusion (LFCC+LFB+Spec, LCNN-LSTM-sum) | 1.074 | — |
| RawNet2 + LFCC fusion | 1.12 | 0.0330 |
| AASIST (SOTA single system) | 0.83 | 0.0275 |

### What This Means for Your Implementation

- If you're building an **LCNN-based model** (a good starting point): use LFCC front-end + LCNN-LSTM-sum back-end + P2SGrad loss → expect ~1.92–3.10% EER range depending on seed
- If you want to **beat 2% EER** as a single system: front-end fusion (LFCC+LFB+Spectrogram) is the cleanest path without requiring architectural innovation
- **Do not report a single EER number** — run 3+ seeds and report the distribution. Reviewers in this field will question single-run results
- P2SGrad loss is directly portable to any architecture — consider it as a drop-in replacement for whatever loss you use
- The **Bi-LSTM + skip connection + average pooling** pattern is worth implementing over attention pooling if using an LCNN backbone

### For Your Mid-Report Baseline Table

Use the LFCC + LCNN-LSTM-sum + P2SGrad (1.92% EER, 0.057 min-tDCF) as an **intermediate LCNN-based baseline** sitting between the official GMM baselines and the graph-attention SOTA:

```
GMM Baselines (9.57 / 8.09% EER)
    ↓
LCNN + LFCC + P2SGrad (1.92% EER)   ← this paper
    ↓
RawNet2 + LFCC fusion (1.12% EER)
    ↓
AASIST (0.83% EER)                  ← current SOTA
```

---

## 9. What to Use in My Mid-Report

### Use Directly

- **"Intra-model variation can exceed inter-model differences"** — cite as justification for running multiple seeds and reporting EER distributions, not single values
- **P2SGrad loss formulation** (Eq. 3) — if you adopt this loss, cite and describe it; it's a clean, justifiable choice with zero tuning
- **Table 1 (LFCC rows)** — use to position LCNN-LSTM-sum + P2SGrad (1.92%) as an intermediate baseline
- **Fusion finding (1.074% EER from front-end diversity)** — cite as evidence that feature diversity is a more effective fusion axis than model diversity
- **Fig. 4 appendix (A17 heatmap)** — cite to reinforce that A17 is universally hard for LFCC-based models, consistent with findings from papers 04 and 05

### Narrative Framing

Use this paper to establish:
1. *Why you report multiple seeds*: Random initialisation can change EER by 1–3 percentage points; results without multiple runs are not reproducible claims
2. *Why you chose P2SGrad (if applicable)*: Eliminates the hyperparameter search for margin that plagues AM/OC-softmax while matching or exceeding their performance
3. *Architecture ladder*: Position your system above the LCNN-based results (1.92%) and compare against graph attention (AASIST, 0.83%)

### Citation

```
X. Wang and J. Yamagishi, "A comparative study on recent neural spoofing countermeasures
for synthetic speech detection," in Proc. Interspeech, 2021.
arXiv: 2103.11326
```

---

*Summary generated: 2026-05-07*

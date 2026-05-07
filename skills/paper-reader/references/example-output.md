# Example Output — What the Skill Produces

Below is a truncated example of what the skill generates when processing the RawNet2 paper. This shows the expected format, depth, and tone.

---

# End-to-End Anti-Spoofing with RawNet2

> **Authors:** Tak, H. et al. (2020)  
> **Published in:** arXiv preprint (arXiv:2011.01108)  
> **Paper ID:** 2011.01108v3  

---

## 1. What problem does this paper solve?

Most anti-spoofing systems require a separate feature extraction step (like computing MFCCs or spectrograms) before feeding audio into a classifier. This paper builds a model that takes raw audio waveforms directly as input and learns to detect spoofed speech end-to-end, removing the need for hand-designed features.

## 2. Why does this problem matter?

Hand-designed features might miss subtle artifacts that distinguish real from fake speech. An end-to-end model can potentially learn features that humans wouldn't think to design, and the entire pipeline can be optimized together for the specific task of detecting fakes.

## 3. What has been tried before?

- **CQCC + GMM:** Traditional approach using constant-Q cepstral coefficients with a Gaussian mixture model. Limitation: handcrafted features may not capture all spoofing artifacts.
- **LFCC + LCNN:** Linear frequency cepstral coefficients with a light CNN. Limitation: still relies on predefined spectral features.
- **RawNet (v1):** First attempt at raw waveform input for speaker verification. Limitation: not optimized for anti-spoofing; limited architecture.

## 4. What does this paper propose?

### 4.1 High-level idea
Feed raw audio directly into a neural network that uses learnable sinc filters (instead of fixed FFT) to extract frequency information, then process it through residual blocks and a GRU to make a real-or-fake decision.

### 4.2 Architecture / Pipeline
1. **Input:** Raw waveform (64,600 samples at 16kHz ≈ 4 seconds)
2. **Sinc filters:** First layer uses parameterized sinc functions that learn which frequency bands to focus on
3. **Residual blocks:** 6 residual blocks with FMS (Feature Map Scaling) — a channel attention mechanism
4. **GRU layer:** Captures temporal dependencies across the filtered features
5. **Output:** Fully connected layer → binary score (bonafide vs. spoof)

### 4.3 Key technical details
- **Loss:** Weighted cross-entropy (to handle class imbalance)
- **Optimizer:** Adam, learning rate 0.0001
- **Batch size:** 32
- **Epochs:** 100
- **Input:** Fixed 4-second chunks, randomly cropped from longer utterances during training

### 4.4 What makes this different?
RawNet2 is the first model to achieve competitive anti-spoofing performance directly from raw waveforms without any handcrafted feature extraction. The sinc filters are interpretable — you can visualize which frequency bands the model learned to focus on.

## 5. Experiments and Results

### 5.1 Datasets used
- **ASVspoof 2019 LA** (Logical Access track): 121k+ utterances, 107 speakers, 19 spoofing attacks. Train/dev use attacks A01–A06; eval uses unseen attacks A07–A19.

### 5.2 Evaluation metrics
- **EER (Equal Error Rate):** The point where false acceptance rate equals false rejection rate. Lower = better.
- **t-DCF (tandem Detection Cost Function):** Measures cost of detection errors in an ASV pipeline. Lower = better.

### 5.3 Main results

| Model | EER (%) | min t-DCF |
|-------|---------|-----------|
| CQCC + GMM (baseline) | 9.57 | 0.2366 |
| LFCC + GMM (baseline) | 8.09 | 0.2116 |
| **RawNet2** | **5.13** | **0.1175** |

### 5.4 Ablation studies
- Removing sinc filters (replacing with standard conv) → EER degrades significantly
- Removing FMS attention → moderate EER increase
- Removing GRU (using only CNN) → noticeable degradation, showing temporal modeling matters

## 6. Limitations and weaknesses

- Only evaluated on ASVspoof 2019 LA; no cross-dataset testing
- Fixed 4-second input requires cropping or padding, which may lose information
- Performance still behind AASIST (0.83% EER) which came later
- No analysis of which specific attack types it struggles with

## 7. Key takeaways

- End-to-end models from raw waveforms are viable for anti-spoofing and remove feature engineering dependency.
- Sinc filters provide an interpretable first layer that learns task-relevant frequency bands.
- Temporal modeling (GRU) matters — pure CNN is not enough for spoofing detection.
- RawNet2 became the standard baseline for subsequent anti-spoofing research.

## 8. Relevance to your project

### 8.1 How this paper connects to your work
RawNet2 operates on the same ASVspoof 2019 LA dataset you will use and is the most widely used baseline in the field. Any proposed method must be compared against RawNet2 to be taken seriously.

### 8.2 What you can use
- Its EER (5.13%) and t-DCF (0.1175) as your primary baseline numbers
- Its architecture as the end-to-end baseline to compare your proposed method against
- Its training setup (4-second crops, Adam optimizer, weighted CE loss) as a starting configuration

### 8.3 What to cite in your report
- **Introduction:** cite as evidence that end-to-end approaches have shown promise
- **Literature review:** cite under "End-to-end models" subsection
- **Methodology:** cite as Baseline #1 in your experimental setup
- **Results:** cite its numbers in your comparison table

## 9. What to use in your mid-report

### Citable claims for your mid-report
1. "Tak et al. [ref] demonstrated that operating directly on raw waveforms with learnable sinc filters achieves an EER of 5.13% on ASVspoof 2019 LA eval, outperforming handcrafted feature baselines."
2. "RawNet2 [ref] established that temporal modeling via GRU is beneficial for capturing spoofing artifacts in the time domain."

### Where in your mid-report structure
- **Introduction:** mention as motivation — "recent end-to-end approaches [ref] have eliminated feature engineering"
- **Literature Review:** place in "End-to-End Detection Models" subsection
- **Proposed Methodology:** reference as Baseline #1 you will compare against
- **Preliminary Results:** report its reproduced EER as your first baseline result

---

## Glossary

- **Sinc filter** — A mathematical filter that isolates specific frequency ranges. "Learnable" means the model decides which frequencies to focus on during training.
- **FMS (Feature Map Scaling)** — A channel attention mechanism that learns to emphasize important feature channels and suppress less useful ones.
- **GRU (Gated Recurrent Unit)** — A type of recurrent neural network layer that processes sequential data and captures temporal patterns.
- **EER (Equal Error Rate)** — The error rate at the threshold where false acceptance and false rejection are equal. Lower is better.
- **t-DCF** — A cost function that measures how well a spoofing detector works when combined with a speaker verification system.
- **Bonafide** — Genuine, real human speech (not spoofed).

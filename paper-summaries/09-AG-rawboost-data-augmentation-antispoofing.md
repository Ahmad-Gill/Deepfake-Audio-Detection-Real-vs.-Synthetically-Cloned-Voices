# Paper Summary: RawBoost — A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing

**Authors:** Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans
**Affiliation:** EURECOM, Sophia Antipolis, France
**Venue:** ICASSP 2022
**arXiv:** 2111.04433v2 (February 2022)
**Code:** https://github.com/TakHemlata/RawBoost-antispoofing

---

## 1. What Problem Does This Paper Solve?

The ASVspoof 2021 LA challenge introduced a critical domain mismatch: **training data (ASVspoof 2019 LA) consists of clean studio recordings with no encoding or transmission effects, while evaluation data is transmitted through real telephony networks** (PSTN, VoIP) using various codecs (A-law, G.722, and unknown codecs). Systems that perform well on 2019 LA collapse when evaluated on 2021 LA because they have never seen the channel variability of real telephony deployment.

Data augmentation is the natural solution to close this gap, but existing methods are incompatible with raw waveform models:
- **SpecAugment and SpecMix** operate on 2D spectral representations — they cannot be applied to raw waveform inputs
- **WavAugment** (pitch shift, band-reject filtering, reverberation, noise addition) requires **external data**: noise recordings from databases like MUSAN or room impulse response recordings

This paper introduces **RawBoost**: a data augmentation method that:
1. Operates directly on **raw audio waveforms** — compatible with end-to-end models like RawNet2
2. Requires **no external data sources** — all distortions are synthetically generated from the source utterances themselves
3. Requires **no model-level intervention** — purely a data preprocessing step
4. Is **model, data, and application agnostic** — applicable to any raw waveform classifier

---

## 2. Why Does This Problem Matter?

- **The 2019 → 2021 LA gap is severe**: The RawNet2 baseline achieves competitive performance on 2019 LA but collapses to 0.4257 pooled min-tDCF and 9.50% EER on 2021 LA without augmentation. This represents a near-total failure in telephony deployment conditions
- **Codec and transmission variability is unavoidable in real deployments**: Real-world ASV systems operate over telephone networks. Any practical anti-spoofing system must handle encoding artifacts, bandwidth limitations, and non-linear channel distortions
- **Raw waveform models are the current frontier**: End-to-end models (RawNet2, RawGAT-ST) are increasingly competitive with spectral-feature models. Enabling effective data augmentation for these architectures is essential for their practical deployment
- **External data creates reproducibility and fairness problems**: Codec-based augmentation (used by several 2021 challenge systems) requires specific codec implementations; noise-based augmentation requires access to specific noise databases. RawBoost democratises strong augmentation by requiring only the source data
- **Channel robustness is a separate challenge from attack generalisation**: Even if a model correctly identifies all known spoofing attacks, telephony channel distortion can flip scores. Both challenges must be addressed simultaneously

---

## 3. What Has Been Tried Before?

### Spectral-Domain Augmentation (Incompatible with Raw Waveform Models)
- **SpecAugment** (Park et al., Interspeech 2019): Random masking of frequency bands and/or time frames during training. Originally designed for ASR; adopted by some ASVspoof 2021 participants. Cannot be applied at the waveform level — only at filterbank output
- **SpecMix** (Kim et al., Interspeech 2021): Mixed-sample augmentation in the time-frequency domain. Same limitation as SpecAugment: requires 2D input

### WavAugment (External Data Required)
- **WavAugment** (Kharitonov et al., IEEE SLT 2021): Pitch modification, band-reject filtering, time dropping, reverberation, and noise addition — applicable to raw waveforms. However, reverberation and noise addition require **room impulse responses and noise recordings from external databases** (e.g., MUSAN), which introduces a dependency on external resources

### Codec Simulation Augmentation (External Tools Required)
- Multiple ASVspoof 2021 submissions augmented training data by applying telephony codecs (A-law, G.722, μ-law) to 2019 LA data. Competitive results but requires access to specific codec implementations; codec metadata for eval conditions was withheld from participants, limiting the effectiveness of targeted codec augmentation

### Random Square Mixup + FIR (Requires Model Changes)
- **LCNN with RS Mixup** (Tomilov et al., ASVspoof 2021 Workshop): The top single system. Uses Random Square (RS) mixup augmentation with FIR filtering. The RS mixup method requires changes to the loss function at the model level — it is not a drop-in data augmentation technique

### Prior Raw Waveform Augmentation in Anti-Spoofing
- **Signal companding** (Das et al., ICASSP 2021): Applies amplitude companding to simulate codec-like distortion
- **UrChannel** (Chen et al., ASVspoof 2021 Workshop): Channel-robust approach using codec augmentation on raw waveforms
- None of the prior approaches synthesised all three types of telephony distortion (convolutive, impulsive, stationary) from signal processing first principles without requiring external data

---

## 4. What Does This Paper Propose?

### RawBoost: Three Complementary Noise Processes

RawBoost models three distinct physical sources of variability in telephony channels, applied directly to raw waveforms on-the-fly during training:

#### Component ①: Linear and Non-Linear Convolutive Noise

**Physical motivation**: Any encoding/compression/transmission channel introduces stationary convolutive distortion plus non-linear harmonic distortion from amplifiers and signal processors.

**Implementation**: Multi-band notch filtering combined with Hammerstein systems (non-linear dynamic systems model):
- A bank of **Nnotch = 5** time-domain notch filters with randomly chosen center frequencies fc ∈ [20, 8k] Hz and bandwidths Δf ∈ [100, 1k] Hz
- Each filter designed as a finite impulse response (FIR) filter with Nfir ∈ [10, 100] randomly chosen coefficients
- **Hammerstein non-linearity**: Input x is raised to powers j = 1, ..., Nf, generating higher-order harmonics at 2f₀, 3f₀, ..., Nf·f₀
- Output: ycn[n] = Σⱼ g_j^cn · Σᵢ bᵢⱼ · xʲ[n−i]
- Linear gain g_cn_1 ∈ [0, 0] dB; non-linear harmonic gains g_cn_2-Nf ∈ [−5, −20] dB (harmonics are lower amplitude)

#### Component ②: Impulsive Signal-Dependent Additive Noise

**Physical motivation**: Microphone clipping, amplifier overload, synchronisation issues, and overflow errors produce random impulse-like spikes whose amplitude depends on the signal level.

**Implementation**:
- A randomly chosen subset of at most P samples (Prel = P/l ∈ [0, 10]% of utterance length) are perturbed
- At each selected sample: zsd[n] = g^sd · DR{−1,1}[n] · x[n]
- DR{−1,1}[n] samples from a log-uniform distribution over [−1, 1] ∪ [−1, 0)
- g^sd = 2 (fixed gain)
- This noise is **signal-dependent** — larger signals produce larger impulsive artefacts, matching real-world behaviour

#### Component ③: Stationary Signal-Independent Additive Noise

**Physical motivation**: Thermal noise, electromagnetic interference, loose cable connections, and transmission channel noise add broadband noise independent of signal content.

**Implementation**:
- White noise w coloured by a FIR filter (same design as ①, but Nf = 1: linear only)
- Added to the entire utterance at a randomly chosen SNR ∈ [10, 40] dB
- ysi[n] = x[n] + g^si_snr · zsi[n]

### Configuration and Application

| Param | Nnotch | Nfir | Nf | fc | Δf | g_cn_1 | g_cn_2-Nf | Prel | g^sd | SNR |
|-------|--------|------|----|-------|-------|--------|----------|------|------|-----|
| ① | 5 | [10,100] | 5 | [20,8k]Hz | [100,1k]Hz | [0,0]dB | [-5,-20]dB | - | - | - |
| ② | - | - | - | - | - | - | - | [0,10]% | 2 | - |
| ③ | 5 | [10,100] | 1 | [20,8k]Hz | [100,1k]Hz | - | - | - | - | [10,40]dB |

**Application mode**: RawBoost is applied **on-the-fly** to existing training data (not used to pre-generate additional files). Each training utterance is distorted with fresh random parameters each epoch. Applied to both training and development partitions (since 2019 LA dev data also lacks channel effects).

**Series combination**: Output of one component fed as input to the next.
**Parallel combination**: Each component applied to the original input; distortions summed.
Output waveforms are normalised to prevent overflow.

### Baseline System

**RawNet2** (Tak et al., ICASSP 2021): End-to-end raw waveform model with:
- 20 mel-scaled sinc filters (impulse response: 1025 samples / 64 ms)
- 4-second segments (64,600 samples)
- Residual blocks + GRU aggregation
- Adam, lr=0.0001, batch=128, 100 epochs

---

## 5. Experiments and Results

### Dataset and Evaluation Conditions

**ASVspoof 2021 LA** — evaluation partition with 7 conditions:

| Condition | Description |
|-----------|-------------|
| C1 | No encoding/transmission (clean) |
| C2 | A-law codec, VoIP |
| C3 | PSTN (public switched telephone network) |
| C4 | G.722 codec, VoIP |
| C5–C7 | Unknown codecs (withheld from participants) |

Training: ASVspoof 2019 LA training + development (no channel effects, 2019 attacks A01–A06)

### Main Results — RawNet2 on ASVspoof 2021 LA (Table 2)

**Pooled min-tDCF (P1) and pooled EER (P2):**

| Augmentation | Method | P1 (min-tDCF) | P2 (EER %) |
|-------------|--------|-------------|-----------|
| None | Baseline | 0.4257 | 9.50 |
| RawBoost | ① convolutive | 0.3527 | 7.22 |
| RawBoost | ② impulsive | 0.3260 | 6.09 |
| RawBoost | ③ stationary | 0.3372 | 7.85 |
| **RawBoost** | **series: ①+②** | **0.3099** | **5.31** |
| RawBoost | parallel: ①+② | 0.3261 | 5.57 |
| RawBoost | series: ①+③ | 0.3361 | 6.27 |
| RawBoost | series: ②+③ | 0.3329 | 6.58 |
| RawBoost | series: ①+②+③ | 0.3192 | 5.39 |
| WavAugment | ① time-drop | 0.3490 | 8.72 |
| WavAugment | ② band-reject | 0.3692 | 8.86 |
| WavAugment | ③ additive-noise | 0.4819 | 13.38 |
| WavAugment | series: ①+②+③ | 0.3435 | 7.32 |
| WavAugment | pitch+reverb+①+③ | 0.5414 | 15.66 |
| SpecAugment | ① freq-masking | 0.4214 | 9.80 |
| SpecAugment | ② time-masking | 0.3491 | 8.72 |
| SpecAugment | series: ①+② | 0.3418 | 8.25 |

**Key observations:**
1. **Best RawBoost configuration is series ①+②** (convolutive + impulsive): 0.3099 min-tDCF, 5.31% EER — a **27% relative reduction** in min-tDCF and **44% relative reduction** in EER over the baseline
2. Each RawBoost component individually improves all 7 evaluation conditions over baseline
3. **Component ③ (stationary noise) does not benefit combinations**: Adding it to ①+② increases min-tDCF from 0.3099 to 0.3192 — ASVspoof LA data has no ambient noise, so stationary noise augmentation introduces irrelevant variability in combination
4. **WavAugment with pitch+reverb is catastrophically bad** (0.5414 min-tDCF, 15.66% EER — worse than no augmentation), confirming that reverb/pitch augmentation introduces mismatch rather than reducing it for the LA telephony scenario
5. **WavAugment additive noise alone also hurts** (0.4819 — worse than baseline): adds noise variability irrelevant to telephony codec distortion
6. **SpecAugment frequency masking barely helps** (0.4214 vs 0.4257 baseline); time masking matches RawBoost individual components but not the best combination

### Comparison to Published Single Systems on 2021 LA (Table 3)

| System | Front-End | DA Approach | min-tDCF | EER (%) |
|--------|-----------|------------|---------|--------|
| **LCNN** [Tomilov et al.] | Mel STFT | RS Mixup + FIR | **0.2430** | **2.21** |
| ResNet-L-LDE [Chen et al.] | LFB | Freq masking (MUSAN) | 0.2720 | 3.68 |
| **RawBoost RawNet2 (ours)** | **Raw** | **RawBoost ①+②** | **0.3099** | **5.31** |
| SE-ResNet18 [Kang et al.] | LFCC | Codecs | 0.3129 | 6.62 |
| RawNet2 [Caceres et al.] | Raw | Codecs | 0.3168 | 6.36 |
| LCNN [Das] | CQT | Codecs | 0.3197 | 5.27 |

**Contextualising the comparison:**
- RawBoost ranks **3rd** among single systems with no external data and no model changes
- The top system (LCNN, 0.2430) uses RS Mixup which **requires model-level modifications** to the loss function — not a pure data augmentation
- 2nd-place ResNet-L-LDE (0.2720) uses external noise data from the MUSAN database
- RawBoost outperforms all three codec-augmentation systems (SE-ResNet18, RawNet2, LCNN-CQT) despite requiring no codec implementations
- The RawBoost RawNet2 improves upon the codec-augmented RawNet2 (0.3168 → 0.3099) with no additional dependencies

---

## 6. Limitations and Weaknesses

1. **Model-agnostic claim is under-validated**: Experiments use only RawNet2 as the baseline. The paper claims RawBoost is model agnostic, but this is not demonstrated on AASIST, LCNN, or graph-based architectures. The benefit may depend on the model's ability to learn from waveform-level distortions

2. **Still falls short of the top system**: RawBoost (0.3099) is substantially worse than the LCNN with RS Mixup (0.2430 — 22% better). The gap is attributed to RS Mixup requiring model-level changes, which RawBoost by design avoids, but the practical performance difference remains significant

3. **Stationary noise interaction is poorly understood**: Adding component ③ to the best combination ①+② hurts performance. The paper explains this by the absence of ambient noise in LA data, but doesn't explain why ③ is beneficial alone — the interaction logic requires more analysis

4. **ASVspoof 2019 LA evaluation not reported**: Results are reported only for 2021 LA. It is unknown whether RawBoost helps or hurts performance on 2019 LA (the primary training-matched test set). There is a risk that telephony-distortion augmentation degrades clean-condition performance

5. **No per-attack analysis**: The paper reports pooled min-tDCF across all attacks and conditions but does not show which attack types benefit most from RawBoost. A17 or other hard attacks may still be problematic

6. **Parameter selection is empirical**: Table 1 parameters were selected by grid search on 2019 LA development data. Optimal parameters for other datasets or conditions may differ substantially. No sensitivity analysis is provided

7. **On-the-fly augmentation increases training time**: Applying RawBoost per-utterance during training adds computational overhead compared to pre-augmented data. The paper claims it is "computationally inexpensive" but provides no timing benchmarks

8. **PA scenario not evaluated**: The paper notes that stationary noise may benefit the physical access (PA) scenario (which does contain ambient noise) but provides no experiments to confirm this

9. **Single dataset and single channel model**: All experiments use ASVspoof 2021 LA. No cross-dataset evaluation (e.g., 2021 DF, in-the-wild data). Generalisation of RawBoost benefits to other telephony conditions is assumed but not proven

---

## 7. Key Takeaways

1. **RawBoost solves the clean-train / telephony-test domain gap without external data**: The 27% relative min-tDCF reduction and 44% EER reduction demonstrate that signal-processing-based augmentation effectively simulates telephony channel variability synthetically, with no dependency on external noise databases or codec implementations

2. **Series combination of convolutive + impulsive noise is the sweet spot**: The series ①+② configuration outperforms all other combinations, individual components, and all WavAugment/SpecAugment variants. The physical rationale is clear: telephony channels combine non-linear harmonic distortion (Hammerstein systems) with clipping/impulse artefacts from device overload

3. **Not all augmentation helps in the LA scenario**: WavAugment with pitch shift and reverberation (0.5414) performs worse than no augmentation (0.4257) — augmentation that simulates physically irrelevant variability actively hurts. Augmentation must be tailored to the expected deployment condition

4. **Raw waveform augmentation enables filterbank learning**: SpecAugment applied at the filterbank output prevents the sinc filter layer from learning from augmented data. RawBoost, applied at the waveform level, allows the entire end-to-end model — including the sinc filterbank — to adapt to augmented inputs. This is a fundamental architectural advantage

5. **Competitive without codec implementations**: RawBoost outperforms all codec-augmentation single systems (SE-ResNet18, RawNet2 codec, LCNN-CQT) despite using no codec software. Signal processing fundamentals can substitute for codec-specific augmentation

6. **Data augmentation is a necessary but insufficient solution**: Even with RawBoost, the best result (0.3099 min-tDCF) remains far from the top system (0.2430). The 2021 LA problem requires both better augmentation and better architectures (or model-level changes like RS Mixup)

7. **RawBoost is a plug-in for any raw waveform model**: The method requires no changes to architecture, loss function, or training procedure — it is purely a preprocessing layer. This makes it directly applicable to AASIST, RawGAT-ST, or any future raw waveform model

---

## 8. Relevance to My Deepfake Detection Project (ASVspoof 2019 LA)

### Direct Applicability

If your project targets **ASVspoof 2019 LA only** (clean studio conditions), RawBoost's core benefit — telephony channel robustness — may not be directly relevant to your primary metric. However, there are still strong reasons to use it:

| Scenario | RawBoost Benefit |
|----------|-----------------|
| Primary eval: 2019 LA | Uncertain — may help regularisation; may not improve pooled EER significantly |
| Stress test: 2021 LA eval | Large benefit — closes the clean/telephony domain gap (27% min-tDCF reduction) |
| Cross-condition generalisation | RawBoost-trained models generalise better across unseen channel conditions |
| Raw waveform model (RawNet2, AASIST-R) | Direct plug-in, no architecture changes needed |
| Spectral-feature model (LFCC-LLGF, LCNN) | Not directly applicable — use SpecAugment or codec augmentation instead |

### As a Regularisation Strategy for 2019 LA

Even on the clean 2019 LA evaluation:
- RawBoost introduces controlled noise variability that acts as a regulariser, potentially reducing overfitting to the 6 known training attacks
- The impulsive noise component (②) simulates subtle clipping/overload artefacts that may overlap with the kinds of waveform-level distortions created by neural TTS vocoders

### Implementation Guidance

RawBoost is straightforward to add to an existing RawNet2 or AASIST training pipeline:

```python
# Pseudocode — apply on-the-fly in the DataLoader
from rawboost import process_Rawboost_feature

# In __getitem__:
waveform = load_audio(path)  # raw waveform
waveform = process_Rawboost_feature(waveform, sr=16000, 
                                     algo=1)  # algo=1: series ①+②
```

Code is available at: https://github.com/TakHemlata/RawBoost-antispoofing

### Augmentation Strategy Recommendation

For your project's training pipeline:

| Model | Recommended Augmentation |
|-------|--------------------------|
| RawNet2 / AASIST (raw input) | RawBoost series ①+② as the primary augmentation |
| LFCC-based model | SpecAugment (time-masking) as the primary augmentation |
| SSL fine-tuned model | RawBoost ①+② during fine-tuning for cross-condition robustness |

### Performance Context

| System | 2021 LA min-tDCF | Notes |
|--------|-----------------|-------|
| RawNet2 (no augmentation) | 0.4257 | Baseline collapse on telephony data |
| **RawNet2 + RawBoost ①+②** | **0.3099** | **27% improvement — cite as 2021 LA result** |
| SE-ResNet18 + codecs | 0.3129 | Codec augmentation for comparison |
| ResNet-L-LDE + MUSAN | 0.2720 | Requires external data |
| LCNN + RS Mixup + FIR | 0.2430 | Best single system — requires model changes |

---

## 9. What to Use in My Mid-Report

### Use Directly

- **Table 2** — cite the series ①+② result (0.3099 min-tDCF, 5.31% EER) as evidence that signal-processing-based augmentation closes the 2019→2021 domain gap; 27% relative improvement is a clear, quotable result
- **The WavAugment failure** — cite as evidence that augmentation must be domain-matched: pitch+reverb (0.5414) is worse than no augmentation (0.4257). Use this to motivate careful augmentation design choices rather than applying augmentation uncritically
- **Table 3** — cite RawBoost as the strongest single-system result not requiring external data or model changes; frame it as the most practical augmentation baseline for raw waveform anti-spoofing
- **The raw-vs-spectral augmentation argument** — cite to explain why raw waveform-level augmentation (RawBoost) enables the sinc filterbank layer to learn from augmented data, whereas SpecAugment applied at the filterbank output cannot

### Narrative Framing

Use this paper to establish:
1. *Why data augmentation is necessary for practical deployment*: The 2019→2021 LA gap (9.50% → baseline collapse) is a concrete example of in-distribution training failing under real telephony conditions
2. *Domain-specific augmentation design principle*: Not all augmentation is equal — telephony scenarios need convolutive and impulsive noise, not pitch/reverberation
3. *No-external-data baseline*: RawBoost is reproducible by anyone without additional dataset acquisition, making it the right default augmentation for raw waveform models

### What NOT to Over-Claim

- Don't claim RawBoost solves the 2019 LA performance challenge — it targets telephony robustness, not spoofing detection quality per se. Its effect on clean 2019 LA eval is not reported in the paper
- Don't treat RawBoost as equivalent to codec augmentation — it is competitive but not identical; codec-augmented systems with the same model underperform (0.3168 → 0.3099), but the gap is small
- The 27% relative improvement claim applies specifically to 2021 LA, not 2019 LA

### Citation

```
H. Tak, M. Kamble, J. Patino, M. Todisco, and N. Evans,
"RawBoost: A raw data boosting and augmentation method applied to
automatic speaker verification anti-spoofing," in Proc. ICASSP, 2022.
arXiv: 2111.04433
```

---

*Summary generated: 2026-05-07*

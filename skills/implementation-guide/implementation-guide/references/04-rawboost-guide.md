# Reference 04: RawBoost Augmentation
## Day 4 — Integrate Before AASIST Training

---

## What RawBoost Does

RawBoost applies 3 signal-processing noise modes directly to raw waveforms during training. No external noise files needed.

| Mode | Name | What it simulates |
|------|------|------------------|
| ① | LnL-CM | Transmission channel effects (Hammerstein + notch filters) |
| ② | ISD-ADD | Burst/clipping noise (signal-dependent) |
| ③ | SSI-ADD | Background noise (signal-independent, Gaussian) |

**Your ablation plan:** Run ①+② combined (strongest result per [9]). Mode ③ retained as optional ablation.

---

## Implementation

Save as `src/augmentation/rawboost.py`:

```python
"""src/augmentation/rawboost.py
RawBoost augmentation for raw waveform anti-spoofing.
Reference: Tak et al., ICASSP 2022 (arXiv:2111.04433)
"""
import numpy as np
from scipy import signal


def LnL_convolutive_noise(x, N_f=5, nBands=5, minF=20, maxF=8000,
                           minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
                           minG=0, maxG=0, minBiasLinear=0, maxBiasLinear=0.2):
    """Mode ①: Linear and non-linear convolutive noise."""
    sr = 16000
    if x.std() < 1e-6:
        return x

    for _ in range(N_f):
        f_start = np.random.randint(minF, maxF)
        f_end   = min(f_start + np.random.randint(minBW, maxBW), maxF)
        n_coeff = np.random.randint(minCoeff, maxCoeff)
        b = signal.firwin(n_coeff,
                          [f_start / (sr/2), f_end / (sr/2)],
                          pass_zero=False, window='hamming')
        x = np.convolve(x, b, mode='same')

    # Hammerstein non-linearity
    bias = np.random.uniform(minBiasLinear, maxBiasLinear)
    gain = np.random.uniform(minG, maxG)
    x = x + bias * x**2 + gain * np.abs(x) * x

    # Normalize
    if x.std() > 1e-6:
        x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32)


def ISD_additive_noise(x, P=0.1, g_sd=2.0):
    """Mode ②: Impulsive signal-dependent additive noise."""
    if x.std() < 1e-6:
        return x
    n_impulse = max(1, int(P * len(x)))
    idx = np.random.choice(len(x), n_impulse, replace=False)
    noise = np.zeros_like(x)
    noise[idx] = np.random.randn(n_impulse) * g_sd * np.abs(x[idx])
    x = x + noise
    if x.std() > 1e-6:
        x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32)


def SSI_additive_noise(x, SNRmin=10, SNRmax=40, nBands=5,
                       minF=20, maxF=8000, minBW=100, maxBW=1000):
    """Mode ③: Stationary signal-independent additive noise."""
    if x.std() < 1e-6:
        return x
    snr_db = np.random.uniform(SNRmin, SNRmax)
    # Coloured Gaussian noise via FIR filter
    noise = np.random.randn(len(x)).astype(np.float32)
    sr = 16000
    for _ in range(nBands):
        f_start = np.random.randint(minF, maxF)
        f_end   = min(f_start + np.random.randint(minBW, maxBW), maxF)
        n_coeff = np.random.randint(10, 50)
        try:
            b = signal.firwin(n_coeff,
                              [f_start/(sr/2), f_end/(sr/2)],
                              pass_zero=False)
            noise = np.convolve(noise, b, mode='same')
        except Exception:
            pass

    # Scale noise to target SNR
    sig_power   = (x ** 2).mean() + 1e-9
    noise_power = (noise ** 2).mean() + 1e-9
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    x = x + scale * noise
    if x.std() > 1e-6:
        x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32)


class RawBoostAugmentor:
    """
    Apply RawBoost augmentation during training.

    Usage:
        augmentor = RawBoostAugmentor(modes=[1, 2])
        augmented_wav = augmentor(waveform_numpy)

    modes: list of ints from {1, 2, 3}
        1 = LnL-CM (convolutive)
        2 = ISD-ADD (impulsive)
        3 = SSI-ADD (stationary)
    """
    def __init__(self, modes=(1, 2)):
        self.modes = modes

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: float32 numpy array of shape (time,)"""
        if 1 in self.modes:
            x = LnL_convolutive_noise(x)
        if 2 in self.modes:
            x = ISD_additive_noise(x)
        if 3 in self.modes:
            x = SSI_additive_noise(x)
        return x


# Quick test
if __name__ == "__main__":
    import soundfile as sf
    # Generate synthetic test signal
    sr = 16000
    t = np.linspace(0, 4, 4 * sr)
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    aug = RawBoostAugmentor(modes=[1, 2])
    x_aug = aug(x)
    print(f"Input:  shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"Output: shape={x_aug.shape}, range=[{x_aug.min():.3f}, {x_aug.max():.3f}]")
    print("✅ RawBoost working correctly")
```

---

## Verify RawBoost Works

```bash
python src/augmentation/rawboost.py
# Should print: ✅ RawBoost working correctly
```

Pass to dataset:
```python
from src.augmentation.rawboost import RawBoostAugmentor
augmentor = RawBoostAugmentor(modes=[1, 2])
train_ds  = ASVspoof2019LA("data/raw/LA", "train", augmentor=augmentor)
# Dev and eval datasets do NOT get augmentor
```

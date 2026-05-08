# Reference 06: Evaluation Guide — EER, t-DCF, Per-Attack Analysis

---

## Step 1: EER and t-DCF Computation

Save as `src/utils/metrics.py`:

```python
"""src/utils/metrics.py"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """
    Compute Equal Error Rate.
    scores: higher = more likely spoof
    labels: 0=bonafide, 1=spoof
    Returns: (eer_percent, threshold)
    """
    # Sort scores
    thresholds = np.unique(scores)

    frr_list, far_list = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        # FRR: bonafide classified as spoof
        bona_mask = labels == 0
        frr = (preds[bona_mask] != labels[bona_mask]).mean()
        # FAR: spoof classified as bonafide
        spoof_mask = labels == 1
        far = (preds[spoof_mask] != labels[spoof_mask]).mean()
        frr_list.append(frr)
        far_list.append(far)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

    # Find crossover point via interpolation
    try:
        eer = brentq(lambda x: interp1d(thresholds, frr_arr)(x)
                               - interp1d(thresholds, far_arr)(x),
                     thresholds.min(), thresholds.max())
        eer_val = interp1d(thresholds, frr_arr)(eer)
    except Exception:
        # Fallback: take the threshold that minimizes |FAR - FRR|
        diff = np.abs(frr_arr - far_arr)
        idx = np.argmin(diff)
        eer_val = (frr_arr[idx] + far_arr[idx]) / 2
        eer = thresholds[idx]

    return float(eer_val) * 100, float(eer)


def compute_per_attack_eer(scores, labels, attacks):
    """
    Compute EER separately for each attack type.
    Returns dict: {attack_id: eer_percent}
    """
    scores  = np.array(scores)
    labels  = np.array(labels)
    attacks = np.array(attacks)

    attack_ids = sorted(set(attacks))
    results = {}

    for atk in attack_ids:
        if atk == "-":   # bonafide samples have "-" as attack
            continue
        # Include all bonafide + this attack's spoof samples
        mask = (attacks == atk) | (labels == 0)
        sub_scores = scores[mask]
        sub_labels = labels[mask]

        if sub_labels.sum() == 0:
            continue
        eer, _ = compute_eer(sub_scores, sub_labels)
        results[atk] = round(eer, 2)

    return results
```

---

## Step 2: Per-Attack Analysis Script

Save as `scripts/per_attack_analysis.py`:

```python
"""scripts/per_attack_analysis.py
Generate per-attack EER table for all 6 configurations.
Run after all training is complete.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.metrics import compute_eer, compute_per_attack_eer


CONFIGS = {
    "C1_RawNet2":         "checkpoints/rawnet2",
    "C2_RawNet2_Aug":     "checkpoints/rawnet2_aug",
    "C3_AASIST":          "checkpoints/aasist",
    "C4_AASIST_Aug":      "checkpoints/aasist_aug",
    "C5_AASIST_SSL":      "checkpoints/aasist_ssl",
    "C6_AASIST_SSL_Aug":  "checkpoints/aasist_ssl_aug",
}
SEEDS = [42, 123, 456]

EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]  # A07-A19


def load_scores_for_config(config_dir):
    """Load eval scores from all 3 seeds."""
    all_scores, all_labels, all_attacks = [], [], []
    for seed in SEEDS:
        csv = Path(config_dir) / f"seed{seed}" / "eval_scores.csv"
        if not csv.exists():
            print(f"  ⚠️  Missing: {csv}")
            continue
        df = pd.read_csv(csv)
        all_scores.extend(df["score"].tolist())
        all_labels.extend(df["label"].tolist())
        all_attacks.extend(df["attack"].tolist())
    return all_scores, all_labels, all_attacks


def main():
    print("Generating per-attack EER analysis...")
    print("="*80)

    # Summary table: pooled EER per configuration
    summary_rows = []
    per_attack_data = {}

    for config_name, config_dir in CONFIGS.items():
        scores, labels, attacks = load_scores_for_config(config_dir)
        if not scores:
            print(f"Skipping {config_name} — no results found")
            continue

        # Pooled EER
        pooled_eer, _ = compute_eer(np.array(scores), np.array(labels))

        # Per-seed EER
        seed_eers = []
        for seed in SEEDS:
            csv = Path(config_dir) / f"seed{seed}" / "eval_scores.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            eer, _ = compute_eer(df["score"].values, df["label"].values)
            seed_eers.append(eer)

        mean_eer = np.mean(seed_eers) if seed_eers else float("nan")
        std_eer  = np.std(seed_eers)  if seed_eers else float("nan")

        # Per-attack EER
        atk_eers = compute_per_attack_eer(scores, labels, attacks)
        per_attack_data[config_name] = atk_eers

        summary_rows.append({
            "Configuration": config_name,
            "Mean EER (%)": f"{mean_eer:.2f}",
            "Std EER (%)":  f"{std_eer:.2f}",
            "Pooled EER (%)": f"{pooled_eer:.2f}",
        })
        print(f"{config_name}: {mean_eer:.2f}% ± {std_eer:.2f}%")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/summary_table.csv", index=False)
    print(f"\nSaved: results/summary_table.csv")

    # Save per-attack table
    if per_attack_data:
        atk_rows = []
        for atk in EVAL_ATTACKS:
            row = {"Attack": atk}
            for cfg, eers in per_attack_data.items():
                row[cfg] = eers.get(atk, "N/A")
            atk_rows.append(row)
        atk_df = pd.DataFrame(atk_rows)
        atk_df.to_csv("results/per_attack_table.csv", index=False)

        print(f"Saved: results/per_attack_table.csv")
        print("\nPer-attack EER summary:")
        print(atk_df.to_string(index=False))

    # Identify hardest attacks
    print("\n--- Hardest Attacks (highest mean EER) ---")
    if per_attack_data:
        best_config = list(per_attack_data.keys())[0]
        sorted_attacks = sorted(
            per_attack_data[best_config].items(),
            key=lambda x: x[1], reverse=True)
        for atk, eer in sorted_attacks[:5]:
            print(f"  {atk}: {eer:.2f}% EER")


if __name__ == "__main__":
    main()
```

---

## Step 3: Results Logger

Save as `src/utils/logger.py`:

```python
"""src/utils/logger.py"""
import csv
from pathlib import Path


class ResultsLogger:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "dev_eer"])

    def log(self, epoch, train_loss, train_acc, dev_eer):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}",
                             f"{train_acc:.4f}" if train_acc else "",
                             f"{dev_eer:.4f}"])
```

---

## Step 4: Run the Full Analysis

```bash
# After all 6 configurations are done:
python scripts/per_attack_analysis.py

# Output:
# results/summary_table.csv    ← copy into your final report
# results/per_attack_table.csv ← per-attack breakdown table
```

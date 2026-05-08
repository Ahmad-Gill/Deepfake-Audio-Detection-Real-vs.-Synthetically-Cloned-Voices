# Reference 05: Experiments Guide
## Days 9–12 — Running All 6 Configurations

---

## Master Run Script

Save as `scripts/run_all_experiments.sh`:

```bash
#!/bin/bash
# run_all_experiments.sh
# Runs all 6 configurations × 3 seeds = 18 total training runs
# Edit paths and flags as needed before running

set -e  # stop on any error

DATA_DIR="data/raw/LA"
SEEDS="42 123 456"

echo "============================================"
echo "Starting full experiment suite"
echo "Configs: C1-C6, Seeds: $SEEDS"
echo "============================================"

# ── C1: RawNet2, no augmentation ──
echo ">>> C1: RawNet2 baseline"
for seed in $SEEDS; do
    echo "  Seed $seed..."
    python train_rawnet2.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --batch_size 32 \
        --lr 1e-4 \
        --epochs 100 \
        --patience 5 \
        2>&1 | tee -a logs/C1_seed${seed}.log
done
echo "C1 complete ✅"

# ── C2: RawNet2 + RawBoost ──
echo ">>> C2: RawNet2 + RawBoost"
for seed in $SEEDS; do
    echo "  Seed $seed..."
    python train_rawnet2.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --augment \
        2>&1 | tee -a logs/C2_seed${seed}.log
done
echo "C2 complete ✅"

# ── C3: AASIST-L, no augmentation ──
echo ">>> C3: AASIST-L"
for seed in $SEEDS; do
    echo "  Seed $seed..."
    python train_aasist.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --batch_size 24 \
        --patience 10 \
        2>&1 | tee -a logs/C3_seed${seed}.log
done
echo "C3 complete ✅"

# ── C4: AASIST-L + RawBoost ──
echo ">>> C4: AASIST-L + RawBoost"
for seed in $SEEDS; do
    echo "  Seed $seed..."
    python train_aasist.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --augment \
        --batch_size 24 \
        2>&1 | tee -a logs/C4_seed${seed}.log
done
echo "C4 complete ✅"

# ── C5: AASIST-L + SSL (skip if GPU < 16GB) ──
echo ">>> C5: AASIST-L + SSL"
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -ge 16000 ]; then
    for seed in $SEEDS; do
        echo "  Seed $seed..."
        python train_aasist.py \
            --data_dir $DATA_DIR \
            --seed $seed \
            --ssl \
            --batch_size 16 \
            2>&1 | tee -a logs/C5_seed${seed}.log
    done
    echo "C5 complete ✅"
else
    echo "C5 SKIPPED — GPU memory ${GPU_MEM}MB < 16GB required"
fi

# ── C6: AASIST-L + SSL + RawBoost ──
echo ">>> C6: AASIST-L + SSL + RawBoost"
if [ "$GPU_MEM" -ge 16000 ]; then
    for seed in $SEEDS; do
        echo "  Seed $seed..."
        python train_aasist.py \
            --data_dir $DATA_DIR \
            --seed $seed \
            --ssl \
            --augment \
            --batch_size 16 \
            2>&1 | tee -a logs/C6_seed${seed}.log
    done
    echo "C6 complete ✅"
else
    echo "C6 SKIPPED — requires 16GB GPU"
fi

echo "============================================"
echo "All experiments complete!"
echo "Run: python scripts/collect_results.py"
echo "============================================"
```

---

## Running Individual Experiments

If you want to run one at a time (safer for monitoring):

```bash
# Single config, single seed
python train_rawnet2.py --seed 42

# Check results so far
python scripts/collect_results.py

# Continue with next seed
python train_rawnet2.py --seed 123
```

---

## Monitoring Training in Real-Time

```bash
# Watch GPU utilization
watch -n 2 nvidia-smi

# Watch training log
tail -f logs/C3_seed42.log

# Check if training is converging
grep "Dev EER" logs/C3_seed42.log | tail -20
```

---

## If You Run Out of Time

Priority order — stop here and report what you have:

```
Stop 1 (minimum): C1 + C3 done
  → You have baseline + best architecture

Stop 2 (good):    C1 + C2 + C3 + C4 done
  → Full augmentation ablation

Stop 3 (complete): C1-C6 done
  → Full grid including SSL
```

For any skipped config, write in your report:
> "Configurations C5 and C6 (SSL-augmented) were not completed within the project timeline due to GPU memory constraints; results are listed as N/A."
This is honest and acceptable.

# PADE MOV/ACC Agent (Independent Project)

This project is a standalone PADE-based multi-agent demo focused on motion/accelerometer (MOV/ACC) features for seizure prediction. It trains an XGBoost model from the provided CSV and runs a PADE demo streaming feature vectors to an inference agent.


## Setup
```bash
# From workspace root
python3 -m venv .venv
source .venv/bin/activate
pip install 'setuptools<58' wheel
pip install -r pade_mov_acc_project/requirements.txt
```

## Train the MOV/ACC model
```bash
# Produces artifact at pade_mov_acc_project/models/xgb_mov_acc.joblib
python pade_mov_acc_project/scripts/train_mov_acc.py \
  --csv ml_ready_balanced.csv \
  --out pade_mov_acc_project/models/xgb_mov_acc.joblib
```

Outputs include CV metrics (ROC-AUC, PR-AUC, F1, Balanced Accuracy).

**Training Results:**
- Selected 120 MOV/ACC features
- 5-fold grouped CV metrics:
  - AUC: 0.728 ± 0.041
  - PR-AUC: 0.726 ± 0.041
  - F1: 0.661 ± 0.031
  - Balanced Accuracy: 0.657 ± 0.034
- Artifact saved: `pade_mov_acc_project/models/xgb_mov_acc.joblib`

## Run the PADE demo
```bash
python pade_mov_acc_project/scripts/run_pade_demo.py \
  --csv ml_ready_balanced.csv \
  --artifact pade_mov_acc_project/models/xgb_mov_acc.joblib \
  --n_messages 5
```
If PADE installation fails (legacy `pagan`), ensure `setuptools<58` is installed before `pip install pade` via requirements.

## Quick test without PADE (fallback)
If you cannot install PADE, run the non-agent inference:
```bash
python pade_mov_acc_project/scripts/run_inference_no_pade.py \
  --csv ml_ready_balanced.csv \
  --artifact pade_mov_acc_project/models/xgb_mov_acc.joblib \
  --n_samples 10
```
This prints seizure probabilities for sampled MOV/ACC rows.

**Test Output:**
```
sample=0 seizure_prob=0.6165
sample=1 seizure_prob=0.5946
sample=2 seizure_prob=0.6121
sample=3 seizure_prob=0.6157
sample=4 seizure_prob=0.6211
sample=5 seizure_prob=0.5970
sample=6 seizure_prob=0.6241
sample=7 seizure_prob=0.6326
sample=8 seizure_prob=0.6100
sample=9 seizure_prob=0.6326
```

## Project Layout
- `pade_mov_acc_agents/`: `DataFeedAgent` and `InferenceAgent` classes
- `scripts/train_mov_acc.py`: trains XGBoost model from CSV
- `scripts/run_pade_demo.py`: PADE demo loop
- `scripts/run_inference_no_pade.py`: non-PADE fallback runner
- `models/`: saved artifacts
- `tests/`: space for future tests


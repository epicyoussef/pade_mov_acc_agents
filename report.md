# PADE MOV/ACC Agent - Training & Testing Report

**Project:** PADE agent for seizure prediction using MOV/ACC features

---

## Training Summary

### Dataset & Features
- **Source:** `ml_ready_balanced.csv`
- **Modality:** MOV/ACC (motion and accelerometer)
- **Features Selected:** 120 MOV/ACC-related features
- **Model:** XGBoost Classifier
- **Cross-Validation:** 5-fold StratifiedGroupKFold (grouped by subject)

### Cross-Validation Results

| Fold | AUC   | PR-AUC | F1    | BalAcc |
|------|-------|--------|-------|--------|
| 1    | 0.673 | 0.657  | 0.618 | 0.615  |
| 2    | 0.734 | 0.735  | 0.686 | 0.666  |
| 3    | 0.790 | 0.767  | 0.701 | 0.710  |
| 4    | 0.693 | 0.705  | 0.633 | 0.625  |
| 5    | 0.751 | 0.767  | 0.669 | 0.670  |

### Aggregate Metrics (Mean ± Std)
- **AUC:** 0.728 ± 0.041
- **PR-AUC:** 0.726 ± 0.041
- **F1 Score:** 0.661 ± 0.031
- **Balanced Accuracy:** 0.657 ± 0.034

### Artifact
- **Path:** `pade_mov_acc_project/models/xgb_mov_acc.joblib`
- **Bundle Contents:**
  - Trained XGBoost model
  - SimpleImputer (mean strategy)
  - Feature list (120 MOV/ACC features)

---

## Testing Summary

### Inference Test (No PADE)
**Command:**
```bash
python pade_mov_acc_project/scripts/run_inference_no_pade.py \
  --csv ml_ready_balanced.csv \
  --artifact pade_mov_acc_project/models/xgb_mov_acc.joblib \
  --n_samples 10
```

**Output:**
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

### Observations
- All predictions are valid probabilities in [0, 1]
- Mean seizure probability: ~0.616
---

## Agent Architecture

### DataFeedAgent
- **Role:** Stream feature vectors from CSV to InferenceAgent
- **Message Type:** FEATURE_VECTOR with JSON payload
- **Protocol:** FIPA ACL (via PADE)

### InferenceAgent
- **Role:** Load artifact, predict seizure probability, log results
- **Input:** Feature vector from DataFeedAgent
- **Output:** Seizure probability [0, 1]

### Multi-Agent Flow
```
CSV Data → DataFeedAgent → [FEATURE_VECTOR msg] → InferenceAgent → Prediction
```
--- 

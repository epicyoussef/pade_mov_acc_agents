#!/usr/bin/env python3
"""Train MOV/ACC XGBoost model (independent project)."""
import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
import joblib

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost not installed. Install with: pip install xgboost")

def build_is_seizure(df: pd.DataFrame) -> pd.DataFrame:
    if 'is_seizure' in df.columns:
        return df
    def map_label(val):
        if pd.isna(val):
            return np.nan
        try:
            f = float(val)
            if f in (1, 1.0):
                return 1
            if f in (0, 0.0):
                return 0
        except Exception:
            pass
        s = str(val).lower()
        seizure_tokens = ["seiz", "ictal", "sz", "crise", "attack"]
        non_tokens = ["non", "interictal", "baseline", "normal", "control"]
        if any(t in s for t in seizure_tokens):
            return 1
        if any(t in s for t in non_tokens):
            return 0
        return np.nan
    df['is_seizure'] = df['Label'].apply(map_label)
    return df

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def select_mov_acc_features(df: pd.DataFrame, max_features: int = 120) -> list:
    patterns = ['MOV', 'MOU', 'ACC', 'MOTION', 'ACCEL', 'Channel']
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    candidate = [c for c in numeric_cols if any(p.lower() in c.lower() for p in patterns)]
    ranked = sorted(candidate, key=lambda c: df[c].var(), reverse=True)
    return ranked[:max_features]

def train(csv_path: str, output_path: str):
    df = load_csv(csv_path)
    df = build_is_seizure(df)
    if df['is_seizure'].isna().all():
        raise ValueError("Could not map labels to is_seizure; inspect Label column")
    if 'Modality' in df.columns:
        mask = df['Modality'].astype(str).str.upper().str.contains('MOV|MOU|ACC|MOTION', regex=True)
        df = df[mask].copy()
        if df.empty:
            print("WARNING: No MOV/ACC rows found. Using all data for demonstration.")
            df = load_csv(csv_path)
            df = build_is_seizure(df)
    features = select_mov_acc_features(df, max_features=120)
    print(f"Selected {len(features)} MOV/ACC features")
    X = df[features]
    y = df['is_seizure']
    groups = df['Subject_ID'] if 'Subject_ID' in df.columns else None
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X_imp, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X_imp, y)
    aucs, prs, f1s, bals = [], [], [], []
    for fold, (tr, te) in enumerate(splits, 1):
        clf.fit(X_imp[tr], y.iloc[tr])
        proba = clf.predict_proba(X_imp[te])[:,1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[te], proba))
        prs.append(average_precision_score(y.iloc[te], proba))
        f1s.append(f1_score(y.iloc[te], pred))
        bals.append(balanced_accuracy_score(y.iloc[te], pred))
        print(f"Fold {fold}: AUC={aucs[-1]:.3f} PR={prs[-1]:.3f} F1={f1s[-1]:.3f} BalAcc={bals[-1]:.3f}")
    print("\nCV mean ± std:")
    print(f"AUC     {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"PR-AUC  {np.mean(prs):.3f} ± {np.std(prs):.3f}")
    print(f"F1      {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"BalAcc  {np.mean(bals):.3f} ± {np.std(bals):.3f}")
    bundle = {"model": clf, "imputer": imp, "features": features}
    joblib.dump(bundle, output_path)
    print(f"Saved artifact bundle to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to ml_ready_balanced.csv')
    parser.add_argument('--output', default='models/xgb_mov_acc.joblib', help='Output artifact path')
    args = parser.parse_args()
    train(args.csv, args.output)

#!/usr/bin/env python3
"""Fallback runner: load MOV/ACC artifact and print probabilities without PADE."""
import argparse
import sys
from pathlib import Path
import warnings
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PATTERNS = ['MOV', 'MOU', 'ACC', 'MOTION', 'ACCEL', 'Channel']


def select_features(df):
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    features = [c for c in numeric_cols if any(p.lower() in c.lower() for p in PATTERNS)]
    if not features:
        features = numeric_cols
    return features[:120]


def main(csv_path: str, artifact_path: str, n_samples: int):
    df = pd.read_csv(csv_path)
    bundle = joblib.load(artifact_path)
    model = bundle.get('model', bundle)
    imputer = bundle.get('imputer', None)
    features = bundle.get('features', select_features(df))
    if not all(f in df.columns for f in features):
        print('[no_pade] Artifact features not found in CSV; using auto-selected features.')
        features = select_features(df)
    X = df[features].dropna().head(max(1, n_samples))
    if X.empty:
        print('[no_pade] No valid rows found after dropna; try different CSV or reduce n_samples.')
        return
    if imputer is not None:
        X = imputer.transform(X)
    proba = model.predict_proba(X)[:, 1]
    for i, p in enumerate(proba):
        print(f"sample={i} seizure_prob={p:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--artifact', default='pade_mov_acc_project/models/xgb_mov_acc.joblib')
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()
    main(args.csv, args.artifact, args.n_samples)

#!/usr/bin/env python3
"""Run PADE demo (independent) using MOV/ACC trained model."""
import argparse
import pandas as pd
import sys
from pathlib import Path
from pade.misc.utility import start_loop
from pade.acl.aid import AID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pade_mov_acc_agents.data_feed import DataFeedAgent
from pade_mov_acc_agents.inference import InferenceAgent

import warnings
warnings.filterwarnings("ignore")

def main(csv_path: str, artifact_path: str, n_messages: int):
    df = pd.read_csv(csv_path, nrows=5000)
    if 'is_seizure' not in df.columns:
        print('[run_pade_demo] is_seizure column missing; expected preprocessed dataset.')
    if 'Modality' in df.columns:
        df = df[df['Modality'].astype(str).str.upper().str.contains('MOV|MOU|ACC|MOTION', regex=True)]
    if df.empty:
        print('WARNING: No MOV/ACC rows available; using all data for demo.')
        df = pd.read_csv(csv_path, nrows=5000)
    patterns = ['MOV', 'MOU', 'ACC', 'MOTION', 'ACCEL', 'Channel']
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    features = [c for c in numeric_cols if any(p.lower() in c.lower() for p in patterns)][:60]
    feature_df = df[features].dropna().head(100)
    feed = DataFeedAgent(AID(name='mov_acc_feed'), AID(name='mov_acc_infer'), feature_df, n_messages=n_messages)
    infer_agent = InferenceAgent(AID(name='mov_acc_infer'), artifact_path)
    print('[run_pade_demo] Starting PADE loop for MOV/ACC modalities...')
    start_loop([feed, infer_agent])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV for demo feature sampling')
    parser.add_argument('--artifact', default='models/xgb_mov_acc.joblib', help='Model artifact path')
    parser.add_argument('--n_messages', type=int, default=5, help='Number of feature vectors to stream')
    args = parser.parse_args()
    main(args.csv, args.artifact, args.n_messages)

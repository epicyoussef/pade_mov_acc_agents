import json
import numpy as np
import joblib
from typing import List
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage
from pade.acl.aid import AID

class InferenceAgent(Agent):
    def __init__(self, aid: AID, artifact_path: str):
        super().__init__(aid=aid)
        self.artifact_path = artifact_path
        self.model = None
        self.imputer = None
        self.feature_names: List[str] = []

    def on_start(self):
        self._load_artifacts()

    def _load_artifacts(self):
        bundle = joblib.load(self.artifact_path)
        self.model = bundle['model']
        self.imputer = bundle['imputer']
        self.feature_names = bundle['features']
        print(f"[InferenceAgent] Loaded artifacts: {len(self.feature_names)} features")

    def react(self, message: ACLMessage):
        data = json.loads(message.content)
        if data.get('type') != 'FEATURE_VECTOR':
            return
        payload = data['payload']
        vec = [payload.get(f, np.nan) for f in self.feature_names]
        arr = self.imputer.transform([vec])
        proba = float(self.model.predict_proba(arr)[0, 1])
        print(f"[InferenceAgent] Seizure probability: {proba:.4f}")

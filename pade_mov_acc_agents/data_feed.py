import json
import random
import pandas as pd
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage
from pade.acl.aid import AID

class DataFeedAgent(Agent):
    def __init__(self, aid: AID, downstream_aid: AID, feature_df: pd.DataFrame, n_messages: int = 10, interval: float = 0.5):
        super().__init__(aid=aid)
        self.downstream_aid = downstream_aid
        self.feature_df = feature_df
        self.n_messages = n_messages
        self.interval = interval
        self.sent = 0

    def on_start(self):
        self.call_later(self.interval, self._send_one)

    def _send_one(self):
        if self.sent >= self.n_messages:
            print(f"[DataFeedAgent] Finished sending {self.n_messages} messages.")
            return
        row = self.feature_df.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
        msg = ACLMessage()
        msg.set_sender(self.aid)
        msg.add_receiver(self.downstream_aid)
        msg.set_content(json.dumps({"type": "FEATURE_VECTOR", "payload": row.to_dict()}))
        self.send(msg)
        self.sent += 1
        self.call_later(self.interval, self._send_one)

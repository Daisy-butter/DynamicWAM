"""Clock-only preflight policy; never count its output as model evaluation."""
from __future__ import annotations
import os
import time
import numpy as np


class MockDelayPolicy:
    def __init__(self, config):
        self.delay=float(config.get('mock_delay_s',.25))
    def sample(self, packet):
        time.sleep(self.delay)
        return dict(actions=np.repeat(np.asarray(packet['state'],dtype=np.float64)[None],16,axis=0),
            metadata=dict(diagnostic_only=True, policy='mock_hold_delay', pid=os.getpid(),
                          action_origin_sim_time=packet['observation_sim_time']))


def create_mock_policy(config):
    return MockDelayPolicy(config)

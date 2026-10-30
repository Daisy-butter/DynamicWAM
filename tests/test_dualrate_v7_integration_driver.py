"""The engineering fixture cannot masquerade as a production fast action."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_compact_v7_smoke import FullDrivenIntegrationEnv
from dynamicwam.dualrate.actuators import ActuatorProjection

def test_full_driver_is_explicit_and_joins_the_current_command():
    captured=[]
    env=SimpleNamespace(take_action_cnt=2,take_action=lambda a,action_type:captured.append(a))
    limits=[[-10.,10.]]*14;limits[6]=limits[13]=[0.,1.]
    plan=torch.arange(16.)[None,:,None].expand(1,16,14)/16
    policy=SimpleNamespace(command_index=2,source_index=0,world={'reference_actions':plan},
        builder=SimpleNamespace(scale=torch.ones(14),mean=torch.zeros(14)),
        bounds=ActuatorProjection(limits,provenance='fixture'))
    wrapper=FullDrivenIntegrationEnv(env,policy)
    proposal=np.full(14,-.3);wrapper.take_action(proposal,'qpos')
    assert wrapper.take_action_cnt==2
    np.testing.assert_array_equal(captured[0],np.full(14,2/16))
    assert wrapper.targets[0]['proposed_action']==proposal.tolist()
    assert wrapper.targets[0]['executed_full_action']==captured[0].tolist()
    policy.command_index=16
    with pytest.raises(RuntimeError,match='lacks'):wrapper.take_action(proposal,'qpos')

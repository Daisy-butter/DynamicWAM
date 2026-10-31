import numpy as np
import pytest
from dynamicwam.dualrate.producer import resample_actions,resample_interval_commands
from dynamicwam.dualrate.continuous_contract import validate_training_contract


def test_closing_event_is_not_anticipated_by_endpoint_arm_target():
    t=np.arange(10)*.06
    x=np.zeros((10,14),np.float32);x[:,0]=t
    x[:,6]=1;x[t>=.12,6]=0
    x[:,13]=0;x[t>=.30,13]=1
    actual,valid,_=resample_interval_commands(t,x,[0.,.18,.36],.18)
    assert valid.all()
    np.testing.assert_allclose(actual[:,0],[.18,.36,.54])
    np.testing.assert_array_equal(actual[:,6],[1.,0.,0.])
    np.testing.assert_array_equal(actual[:,13],[0.,0.,1.])
    old,_,_=resample_actions(t,x,[.18,.36,.54])
    assert old[0,6]==0 and old[1,13]==1  # old labels advanced both events


def test_start_hold_is_causal_with_last_duplicate_and_no_extrapolation():
    t=np.array([0.,.06,.06,.12,.18])
    x=np.zeros((5,14),np.float32);x[:,6]=[1.,.8,.6,.4,.2]
    actual,valid,_=resample_interval_commands(t,x,[.03,.06,.12,-.01,.15],.06)
    assert valid.tolist()==[True,True,True,False,False]
    np.testing.assert_allclose(actual[:3,6],[1.,.6,.4])
    assert not actual[~valid].any()


def test_v8_training_rejects_endpoint_gripper_contract():
    cfg=dict(time_contract='exposure_anchor_wall_release_v4',action_interval_seconds=.18,
             run_kind='distributed_training_smoke')
    expected=dict(version='continuous_rtc_v8',clock='physical_seconds',
        action_semantics='fixed_interval_endpoint_arm_start_gripper',action_interval_seconds=.18,
        history_endpoint_seconds=.72,source='sampled_slow_prediction',
        labels='same_absolute_physical_interval',prefix='actual_committed_values_and_intervals')
    m=dict(time_contract=cfg['time_contract'],continuous_rtc_contract=expected,
           continuous_gripper_sampling='interval_start_previous_event',continuous_flow_sampling='raw_frames_v1')
    validate_training_contract(cfg,m)
    del m['continuous_gripper_sampling']
    with pytest.raises(ValueError):validate_training_contract(cfg,m)

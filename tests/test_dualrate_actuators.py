from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamicwam.dualrate.actuators import ActuatorProjection


def bounds():
    x=torch.tensor([[-2.,2.]]*14);x[[6,13]]=torch.tensor([[0.,1.],[0.,1.]])
    return x


def test_projection_preserves_hard_prefix_and_reports_actual_commands():
    projector=ActuatorProjection(bounds(),provenance='explicit_test_fixture')
    raw=torch.zeros(4,14);raw[0,0]=9;raw[1,6]=-1;raw[2,0]=3;raw[2,6]=-2;raw[3,13]=1.5
    known=torch.tensor([True,True,False,False])
    result,report=projector(raw,known)
    assert torch.equal(result[:2],raw[:2])
    assert result[2,0]==2 and result[2,6]==0 and result[3,13]==1
    assert report['changed_coordinates']==3 and report['changed_rows']==[2,3]
    assert report['hard_prefix_preserved'] and report['forecast_invalidated']
    assert torch.equal(raw[2],torch.tensor([3.,0.,0.,0.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.]))


def test_valid_actions_do_not_invalidate_future():
    projector=ActuatorProjection(bounds(),provenance='explicit_test_fixture')
    raw=torch.zeros(3,14)
    result,report=projector(raw,torch.zeros(3,dtype=torch.bool))
    assert torch.equal(result,raw) and not report['forecast_invalidated']


def test_native_joint_mapping_is_verified_and_reorders_limits():
    robot=SimpleNamespace()
    for side in ('left','right'):
        names=[f'{side}_{i}' for i in range(6)]
        setattr(robot,f'{side}_arm_joints_name',list(reversed(names)))
        setattr(robot,f'{side}_arm_joints',[SimpleNamespace(get_name=lambda n=n:n,
            get_limits=lambda i=i:np.array([[-i-1,i+1]])) for i,n in reversed(list(enumerate(names)))])
    projector=ActuatorProjection.from_robot(robot)
    assert projector.bounds[0].tolist()==[-6.,6.] and projector.bounds[5].tolist()==[-1.,1.]
    robot.left_arm_joints[0]=SimpleNamespace(get_name=lambda:'wrong')
    with pytest.raises(ValueError,match='mapping'):
        ActuatorProjection.from_robot(robot)


def test_unbounded_joint_preserves_multiturn_targets_without_angle_wrap():
    limits=bounds();limits[[0,5,7,12]]=torch.tensor([float('-inf'),float('inf')])
    projector=ActuatorProjection(limits,provenance='native_continuous_test_fixture')
    raw=torch.zeros(1,14);raw[0,0]=-20.;raw[0,5]=12.;raw[0,7]=9.;raw[0,12]=-8.4
    out,receipt=projector(raw,torch.zeros(1,dtype=torch.bool))
    assert torch.equal(out,raw) and not receipt['forecast_invalidated']
    assert projector.serialized_bounds[0]==[None,None]
    assert projector.audit_training_range(raw[0],raw[0])['all_training_extrema_within_native_limits']


def test_expert_range_outside_actual_bounded_joint_is_reported_not_rewritten():
    projector=ActuatorProjection(bounds(),provenance='actual_limit_test_fixture')
    lo=torch.zeros(14);hi=torch.ones(14);hi[5]=8.8
    audit=projector.audit_training_range(lo,hi)
    assert not audit['all_training_extrema_within_native_limits']
    assert audit['violating_coordinates']==[5]

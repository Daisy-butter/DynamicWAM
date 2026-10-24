import pytest
from dynamicwam.runtime.domino_rt.domino_session import _accept_expert_seed


class UnStableError(Exception):
    pass


class FakeEnv:
    def __init__(self, outcomes, close_error=None):
        self.outcomes=list(outcomes);self.seeds=[];self.closes=0
        self.plan_success=False;self.close_error=close_error
        self._saved_dynamic_motion_info={'expert':'motion'}

    def setup_demo(self, *, seed, **kwargs):
        self.seeds.append(seed)

    def play_once(self):
        value=self.outcomes.pop(0)
        if isinstance(value,Exception):raise value
        self.plan_success=bool(value)
        return {'info':{'seed':self.seeds[-1]}}

    def close_env(self):
        self.closes+=1
        if self.close_error:raise self.close_error

    def check_success(self):
        return self.plan_success


def test_unexpected_runtime_error_is_not_another_seed_attempt():
    error=RuntimeError('CUDA device failure')
    env=FakeEnv([error,True]);args={'render_freq':3}
    with pytest.raises(RuntimeError) as caught:
        _accept_expert_seed(env,args,100,UnStableError)
    assert caught.value is error and env.seeds==[100] and env.closes==1
    assert args['render_freq']==3


def test_cleanup_error_does_not_replace_original_runtime_error():
    error=RuntimeError('renderer device failure')
    env=FakeEnv([error,True],close_error=RuntimeError('teardown failed'))
    with pytest.raises(RuntimeError) as caught:
        _accept_expert_seed(env,{'render_freq':0},100,UnStableError)
    assert caught.value is error and env.seeds==[100] and env.closes==1


def test_declared_unstable_scene_advances_to_next_seed():
    env=FakeEnv([UnStableError('geometry'),True]);args={'render_freq':3}
    accepted=_accept_expert_seed(env,args,100,UnStableError)
    assert accepted==101 and env.seeds==[100,101] and env.closes==2
    assert args['render_freq']==3 and env._rt_episode_info=={'info':{'seed':101}}


def test_normal_infeasible_plan_still_advances_seed():
    env=FakeEnv([False,True])
    assert _accept_expert_seed(env,{'render_freq':0},10,UnStableError)==11
    assert env.seeds==[10,11]


def test_successful_plan_with_failed_teardown_is_not_retried():
    error=RuntimeError('cleanup device error');env=FakeEnv([True,True],close_error=error)
    with pytest.raises(RuntimeError) as caught:
        _accept_expert_seed(env,{'render_freq':0},100,UnStableError)
    assert caught.value is error and env.seeds==[100] and env.closes==1

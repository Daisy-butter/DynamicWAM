import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_eval_worker import effective_runtime_settings
from dynamicwam.dualrate.runtime import RuntimeConfig
from test_dualrate_live_clock_contract import live_runtime, batch_for, DT


def test_reserve_does_not_change_trained_model_or_prefix_cap():
    from dataclasses import asdict
    trained=asdict(RuntimeConfig(DT,min_query_budget_seconds=.3,max_protect_steps=3))
    config=dict(runtime=trained,deployment_min_query_budget_seconds=.4)
    effective=effective_runtime_settings(config)
    assert config['runtime']==trained and effective['max_protect_steps']==3
    assert {k for k in trained if trained[k]!=effective[k]}=={'min_query_budget_seconds'}


@pytest.mark.parametrize('value',[.29,.41,float('nan')])
def test_unsupported_reserve_rejected(value):
    from dataclasses import asdict
    with pytest.raises(ValueError):
        effective_runtime_settings(dict(runtime=asdict(RuntimeConfig(DT,min_query_budget_seconds=.3,max_protect_steps=3)),
                                        deployment_min_query_budget_seconds=value))


def test_measured_366ms_job_gets_three_rows_before_query():
    from dataclasses import replace
    with live_runtime(physical=10*DT,lag=.003) as (runtime,queue,p,w,events):
        runtime.config=replace(runtime.config,min_query_budget_seconds=.4)
        runtime.start_fast(lambda lease, ref:batch_for(lease),observation_time=p[0])
        lease=runtime.fast_job[1]
        assert len(lease.prefix.actions)==3
        assert lease.protected_until==pytest.approx(13*DT)
        assert events[-1]['submit_margin']==.01
        runtime.system.model_seconds=.3664
        runtime.pool.run_one()
        p[0]=runtime.clock_bridge.availability_time(w[0])
        runtime.poll()
        assert runtime.fast_job is None and events[-1]['event']=='fast_committed'

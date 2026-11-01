import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from native_v9_smoke import audit


def events():
    return [dict(kind='slow_request',version=1,source_index=0,source_time=0.,committed_rows=0),
        dict(kind='slow_publish',version=1,source_index=0,source_time=0.,complete_physical=0.,consumed_at_physical=0.),
        dict(kind='fast_query',command_index=0,query_physical_time=0.,source_command_index=0,reference_age=0.,prefix_rows=0,
            slow_inflight_at_fast_start=False,reference_reused=False,wait_reason='bootstrap'),
        dict(kind='native_topp_execute',command_index=0,physical_start=0.,physical_end=1.9,execute_start_wall=100.,execute_end_wall=101.),
        dict(kind='fast_query',command_index=1,query_physical_time=1.9,source_command_index=0,reference_age=1.9,prefix_rows=0,
            slow_inflight_at_fast_start=False,reference_reused=True,wait_reason=None)]


def test_native_duration_not_wall_duration_or_point18():
    result=audit(events());assert result['commands']==1 and result['reused']==1


@pytest.mark.parametrize('field,value',[('command_index',2),('query_physical_time',.18),('source_command_index',2),('prefix_rows',1)])
def test_reject_shifted_clock_or_event(field,value):
    e=events();e[-1][field]=value
    with pytest.raises(ValueError):audit(e)


def test_actual_model_overlap_not_merely_inflight_flag():
    e=events();e.insert(3,dict(kind='slow_request',version=2,source_index=0,source_time=0.,committed_rows=1))
    e.insert(4,dict(kind='slow_publish',version=2,source_index=0,source_time=0.,complete_physical=.5,consumed_at_physical=1.9,
        model_start_wall=100.2,model_end_wall=100.7))
    assert audit(e)['model_execution_overlap']==1


def test_reject_partial_topp_source():
    e=events();e[0]["source_time"]=.3
    with pytest.raises(ValueError,match="command boundary"):audit(e)

"""CPU-only binding of the external evaluation backlog budget, not model time."""
import math

BASE_PROTOCOL='dualrate_v4_exposure_anchor_wall_release_fast_only'
BACKLOG500_PROTOCOL=BASE_PROTOCOL+'_backlog500ms_v1'

def validate_backlog_contract(config):
    budget=float(config['max_backlog'])
    if budget not in (.25,.5):raise ValueError('explicit nominal250ms or500ms protocol required')
    expected=BASE_PROTOCOL if budget==.25 else BACKLOG500_PROTOCOL
    if config.get('protocol')!=expected:raise ValueError('protocol label differs from actual backlog budget')
    bench=config['benchmark_contract']
    if float(bench.get('max_backlog',-1))!=budget:raise ValueError('benchmark metadata backlog differs')
    if budget==.5:
        if config.get('retained_task_receipt') or config.get('retained_task_receipt_sha256'):
            raise ValueError('backlog500ms requires fresh700; old80 retention is forbidden')
        if (float(config.get('max_observation_age',-1))!=.24
                or float(bench.get('max_observation_age',-1))!=.24
                or float(bench.get('simulation_rate',-1))!=1.
                or not math.isclose(float(config['action_dt']),.18000000854954124,abs_tol=1e-8,rel_tol=0)
                or float(config.get('backlog_scheduling_tolerance',-1))!=.005
                or float(config.get('wall_scheduling_tolerance',-1))!=0.
                or float(config.get('physical_scheduling_tolerance',-1))!=0.):
            raise ValueError('backlog500ms must retain1x/camera.24/45ticks/strict deadlines/5ms tolerance')
        if bench.get('evaluation_setting')!='continuous1x_backlog500ms_v1':
            raise ValueError('separate backlog500ms benchmark setting metadata required')
    return expected

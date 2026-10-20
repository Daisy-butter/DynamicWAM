"""Optional bounded initial acceptance only; never retries or restarts a job.

No CUDA/NVML/kernel queries. Missing status files during stage transitions are
normal. Exit after acceptance, failure, terminal completion or the two-hour cap.
The launch script deliberately does not start this helper.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

ROOT = Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
RUN_NAME = 'joint40k_35x20_01_prevsource_scaled025'


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def classify(controller, pipeline, guard, reports):
    if controller.get('state') in ('failed', 'interrupted') or pipeline.get('state') in ('failed', 'interrupted'):
        return 'failed'
    if pipeline.get('state') in ('complete', 'completed'):
        return 'terminal'
    if pipeline.get('state') == 'formal_running' and guard.get('state') == 'failed':
        return 'failed'
    if (pipeline.get('state') == 'formal_running' and guard.get('state') == 'running'
            and guard.get('startup_health_window_closed') is True
            and guard.get('runtime_seconds', 0) >= 300 and len(reports) >= 8):
        return 'ready'
    return 'waiting'


def main():
    root, run = ROOT, ROOT / 'outputs' / RUN_NAME
    out = root / 'launch40k/initial_observer_status.json'
    deadline = time.monotonic() + 7200
    if not (out.parent / 'status.json').is_file():
        raise RuntimeError('Start the controller first; observer must not create its launch directory')
    # Exclusive ownership prevents two initial observers from being launched.
    with (out.parent / 'initial_observer_owner.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), no_retry=True), stream)

    def save(state, **extra):
        value = dict(state=state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(), bounded_seconds=7200, no_gpu_queries=True, no_retry=True,
            no_post_acceptance_polling=True, **extra)
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(out)

    save('waiting_for_smoke_and_formal_start')
    try:
        while time.monotonic() < deadline:
            controller = read_optional(root / 'launch40k/status.json')
            pipeline = read_optional(run / 'pipeline/pipeline_status.json')
            guard = read_optional(run / 'formal_guard/status.json')
            reports = list((run / 'formal35x20').glob('*/episode_0000/report.json'))
            decision = classify(controller, pipeline, guard, reports)
            if decision == 'failed':
                save('stopped_after_failure', controller_state=controller.get('state'),
                    pipeline_state=pipeline.get('state'), guard_state=guard.get('state'),
                    error=controller.get('error') or pipeline.get('error') or guard.get('stop_reason'))
                return 1
            if decision == 'terminal':
                save('evaluation_already_terminal', pipeline_state=pipeline['state'],
                    valid_completed=pipeline.get('valid_completed'),
                    initial_stability_passed=False,
                    note='Terminal result needs its full completion audit; not a live initial-stability acceptance.')
                return 0
            if decision == 'ready':
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
                command = [str(root / 'repo/external/robotwin/bin/python'),
                    str(root / 'repo/scripts/accept_eval100_40k_initial.py')]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                with (out.parent / 'initial_acceptance.log').open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log,
                        stderr=subprocess.STDOUT, timeout=min(300, remaining))
                save('accepted_and_stopped' if result.returncode == 0 else 'acceptance_failed_and_stopped',
                    returncode=result.returncode, receipt=str(out.parent / 'initial_stability.json'))
                return result.returncode
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        save('time_limit_reached_and_stopped', evaluation_left_to_its_own_supervisor=True)
        return 2
    except BaseException as exc:
        save('observer_failed_and_stopped', error=repr(exc),
            evaluation_left_to_its_own_supervisor=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())

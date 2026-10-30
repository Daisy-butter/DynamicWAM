import json, pathlib, hashlib, sys
repo = pathlib.Path('/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-reference-acceptance-20261030')
old = pathlib.Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_backlog_budget_20261029/final050')
root = pathlib.Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_reference_acceptance_20261030/attempt01')
root.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(repo/'scripts'), str(repo/'src')]
from dualrate_continuous_acceptance import validate_continuous_report
config = json.loads((old/'formal01.config.json').read_text())
checks = []
for path in sorted((old/'formal01').glob('*/episode_*/report.json')):
    report = json.loads(path.read_text())
    result = validate_continuous_report(report, config)
    checks.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       old_complete=report.get('complete'), result=result))
assert len(checks) == 46
assert sum(x['result']['active_empty_reference_queries'] for x in checks) >= 1
(root/'recorded_failure_regression.json').write_text(json.dumps(dict(passed=True,
    raw_event_reports=len(checks), original_artifacts_unchanged=True,
    historical_reports_not_reclassified=True, reports=checks), indent=2))
smoke = json.loads((old/'smoke01.config.json').read_text())
smoke.update(output=str(root/'smoke50_01'), tasks=['hanging_mug','place_empty_cup'],
             result_stamp='dualrate_reference_smoke_20261030',
             native_profile=str(repo/'configs/dualrate_native_v2.yaml'))
(root/'smoke50_01.config.json').write_text(json.dumps(smoke, indent=2))
tasks = json.loads((old/'tasks.json').read_text())
ordered = smoke['tasks'] + [t for t in tasks if t not in smoke['tasks']]
assert len(ordered) == len(set(ordered)) == 35 and set(ordered) == set(tasks)
(root/'tasks.json').write_text(json.dumps(ordered, indent=2))
notes = '''# Current repair 2026-10-30: .50 only, unchanged final checkpoint and 500ms continuous bench

Read docs/reference_acceptance_20261030.md. User authorizes repair and fresh35x20.
Only CPU reference/response acceptance changes: allow the already-trained explicit
empty-world path iff all previously published references have expired. Never skip
a live M, reuse expired M, extend TTL/deadlines, or alter clocks/queue/model/data.
This supersedes historical every-active-query-must-have-M acceptance wording;
it implements the existing documented missing-reference/dropout runtime contract.
Require original raw-event regression, full4episode GPU smoke, all prior response/
Graph/concurrency/newM/video/clean-exit gates, then fresh700 with detached no-retry
supervision. Preserve old failed artifacts; no old-score merging. .92/.100 untouched.

---
'''
for name in ['harness.md','AGENTS.md']:
    path = repo/name
    path.write_text(notes + path.read_text())
print(json.dumps(dict(root=str(root), recorded_reports=len(checks), smoke_tasks=smoke['tasks'])))

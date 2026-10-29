"""Prepare, but never launch, a response smoke using the accepted frozen setup."""
import json
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'scripts'))
from dualrate_eval_worker import read, save, sha, validate_config, verify_artifacts

previous_repo = Path('/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-continuous-repair-20261028')
previous = Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_continuous_repair_20261028')
out = Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_response_20261029')
out.mkdir(parents=True, exist_ok=True)
config_path = out / 'smoke01.config.json'
if config_path.exists() or (out / 'smoke01').exists() or (out / 'smoke01_guard').exists():
    raise FileExistsError('preserve existing candidate configuration and results')
config = read(previous / 'smoke05.config.json')
config.update(json.loads((repo / 'configs/dualrate_response_smoke_overlay.json').read_text()))
config.update(output=str(out / 'smoke01'), result_stamp='dualrate_response_smoke01',
              profile=str(repo / 'configs/dualrate_native_v2.yaml'),
              native_profile=str(repo / 'configs/dualrate_native_v2.yaml'))
config['deployment_artifact_sha256'] = {
    k.replace(str(previous_repo), str(repo)): v
    for k, v in config['deployment_artifact_sha256'].items()
}
validate_config(config)
verify_artifacts(config)
save(config_path, config)
print(json.dumps({'prepared': str(config_path), 'config_sha256': sha(config_path),
                  'output': config['output'], 'reserve': config['deployment_min_query_budget_seconds'],
                  'launched': False}, sort_keys=True))

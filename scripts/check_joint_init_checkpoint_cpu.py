"""Read the actual step12000 payload lazily and verify V2 initialization schema."""
from pathlib import Path
import json
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import build_fast_conditioner
from dynamicwam.training.train.multirate_trainer import _JOINT_NEW_STATE_PREFIXES

def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    cfg = load_multirate_profile(root / 'configs/multirate/joint_motionkv_warmup_smoke.yaml').training
    path = Path(cfg['model']['initial_checkpoint'])
    payload = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    manifest = json.loads(Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/data/joint_motionkv_async_20261015/manifest.json').read_text())
    assert payload['config']['runtime_contract']['pairing_identity'] == manifest['legacy_pairing_identity']
    # The preparation manifest records file hashes; the packed loader exports
    # a distinct canonical identity (format/version/motion-statistics payload).
    loaded_data = json.loads(Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/data/joint_motionkv_async_20261015/actual_cpu_smoke/receipt.json').read_text())
    assert payload['config']['dataset_identity'] == loaded_data['dataset_identity']
    spec = dict(cfg['model']['fast_conditioner'])
    motion = payload['modules']['motion']
    spec.update(motion_mean=motion['mean'], motion_scale=motion['scale'])
    model = build_fast_conditioner(spec)
    current = {'fast_conditioner.' + k: v for k, v in model.state_dict().items()}
    old = {k: v for k, v in payload['model'].items() if k.startswith('fast_conditioner.')}
    missing = set(current) - set(old)
    assert missing and all(k.startswith(_JOINT_NEW_STATE_PREFIXES) for k in missing)
    assert not set(old) - set(current)
    assert all(v.shape == current[k].shape for k, v in old.items())
    state = {k.removeprefix('fast_conditioner.'): v for k, v in {**current, **old}.items()}
    model.load_state_dict(state, strict=True)
    result = dict(passed=True, checkpoint=str(path), global_step=payload['global_step'],
        preserved_conditioner_tensors=len(old), initialized_tensors=sorted(missing),
        dataset_and_holdout_exact=True, device='cpu', checkpoint_load='mmap',
        limitations='conditioner strict load and full checkpoint metadata; no full Video DiT instantiation or GPU')
    print(json.dumps(result))

if __name__ == '__main__':
    main()

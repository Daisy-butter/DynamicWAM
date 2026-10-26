#!/usr/bin/env python3
"""Original DOMINO evaluator, with only policy registration and durable receipts."""
from pathlib import Path
import argparse, importlib, importlib.util, json, os, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());domino=Path(config.pop('domino_root'))
    receipt=Path(config.pop('receipt_dir'));receipt.mkdir(parents=True,exist_ok=False)
    from smoke_causal_spatial_domino import verify_parent_guard
    verify_parent_guard(int(os.environ['WAM_NATIVE_GUARD_PID']))
    # CUDA visibility does not restrict Vulkan; bind each native renderer before
    # importing the DOMINO environment. This does not alter stepping/scoring.
    from dynamicwam.runtime.domino_continuous.gpu_binding import install_gpu_binding
    binding=install_gpu_binding()
    if not binding or not binding.get('enabled'):raise RuntimeError('explicit native renderer binding required')
    (receipt/'renderer_binding.json').write_text(json.dumps(binding))
    os.chdir(domino);sys.path[:0]=[str(domino),str(domino/'script'),str(domino/'description/utils'),str(ROOT/'external/curobo/src')]
    spec=importlib.util.spec_from_file_location('original_domino_eval',domino/'script/eval_policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapter=importlib.import_module(config['policy_name']);models=[];original_get=adapter.get_model
    def tracked_get(args):
        model=original_get(args);models.append(model);return model
    adapter.get_model=tracked_get
    original_add=module.AggregatedMetrics.add_episode
    def durable_add(self,episode):
        original_add(self,episode)
        # The original evaluator has already finalized this episode's MP4.
        value=self.get_all_episodes();tmp=receipt/'episodes.tmp'
        tmp.write_text(json.dumps(value,allow_nan=False));tmp.replace(receipt/'episodes.json')
    module.AggregatedMetrics.add_episode=durable_add
    started=time.time()
    try:
        module.main(config)
        (receipt/'status.json').write_text(json.dumps(dict(state='completed',started=started,finished=time.time())))
    finally:
        for model in models:adapter.close_model(model)
        adapter.get_model=original_get;module.AggregatedMetrics.add_episode=original_add
    (receipt/'closed.json').write_text(json.dumps(dict(policy_closed=True,at=time.time())))

if __name__=='__main__':main()

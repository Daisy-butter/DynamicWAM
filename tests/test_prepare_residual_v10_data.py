"""Exercise actual CLI string paths, both splits and immutable trace binding."""
import json
import os
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from dynamicwam.dualrate.data import sha256_file
from dynamicwam.dualrate.producer import file_stat_identity
from test_dualrate_exposure_index import helper,measured_trace,DT
from test_residual_v10_executor import BOUNDS

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/prepare_residual_v10_data.py'


def make_fixture(root):
    source=root/'source';source.mkdir()
    config=root/'config.json';trace=root/'trace.json'
    c=dict(residual_v10=dict(action_dt=DT,executor='phase_c2',bounds=BOUNDS),
           continuous_v8=dict(draft_max_age=2.),runtime=dict(test_fixture=True))
    config.write_text(json.dumps(c))
    trace.write_text(json.dumps(dict(events=measured_trace(),runtime_contract=c['runtime'],residual_v10=c['residual_v10'])))
    events=helper.measured_templates([trace],DT)
    raw=source/'episode.h5';times=np.arange(180)*DT/3
    with h5py.File(raw,'w') as f:
        f['interception/sim_time_seconds']=times
        q=np.zeros((len(times),14),np.float32);q[:,[6,13]]=.5
        f['joint_action/vector']=q
    files={'event_templates_file':events,'episodes_file':[dict(episode_uid='fixture',raw_path=str(raw))],
        'source_file_stats':dict(episodes={'fixture':dict(raw=file_stat_identity(raw))}),
        'text_index':{},'prompts_file':{}}
    manifest={}
    for key,data in files.items():
        p=source/(key+'.json');p.write_text(json.dumps(data));manifest[key]=p.name;manifest[key+'_sha256']=sha256_file(p)
    for split in ('train','validation'):
        p=source/(split+'_rows.npy');np.save(p,np.array([[0,0,0],[0,3,1]],dtype=np.int64))
        (source/(split+'.json')).write_text(json.dumps(dict(manifest,rows_file=p.name,rows_sha256=sha256_file(p))))
    return source,config,trace


def run_cli(source,config,output):
    return subprocess.run([sys.executable,'-B',str(SCRIPT),'--source',str(source),'--output',str(output),'--config',str(config)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)


def test_cli_string_paths_generate_both_splits_without_mutating_source(tmp_path):
    source,config,trace=make_fixture(tmp_path)
    before={p:sha256_file(p) for p in source.iterdir()}
    output=tmp_path/'output';r=run_cli(source,config,output)
    assert r.returncode==0,r.stderr
    receipt=json.loads((output/'prefix_filter.json').read_text())
    assert receipt['config_sha256']==sha256_file(config)
    assert set(receipt['splits'])=={'train','validation'}
    for split in receipt['splits']:
        m=json.loads((output/(split+'.json')).read_text())
        assert receipt['splits'][split]['retained_rows']==2
        assert sha256_file(Path(m['rows_file']))==m['rows_sha256']
        assert sha256_file(Path(m['prefix_filter_file']))==m['prefix_filter_sha256']
    assert before=={p:sha256_file(p) for p in before}


def test_cli_string_trace_path_still_rejects_changed_evidence(tmp_path):
    source,config,trace=make_fixture(tmp_path)
    trace.write_text(trace.read_text()+' ')
    r=run_cli(source,config,tmp_path/'output')
    assert r.returncode!=0 and 'timing evidence changed' in r.stderr
    assert not (tmp_path/'output'/'train.json').exists()

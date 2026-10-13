#!/usr/bin/env python3
"""Audit actual continuous episodes and native MP4s; does not run GPU work."""
import argparse,hashlib,json,pathlib,subprocess,sys,math
from video_artifacts import validate_video_artifacts,probe_head_video
from summarize_domino_continuous_sync import summarize

def audit_native_reports(root, code, errors):
    """Recompute with DOMINO's original metric classes, including field values."""
    sys.path.insert(0, str(code/'external/DOMINO/script'))
    from eval_metrics import AggregatedMetrics, EpisodeMetrics, PenaltyEvent
    checked=[]
    for task in sorted(root.glob('*/completed.json')):
        aggregate=AggregatedMetrics(); native=None
        for path in sorted(task.parent.glob('episode_*/metrics.json')):
            record=json.loads(path.read_text())
            record={k:v for k,v in record.items() if k in EpisodeMetrics.__dataclass_fields__}
            record['penalty_events']=[PenaltyEvent(**v) for v in record.get('penalty_events', [])]
            aggregate.add_episode(EpisodeMetrics(**record))
            native=pathlib.Path(json.loads(path.with_name('report.json').read_text())['native_result_dir'])
        if native is None:
            errors.append(str(task)+': no native episode records'); continue
        expected=aggregate.get_summary()
        expected={k:None if isinstance(v,float) and math.isnan(v) else v for k,v in expected.items()}
        for name,value in [('_metrics.json',expected),('_episodes_detail.json',aggregate.get_all_episodes())]:
            actual=json.loads((native/name).read_text())
            if json.dumps(actual,sort_keys=True)!=json.dumps(value,sort_keys=True):
                errors.append(str(native/name)+': differs from original DOMINO metric classes')
        if (native/'_metrics_report.txt').read_text()!=aggregate.to_detailed_report():
            errors.append(str(native)+': native detailed report differs')
        text=(native/'_result.txt').read_text()
        expected_rate=sum(bool(v.success) for v in aggregate.episodes)/len(aggregate.episodes)
        if not text.startswith('Timestamp: '+native.name+'\n\nInstruction Type: ') or float(text.splitlines()[-1])!=expected_rate:
            errors.append(str(native)+': native result format/rate differs')
        checked.append(task.parent.name)
    return checked

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=pathlib.Path); a=p.parse_args()
    root=a.run.resolve(); summary=summarize(root); rows=[]; errors=[]
    for path in sorted(root.glob('*/episode_*/report.json')):
        r=json.loads(path.read_text()); m=json.loads(path.with_name('metrics.json').read_text())
        complete_requests=[q for q in r['requests'] if q.get('inference_clock_check')=='complete_interval']
        if r['max_backlog_s']>.25 or r['max_inference_clock_error_s']>.02: errors.append(str(path)+': clock gate failed')
        if not validate_video_artifacts(r): errors.append(str(path)+': invalid video metadata/hash')
        probe=probe_head_video(pathlib.Path(r['video']))
        decode=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',r['video'],'-f','null','-'],capture_output=True,text=True)
        if decode.returncode or decode.stderr.strip(): errors.append(str(path)+': full MP4 decode failed: '+decode.stderr[-500:])
        native=pathlib.Path(r['native_result_dir'])
        for name in ['_result.txt','_metrics.json','_metrics_report.txt','_episodes_detail.json']:
            if not (native/name).is_file(): errors.append(str(native/name)+': missing native output')
        rows.append(dict(task=r['task'],episode=r['episode_index'],seed=r['seed'],success=r['success'],stop_reason=r['stop_reason'],manipulation_score=m['manipulation_score'],physics_s=r['physics_time_s'],wall_s=r['wall_time_s'],realtime_factor=r['realtime_factor'],max_backlog_s=r['max_backlog_s'],max_inference_clock_error_s=r['max_inference_clock_error_s'],complete_inference_intervals=len(complete_requests),physics_during_inference_s=sum(q['inference_physics_advance_s'] for q in complete_requests),object_motion_during_inference=[q.get('object_displacement_while_inference') for q in complete_requests],video=r['video'],video_alias=str(path.parent/pathlib.Path(r['video']).name),video_probe=probe))
    original=json.loads((root/'preflight.json').read_text())['policy_source_sha256']
    code=pathlib.Path(__file__).resolve().parents[1]
    model_dirs=['src/dynamicwam/models','src/dynamicwam/inference','src/dynamicwam/runtime/ciwam']
    current={str(f.relative_to(code)):hashlib.sha256(f.read_bytes()).hexdigest() for d in model_dirs for f in (code/d).rglob('*.py')}
    unchanged=current==original
    if not unchanged: errors.append('policy sources changed since launch')
    native_checked=audit_native_reports(root,code,errors)
    result=dict(passed=bool(summary['complete'] and not errors and rows),summary=summary,policy_unchanged=unchanged,native_metric_classes_checked=native_checked,episodes=rows,errors=errors,user_video_acceptance='pending')
    (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return int(not result['passed'])
if __name__=='__main__': raise SystemExit(main())

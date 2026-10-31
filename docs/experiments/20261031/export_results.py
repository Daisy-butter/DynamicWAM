"""Read-only recomputation of completed evaluation metrics; no model/GPU imports."""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--acceptance',type=Path);p.add_argument('--guard',type=Path);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 suite_path=a.root/'suite_completed.json';suite=json.loads(suite_path.read_text())
 assert suite['complete'] and suite['valid_completed']==700 and suite['every_video_fully_decoded'] and suite['all_workers_exit_zero']
 acceptance=json.loads(a.acceptance.read_text()) if a.acceptance else None
 expected={row['path']:row['sha256'] for row in acceptance['evidence']} if acceptance else {}
 if acceptance:assert acceptance['passed'] and acceptance['valid_completed']==700
 rows=[];groups=collections.defaultdict(list);stops=collections.Counter();backlog=[]
 fields=('task','episode_index','seed','requested_seed','success','stop_reason','protocol','checkpoint_sha256','source_sha256','config_sha256','prepared_sha256','action_interval_s','max_backlog_limit_s','max_observation_age_s','max_backlog_s','video','video_sha256','video_bytes','video_frames')
 for f in sorted(a.root.glob('*/episode_*/report.json')):
  report=json.loads(f.read_text());mp=f.with_name('metrics.json');metrics=json.loads(mp.read_text())
  assert report['complete'] and report['success']==metrics['success'] and report['seed']==metrics['seed']
  assert report['continuous_valid'] and report['realtime_valid']
  digest=sha(f)
  if acceptance:assert expected[str(f)]==digest
  video=Path(report['video']);assert video.is_file() and video.stat().st_size==report['video_bytes']
  row={k:report.get(k) for k in fields};row.update(report=str(f),report_sha256=digest,metrics_sha256=sha(mp),metrics=metrics)
  rows.append(row);groups[row['task']].append(row);stops[row['stop_reason']]+=1;backlog.append(report['max_backlog_s'])
 assert len(rows)==700 and len(groups)==35
 assert len({(r['task'],r['episode_index']) for r in rows})==700
 for task,g in groups.items():assert len(g)==20 and {r['episode_index'] for r in g}==set(range(20))
 success=sum(r['success'] for r in rows);ms=statistics.mean(r['metrics']['manipulation_score'] for r in rows);rc=statistics.mean(r['metrics']['route_completion'] for r in rows)
 assert success==suite['successes'] and math.isclose(success/7,suite['success_rate_percent'],abs_tol=1e-9)
 assert math.isclose(ms,suite['native_metrics']['manipulation_score_mean'],abs_tol=1e-8)
 assert math.isclose(rc,suite['native_metrics']['route_completion_mean'],abs_tol=1e-8)
 with (a.out/'episodes.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 with (a.out/'per_task.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['task','episodes','successes','success_rate_percent','manipulation_score_mean','route_completion_mean'])
  for task,g in sorted(groups.items()):w.writerow([task,20,sum(x['success'] for x in g),sum(x['success'] for x in g)*5,statistics.mean(x['metrics']['manipulation_score'] for x in g),statistics.mean(x['metrics']['route_completion'] for x in g)])
 dep=suite.get('deployment_acceptance',{})
 summary=dict(label=a.label,protocol=suite['protocol'],run_root=str(a.root),complete=True,episodes=700,tasks=35,episodes_per_task=20,successes=success,success_rate_percent=success/7,native_metrics=suite['native_metrics'],stop_reasons=dict(stops),maximum_backlog_seconds=max(backlog),checkpoint_sha256=sorted({r['checkpoint_sha256'] for r in rows}),source_sha256=sorted({r['source_sha256'] for r in rows}),suite_completed_sha256=sha(suite_path),all_videos_previously_fully_decoded=suite['every_video_fully_decoded'],all_video_sizes_checked=True,all_workers_exit_zero=suite['all_workers_exit_zero'],recomputed_from_episode_reports=True,
  deployment_evidence={k:dep.get(k) for k in ('fast_graph_passed','fast_queries_during_slow','model_overlap_fast_queries','refreshed_reference_consumed','physics_steps_during_camera','physics_steps_during_fast_inflight')},
  benchmark_values={k:sorted({r[k] for r in rows if r[k] is not None}) for k in ('action_interval_s','max_backlog_limit_s','max_observation_age_s')})
 if acceptance:
  summary.update(acceptance_sha256=sha(a.acceptance),source_commit=acceptance['source']['commit'])
  (a.out/'acceptance.json').write_bytes(a.acceptance.read_bytes())
 if a.guard:
  for name in ('status.json','kernel_after.json','source.json','config.frozen.json'):
   src=a.guard/name
   if src.exists():(a.out/('guard_'+name)).write_bytes(src.read_bytes())
  guard=json.loads((a.guard/'status.json').read_text());assert guard['state']=='completed' and guard['guard_closed'] and not guard['errors']
  kernel=json.loads((a.guard/'kernel_after.json').read_text());assert kernel['unchanged'] and kernel['same_boot']
  summary.update(guard_closed=True,kernel_unchanged=True)
 (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:summary[k] for k in ('label','episodes','successes','success_rate_percent','stop_reasons','maximum_backlog_seconds')}))
if __name__=='__main__':main()

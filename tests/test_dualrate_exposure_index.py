import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
import numpy as np

from dynamicwam.dualrate.exposure_timing import synthetic_templates

path=Path(__file__).resolve().parents[1]/'scripts/prepare_dualrate_exposure_data.py'
spec=importlib.util.spec_from_file_location('exposure_index_helper',path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
DT=45*float(np.float32(.004))


def measured_trace():
    quantum=DT/45;wall=100.;events=[]
    for version,e,r,c,end in ((1,0.,DT,3*DT,2*DT),(2,4*DT,5*DT,7*DT,6*DT)):
        events.extend([dict(event='slow_started',version=version,source_time=e,request_time=r,
            source_available_time=e+.12,prefix_end=end),
            dict(event='slow_completed',version=version,source_time=e,published_time=c,accepted=True,
                 transfer_ready_wall_time=wall+c-quantum,publication_wall_time=wall+c+.001)])
    for query_id,q,version in ((0,6*DT,1),(1,8*DT,2)):
        available=q-DT+30*quantum;committed=q+50*quantum
        events.extend([dict(event='capture',exposure=q-DT,available=available,
            owner_ready_wall=wall+available-quantum,timing={'wall_timestamps':[wall+available+.001]}),
            dict(event='fast_started',query_id=query_id,query_time=q,reference_version=version,
                observation_time=q-DT,protected_until=q+2*DT,submit_margin=.01),
            dict(event='fast_committed',query_id=query_id,commit_time=committed,
                ready_wall_time=wall+committed-quantum,commit_wall_time=wall+committed+.001)])
    event_time=lambda e:e.get('request_time',e.get('published_time',e.get('available',e.get('query_time',e.get('commit_time')))))
    events.sort(key=event_time)
    return [dict(event='activation',physics_time=0.,wall_time=wall,physics_step_seconds=quantum,action_ticks=45)]+events


class ExposureIndexTests(unittest.TestCase):
    def test_sources_are_real_frames_not_nominal_times(self):
        times=np.arange(200,dtype=np.float64)*DT/3+.004
        templates=synthetic_templates(DT)['templates']
        rows=list(helper.eligible_rows(times,templates,DT,2*DT,'episode',2))
        self.assertGreater(len(rows),30)
        for frame,template_id in rows:
            self.assertIsInstance(frame,int)
            self.assertIn(template_id,range(3))
            self.assertAlmostEqual((times[frame]-.004)/(DT/3),frame)

    def test_duplicate_source_row_uses_last_whole_row(self):
        regular=np.arange(160,dtype=np.float64)*DT/3
        times=np.insert(regular,0,regular[0])
        rows=list(helper.eligible_rows(times,synthetic_templates(DT)['templates'],DT,2*DT,'x',1))
        self.assertEqual(rows[0][0],1)

    def test_real_event_correlation_and_parallel_refresh(self):
        events=measured_trace()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.json';path.write_text(json.dumps(events))
            value=helper.measured_templates([path],DT)
            self.assertEqual(value['kind'],'measured_parallel')
            self.assertEqual(value['fast_queries_while_slow_running'],1)
            self.assertEqual(value['consumed_reference_versions'],2)
            self.assertAlmostEqual(value['templates'][1]['query_time'],4*DT)
            self.assertAlmostEqual(value['templates'][1]['request_time'],DT)

    def test_release_cannot_be_retroactively_dated_before_wall_ready(self):
        events=measured_trace()
        event=next(e for e in events if e['event']=='slow_completed')
        event['transfer_ready_wall_time']=100.+event['published_time']+.008
        event['publication_wall_time']=event['transfer_ready_wall_time']+.001
        with self.assertRaisesRegex(ValueError,'predates wall readiness'):
            helper.audit_wall_release(events,DT)

    def test_small_physics_commit_does_not_hide_wall_deadline_miss(self):
        events=measured_trace()
        commit=next(e for e in events if e['event']=='fast_committed')
        start=next(e for e in events if e['event']=='fast_started')
        commit['commit_wall_time']=100.+start['protected_until']
        with self.assertRaisesRegex(ValueError,'wall deadline'):
            helper.audit_wall_release(events,DT)

    def test_equal_physics_tick_still_respects_publication_event_order(self):
        events=measured_trace()
        done=next(e for e in events if e['event']=='slow_completed' and e['version']==2)
        events.remove(done);done.update(published_time=6*DT,transfer_ready_wall_time=100.+6*DT-DT/45,
                                       publication_wall_time=100.+6*DT+.001)
        query_index=next(i for i,e in enumerate(events) if e['event']=='fast_started' and e['query_id']==0)
        events.insert(query_index+1,done)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.json';path.write_text(json.dumps(events))
            value=helper.measured_templates([path],DT)
            self.assertEqual(value['templates'][0]['reference_version'],1)
            self.assertEqual(value['fast_queries_while_slow_running'],1)
            events.remove(done);events.insert(query_index,done)
            path.write_text(json.dumps(events))
            with self.assertRaisesRegex(ValueError,'latest published'):
                helper.measured_templates([path],DT)

    def test_expired_reference_query_retained_with_separate_slow_supervision(self):
        events=measured_trace();q=13*DT;quantum=DT/45;available=q-DT+30*quantum
        events.extend([dict(event='capture',exposure=q-DT,available=available,
            owner_ready_wall=100.+available-quantum,timing={'wall_timestamps':[100.+available+.001]}),
            dict(event='fast_started',query_id=2,query_time=q,reference_version=-1,
                observation_time=q-DT,protected_until=q+2*DT,submit_margin=.01),
            dict(event='fast_committed',query_id=2,commit_time=q+50*quantum,
                ready_wall_time=100.+q+49*quantum,commit_wall_time=100.+q+50*quantum+.001)])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.json';path.write_text(json.dumps(events))
            value=helper.measured_templates([path],DT)
            self.assertEqual(value['no_reference_queries'],1)
            row=value['templates'][-1]
            self.assertFalse(row['reference_available'])
            self.assertEqual(row['reference_version'],-1)
            self.assertEqual(row['slow_supervision_version'],2)
            self.assertAlmostEqual(row['query_time'],9*DT)
            self.assertAlmostEqual(row['publish_time'],3*DT)
            self.assertAlmostEqual(row['request_time'],DT)


if __name__=='__main__':unittest.main()

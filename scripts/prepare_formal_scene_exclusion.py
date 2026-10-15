#!/usr/bin/env python3
"""Read-only hash inventory of the frozen700 formal scenes for replay exclusion."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.training.data.physical_grid import sha256_file,json_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--formal-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.formal_root).resolve();out=Path(a.output)
    scenes=[];accepted={}
    for path in sorted(source.glob('*/accepted_manifest.json')):
        rows=json.loads(path.read_text());assert len(rows)==20
        accepted[str(path.relative_to(source))]=sha256_file(path)
        for row in rows:
            ep=path.parent/f"episode_{int(row['episode_index']):04d}"
            receipt=json.loads((ep/'prepared.json').read_text())
            assert sha256_file(ep/'manifest.json')==receipt['manifest_sha256']
            assert sha256_file(ep/'expert_motion.pkl')==receipt['motion_sha256']
            scenes.append(dict(task=row['task'],seed=int(row['seed']),episode_index=int(row['episode_index']),
                scene_id=f"{row['task']}:{int(row['seed'])}",scene_sha256=receipt['manifest_sha256'],motion_sha256=receipt['motion_sha256']))
    if len(scenes)!=700 or len(accepted)!=35:raise ValueError('expected original35x20 frozen scenes')
    payload=dict(schema_version=1,kind='formal700_scene_exclusion',source_root=str(source),
        accepted_manifest_sha256=accepted,scenes=scenes)
    payload['identity']=json_hash(payload)
    if out.exists() and json.loads(out.read_text())!=payload:raise FileExistsError('existing exclusion inventory differs')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha256_file(out),scenes=len(scenes))))
if __name__=='__main__':main()

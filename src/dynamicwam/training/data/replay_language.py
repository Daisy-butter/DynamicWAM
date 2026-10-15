"""Exact recorded-instruction lookup for replay; never substitute a paraphrase."""
from pathlib import Path
import json
import yaml
from dynamicwam.language import generate_domino_language_prompts
from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS
from dynamicwam.training.data.physical_grid import json_hash,sha256_file

def build_replay_language_catalog(packed_root):
    packed=Path(packed_root);project=packed.resolve().parents[2]
    code_root=Path(__file__).resolve().parents[4]
    profile=code_root/'configs/absolute_motion_v2.yaml';cfg=yaml.safe_load(profile.read_text())
    lang_path=packed/'lang/lang.json';bank=json.loads(lang_path.read_text())
    prefix=cfg['inference']['scene_prefix'];groups={};sources={}
    for group in bank['groups']:
        task=group['task_name'];index=OFFICIAL_LEVEL1_TASKS.index(task)
        scene=project/'data/raw'/group['split']/task/cfg['collection']['clean_config_name']/'scene_info.json'
        prompts=generate_domino_language_prompts(domino_root=project/'external/DOMINO',scene_info_path=scene,
            task=task,expected_episodes=cfg['collection']['clean_episodes_per_task'],
            prompts_per_task=cfg['data']['language_prompts_per_task'],seed=cfg['data']['language_prompt_seed']+index,scene_prefix=prefix)
        if len(prompts)!=len(group['lang_ids']):raise ValueError('language prompt count differs from packed bank')
        rows=[]
        for position,(prompt,lang_id) in enumerate(zip(prompts,group['lang_ids'])):
            item=bank['items'][int(lang_id)]
            if item['instruction_idx']!=position or item['lang_group_id']!=group['lang_group_id']:
                raise ValueError('packed language ordering differs from canonical prompt generator')
            rows.append(dict(instruction=prompt[len(prefix):],prompt=prompt,language_index=position,
                lang_group_id=int(group['lang_group_id']),lang_id=int(lang_id)))
        groups.setdefault(task,[]).extend(rows)
        sources[str(scene)]=sha256_file(scene)
        description=project/'external/DOMINO/description/task_instruction'/f'{task}.json'
        sources[str(description)]=sha256_file(description)
    generator=project/'external/DOMINO/description/utils/generate_episode_instructions.py'
    identity=json_hash(dict(lang_metadata_sha256=sha256_file(lang_path),profile_sha256=sha256_file(profile),
        generator_sha256=sha256_file(generator),sources=sources,groups=groups))
    return dict(groups=groups,identity=identity,scene_prefix=prefix)

def match_replay_instruction(catalog,task,instruction):
    if not isinstance(instruction,str) or not instruction:raise ValueError('actual recorded instruction is missing')
    matches=[row for row in catalog['groups'].get(task,[]) if row['instruction']==instruction]
    if not matches:
        raise ValueError('actual instruction has no exact packed T5 embedding; an actual-instruction T5 cache is required')
    # Duplicate identical texts are semantically identical; preserve their first
    # deterministic original bank index, never substitute a different sentence.
    return dict(min(matches,key=lambda r:(r['lang_group_id'],r['language_index'])))

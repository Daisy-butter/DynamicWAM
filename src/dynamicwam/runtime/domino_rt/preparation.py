"""Fixed-seed, accepted-expert replay preparation for continuous DOMINO."""
from __future__ import annotations
import os
import ast
import copy
import hashlib
import pickle
import random
import sys
from pathlib import Path
from typing import Any
import numpy as np
from .motion import adopt_level1_realtime_motion


def _task_state(env: Any, task_file: Path) -> dict:
    """Only task-defined, portable scalar/array fields; never pickle SAPIEN."""
    names = {node.attr for node in ast.walk(ast.parse(task_file.read_text()))
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == 'self' and isinstance(node.ctx, ast.Store)}
    state = {}
    for name in sorted(names):
        if not hasattr(env, name):
            continue
        value = getattr(env, name)
        if type(value).__name__ == 'ArmTag':
            state[name] = dict(kind='ArmTag', value=str(value))
        elif value is None or isinstance(value, (str, bool, int, float, np.number)):
            state[name] = dict(kind='scalar', value=copy.deepcopy(value))
        elif isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            state[name] = dict(kind='array', value=value.copy())
    return state


def _restore_task_state(env: Any, state: dict, task: str) -> None:
    for name, item in state.items():
        if item['kind'] == 'ArmTag':
            from envs.utils.action import ArmTag
            value = ArmTag(item['value'])
        elif item['kind'] in ('scalar', 'array'):
            value = copy.deepcopy(item['value'])
        else:
            raise ValueError(f'unknown portable task-state kind: {item}')
        setattr(env, name, value)
    if task in ('put_object_cabinet', 'place_object_scale') and not hasattr(env, 'arm_tag'):
        raise RuntimeError(f'{task} requires expert-selected arm_tag for check_success')


def prepare_fixed_scene(domino_root: Path, *, task: str, task_config: str,
                        seed: int, instruction_type: str,
                        record_path: Path, prepare_only: bool = False,
                        max_attempts: int = 10) -> tuple[Any, dict, dict]:
    """Prepare one exact seed, or replay one accepted expert record.

    Expert instability retries the SAME seed at most max_attempts. Formal
    replay never re-runs the planner, never substitutes a scene, and restores
    expert-owned state (notably arm_tag) before dynamic initialization.
    The pickle is a trusted pipeline artifact, not an untrusted upload.
    """
    root = domino_root.expanduser().resolve()
    os.chdir(root)
    for entry in (root, root/'script', root/'policy', root/'description'/'utils'):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import eval_policy
    from envs import CONFIGS_PATH
    config_path = Path(task_config)
    if not config_path.is_absolute():
        config_path = root/'task_config'/f'{task_config}.yml'
    args = eval_policy.yaml.load(config_path.read_text(), Loader=eval_policy.yaml.FullLoader)
    args.update(task_name=task, task_config=config_path.stem,
                ckpt_setting='continuous-wall', policy_name='domino_continuous',
                eval_mode=True, eval_video_log=False, render_freq=0)
    if not args.get('use_dynamic', False):
        raise ValueError('continuous benchmark is only defined for Level-1 dynamic tasks')
    _install_embodiment(args, CONFIGS_PATH, eval_policy.yaml)
    paths = [root/'envs'/f'{task}.py', root/'envs'/'_base_task.py',
             root/'envs'/'robot'/'robot.py', root/'script'/'eval_metrics.py', config_path]
    identity = {str(path.relative_to(root)) if path.is_relative_to(root) else str(path):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if prepare_only and record_path.exists():
        raise FileExistsError(f'accepted record is immutable: {record_path}')
    if not prepare_only:
        with record_path.open('rb') as handle:
            record = pickle.load(handle)
        if record.get('format') != 'domino_continuous_accepted_scene' or record.get('version') != 1:
            raise ValueError('unsupported accepted-record format')
        if (record['task'], record['seed'], record['instruction_type']) != (task, seed, instruction_type):
            raise ValueError('accepted-record scene or instruction identity mismatch')
        if record['source_sha256'] != identity:
            raise ValueError('DOMINO source/config changed since accepted scene preparation')
    else:
        record = None
    env = None
    failures = []
    for attempt in range(max_attempts if prepare_only else 1):
        env = eval_policy.class_decorator(task)
        try:
            if prepare_only:
                env.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **args)
                episode_info = env.play_once()
                accepted = bool(env.plan_success and env.check_success())
                saved = copy.deepcopy(getattr(env, '_saved_dynamic_motion_info', None))
                owned = _task_state(env, root/'envs'/f'{task}.py')
                env.close_env()
                if not accepted:
                    raise RuntimeError('expert plan or task success check failed')
            else:
                saved = copy.deepcopy(record['saved_motion']); owned = record['task_state']
            env.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **args)
            env._saved_dynamic_motion_info = saved
            if prepare_only:
                # Same env object as native: setup may intentionally overwrite
                # some expert fields. Save the post-setup values, not stale ones.
                owned = _task_state(env, root/'envs'/f'{task}.py')
            _restore_task_state(env, owned, task)
            if prepare_only:
                descriptions = eval_policy.generate_episode_descriptions(task, [episode_info['info']], 1)
                instruction = str(np.random.choice(descriptions[0][instruction_type]))
                init_rng = dict(numpy=np.random.get_state(), python=random.getstate())
            else:
                instruction = record['instruction']; init_rng = record['initialization_rng']
                np.random.set_state(init_rng['numpy']); random.setstate(init_rng['python'])
            env.set_instruction(instruction=instruction)
            if not env.init_dynamic_motion_for_eval():
                raise RuntimeError('dynamic motion initialization rejected fixed scene')
            velocity = adopt_level1_realtime_motion(env)
            if abs(float(env.scene.get_timestep())-.004) > 1e-8:
                raise RuntimeError('DOMINO physics timestep is not 4 ms')
            env.take_action_cnt = 0
            env._scene_step_clock.step_index = 0; env._scene_step_clock.time_seconds = 0.
            dyn = env.get_dynamic_motion_config() or {}
            actor = dyn.get('check_z_actor', dyn.get('target_actor'))
            threshold = dyn.get('check_z_threshold')
            prepared = dict(instruction=instruction, seed=seed, velocity=velocity,
                stop_on_contact=bool(dyn.get('stop_on_contact', True)), check_z_threshold=threshold,
                lift_actor=actor, start_z=None if threshold is None else float(actor.get_pose().p[2]))
            target_position = np.array(env._eval_target_actor.actor.get_pose().p, copy=True)
            if prepare_only:
                record = dict(format='domino_continuous_accepted_scene', version=1,
                    task=task, seed=int(seed), task_config=config_path.stem, instruction_type=instruction_type,
                    instruction=instruction, saved_motion=saved, task_state=owned,
                    initialization_rng=init_rng, source_sha256=identity,
                    target_position=target_position, velocity=velocity,
                    attempt=attempt+1, preceding_failures=failures,
                    control_period_s=.1, physics_dt=.004)
                record_path.parent.mkdir(parents=True, exist_ok=True)
                with record_path.open('xb') as handle:
                    pickle.dump(record, handle, protocol=4)
            elif not np.allclose(target_position, record['target_position'], rtol=0., atol=1e-7):
                raise RuntimeError('replayed target start pose differs from accepted scene')
            return env, prepared, record
        except BaseException as exc:
            failures.append(dict(attempt=attempt+1, seed=seed, error=repr(exc)))
            print(f'FIXED-SCENE-RETRY task={task} seed={seed} attempt={attempt+1}: {exc}', flush=True)
            if env is not None:
                try:
                    env.close_env(); env.release_episode_resources()
                except Exception:
                    pass
            env = None
            if not prepare_only or attempt+1 >= max_attempts:
                raise RuntimeError(f'fixed scene {task}:{seed} failed; no replacement seed: {failures}') from exc
    raise AssertionError('unreachable')

def _install_embodiment(args: dict[str, Any], configs_path: str, yaml_module: Any) -> None:
    embodiment_type = args.get("embodiment")
    with open(os.path.join(configs_path, "_embodiment_config.yml"), encoding="utf-8") as handle:
        embodiments = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    with open(os.path.join(configs_path, "_camera_config.yml"), encoding="utf-8") as handle:
        cameras = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head_type]["h"]
    args["head_camera_w"] = cameras[head_type]["w"]

    def embodiment_file(name: str) -> str:
        robot_file = embodiments[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"embodiment {name} has no file")
        return str(robot_file)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment must contain 1 or 3 items")
    args["left_embodiment_config"] = _read_yaml(
        os.path.join(args["left_robot_file"], "config.yml"),
        yaml_module,
    )
    args["right_embodiment_config"] = _read_yaml(
        os.path.join(args["right_robot_file"], "config.yml"),
        yaml_module,
    )

def _read_yaml(path: str, yaml_module: Any) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"expected a mapping in {path}")
    return loaded

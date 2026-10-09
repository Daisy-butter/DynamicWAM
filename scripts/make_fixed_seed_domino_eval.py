#!/usr/bin/env python3
"""Generate an isolated, audited DOMINO evaluator for an explicit seed manifest.

The pinned input is the existing evaluator downloaded on 2026-10-05. Source
drift must be reviewed; this tool deliberately does not guess new patch sites.
The original evaluator, policy, expert checks and instruction selection are
unchanged except for bounded retries and traversal of the supplied seed list.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "fec0002fe1e30833b225d3559b09b583a9a4f7a22c29bc54a3817eb70d1aea38"
MAX_ATTEMPTS_PER_SEED = 10

VALIDATOR = '''def _validate_fixed_episode_seeds(seeds, test_num, st_seed):
    if not isinstance(seeds, list) or len(seeds) != test_num:
        raise ValueError("fixed_episode_seeds must be a list with exactly test_num entries")
    if any(type(seed) is not int or seed < st_seed for seed in seeds):
        raise ValueError("fixed_episode_seeds must contain integers >= start_seed (no bools)")
    if seeds != sorted(set(seeds)):
        raise ValueError("fixed_episode_seeds must be unique and strictly ascending")
    return list(seeds)


'''


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(source: Path) -> tuple[str, dict]:
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if digest(text) != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "DOMINO evaluator source changed; review before regenerating. "
            f"Expected normalized SHA256 {EXPECTED_SOURCE_SHA256}, got {digest(text)}"
        )
    changes = []

    def replace(label: str, old: str, new: str, count: int = 1) -> None:
        nonlocal text
        found = text.count(old)
        if found != count:
            raise ValueError(f"{label}: expected {count} exact patch sites, found {found}")
        changes.append(dict(name=label, source_fragment_sha256=digest(old),
                            replacement_sha256=digest(new), replacements=found))
        text = text.replace(old, new)

    # Python otherwise searches for eval_metrics beside the generated copy.
    # Config lookup must likewise remain relative to the ORIGINAL script.
    replace("original_script_import_directory", 'sys.path.append("./")',
            f"sys.path.insert(0, {str(source.parent)!r})\nsys.path.append(\"./\")")
    replace("original_config_directory", "current_file_path = os.path.abspath(__file__)",
            f"current_file_path = {str(source)!r}")
    replace("manifest_validator", "def main(usr_args):\n", VALIDATOR + "def main(usr_args):\n")
    replace("forward_manifest_to_environment_args", "    topk = 1\n\n    model = get_model(usr_args)",
            '    args["fixed_episode_seeds"] = _validate_fixed_episode_seeds(\n'
            '        usr_args.get("fixed_episode_seeds"), test_num, st_seed\n'
            '    )\n    topk = 1\n\n    model = get_model(usr_args)')
    replace("initialize_manifest_tracking", "    suc_test_seed_list = []\n",
            '    suc_test_seed_list = []\n'
            '    fixed_episode_seeds = _validate_fixed_episode_seeds(\n'
            '        args.get("fixed_episode_seeds"), test_num, st_seed\n'
            '    )\n'
            '    fixed_seed_attempts = {}\n'
            '    fixed_reported_seed_list = []\n')
    replace("bounded_same_seed_attempt", "    while succ_seed < test_num:\n",
            '    while succ_seed < test_num:\n'
            '        now_seed = fixed_episode_seeds[succ_seed]\n'
            '        attempt = fixed_seed_attempts.get(now_seed, 0) + 1\n'
            f'        if attempt > {MAX_ATTEMPTS_PER_SEED}:\n'
            '            raise RuntimeError(\n'
            f'                f"Fixed episode seed {{now_seed}} exhausted {MAX_ATTEMPTS_PER_SEED} attempts; "\n'
            '                "refusing to substitute another seed"\n'
            '            )\n'
            '        fixed_seed_attempts[now_seed] = attempt\n'
            '        print("FIXED-SEED-ATTEMPT " + json.dumps(dict(\n'
            '            episode_index=succ_seed, seed=now_seed, attempt=attempt\n'
            '        )), flush=True)\n')
    replace("expert_exception_does_not_advance_seed",
            '                now_seed += 1\n                args["render_freq"] = render_freq\n',
            '                args["render_freq"] = render_freq\n', count=2)
    replace("expert_acceptance_does_not_commit_episode",
            '        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):\n'
            '            succ_seed += 1\n'
            '            suc_test_seed_list.append(now_seed)\n'
            '        else:\n'
            '            now_seed += 1\n'
            '            args["render_freq"] = render_freq\n'
            '            continue\n',
            '        if not ((not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success())):\n'
            '            print(f"FIXED-SEED-RETRY seed={now_seed} reason=expert_plan_or_success", flush=True)\n'
            '            args["render_freq"] = render_freq\n'
            '            continue\n')
    replace("instruction_audit",
            '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n',
            '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n'
            '        print("FIXED-SEED-INSTRUCTION " + json.dumps(dict(\n'
            '            episode_index=succ_seed, seed=now_seed, attempt=attempt,\n'
            '            instruction=str(instruction)\n'
            '        ), ensure_ascii=False), flush=True)\n')
    replace("dynamic_initialization_retries_same_seed",
            '                print(f"Error: Failed to initialize dynamic motion for seed {now_seed}, skipping...")\n'
            '                TASK_ENV.close_env()\n'
            '                TASK_ENV.release_episode_resources()\n'
            '                now_seed += 1\n'
            '                succ_seed -= 1\n'
            '                continue\n',
            '                print(f"FIXED-SEED-RETRY seed={now_seed} reason=dynamic_initialization", flush=True)\n'
            '                TASK_ENV.close_env()\n'
            '                TASK_ENV.release_episode_resources()\n'
            '                continue\n')
    replace("commit_only_reported_episodes",
            '        aggregated_metrics.add_episode(episode_metrics)\n',
            '        aggregated_metrics.add_episode(episode_metrics)\n'
            '        fixed_reported_seed_list.append(now_seed)\n'
            '        suc_test_seed_list.append(now_seed)\n'
            '        succ_seed += 1\n'
            '        print("FIXED-SEED-REPORTED " + json.dumps(dict(\n'
            '            episode_index=succ_seed - 1, seed=now_seed, success=bool(succ),\n'
            '            instruction=str(instruction), attempt=attempt\n'
            '        ), ensure_ascii=False), flush=True)\n')
    replace("end_of_manifest_next_seed",
            '        now_seed += 1\n\n    # The adapter normally finalizes',
            '        now_seed = (fixed_episode_seeds[succ_seed] if succ_seed < test_num\n'
            '                    else fixed_episode_seeds[-1] + 1)\n\n'
            '    if suc_test_seed_list != fixed_episode_seeds or fixed_reported_seed_list != fixed_episode_seeds:\n'
            '        raise RuntimeError("Reported episode seeds differ from the fixed seed manifest")\n'
            '    print("FIXED-SEED-COMPLETE " + json.dumps(dict(\n'
            '        seeds=fixed_reported_seed_list, attempts=fixed_seed_attempts\n'
            '    )), flush=True)\n\n'
            '    # The adapter normally finalizes')
    ast.parse(text, filename=str(source))
    manifest = dict(
        generator_version=1, source=str(source), original_script_directory=str(source.parent),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        normalized_source_sha256=EXPECTED_SOURCE_SHA256,
        generated_sha256=digest(text), max_attempts_per_seed=MAX_ATTEMPTS_PER_SEED,
        patches=changes,
        protocol="explicit_fixed_seeds_same_seed_expert_retry_policy_failure_counts",
    )
    return text, manifest


def generate(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or output.parent == source.parent:
        raise ValueError("Generated evaluator must be outside the original script directory")
    text, manifest = build(source)
    manifest["output"] = str(output)
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if output.exists() or sidecar.exists():
        if (output.is_file() and sidecar.is_file()
                and output.read_bytes() == text.encode("utf-8")
                and sidecar.read_bytes() == payload.encode("utf-8")):
            return manifest
        raise FileExistsError("Refusing to overwrite a different or incomplete generated evaluator")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing another process's immutable copy.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

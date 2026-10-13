"""CPU-only RTC export/deployment gates; no checkpoint files or GPU work."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.inference.fast_runtime import require_rtc_deployment_mode
from dynamicwam.inference.multirate_model_loader import rtc_prefix_contract


def fixture():
    rtc = dict(version=1, mode="training_time_action_prefix", max_prefix_actions=6,
        clean_timestep=0.0, loss="per_sample_suffix_mean", includes_zero_prefix=True,
        queue_clock="absolute_action_effect_times", overflow="reject_and_refresh",
        execute_suffix_actions=4)
    return dict(modules=dict(architecture="native", native_multirate=True,
        action_expert=dict(chunk_size=16), runtime_contract=dict(rtc=rtc)),
        config=dict(runtime_contract=dict(rtc=deepcopy(rtc)),
            training_recipe=dict(training=dict(rtc_prefix_max=6))))


def rejected(fn, label):
    try:
        fn()
    except (RuntimeError, TypeError):
        return
    raise AssertionError("expected rejection: " + label)


def run():
    payload = fixture()
    rejected(lambda: require_rtc_deployment_mode(payload), "legacy deployment default")
    rtc = require_rtc_deployment_mode(payload, allow_zero_prefix_ablation=True)
    assert rtc["max_prefix_actions"] == 6
    rtc["max_prefix_actions"] = 1
    assert rtc_prefix_contract(payload)["max_prefix_actions"] == 6
    rejected(lambda: require_rtc_deployment_mode(payload, allow_zero_prefix_ablation="true"),
             "truthy string does not authorize ablation")
    fast = dict(modules=deepcopy(payload["modules"]))
    rejected(lambda: require_rtc_deployment_mode(fast), "fast-only export gate")
    assert require_rtc_deployment_mode(fast, allow_zero_prefix_ablation=True)["max_prefix_actions"] == 6
    broken = deepcopy(payload)
    del broken["modules"]["runtime_contract"]["rtc"]
    rejected(lambda: rtc_prefix_contract(broken), "missing module RTC metadata")
    broken = deepcopy(payload)
    del broken["config"]["runtime_contract"]["rtc"]
    rejected(lambda: rtc_prefix_contract(broken), "missing full RTC metadata")
    broken = deepcopy(payload)
    del broken["modules"]["runtime_contract"]["rtc"]
    del broken["config"]["runtime_contract"]["rtc"]
    rejected(lambda: rtc_prefix_contract(broken), "training recipe cannot lose RTC metadata")
    for field, value in (("max_prefix_actions", 16), ("max_prefix_actions", -1),
                         ("max_prefix_actions", 6.0), ("queue_clock", "guessed_fixed_dt"),
                         ("clean_timestep", 1000.), ("includes_zero_prefix", False)):
        broken = deepcopy(payload)
        broken["modules"]["runtime_contract"]["rtc"][field] = value
        broken["config"]["runtime_contract"]["rtc"][field] = value
        rejected(lambda: rtc_prefix_contract(broken), field)
    assert require_rtc_deployment_mode({"modules": {}}) is None
    print({"passed": True, "checks": 14,
        "scope": "CPU RTC contract validation; controller support is not claimed"})


if __name__ == "__main__":
    run()

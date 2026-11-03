import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_feedback_reuse import inactive_loader_equivalent


def test_only_inactive_checkpoint_branch_may_change():
    old="def child(config):\n    setup(1)\n    if config.get('checkpoint'):\n        load_old(config)\n    predict()\n"
    new="def load_trained_state(model, config):\n    verified_load(model,config)\ndef child(config):\n    setup(1)\n    if config.get('checkpoint'):\n        load_trained_state(model,config)\n    predict()\n"
    assert inactive_loader_equivalent(old,new)
    assert not inactive_loader_equivalent(old,new.replace('setup(1)','setup(2)'))
    assert not inactive_loader_equivalent(old,new.replace('predict()','predict(steps=1)'))

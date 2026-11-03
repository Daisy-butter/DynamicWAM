import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_v9_slow_service import load_trained_state
from dynamicwam.dualrate.data import sha256_file


@pytest.mark.parametrize('as_string',[True,False])
def test_trained_child_loads_exact_state_from_json_string_or_path(tmp_path,as_string):
    source=torch.nn.Linear(3,2);target=torch.nn.Linear(3,2)
    with torch.no_grad():source.weight.fill_(.375);source.bias.fill_(-.25)
    p=tmp_path/'trained.pt';torch.save({'model':source.state_dict()},p)
    load_trained_state(target,dict(checkpoint=str(p) if as_string else p,checkpoint_sha256=sha256_file(p)))
    for k,v in source.state_dict().items():assert torch.equal(v,target.state_dict()[k])


def test_bad_digest_never_loads_or_falls_back(tmp_path):
    source=torch.nn.Linear(3,2);p=tmp_path/'trained.pt';torch.save({'model':source.state_dict()},p)
    target=torch.nn.Linear(3,2);old=target.weight.detach().clone()
    with pytest.raises(ValueError,match='changed'):load_trained_state(target,dict(checkpoint=str(p),checkpoint_sha256='wrong'))
    assert torch.equal(old,target.weight)


def test_bad_model_schema_is_strict(tmp_path):
    p=tmp_path/'bad.pt';torch.save({'model':{'wrong':torch.zeros(1)}},p)
    with pytest.raises(RuntimeError):load_trained_state(torch.nn.Linear(3,2),dict(checkpoint=str(p),checkpoint_sha256=sha256_file(p)))

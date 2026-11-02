"""Prevent a clean diagnostic or unrelated source from authorizing training."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import guarded_dualrate_run as guard


def test_unresolved_tracking_defect_cannot_authorize_formal(tmp_path):
    config={'residual_v10':{'deterministic':True}}
    source={'source_sha256':'new-controller'}
    with pytest.raises(ValueError,match='absent'):
        guard.validate_tracking_review(config,{},source)
    evidence=tmp_path/'physical.json';evidence.write_text('{"complete": true}')
    review=tmp_path/'review.json'
    data=dict(passed=False,source_sha256='new-controller',production_contacts_enabled=True,
        baseline_comparison_complete=True,evidence=[dict(path=str(evidence),sha256=guard.base.file_sha256(evidence))])
    def receipt():
        review.write_text(json.dumps(data))
        return {'actuator_tracking_review':dict(path=str(review),sha256=guard.base.file_sha256(review))}
    with pytest.raises(ValueError,match='unresolved'):
        guard.validate_tracking_review(config,receipt(),source)
    data['passed']=True;data['source_sha256']='old-controller'
    with pytest.raises(ValueError,match='unresolved'):
        guard.validate_tracking_review(config,receipt(),source)
    data['source_sha256']='new-controller'
    guard.validate_tracking_review(config,receipt(),source)
    r=receipt();evidence.write_text('{"complete": false}')
    with pytest.raises(ValueError,match='evidence changed'):
        guard.validate_tracking_review(config,r,source)

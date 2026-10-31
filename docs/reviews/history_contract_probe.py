"""Read-only CPU counterexample for training and native deployment history spans."""
import json
from types import SimpleNamespace
import numpy as np
from dynamicwam.dualrate.online import history_endpoints
from dynamicwam.dualrate.producer import flow_history_selection

# Same latest physical time; raw expert exposures are 60ms apart. Native
# TOPP is permitted to take 495ms per waypoint. No renderer or GPU is used.
raw_times = np.arange(199, dtype=np.float64) * .06
anchor = float(raw_times[-1])
frames = tuple(SimpleNamespace(exposure=float(t), available=float(t), frame_id=i)
               for i, t in enumerate(np.linspace(0., anchor, 25)))
selected, matched = history_endpoints(frames, native_ordinal=True)
indices, valid = flow_history_selection(raw_times, len(raw_times)-1, interval_seconds=.72)
report = dict(kind='deterministic_counterexample_not_measured_episode',
              anchor_seconds=anchor,
              training_endpoints=raw_times[indices].tolist(), training_valid=list(map(bool, valid)),
              deployment_endpoints=[x.exposure for x in selected], deployment_valid=list(matched),
              training_interval_seconds=np.diff(raw_times[indices]).tolist(),
              deployment_interval_seconds=np.diff([x.exposure for x in selected]).tolist(),
              conclusion='Real dt is preserved, but history span is not equivalent under variable-duration waypoints.')
print(json.dumps(report, indent=2))

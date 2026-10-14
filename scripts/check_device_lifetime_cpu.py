"""CPU mocks only; never opens a GPU device."""
import ast
from pathlib import Path
from unittest.mock import patch
import device_lifetime_guard as module

probe = Path(__file__).with_name('probe_multigpu_teardown.py')
ast.parse(probe.read_text())
for failing_log in (False, True):
    def record(events):
        if failing_log:
            raise OSError('simulated log failure')
    guard = module.DeviceLifetimeGuard(['a', 'b'], record)
    guard.handles = [(123, 'a', '/dev/mock0'), (124, 'b', '/dev/mock1')]
    closed = []
    with patch.object(module.os, 'close', side_effect=closed.append):
        try:
            guard.close()
        except OSError:
            assert failing_log
        else:
            assert not failing_log
        guard.close()
    assert closed == [124, 123] and not guard.handles

guard = module.DeviceLifetimeGuard([], lambda events: None)
guard.handles = [(123, 'a', '/dev/mock0'), (124, 'b', '/dev/mock1')]
closed = []
def fail_close(fd):
    closed.append(fd)
    if fd == 124:
        raise OSError('simulated ambiguous close')
with patch.object(module.os, 'close', side_effect=fail_close):
    try:
        guard.close()
    except OSError:
        pass
    else:
        raise AssertionError('close failure swallowed')
    guard.close()
assert closed == [124, 123] and not guard.handles
print('PASS: syntax, all guards close despite logging/close failure, no double close; CPU only')

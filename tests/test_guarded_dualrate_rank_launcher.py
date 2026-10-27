"""Real eight-child CPU session/failure cleanup; never imports or uses CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dualrate_rank_launcher.py"
spec = importlib.util.spec_from_file_location("dualrate_rank_launcher_cpu_test", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

DUMMY = r'''
import json,os,signal,sys,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(os.environ['RANK']);fail=int(sys.argv[2])
def stop(sig,frame):
    (root/f'{rank}.signal').write_text(str(sig));raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
(root/f'{rank}.started').write_text(json.dumps(dict(rank=rank,pid=os.getpid(),sid=os.getsid(0),
    pgrp=os.getpgrp(),local_rank=os.environ['LOCAL_RANK'],world=os.environ['WORLD_SIZE'])))
deadline=time.monotonic()+15
while len(list(root.glob('*.started')))<8:
    if time.monotonic()>deadline:raise SystemExit(98)
    time.sleep(.01)
if fail<0:raise SystemExit(0)
if rank==fail:raise SystemExit(17)
while True:time.sleep(.05)
'''


@pytest.mark.skipif(os.name != "posix", reason="Linux process/session contract")
@pytest.mark.parametrize("failed_rank", [-1, 3])
def test_eight_actual_children_keep_session_and_failure_leaves_no_rank(failed_rank):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                    start_new_session=True, stdin=subprocess.DEVNULL)
        calls, events = [], []
        def factory(argv, **kwargs):
            calls.append((argv, kwargs["env"].copy(), kwargs["start_new_session"]))
            assert kwargs["start_new_session"] is False
            return subprocess.Popen([sys.executable,"-c",DUMMY,folder,str(failed_rank)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        try:
            result = launcher.run_rank_group(root/"config.json", root/"authorization.json",
                popen_factory=factory, emit=events.append, poll_seconds=.01, grace_seconds=2.)
            assert result["passed"] is (failed_rank < 0)
            assert len(result["ranks"]) == len(calls) == 8
            rows = [json.loads((root/f"{rank}.started").read_text()) for rank in range(8)]
            assert {row["sid"] for row in rows} == {os.getsid(0)}
            assert {row["pgrp"] for row in rows} == {os.getpgrp()}
            assert all(row["world"] == "8" and row["local_rank"] == str(row["rank"]) for row in rows)
            assert len({env["MASTER_PORT"] for _,env,_ in calls}) == 1
            assert all(env["MASTER_ADDR"] == "127.0.0.1" for _,env,_ in calls)
            assert all(argv[1:4] == ["-B","-m","dynamicwam.dualrate.train"] for argv,_,_ in calls)
            assert all(item["returncode"] is not None for item in result["ranks"])
            assert all(not Path(f"/proc/{row['pid']}").exists() for row in rows)
            assert sentinel.poll() is None, "unrelated session/process was signalled"
            if failed_rank >= 0:
                assert result["ranks"][failed_rank]["returncode"] == 17
                assert len(list(root.glob("*.signal"))) == 7
                assert "without retry" in result["error"]
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=10)


def test_launcher_module_has_no_torch_import_or_elastic_dependency():
    result = subprocess.run([sys.executable,"-c",
        "import runpy,sys;runpy.run_path(sys.argv[1],run_name='audit_import');"
        "assert 'torch' not in sys.modules;assert 'pynvml' not in sys.modules",str(SCRIPT)],
        capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stderr

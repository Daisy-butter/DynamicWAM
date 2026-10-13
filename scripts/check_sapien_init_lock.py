"""Scene preparation may overlap across GPUs; native contexts remain serialized."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.runtime.domino_continuous.sapien_init_lock import SapienInitLock, fcntl


class Locks(unittest.TestCase):
    def test_gpu_scopes_and_shared_context_scope(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'TMPDIR':tmp}):
            os.environ.pop('DOMINO_SAPIEN_INIT_LOCK',None)
            with patch.dict(os.environ,{'DYNAMICWAM_GPU_PCI':'00000000:1B:00.0'}):
                a=SapienInitLock().path;c=SapienInitLock(device_context=True).path
            with patch.dict(os.environ,{'DYNAMICWAM_GPU_PCI':'00000000:1C:00.0'}):
                self.assertNotEqual(a,SapienInitLock().path)
                self.assertEqual(c,SapienInitLock(device_context=True).path)
                self.assertNotEqual(c,SapienInitLock().path)

    @unittest.skipIf(fcntl is None,'POSIX file locks')
    def test_actual_lock_isolation_and_context_exclusion(self):
        code='''from dynamicwam.runtime.domino_continuous.sapien_init_lock import SapienInitLock
import sys
try:
 with SapienInitLock(device_context=sys.argv[1]=='context', timeout_s=.05):pass
except TimeoutError:sys.exit(2)
'''
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'TMPDIR':tmp,'DYNAMICWAM_GPU_PCI':'00000000:1B:00.0'}):
            os.environ.pop('DOMINO_SAPIEN_INIT_LOCK',None)
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),DYNAMICWAM_GPU_PCI='00000000:1C:00.0')
            with SapienInitLock():
                result=subprocess.run([sys.executable,'-c',code,'scene'],env=env,timeout=5)
                self.assertEqual(result.returncode,0)
            with SapienInitLock(device_context=True):
                result=subprocess.run([sys.executable,'-c',code,'context'],env=env,timeout=5)
                self.assertEqual(result.returncode,2)


if __name__=='__main__':unittest.main()

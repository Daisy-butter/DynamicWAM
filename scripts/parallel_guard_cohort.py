"""CPU-only startup/drain barrier for jointly owned GPU workers.
All guards retain device descriptors until every peer worker has exited. Idle
checks stay strict: peer graphics contexts are never exempted from telemetry.
"""
import json,time,os
from pathlib import Path
from guarded_dualrate_smoke import atomic,proc_identity

class ParallelGuardCohort:
    def __init__(self,config,storage):
        self.root=Path(config['directory']).resolve(strict=True)
        if not self.root.is_relative_to(Path(storage).resolve()):raise ValueError('cohort outside owned storage')
        spec=json.loads((self.root/'manifest.json').read_text())
        self.members=spec['members'];self.member=config['member']
        if self.member not in self.members or len(set(self.members))!=len(self.members):raise ValueError('cohort membership')
    def abort(self,error):
        try:
            with (self.root/'abort.json').open('x') as f:json.dump(dict(member=self.member,error=str(error)),f)
        except FileExistsError:pass
    def barrier(self,phase,*,timeout,honor_abort=True):
        if phase not in ('ready','drained'):raise ValueError('unknown guard barrier')
        directory=self.root/phase
        directory.mkdir(parents=True,exist_ok=True)
        atomic(directory/(self.member+'.json'),dict(owner=proc_identity(os.getpid()),phase=phase))
        deadline=time.monotonic()+timeout
        while True:
            if honor_abort and (self.root/'abort.json').exists():raise RuntimeError('parallel GPU cohort aborted')
            if all((directory/(member+'.json')).is_file() for member in self.members):return
            if time.monotonic()>deadline:raise TimeoutError('parallel guard '+phase+' barrier')
            time.sleep(.1)

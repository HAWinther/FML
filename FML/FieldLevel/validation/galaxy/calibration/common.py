"""Frozen-study identity, atomic receipts and conservative sampling-time accounting."""
from pathlib import Path
import hashlib
import json
import math
import os

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)

class Budget:
    """Charge a whole batch reservation before launch; crashed batches get no refund."""
    def __init__(self,path,limit,reservation):
        self.path=Path(path);self.limit=float(limit);self.reservation=float(reservation)
        if not all(math.isfinite(v) and v>0 for v in (self.limit,self.reservation)) or self.reservation>self.limit:
            raise ValueError('Positive finite budget >= batch reservation required')
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if (self.state['limit'],self.state['reservation'])!=(self.limit,self.reservation):
                raise ValueError('Sampling time budget changed')
            if self.state['pending']:
                self.state['unrefunded_crashed_batches']+=1;self.state['pending']=False
        else:self.state=dict(limit=self.limit,reservation=self.reservation,spent=0.,pending=False,unrefunded_crashed_batches=0,batches=0)
        if not math.isfinite(self.state['spent']) or self.state['spent']<0:raise ValueError('Invalid budget ledger')
        write(self.path,self.state)
    def reserve(self):
        if self.state['pending']:raise ValueError('Batch already pending')
        if self.state['spent']+self.reservation>self.limit:return False
        self.state['spent']+=self.reservation;self.state['pending']=True;write(self.path,self.state);return True
    def finish(self,seconds):
        if not self.state['pending'] or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid batch completion')
        self.state['spent']+=seconds-self.reservation;self.state['pending']=False;self.state['batches']+=1;write(self.path,self.state)

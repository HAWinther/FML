#!/usr/bin/env python3
"""Prepare the pinned public Box-1 catalog; never invent missing initial conditions."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,urllib.request
HERE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def write(path,data):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def verify_catalog(path,plan):
 if path.stat().st_size!=plan['bytes'] or sha(path)!=plan['sha256']:raise ValueError('Catalog differs from pinned Box 1')
 h=hashlib.sha1(('blob '+str(plan['bytes'])+'\0').encode())
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if h.hexdigest()!=plan['git_blob_sha1']:raise ValueError('Git blob identity differs')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter',type=Path,default=HERE.parent/'catalog_hdf5.py');a=p.parse_args()
 plan=json.loads((HERE/'plan.json').read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);exe=a.exe.resolve();adapter=a.adapter.resolve();catalog=a.input.resolve() if a.input else out/'input'/Path(plan['catalog']).name
 url='https://raw.githubusercontent.com/ANSalcedo/Beyond2ptMock/'+plan['commit']+'/'+plan['catalog']
 if not catalog.exists():
  if a.input:raise FileNotFoundError(catalog)
  catalog.parent.mkdir(exist_ok=True);temporary=catalog.with_suffix('.partial')
  with urllib.request.urlopen(url,timeout=120) as src,temporary.open('wb') as dst:
   count=0
   while True:
    block=src.read(1024*1024)
    if not block:break
    count+=len(block)
    if count>plan['bytes']:raise ValueError('Download exceeds pinned length')
    dst.write(block)
  verify_catalog(temporary,plan);temporary.replace(catalog)
 verify_catalog(catalog,plan)
 identity=dict(catalog=str(catalog),catalog_sha256=sha(catalog),exe=str(exe),exe_sha256=sha(exe),adapter=str(adapter),adapter_sha256=sha(adapter),plan=plan,source_sha256={n:sha(HERE/n) for n in ['prepare.py','analyze.py','plan.json']})
 if (out/'INPUTS.json').exists():
  if json.loads((out/'INPUTS.json').read_text())!=identity:raise ValueError('Changed application inputs; use a fresh output directory')
 else:write(out/'INPUTS.json',identity)
 for n in plan['meshes']:
  directory=out/f'modes-{n}';directory.mkdir(exist_ok=True);receipt=directory/'completed.json'
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if not old['complete'] or any(sha(Path(path))!=h for path,h in old['hashes'].items()):raise ValueError('Changed completed mode extraction')
   print('Verified completed extraction',n,flush=True);continue
  prefix=directory/'observed'
  if prefix.with_suffix('.json').exists() or Path(str(prefix)+'.rank0.modes.csv').exists():raise ValueError('Incomplete extraction exists; inspect it before retrying in a fresh directory')
  command=[sys.executable,str(adapter),'--input',str(catalog),'--exe',str(exe),'--output-prefix',str(prefix),'--mesh',str(n),'--boxsize',str(plan['boxsize']),'--cutoff',str(max(plan['cutoffs'])),'--geometry',plan['geometry'],'--epsilon',str(plan['epsilon']),'--batch-rows',str(plan['batch_rows'])]
  start=time.monotonic()
  with (directory/'extract.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
  files=[prefix.with_suffix('.json'),Path(str(prefix)+'.rank0.modes.csv')]
  write(receipt,dict(complete=True,mesh=n,seconds=time.monotonic()-start,command=command,hashes={str(x):sha(x) for x in files}));print('Extracted catalog mesh',n,flush=True)
 import analyze
 analyze.run(out)
if __name__=='__main__':main()

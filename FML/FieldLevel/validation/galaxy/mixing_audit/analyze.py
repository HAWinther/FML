#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json,hashlib,os
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description='Report all predeclared dataset-3 geometry and exchange diagnostics');ap.add_argument('audit',type=Path);args=ap.parse_args();out=args.audit.resolve();f=json.loads((out/'FROZEN.json').read_text());plan=json.loads((out/'plan.json').read_text());source=Path(f['source'])
 if sha(out/'plan.json')!=f['plan_sha256']:raise ValueError('Frozen audit plan changed')
 for group in ['source_hashes','implementation']:
  for path,digest in f[group].items():
   if sha(path)!=digest:raise ValueError('Source changed: '+path)
 for name,digest in f['inputs'].items():
  if sha(out/name)!=digest:raise ValueError('Frozen geometry input changed')
 geometry=[];controls=[]
 for draw in plan['endpoint_draws']:
  rows=list(csv.DictReader((out/f'results{draw}.csv').open()));assert len(rows)==1010
  paths=list(dict.fromkeys(r['path'] for r in rows))
  for name in paths:
   values=[r for r in rows if r['path']==name];assert len(values)==101 and [int(v['index']) for v in values]==list(range(101))
   direct=np.array([float(r['u_direct']) for r in values]);profile=np.array([float(r['u_profile']) for r in values]);q=np.array([float(r['quadratic_profile']) for r in values])
   if name=='rescale-control':
    variation=float(np.ptp(q));assert variation<1e-8;controls.append(dict(draw=draw,max_profile_quadratic_variation=variation));continue
   geometry.append(dict(draw=draw,path=name,direct_barrier_above_higher_endpoint=float(direct.max()-max(direct[0],direct[-1])),profiled_barrier_above_higher_endpoint=float(profile.max()-max(profile[0],profile[-1])),profiled_endpoint_gap=float(profile[-1]-profile[0]),max_kkt=max(float(v['kkt']) for v in values),max_profile_full_error=max(float(v['profile_full_error']) for v in values)))
 transport=[];traces=[]
 for e in range(4):
  summary=json.loads((source/f'ensemble{e}/summary.json').read_text());betas=np.array(summary['betas']);edges=len(betas)-1;attempt=np.zeros(edges,dtype=int);accepted=attempt.copy();loga=[[] for _ in range(edges)]
  for draw in range(2048):
   row=json.loads((source/f'ensemble{e}/draws/draw-{draw}.rank-0.json').read_text());q=np.array(row['quadratic']);final=q.copy()
   if row['draw']!=draw or len(row['exchange_trace'])!=4:raise ValueError('Unexpected transport record')
   # Reverse only accepted swaps to obtain the energies before each attempted pass.
   for event in reversed(row['exchange_trace']):
    pb=np.array(event['accepted'],dtype=int);pa=np.array(event['attempts'],dtype=int)
    for j in np.flatnonzero(pb):q[j],q[j+1]=q[j+1],q[j]
    for j in np.flatnonzero(pa):loga[j].append(float((betas[j]-betas[j+1])*(q[j]-q[j+1])))
    attempt+=pa;accepted+=pb
   for event in row['exchange_trace']:
    for j in np.flatnonzero(event['accepted']):q[j],q[j+1]=q[j+1],q[j]
   if not np.array_equal(q,final):raise ValueError('Energy reverse/forward audit failed')
  assert attempt.tolist()==summary['swap_attempts'] and accepted.tolist()==summary['swap_accepted']
  edge_rows=[]
  for j,values in enumerate(loga):
   x=np.array(values);edge_rows.append(dict(edge=j,betas=[float(betas[j]),float(betas[j+1])],attempts=int(attempt[j]),accepted=int(accepted[j]),acceptance=float(accepted[j]/attempt[j]),conditional_mean_acceptance_probability=float(np.exp(np.minimum(0,x)).mean()),log_acceptance_q05_q50_q95=np.quantile(x,[.05,.5,.95]).tolist()))
  x=np.loadtxt(source/f'calibration.chain{e}.csv',delimiter=',',skiprows=1);assert x.shape==(2048,13) and np.array_equal(x[:,0],np.arange(2048));traces.append(x[:,1:])
  transport.append(dict(ensemble=e,roundtrips=sum(summary['roundtrips']),edges=edge_rows,projection_means=x[:,6:12].mean(axis=0).tolist(),projection_sds=x[:,6:12].std(axis=0,ddof=1).tolist()))
 result=dict(scope=plan['scope'],geometry=geometry,rescaling_controls=controls,transport=transport,all_original_records_unchanged=True,profile_is_not_marginalization=True,limits=plan['limits'])
 (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');plot(out,geometry,transport,np.stack(traces),plan)
 print(json.dumps(result,indent=2))

def plot(out,geometry,transport,traces,plan):
 os.environ.setdefault('MPLCONFIGDIR',str(out/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(out/'cache'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=plt.get_cmap('tab10').colors
 for item in transport:
  e=item['ensemble'];axes[0,0].plot(range(10),[r['acceptance'] for r in item['edges']],'o-',ms=3,color=colors[e],label=f"Ensemble {e}: {item['roundtrips']} trips")
  axes[1,0].plot(traces[e,:,10],lw=.6,color=colors[e],label=f'Ensemble {e}')
 axes[0,0].axhline(.05,ls='--',color='black',lw=.8);axes[0,0].set(xlabel='Temperature edge (0 = cold end)',ylabel='Observed exchange acceptance');axes[0,0].legend(fontsize=8)
 axes[1,0].set(xlabel='Retained draw',ylabel='sin001 white-field projection',title='Cold-chain separation (original failed run)')
 for k,draw in enumerate(plan['endpoint_draws']):
  ax=axes[k,1];rows=list(csv.DictReader((out/f'results{draw}.csv').open()))
  for n,name in enumerate(dict.fromkeys(r['path'] for r in rows if r['path']!='rescale-control')):
   values=[r for r in rows if r['path']==name];u=np.array([float(v['u_profile']) for v in values]);ax.plot(np.linspace(0,1,101),u-max(u[0],u[-1]),lw=1,label=name,color=plt.get_cmap('tab20').colors[n])
  ax.set(xlabel='Path fraction',ylabel='Profiled U minus higher endpoint',title=f'Draw {draw}: selected routes from ensemble 1');ax.legend(fontsize=6,ncol=3)
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.suptitle('Dataset 203: exchange-flow and deterministic geometry audit\nSelected paths and profiled point potentials; no evidence or global-barrier claim')
 fig.savefig(out/'mixing-audit.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-hoc temperature-profile and qualified-walker-leg reconstruction."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);a=p.parse_args();root=a.audit.resolve();f=json.loads((root/'FROZEN.json').read_text());source=Path(f['source'])
 for name,h in f['source_hashes'].items():
  if sha(name)!=h:raise ValueError('Original source changed')
 profiles=[]
 for draw in [1024,2047]:
  rows=list(csv.DictReader((root/f'results{draw}.csv').open()))
  for name in dict.fromkeys(r['path'] for r in rows if r['path']!='rescale-control'):
   group=[r for r in rows if r['path']==name];prior=np.array([float(r['white_prior']) for r in group]);q=np.array([float(r['quadratic_profile']) for r in group]);temperatures=[]
   for beta in [1,.5,.25,.125,.0625,0]:
    u=prior+beta*q;temperatures.append(dict(beta=beta,barrier_above_higher_endpoint=float(u.max()-max(u[0],u[-1]))))
   profiles.append(dict(draw=draw,path=name,temperatures=temperatures))
 legs=[]
 for e in range(4):
  summary=json.loads((source/f'ensemble{e}/summary.json').read_text());k=len(summary['betas']);phase=np.zeros(k,dtype=int);trips=np.zeros(k,dtype=int);cold_seen=set();arrivals=0;trip_events=[];arrival_events=[]
  for j in range(2048):
   row=json.loads((source/f'ensemble{e}/draws/draw-{j}.rank-0.json').read_text());label=int(row['refreshed_prior_label'])
   if phase[label]==1:phase[label]=2;arrivals+=1;arrival_events.append(dict(draw=j,label=label))
   for pass_index,event in enumerate(row['exchange_trace']):
    label=int(event['labels'][0]);cold_seen.add(label)
    if phase[label]==2:trips[label]+=1;trip_events.append(dict(draw=j,pass_index=pass_index,label=label))
    phase[label]=1
  if phase.tolist()!=summary['transport_phase'] or trips.tolist()!=summary['roundtrips']:raise ValueError('Qualified journey reconstruction differs from native counters')
  legs.append(dict(ensemble=e,labels_seen_cold=len(cold_seen),cold_to_actual_prior_refresh_legs=arrivals,completed_prior_refresh_to_cold_legs=int(trips.sum()),unfinished_qualified_legs=int(np.sum(phase==2)),arrival_events=arrival_events,trip_events=trip_events))
 result=dict(scope='Post-hoc deterministic supplement; no new sampling or altered verdicts',temperature_profiles=profiles,walker_legs=legs,profile_formula='U_beta=white_prior + beta*Q_profile; bounded-uniform bias optimum is unchanged by positive beta scaling',limitations=['Only inspected paths; no global or free-energy barrier claim','Walker labels measure refreshed-state transport, not persistence of the same physical configuration','No new temperature ladder has been run'],result_sources={str(root/f'results{draw}.csv'):sha(root/f'results{draw}.csv') for draw in [1024,2047]},implementation_sha256=sha(Path(__file__)))
 (root/'supplement.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(legs=[{k:v for k,v in r.items() if k not in ['arrival_events','trip_events']} for r in legs],maximum_inspected_barrier_by_beta={str(beta):max(t['barrier_above_higher_endpoint'] for r in profiles for t in r['temperatures'] if t['beta']==beta) for beta in [1,.5,.25,.125,.0625,0]}),indent=2))
if __name__=='__main__':main()

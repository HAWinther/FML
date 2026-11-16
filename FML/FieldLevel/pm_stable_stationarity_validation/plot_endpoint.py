from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
out=args.input;r=np.genfromtxt(out/'science/probe.directions.csv',names=True,delimiter=',',dtype=None,encoding='utf8');p=np.genfromtxt(out/'science/public.line.csv',names=True,delimiter=',');v=r[(r['point']=='stalled')&(r['direction']==0)];h=sorted(set(v['h']),reverse=True)
left=[];right=[];lc=[];rc=[]
for x in h:
 a=v[(v['h']==x)&(v['sign']==1)][0];b=v[(v['h']==x)&(v['sign']==-1)][0];left.append(-b['stable_delta']/x);right.append(a['stable_delta']/x);lc.append(b['changed_axes']);rc.append(a['changed_axes'])
fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
axs[0].plot(h,left,'o-',ms=3,label='Left slope');axs[0].plot(h,right,'o-',ms=3,label='Right slope');axs[0].axhline(v[0]['analytic'],color='black',ls=':',label='Adjoint');axs[0].axhline(0,color='gray',lw=.7);axs[0].set_xscale('log');axs[0].invert_xaxis();axs[0].set_ylim(-.01,.014);axs[0].set(xlabel='Normalized field perturbation h',ylabel='Directional target slope',title='Negative-gradient direction');axs[0].legend(fontsize=8)
axs[1].plot(h,lc,'o-',ms=3,label='Negative perturbation');axs[1].plot(h,rc,'o-',ms=3,label='Positive perturbation');axs[1].set_xscale('log');axs[1].invert_xaxis();axs[1].set_yscale('symlog',linthresh=1);axs[1].set(xlabel='Normalized field perturbation h',ylabel='Changed particle / node / axis stencils',title='Corresponding CIC branch changes');axs[1].legend(fontsize=8)
axs[2].plot(p['alpha'],p['raw_delta'],'o-',ms=3,label='Ordinary subtraction');axs[2].plot(p['alpha'],p['stable_delta'],'o-',ms=3,label='Stable change');axs[2].plot(p['alpha'],p['armijo_delta'],'k:',label='Armijo bound');axs[2].set_xscale('log');axs[2].invert_xaxis();axs[2].set_yscale('symlog',linthresh=1e-14);axs[2].set(xlabel='Failed L-BFGS search step alpha',ylabel='Target change',title='Stable comparison accepts 0 / 40');axs[2].legend(fontsize=8)
fig.suptitle('Iteration 98: same target and tolerance; no convergence claim')
fig.savefig(args.output/'endpoint.png',dpi=170);fig.savefig(args.output/'endpoint.pdf');plt.close(fig)

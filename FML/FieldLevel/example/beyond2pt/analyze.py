#!/usr/bin/env python3
"""Report observed Box-1 modes and fixed-IC input readiness; no posterior fit."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
import h5py

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def run(out):
 identity=json.loads((out/'INPUTS.json').read_text());plan=identity['plan'];catalog=Path(identity['catalog']);L=plan['boxsize'];volume=L**3
 for key in ['catalog','exe','adapter']:
  if sha(identity[key])!=identity[key+'_sha256']:raise ValueError('Changed '+key)
 for name,h in identity['source_sha256'].items():
  if sha(Path(__file__).resolve().parent/name)!=h:raise ValueError('Changed application source '+name)
 nvec=np.asarray(plan['direct_modes'],dtype=int);direct=np.zeros(len(nvec),dtype=complex);minimum=np.full(3,np.inf);maximum=-minimum;outside=0;seen=0;image=np.zeros((160,160));slab_rows=0
 with h5py.File(catalog,'r') as f:
  ds=f[plan['group']]
  if not isinstance(ds,h5py.Dataset) or ds.ndim!=1 or ds.dtype.names!=tuple(plan['columns']) or len(ds)!=plan['rows']:raise ValueError('Unexpected pinned catalog schema')
  schema=dict(group=plan['group'],shape=list(ds.shape),dtype=ds.dtype.descr,root_attributes={k:str(v) for k,v in f.attrs.items()})
  for start in range(0,len(ds),plan['batch_rows']):
   block=ds[start:start+plan['batch_rows']];xyz=np.column_stack([block[n] for n in plan['columns']]).astype(float)
   if not np.isfinite(xyz).all():raise ValueError('Nonfinite catalog coordinate')
   seen+=len(xyz);minimum=np.minimum(minimum,xyz.min(axis=0));maximum=np.maximum(maximum,xyz.max(axis=0));outside+=int(np.any((xyz<0)|(xyz>=L),axis=1).sum())
   # Direct summation on the COMPLETE catalog; no mesh or NUFFT interpolation.
   for j,n in enumerate(nvec):direct[j]+=np.exp((-2j*np.pi/L)*(xyz@n)).sum(dtype=np.complex128)
   slab=xyz[(xyz[:,2]>=0)&(xyz[:,2]<100)];slab_rows+=len(slab);image+=np.histogram2d(slab[:,0],slab[:,1],bins=160,range=[[0,L],[0,L]])[0]
 direct/=seen;C=1/seen;Pnoise=volume/seen;modes=[];metas=[];receipts=[]
 for n in plan['meshes']:
  d=out/f'modes-{n}';receipt=json.loads((d/'completed.json').read_text());receipts.append(receipt)
  if not receipt['complete'] or any(sha(p)!=h for p,h in receipt['hashes'].items()):raise ValueError('Changed native extraction')
  meta=json.loads((d/'observed.json').read_text());metas.append(meta)
  if meta['mesh']!=n or meta['rows']!=seen or meta['ranks']!=1 or meta['geometry']!='cube' or meta['cutoff']!=max(plan['cutoffs']) or not np.isclose(meta['mode_variance'],C,rtol=1e-12) or not np.isclose(meta['poisson_power'],Pnoise,rtol=1e-12):raise ValueError('Native metadata differs from application contract')
  x=np.loadtxt(d/'observed.rank0.modes.csv',delimiter=',',skiprows=1);x=x[np.lexsort((x[:,2],x[:,1],x[:,0]))]
  nv=x[:,:3].astype(int)
  if not np.array_equal(nv,x[:,:3]) or np.any(np.all(nv==0,axis=1)) or len(np.unique(nv,axis=0))!=len(nv):raise ValueError('Malformed/duplicate/zero observed mode')
  expected_weight=np.where(nv[:,2]==0,1,2)
  if not np.array_equal(x[:,4],expected_weight) or not np.isfinite(x).all():raise ValueError('Invalid Hermitian weights/modes')
  modes.append(x)
 a,b=modes
 if not np.array_equal(a[:,:3],b[:,:3]) or not np.array_equal(a[:,4],b[:,4]):raise ValueError('Observed mode sets differ')
 da=a[:,5]+1j*a[:,6];db=b[:,5]+1j*b[:,6];weights=a[:,4];distance=float(np.sum(weights*np.abs(da-db)**2/C));direct_results=[]
 for j,n in enumerate(nvec):
  where=np.flatnonzero(np.all(a[:,:3]==n,axis=1))
  if len(where)!=1:raise ValueError('Missing direct-sum reference mode')
  i=where[0];errs=[float(abs(x[i,5]+1j*x[i,6]-direct[j])) for x in modes]
  direct_results.append(dict(n=n.tolist(),direct=[float(direct[j].real),float(direct[j].imag)],absolute_errors=errs,max_error_in_noise_sd=max(errs)/np.sqrt(C)))
 cuts=[]
 for cutoff in plan['cutoffs']:
  chosen=np.max(np.abs(a[:,:3]),axis=1)*(2*np.pi/L)<=cutoff;observed=int(weights[chosen].sum());limit=int(np.floor(cutoff*L/(2*np.pi)));expected=(2*limit+1)**3-1
  if observed!=expected:raise ValueError('Incomplete cubic cutoff')
  cuts.append(dict(kmax=cutoff,largest_integer_component=limit,stored_half_complex_modes=int(chosen.sum()),real_mode_degrees=observed,noise_normalized_mesh_distance_squared=float(np.sum(weights[chosen]*np.abs(da[chosen]-db[chosen])**2/C))))
 direct_pass=max(v['max_error_in_noise_sd'] for v in direct_results)<=plan['direct_max_error_in_noise_sd'];max_mesh=float(np.max(np.abs(da-db))/np.sqrt(C));mesh_pass=max_mesh<=plan['mesh_max_error_in_noise_sd']
 r=dict(scope=plan['scope'],observations_prepared=True,fixed_ic_fit_ready=False,posterior_computed=False,input_sha256=sha(out/'INPUTS.json'),catalog_sha256=sha(catalog),catalog_rows=seen,number_density=seen/volume,poisson_power=Pnoise,mode_variance=C,schema=schema,coordinate_min=minimum.tolist(),coordinate_max=maximum.tolist(),rows_outside_primary_box=outside,projection_slab_rows=slab_rows,cutoffs=cuts,native_metadata=metas,extraction_seconds=[v['seconds'] for v in receipts],direct_full_catalog_modes=direct_results,direct_check_passed=bool(direct_pass),mesh_check_passed=bool(mesh_pass),max_mesh_difference_in_noise_sd=max_mesh,noise_normalized_mesh_distance_squared=distance,missing_fit_inputs=plan['missing_fit_inputs'],sampler_campaigns_started=0,limits=['These are observed-data and ingestion checks; no model, sigma8 likelihood or posterior is fitted','Observed-data mesh distance is not a bound on parameter bias or forward-model error','Cubic likelihood cuts retain corner modes above the radial cutoff; the radial power plot uses complete shells only','Power is descriptive, with no likelihood or cosmological interpretation','Public catalog identity does not supply its exact matching phases or verified spectrum'])
 (out/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(out/'INPUT_REQUIREMENTS.json').write_text(json.dumps(dict(fixed_ic_fit_ready=False,missing_fit_inputs=plan['missing_fit_inputs'],do_not_substitute='Random phases or phases reconstructed from the same observed catalog are not the known simulation ICs'),indent=2)+'\n')
 # Preserve all observed modes. The descriptive spectrum uses complete radial shells.
 k=b[:,3];edges=np.linspace(0,max(plan['cutoffs']),31);power=volume*np.abs(db)**2;counts=np.histogram(k,bins=edges,weights=weights)[0];sums=np.histogram(k,bins=edges,weights=weights*power)[0];ks=np.histogram(k,bins=edges,weights=weights*k)[0];valid=counts>0;table=np.column_stack((ks[valid]/counts[valid],sums[valid]/counts[valid],counts[valid]));np.savetxt(out/'observed-power.csv',table,delimiter=',',header='mean_k,raw_power,real_mode_count',comments='')
 os.environ.setdefault('MPLCONFIGDIR',str(out/'mplcache'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');im=axes[0].imshow(np.log10(1+image.T),origin='lower',extent=[0,L,0,L],cmap='magma',interpolation='nearest');fig.colorbar(im,ax=axes[0],label='log10(1 + galaxy count / pixel)');axes[0].set(xlabel='x [Mpc/h]',ylabel='y [Mpc/h]',title=f'Box 1: 0 ≤ z <100 Mpc/h ({slab_rows:,} galaxies)')
 axes[1].plot(table[:,0],table[:,1],label='Observed raw galaxy power');axes[1].plot(table[:,0],table[:,1]-Pnoise,ls='--',label='Raw power minus Poisson term (descriptive)');axes[1].axhline(Pnoise,color='gray',ls=':',label='Poisson power');axes[1].set(yscale='log',xlabel='k [h/Mpc]',ylabel='P(k) [(Mpc/h)^3]',title='Complete spherical shells from the cubic data vector');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2);fig.suptitle(f'Public Beyond2pt catalog: {seen:,} galaxies at z=1\nObserved-data preparation; no initial-condition or cosmological fit');fig.savefig(out/'catalog-overview.png',dpi=160);plt.close(fig)
 if not direct_pass or not mesh_pass:raise ValueError('Observed-data numerical check failed; retain all artifacts')
 print(json.dumps({k:r[k] for k in ['catalog_rows','number_density','poisson_power','direct_check_passed','mesh_check_passed','max_mesh_difference_in_noise_sd','fixed_ic_fit_ready']},indent=2),flush=True)
 return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();run(a.output.resolve())
if __name__=='__main__':main()

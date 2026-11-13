#!/usr/bin/env python3
"""Analyze bounded survey experiments and make standalone scientific figures.
Requires NumPy and Matplotlib; no SciPy dependency. All coverage intervals
resample independent mock fields, never pretend correlated voxels are trials.
"""
import argparse
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import struct

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

LEVELS = np.array([.5, .68, .9, .95])
NORMAL = NormalDist()
COLORS = {'gaussian': '#2463a6', 'poisson': '#d57820'}


def metadata(path):
    return {k: float(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def rhat(chains):
    """Independent rank-normalized split/folded Rhat, checked against native FML."""
    c, n, d = chains.shape
    half = n//2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    flat = split.reshape(-1, d).astype(np.longdouble)
    result = np.zeros(d)
    for folded in (False, True):
        values = np.abs(flat.astype(float)-np.median(flat.astype(float), axis=0)).astype(np.longdouble) if folded else flat
        for j in range(d):
            _, inverse, counts = np.unique(values[:, j], return_inverse=True, return_counts=True)
            ends = np.cumsum(counts)
            ranks = (2*ends-counts+1)/2
            p = (ranks-.375)/(len(flat)+.25)
            scores = np.fromiter((NORMAL.inv_cdf(float(x)) for x in p), float)[inverse].reshape(2*c, half)
            within = np.var(scores, axis=1, ddof=1).mean()
            between = half*np.var(scores.mean(axis=1), ddof=1)
            value = np.sqrt(((half-1)/half*within+between/half)/within) if within > 0 else np.inf
            result[j] = max(result[j], value)
    return result


def basis_matrix(keys, n, power=2000., length=100.):
    coords = np.array(np.unravel_index(np.arange(n**3), (n,n,n))).T
    phase = 2*np.pi/n*coords@keys.T
    matrix = np.empty((n**3, 2*len(keys)))
    matrix[:, 0::2] = np.cos(phase)*np.sqrt(2*power/length**3)
    matrix[:, 1::2] = np.sin(phase)*np.sqrt(2*power/length**3)
    return matrix


def snapshots(directory, n, thin, chains, draws):
    rows = {}
    expected_keys = {(c, i) for c in range(chains) for i in range(0, draws, thin)}
    for file in sorted(directory.glob('snapshots.rank*.bin')):
        raw = file.read_bytes(); offset = 0
        def integer():
            nonlocal offset
            value, = struct.unpack_from('<Q', raw, offset); offset += 8
            return value
        size = integer(); tag = raw[offset:offset+size].decode(); offset += size
        if tag != 'FML survey validation snapshots 1': raise ValueError('Unknown snapshot schema')
        mesh, start, width, stride = (integer() for _ in range(4))
        if mesh != n or stride != thin: raise ValueError('Snapshot layout mismatch')
        seen = set()
        while offset < len(raw):
            c, draw, count = integer(), integer(), integer()
            if count != width*n*n: raise ValueError('Wrong local snapshot length')
            values = np.frombuffer(raw, dtype='<f8', count=count, offset=offset); offset += 8*count
            key = (c, draw)
            if key in seen: raise ValueError('Duplicate snapshot')
            seen.add(key)
            if key not in rows: rows[key] = np.full(n**3, np.nan)
            rows[key][start*n*n:(start+width)*n*n] = values
        if seen != expected_keys: raise ValueError('Incomplete snapshot stream')
    means = np.array([rows[key] for key in sorted(expected_keys)])
    if not np.isfinite(means).all(): raise ValueError('Missing/nonfinite observation slabs')
    return means


def intervals(samples, truth, discrete=False):
    result = []
    for level in LEVELS:
        low, high = np.quantile(samples, [(1-level)/2, (1+level)/2], axis=0,
                                method='inverted_cdf' if discrete else 'linear')
        result.append((truth >= low) & (truth <= high))
    return np.asarray(result)


def read_case(path, full_field_rhat=True):
    if not (path/'complete.txt').exists(): raise ValueError(f'Incomplete case {path}')
    meta = metadata(path/'meta.tsv'); n = int(meta['mesh']); c = int(meta['chains']); draws = int(meta['draws'])
    modes = np.loadtxt(path/'modes.tsv', ndmin=2); keys = modes[:, :3].astype(int); d = 2*len(keys)
    truth = modes[:, 3:5].ravel()
    trace = np.loadtxt(path/'trace.tsv', ndmin=2)
    if trace.shape != (c*draws, d+20): raise ValueError(f'Wrong trace dimensions {path}: {trace.shape}')
    chains = []
    for chain in range(c):
        rows = trace[trace[:, 0] == chain]
        if not np.array_equal(rows[:, 1], np.arange(draws)): raise ValueError('Noncontiguous draws')
        chains.append(rows[:, 2:])
    chains = np.asarray(chains); phi = chains[:, :, :d]; pooled = phi.reshape(-1, d)
    native = np.loadtxt(path/'diagnostics.tsv', ndmin=2)
    independent = rhat(chains)
    if not np.isfinite(independent).all() or np.max(np.abs(independent-native[:, 1])) > 1e-5: raise ValueError(f'Native/independent Rhat disagreement: {path}')
    ppdiag = np.loadtxt(path/'predictive_diagnostics.tsv', ndmin=2)
    chainstats = np.loadtxt(path/'chains.tsv', ndmin=2)
    cells = np.concatenate([np.loadtxt(f, ndmin=2) for f in sorted(path.glob('cells.rank*.tsv'))])
    cells = cells[np.argsort(cells[:, 0])]
    if cells.shape != (n**3, 10) or not np.array_equal(cells[:, 0], np.arange(n**3)): raise ValueError('Invalid cell slabs')
    mask = cells[:, 1].astype(bool); matrix = basis_matrix(keys, n, meta['power'], meta['boxsize'])
    latent = pooled@matrix.T
    if np.max(np.abs(matrix@truth-cells[:, 7])) > 1e-10: raise ValueError('Independent physical-field normalization failed')
    field_rhat = rhat(latent.reshape(c, draws, n**3)) if full_field_rhat else np.full(n**3, np.nan)
    if full_field_rhat and (not np.isfinite(field_rhat).all() or field_rhat.max() >= 1.02):
        raise ValueError(f'Physical-field mixing threshold failed: {path}')
    means = snapshots(path, n, int(meta['thin']), c, draws)
    rng = np.random.default_rng(100000+int(meta['mock'])+1000*int(meta['poisson']))
    rep = rng.poisson(means) if meta['poisson'] else means+rng.normal(size=means.shape)*np.sqrt(cells[:, 3])
    mode_cov = intervals(pooled, truth)
    field_cov = intervals(latent, cells[:, 7])
    # Integrate observation noise more accurately than one replica/state permits.
    # Repeated conditional draws are not additional independent posterior draws.
    active_means=means[:,mask]
    shape=(128,)+active_means.shape
    future_replicas=(rng.poisson(active_means,size=shape) if meta['poisson'] else
                     active_means+rng.normal(size=shape)*np.sqrt(cells[mask,3]))
    future_cov = intervals(future_replicas.reshape(-1,mask.sum()), cells[mask,6], discrete=bool(meta['poisson']))
    coverage_refinement={str(repeats):intervals(future_replicas[:repeats].reshape(-1,mask.sum()), cells[mask,6], bool(meta['poisson'])).mean(axis=1).tolist() for repeats in (32,64,128)}
    replicated_totals=future_replicas.sum(axis=-1).ravel()
    del future_replicas
    predictive = np.loadtxt(path/'predictive.tsv', ndmin=2)
    if predictive.shape != (len(means), 15): raise ValueError('Wrong predictive table shape')
    local_means=means[:, mask];local_rep=rep[:, mask]
    center=local_means.mean(axis=0)
    predvar=local_means.var(axis=0,ddof=1)+(center if meta['poisson'] else cells[mask,3])
    standardized=(local_rep-center)/np.sqrt(predvar)
    future=(cells[mask,6]-center)/np.sqrt(predvar)
    coords=np.array(np.unravel_index(np.arange(n**3),(n,n,n))).T
    position=-50+meta['boxsize']/n*coords
    radial=np.sum(position[mask]**2,axis=1)/38**2;radial-=radial.mean();radial/=np.linalg.norm(radial)
    marginal={'replicated_discrepancy':np.mean(standardized**2,axis=1),
              'holdout_discrepancy':float(np.mean(future**2)),
              'replicated_radial':standardized@radial,'holdout_radial':float(future@radial)}
    scalar = {
        'path': str(path), 'mock': int(meta['mock']), 'chains': c, 'draws_per_chain': draws,
        'collapsed': bool(meta.get('collapsed', 0)), 'max_rhat': float(native[:, 1].max()),
        'min_bulk_ess': float(native[:, 2].min()), 'min_tail_ess': float(native[:, 3].min()),
        'max_predictive_rhat': float(ppdiag[:, 1].max()),
        'max_field_rhat': float(field_rhat.max()) if full_field_rhat else None,
        'min_acceptance': float(chainstats[:, 1].min()), 'divergences': int(chainstats[:, 2].sum()),
        'mean_mode_width_over_prior': float(pooled.std(axis=0, ddof=1).mean()),
        'mode_coverage': mode_cov.mean(axis=1).tolist(),
        'field_inside_coverage': field_cov[:, mask].mean(axis=1).tolist(),
        'field_outside_coverage': field_cov[:, ~mask].mean(axis=1).tolist(),
        'holdout_voxel_coverage': future_cov.mean(axis=1).tolist(),
        'predictive_coverage_refinement': coverage_refinement,
        'mode_truth_quantiles': np.mean(pooled < truth, axis=0).tolist(),
        'initial_rmse_over_prior_zero': float(np.sqrt(np.mean((latent.mean(axis=0)-cells[:, 7])**2)/np.mean(cells[:, 7]**2))),
        'ppc_observed_deviance_p': float(np.mean(predictive[:, 3] >= predictive[:, 2])),
        'ppc_holdout_marginal_discrepancy_p': float(np.mean(marginal['replicated_discrepancy'] >= marginal['holdout_discrepancy'])),
        'holdout_total_quantile': float(np.mean(replicated_totals < predictive[0, 11])+.5*np.mean(replicated_totals == predictive[0, 11])),
        'holdout_total_coverage': intervals(replicated_totals[:, None], np.array([predictive[0, 11]]), bool(meta['poisson']))[:, 0].tolist(),
        'native_passed': (path/'complete.txt').read_text().strip() == 'PASS',
    }
    return {'summary': scalar, 'meta': meta, 'keys': keys, 'phi': phi, 'truth': truth, 'cells': cells,
            'latent': latent, 'field_rhat': field_rhat, 'observation_means': means, 'replicated': rep,
            'predictive': predictive, 'replicated_totals':replicated_totals, 'native': native, 'matrix': matrix, 'marginal': marginal}


def cluster_interval(rows, seed=31):
    rows = np.asarray(rows); rng = np.random.default_rng(seed)
    means = rows[rng.integers(len(rows), size=(4000, len(rows)))].mean(axis=1)
    return rows.mean(axis=0), np.quantile(means, [.025,.975], axis=0)


def exact_linear(case):
    cells, matrix, keys, meta = case['cells'], case['matrix'], case['keys'], case['meta']
    mask = cells[:, 1].astype(bool)
    response = 1.4+.6*keys[:, 2]**2/np.sum(keys*keys, axis=1)
    B = matrix*np.repeat(response, 2)[None, :]*cells[:, 2, None]
    B = B[mask]; variance = cells[mask, 3]; data = cells[mask, 5]
    precision = np.eye(B.shape[1])+B.T@(B/variance[:, None])
    covariance = np.linalg.inv(precision); mean = covariance@(B.T@(data/variance))
    pooled = case['phi'].reshape(-1,len(mean)); actual_mean = pooled.mean(axis=0); actual_cov=np.cov(pooled,rowvar=False)
    # Nonoverlapping batch means provide an independent Monte Carlo error estimate.
    batches = case['phi'][:, :case['phi'].shape[1]//32*32].reshape(-1,32,len(mean)).mean(axis=1)
    mcse = batches.std(axis=0,ddof=1)/np.sqrt(len(batches))
    report = {'max_mean_error_in_posterior_sd': float(np.max(np.abs(actual_mean-mean)/np.sqrt(np.diag(covariance)))),
              'max_mean_error_in_batch_mcse': float(np.max(np.abs(actual_mean-mean)/mcse)),
              'relative_covariance_frobenius_error': float(np.linalg.norm(actual_cov-covariance)/np.linalg.norm(covariance)),
              'min_variance_ratio': float(np.min(np.diag(actual_cov)/np.diag(covariance))),
              'max_variance_ratio': float(np.max(np.diag(actual_cov)/np.diag(covariance)))}
    if report['max_mean_error_in_batch_mcse'] > 5: raise ValueError('Linear posterior mean fails conservative MC-error check')
    return report, mean, covariance, mcse


def quadrature(coefficients, nz=48, nr=32, nphi=96):
    keys=coefficients[:, :3]; coef=coefficients[:,3]+1j*coefficients[:,4]
    z,wz=np.polynomial.legendre.leggauss(nz);z=-25+(z+1)*63/2;wz*=63/2
    t,wt=np.polynomial.legendre.leggauss(nr);t=(t+1)/2;wt/=2
    angle=np.arange(nphi)*2*np.pi/nphi;total=volume=selection_volume=0.
    for zi,wi in zip(z,wz):
        radius2=38**2-zi**2;rho=np.sqrt(radius2*t)
        points=np.column_stack([np.repeat(rho,nphi)*np.tile(np.cos(angle),nr),np.repeat(rho,nphi)*np.tile(np.sin(angle),nr),np.full(nr*nphi,zi)])
        weights=np.repeat(wi*wt*radius2/2*2*np.pi/nphi,nphi)
        phase=2*np.pi*((points+50)/100)@keys.T
        field=2*(np.cos(phase)@coef.real-np.sin(phase)@coef.imag)
        sel=.85*np.exp(-np.sum(points*points,axis=1)/(2*35**2))
        total+=np.sum(weights*sel*np.exp(field))*.015;volume+=weights.sum();selection_volume+=np.sum(weights*sel)
    return {'count': float(total), 'volume': float(volume), 'selection_volume': float(selection_volume)}


def convergence_report(directory):
    data=np.loadtxt(directory/'convergence.tsv',ndmin=2)
    fine=np.loadtxt(directory/'coefficients-N48-W8.tsv',ndmin=2)
    ref=quadrature(fine);coarse=quadrature(fine,24,16,48)
    z0=-25;R=38.;sigma=35.
    analytic_volume=np.pi*(R*R*(R-z0)-(R**3-z0**3)/3)
    analytic_selection=.85*2*np.pi*sigma**2*(sigma*np.sqrt(np.pi/2)*(math.erf(R/(np.sqrt(2)*sigma))-math.erf(z0/(np.sqrt(2)*sigma)))-(R-z0)*np.exp(-R*R/(2*sigma*sigma)))
    if abs(ref['selection_volume']/analytic_selection-1)>1e-10 or abs(ref['volume']/analytic_volume-1)>1e-10:raise ValueError('Independent analytic geometry/quadrature mismatch')
    qerr=abs(coarse['count']/ref['count']-1)
    if qerr>1e-6:raise ValueError('Independent count quadrature not resolved')
    model_errors=[]
    for N in (8,12,16,24,32,48):
        for width in (4,6,8):
            coeff=np.loadtxt(directory/f'coefficients-N{N}-W{width}.tsv',ndmin=2)
            if not np.array_equal(coeff[:,:3],fine[:,:3]):raise ValueError('Inconsistent physical mode keys')
            error=np.sqrt(np.sum((coeff[:,3:]-fine[:,3:])**2)/np.sum(fine[:,3:]**2))
            model_errors.append([N,width,float(error)])
    return {'analytic_volume':float(analytic_volume),'analytic_selection_volume':float(analytic_selection),
            'count_reference':ref['count'],'relative_quadrature_change':qerr,
            'model_errors':model_errors},data


def style(output):
    os.environ.setdefault('MPLCONFIGDIR', str(output/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','legend.frameon':False})
    return plt


def save(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=170, bbox_inches='tight')
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight')


def recovery_plot(cases, output, plt):
    fig,axes=plt.subplots(2,4,figsize=(14,6.8),layout='constrained')
    maxsignal=max(float(np.max(np.abs(c['cells'][:,7]))) for c in cases.values())
    for row,(kind,case) in enumerate(cases.items()):
        n=int(case['meta']['mesh']);length=case['meta']['boxsize'];dx=length/n
        mask=case['cells'][:,1].astype(bool).reshape(n,n,n)[:,n//2,:]
        truth=case['cells'][:,7];mean=case['latent'].mean(axis=0);sd=case['latent'].std(axis=0,ddof=1)
        arrays=[truth,mean,sd,(mean-truth)/sd]
        prior_sd=np.sqrt(2*case['meta']['power']/length**3*len(case['keys']))
        for col,values in enumerate(arrays):
            image=values.reshape(n,n,n)[:,n//2,:].T
            image=np.pad(image,((0,1),(0,1)),mode='wrap')
            kwargs={'cmap':'RdBu_r','vmin':-maxsignal,'vmax':maxsignal} if col<2 else ({'cmap':'viridis','vmin':0,'vmax':prior_sd} if col==2 else {'cmap':'RdBu_r','vmin':-3,'vmax':3})
            im=axes[row,col].imshow(image,origin='lower',extent=(-50-dx/2,50+dx/2,-50-dx/2,50+dx/2),interpolation='nearest',**kwargs)
            angle=np.linspace(np.arcsin(-25/38),np.pi-np.arcsin(-25/38),400)
            xx=38*np.cos(angle);zz=38*np.sin(angle)
            axes[row,col].plot(xx,zz,'k--',lw=.8,alpha=.7)
            edge=np.sqrt(38**2-25**2);axes[row,col].plot([-edge,edge],[-25,-25],'k--',lw=.8,alpha=.7)
            axes[row,col].set(xlim=(-50,50),ylim=(-50,50),aspect='equal')
            axes[row,col].set_xticks([-50,0,50]);axes[row,col].set_yticks([-50,0,50])
            if row==0:axes[row,col].set_title(['True initial density','Posterior mean','Posterior standard deviation','(Mean − truth) / SD'][col])
            if row==1:axes[row,col].set_xlabel(r'$x\ [h^{-1}{\rm Mpc}]$')
            if col==0:axes[row,col].set_ylabel(kind.capitalize()+r'\n$z\ [h^{-1}{\rm Mpc}]$'.replace('\\n','\n'))
            if col in (1,2,3):fig.colorbar(im,ax=axes[row,col],shrink=.82,label=r'$\delta_{\rm ini}$' if col<3 else 'posterior SD')
    fig.suptitle('2LPT masked-survey reconstruction — the same first mock field, two observation models\n'
                 '26 active modes; dashed line: continuous footprint; voxel model uses node selection',fontsize=12)
    save(fig,output,'recovery');plt.close(fig)


def diagnostics_plot(groups, production, output, plt):
    fig,axes=plt.subplots(2,3,figsize=(13.5,6.2),layout='constrained')
    for row,kind in enumerate(('gaussian','poisson')):
        case=production[kind][0] if production.get(kind) else groups[kind][0]
        idx=np.flatnonzero(np.all(case['keys']==[1,0,0],axis=1))[0]*2
        for chain,values in enumerate(case['phi']):axes[row,0].plot(np.arange(len(values)),values[:,idx],lw=.55,alpha=.65,label=f'Chain {chain+1}')
        axes[row,0].axhline(case['truth'][idx],color='black',ls='--',lw=1,label='Truth')
        axes[row,0].set(title=kind.capitalize()+': full-grid retained traces',xlabel='Retained trajectory',ylabel=r'Whitened $k=(1,0,0)$ cosine mode')
        if row==0:axes[row,0].legend(ncol=3,fontsize=8,loc='upper right')
        mocks=[c['summary']['mock'] for c in groups[kind]]
        rh=[c['summary']['max_rhat'] for c in groups[kind]];frh=[c['summary']['max_field_rhat'] for c in groups[kind]]
        axes[row,1].plot(mocks,rh,'o-',color=COLORS[kind],label='26 modes + 16 probes + summaries')
        axes[row,1].plot(mocks,frh,'s--',color='#6f6f6f',label='All 512 initial-density voxels')
        axes[row,1].axhline(1.02,color='black',ls=':',lw=1);axes[row,1].set(xlabel='Independent mock field',ylabel=r'Maximum rank-normalized $\hat R$',title='Repeated-mock reference chains')
        axes[row,1].set_ylim(.998,1.028)
        axes[row,1].legend(fontsize=7,loc='upper left')
        bulk=[c['summary']['min_bulk_ess'] for c in groups[kind]];tail=[c['summary']['min_tail_ess'] for c in groups[kind]]
        axes[row,2].plot(mocks,bulk,'o-',color=COLORS[kind],label='Minimum bulk ESS');axes[row,2].plot(mocks,tail,'s--',color='#5f5f5f',label='Minimum tail ESS')
        axes[row,2].axhline(150,color='black',ls=':',lw=1);axes[row,2].set(xlabel='Independent mock field',ylabel='Effective draws',title='Mode/probe/energy diagnostics')
        axes[row,2].legend(fontsize=8)
    fig.suptitle('Dispersed production chains and 12 independent mock fields per likelihood\nExact active-mode reference integrates out unused white coordinates',fontsize=11)
    save(fig,output,'diagnostics');plt.close(fig)


def calibration_plot(groups, output, plt):
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained');bins=np.linspace(0,1,6)
    for col,kind in enumerate(('gaussian','poisson')):
        cases=groups[kind];ranks=np.array([c['summary']['mode_truth_quantiles'] for c in cases]);rows=np.array([np.histogram(r,bins)[0]/len(r) for r in ranks])
        average,ci=cluster_interval(rows)
        center=(bins[:-1]+bins[1:])/2
        axes[0,col].bar(center,average,width=.18,color=COLORS[kind],alpha=.75)
        axes[0,col].vlines(center,ci[0],ci[1],color='black',lw=1.3);axes[0,col].axhline(.2,color='black',ls='--',lw=1)
        width=np.mean([c['summary']['mean_mode_width_over_prior'] for c in cases])
        axes[0,col].set(title=f'{kind.capitalize()}: initial-mode truth quantiles',xlabel='Posterior CDF at mock truth',ylabel='Fraction in bin',xlim=(0,1))
        axes[0,col].text(.02,.95,f'{len(cases)} independent fields; 26 ranks/field\nMean posterior/prior width: {width:.2f}',transform=axes[0,col].transAxes,va='top',fontsize=9)
        for key,label,color,marker in [('mode_coverage','Initial modes',COLORS[kind],'o'),('field_inside_coverage','Initial density inside footprint','#378351','s'),('field_outside_coverage','Initial density outside footprint','#77559b','^'),('holdout_voxel_coverage','Independent future observations','#d57820','D')]:
            y,interval=cluster_interval([c['summary'][key] for c in cases]);axes[1,col].plot(LEVELS,y,marker+'-',color=color,label=label)
            axes[1,col].fill_between(LEVELS,interval[0],interval[1],color=color,alpha=.1)
        axes[1,col].plot([.4,1],[.4,1],'k--',lw=1);axes[1,col].set(xlim=(.45,.99),ylim=(.35,1.01),xlabel='Nominal central credible probability',ylabel='Empirical coverage',title='95% intervals resample independent mock fields')
        axes[1,col].legend(fontsize=8,loc='lower right')
    fig.suptitle('Conditional calibration with fixed physics and nuisance parameters\nShaded coverage intervals and rank bars are mock-cluster bootstrap intervals; Poisson intervals are discrete',fontsize=11)
    save(fig,output,'calibration');plt.close(fig)


def predictive_plot(cases, output, plt):
    fig,axes=plt.subplots(2,3,figsize=(13.5,6.3),layout='constrained')
    for row,(kind,case) in enumerate(cases.items()):
        p=case['predictive'];color=COLORS[kind]
        axes[row,0].hist(case['replicated_totals'],bins=32,density=True,color=color,alpha=.7,label='Posterior predictive replicates')
        axes[row,0].axvline(p[0,4],color='black',ls='--',label='Fitted observations')
        axes[row,0].axvline(p[0,11],color='#9a365b',lw=1.6,label='Independent future observations')
        true_total=np.sum(case['cells'][case['cells'][:,1].astype(bool),9]);axes[row,0].axvline(true_total,color='#4b4b4b',ls=':',label='True expected total')
        axes[row,0].set(title=kind.capitalize()+': total observed signal',xlabel='Total count' if case['meta']['poisson'] else 'Sum of selected tracer data',ylabel='Predictive density');axes[row,0].legend(fontsize=7)
        marginal=case['marginal']
        axes[row,1].hist(marginal['replicated_discrepancy'],bins=24,density=True,color=color,alpha=.7,label='Predictive replicates')
        axes[row,1].axvline(marginal['holdout_discrepancy'],color='#9a365b',lw=1.6,label='Independent future data')
        axes[row,1].set(title='Marginal predictive residual discrepancy',xlabel='Mean squared residual / predictive variance',ylabel='Density');axes[row,1].legend(fontsize=8)
        axes[row,2].hist(marginal['replicated_radial'],bins=24,density=True,color=color,alpha=.7,label='Predictive replicates')
        axes[row,2].axvline(marginal['holdout_radial'],color='#9a365b',lw=1.6,label='Independent future data')
        axes[row,2].set(title='Marginal radial residual pattern',xlabel='Weighted standardized residual score',ylabel='Density');axes[row,2].legend(fontsize=8)
    fig.suptitle('Posterior predictive checks — first mock in each observation model\nFuture data use new independent noise/count draws at the same true initial field',fontsize=11)
    save(fig,output,'predictive');plt.close(fig)


def convergence_plot(report, data, output, plt):
    fig,axes=plt.subplots(1,3,figsize=(13.5,4),layout='constrained')
    for sub,color in [(1,'#a34840'),(2,'#e2a040'),(4,'#267ba4'),(8,'#398052')]:
        rows=data[(data[:,1]==6)&(data[:,2]==sub)]
        axes[0].plot(rows[:,0],100*np.abs(rows[:,4]/report['analytic_selection_volume']-1),'o-',color=color,label=f'{sub}³ selection samples/voxel')
        axes[1].plot(rows[:,0],100*np.abs(rows[:,5]/report['count_reference']-1),'o-',color=color)
    axes[0].set(ylabel='Absolute relative error [%]',xlabel='Voxel / initial mesh N',title='Selection integral vs analytic result',yscale='log');axes[0].legend(fontsize=7)
    axes[1].set(ylabel='Absolute relative error [%]',xlabel='Voxel / initial mesh N',title='Expected counts vs continuous quadrature',yscale='log')
    errors=np.array(report['model_errors'])
    for width,color in [(4,'#a34840'),(6,'#267ba4'),(8,'#398052')]:
        rows=errors[(errors[:,1]==width)&(errors[:,2]>0)]
        axes[2].plot(rows[:,0],100*rows[:,2],'o-',color=color,label=f'Assignment width {width}')
    axes[2].set(yscale='log',xlabel='Particle / initial mesh N',ylabel='Relative retained-mode error [%]',title='Forward field vs N=48, width=8');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xticks([8,16,24,32,48]);ax.grid(alpha=.2)
    fig.suptitle('Physical-field and selection convergence are separate checks\nSelection averaging improves boundary geometry; the exponential intensity is still sampled at voxel centers',fontsize=11)
    save(fig,output,'convergence');plt.close(fig)


def reference_plot(case, exact, output, plt):
    report,mean,cov,mcse=exact;phi=case['phi'].reshape(-1,len(mean));actual=phi.mean(axis=0);ratio=phi.var(axis=0,ddof=1)/np.diag(cov)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    axes[0].errorbar(np.arange(len(mean)),(actual-mean)/np.sqrt(np.diag(cov)),yerr=2*mcse/np.sqrt(np.diag(cov)),fmt='o',ms=4,color='#2463a6',capsize=2)
    axes[0].axhline(0,color='black',lw=1);axes[0].set(xlabel='Active real Fourier coordinate',ylabel='Mean error / exact posterior SD',title='Means vs independent exact Gaussian posterior')
    axes[1].plot(np.arange(len(mean)),ratio,'o',color='#2463a6');axes[1].axhline(1,color='black',ls='--',lw=1);axes[1].set(xlabel='Active real Fourier coordinate',ylabel='Sampled / exact variance',title='Posterior marginal variances')
    fig.suptitle(f'Masked linear-Gaussian reference: max mean error {report["max_mean_error_in_batch_mcse"]:.2f} batch MCSE; covariance error {report["relative_covariance_frobenius_error"]:.1%}',fontsize=11)
    save(fig,output,'linear_reference');plt.close(fig)


def bridge(a,b):
    d=a['phi'].shape[-1];x=a['phi'].reshape(-1,d);y=b['phi'].reshape(-1,d)
    if not np.allclose(a['cells'][:,[1,2,3,4,5]],b['cells'][:,[1,2,3,4,5]],rtol=1e-10,atol=1e-11):raise ValueError('Bridge datasets differ')
    se=np.sqrt(x.var(axis=0,ddof=1)/a['native'][:d,2]+y.var(axis=0,ddof=1)/b['native'][:d,2])
    max_error=float(np.max(np.abs(x.mean(axis=0)-y.mean(axis=0))/se))
    if not np.isfinite(max_error) or max_error > 5:raise ValueError('Representation bridge mean fails conservative MC-error check')
    return {'max_mean_difference_in_mcse':max_error,
            'relative_covariance_difference':float(np.linalg.norm(np.cov(x,rowvar=False)-np.cov(y,rowvar=False))/np.linalg.norm(np.cov(y,rowvar=False)))}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--production',type=Path)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--full-linear',type=Path);parser.add_argument('--convergence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mpi',type=Path);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);plt=style(args.output)
    groups={};production={};summary={'scope':'2LPT, 26 active initial modes, fixed spectrum/bias/noise/exposure; repeated-mock exact active-mode reference plus production full-grid bridges.','levels':LEVELS.tolist(),'groups':{},'production_bridge':{},'mpi_bridge':{},'production_cases':{},'mpi_cases':{}}
    products={}
    for kind in ('gaussian','poisson'):
        paths=sorted((args.input/('2lpt_'+kind)).glob('mock*'),key=lambda p:int(p.name[4:]))
        groups[kind]=[read_case(p) for p in paths if p.is_dir() and (p/'complete.txt').exists()]
        if not args.partial and len(groups[kind])!=12:raise ValueError(f'Expected 12 completed {kind} mocks, found {len(groups[kind])}')
        if not groups[kind]:raise ValueError(f'No cases: {kind}')
        if not args.partial and [c['summary']['mock'] for c in groups[kind]]!=list(range(12)):raise ValueError('Missing or duplicate mock IDs')
        rows=[c['summary'] for c in groups[kind]]
        if not all(r['native_passed'] for r in rows):raise ValueError(f'Unconverged native chains: {kind}')
        aggregate={key:{'mean':cluster_interval([r[key] for r in rows])[0].tolist(),'cluster_bootstrap95':cluster_interval([r[key] for r in rows])[1].tolist()} for key in ['mode_coverage','field_inside_coverage','field_outside_coverage','holdout_voxel_coverage','holdout_total_coverage']}
        summary['groups'][kind]={'cases':rows,'aggregate':aggregate}
        if args.production:
            paths=sorted((args.production/('2lpt_'+kind)).glob('mock*'),key=lambda p:int(p.name[4:]))
            production[kind]=[read_case(p) for p in paths if p.is_dir() and (p/'complete.txt').exists()]
            summary['production_cases'][kind]=[c['summary'] for c in production[kind]]
            summary['production_bridge'][kind]=[{'mock':c['summary']['mock'],**bridge(c,groups[kind][c['summary']['mock']])} for c in production[kind]]
        if args.mpi:
            paths=sorted((args.mpi/('2lpt_'+kind)).glob('mock*'),key=lambda p:int(p.name[4:]))
            mpi_cases=[read_case(p) for p in paths if p.is_dir() and (p/'complete.txt').exists()]
            if len(mpi_cases)!=1 or not mpi_cases[0]['summary']['native_passed']:raise ValueError('Incomplete MPI bridge')
            summary['mpi_cases'][kind]=[c['summary'] for c in mpi_cases]
            summary['mpi_bridge'][kind]=[{'mock':c['summary']['mock'],**bridge(c,groups[kind][c['summary']['mock']])} for c in mpi_cases]
        first=groups[kind][0];products[kind+'_initial_mean']=first['latent'].mean(axis=0);products[kind+'_initial_sd']=first['latent'].std(axis=0,ddof=1);products[kind+'_cells']=first['cells'];products[kind+'_field_rhat']=first['field_rhat']
    exactcase=read_case(args.reference);exact=exact_linear(exactcase);summary['exact_linear']=exact[0]
    if args.full_linear:
        full=read_case(args.full_linear);summary['exact_linear_production']=exact_linear(full)[0];summary['linear_representation_bridge']=bridge(full,exactcase)
    conv,data=convergence_report(args.convergence);summary['convergence']=conv
    selected={kind:group[0] for kind,group in groups.items()}
    recovery_plot(selected,args.output,plt);diagnostics_plot(groups,production,args.output,plt)
    calibration_plot(groups,args.output,plt);predictive_plot(selected,args.output,plt)
    convergence_plot(conv,data,args.output,plt);reference_plot(exactcase,exact,args.output,plt)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'first_mock_products.npz',**products)
    print(json.dumps({'mocks':{kind:len(cases) for kind,cases in groups.items()},'exact_linear':exact[0],'quadrature_change':conv['relative_quadrature_change']},indent=2))

if __name__=='__main__':main()

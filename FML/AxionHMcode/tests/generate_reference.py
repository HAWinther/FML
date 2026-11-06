#!/usr/bin/env python3
"""Generate independent Python axionHMcode fixtures from physical axionCAMB linear spectra.

Requires numpy==1.26.4 scipy==1.12.0 numba==0.59.1 astropy==6.0.1 and the two upstream checkouts.
Neither Python nor axionCAMB is used by the native library or its regression test.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reference-source', type=Path, required=True)
ap.add_argument('--axioncamb', type=Path, required=True)
ap.add_argument('--work-dir', type=Path, required=True)
ap.add_argument('--linear-only', action='store_true')
ap.add_argument('--reuse-linear', action='store_true')
ap.add_argument('--reuse-transfer', action='store_true', help='reuse the cached axionCAMB transfer file to regenerate linear input')
a = ap.parse_args()
revision = subprocess.check_output(['git', '-C', str(a.reference_source), 'rev-parse', 'HEAD'], text=True).strip()
if revision != 'a85ba2679cba5abe68b4cfd1e49b7a52bdcda424':
    raise RuntimeError('Use the pinned axionHMcode revision a85ba267')
sys.path.insert(0, str(a.reference_source.resolve()))
from axionCAMB_and_lin_PS.axionCAMB_wrapper import axioncamb_params
from axionCAMB_and_lin_PS.lin_power_spectrum import func_power_spec_dic
from cosmology.overdensities import func_D_z_unnorm_int
from halo_model.HMcode_params import HMCode_param_dic
from axion_functions.axion_params import func_axion_param_dic
from halo_model.PS_nonlin_axion import func_full_halo_model_ax

out = Path(__file__).resolve().parent / 'reference'
out.mkdir(exist_ok=True)
a.work_dir.mkdir(parents=True, exist_ok=True)
work = a.work_dir.resolve()
c = dict(h=.67, Omega_b_0=.05, Omega_d_0=.2353, Omega_ax_0=.0317, Omega_db_0=.2853,
         Omega_m_0=.317, Omega_w_0=.683, z=2., m_ax=10**-24.5, ns=.965, As=2.1e-9, k_piv=.05,
         version='dome', transfer_kmax=1000.)
for name in ['b', 'd', 'ax', 'db', 'm']:
    c['omega_'+name+'_0'] = c['Omega_'+name+'_0'] * c['h']**2
if not a.reuse_linear:
    ini = work / 'axion.ini'
    if not a.reuse_transfer:
        axioncamb_params(str(ini), c, output_root=str(work / 'axion'), print_info=False,
                         transfer_k_per_logint=30)
        with (work/'axioncamb.log').open('w') as log:
            subprocess.run([str(a.axioncamb.resolve()), str(ini)], cwd=a.axioncamb.resolve().parent,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
    p = func_power_spec_dic(str(work/'axion_transfer_out.dat'), c)
    k = np.geomspace(1e-5, 1000, 1001)
    def resample(values):
        values = np.maximum(values, 1e-200)
        result = np.exp(np.interp(np.log(k), np.log(p['k']), np.log(values)))
        low = k < p['k'][0]
        result[low] = values[0]*(k[low]/p['k'][0])**c['ns']
        return result
    pc, pa = resample(p['power_cold']), resample(p['power_axion'])
    np.savetxt(out/'linear.txt', np.column_stack([k, pc, pa]), fmt='%.16e',
        header='axionCAMB 891e779cc0bd422e49f97533e6c2fc761149737d; h=.67 Ob=.05 Ocdm=.2353 Oax=.0317 m=10^-24.5 eV z=2 As=2.1e-9 ns=.965\nP~k^ns continuation below the first transfer sample\nk[h/Mpc] P_cold P_axion [(Mpc/h)^3]')
    print('Physical ULA linear spectrum written', flush=True)
if a.linear_only:
    sys.exit(0)
k, pc, pa = np.loadtxt(out/'linear.txt', unpack=True)
power = dict(k=k, power_cold=pc, power_axion=pa)
M = np.geomspace(1e7, 1e18, 65)
c['G_a'] = func_D_z_unnorm_int(c['z'], c['Omega_m_0'], c['Omega_w_0'])
print('G_a=', c['G_a'], flush=True)
for version in ['basic', 'dome']:
    c['version'] = version
    params = HMCode_param_dic(c, k, pc)
    ax = func_axion_param_dic(M, c, power, params, concentration_param=False)
    values = func_full_halo_model_ax(M, power, c, params, ax, alpha=(version=='dome'), eta_given=False,
                                    one_halo_damping=True, two_halo_damping=False,
                                    concentration_param=False, full_2h=False)
    selected = np.where((k >= .001) & (k <= 10))[0][::20]
    reference = np.column_stack([k[selected]]+[v[selected] for v in values])
    if not np.isfinite(reference).all() or not (reference[:, 1:] > 0).all():
        raise RuntimeError('Nonfinite or nonpositive reference component inside the tested k range')
    np.savetxt(out/(version+'.txt'), reference, fmt='%.16e',
        header=f'axionHMcode {revision}, {version}; 65 masses 1e7..1e18; c_min=4; alpha={version=="dome"}; eta=off; full_2h=off\nk[h/Mpc] P_total P_cold P_cold_axion P_axion [(Mpc/h)^3]')
    np.savetxt(out/(version+'_haloes.txt'), np.column_stack([ax['M_int'], ax['M_ax'], ax['central_dens']]), fmt='%.16e',
        header='M_cold M_axion [Msun/h] central_density_parameter')
    diagnostics = [ax['M_cut'], ax['beta2'], ax['frac_cluster'], params['alpha'][0], params['alpha'][1], c['G_a']]
    np.savetxt(out/(version+'_diagnostics.txt'), [[float(np.asarray(v).ravel()[0]) for v in diagnostics]], fmt='%.16e',
        header='Mcut beta2 Fcluster alpha_cold alpha_cross G_a')
    print(version, 'written; Fcluster=', ax['frac_cluster'], 'Mcut=', ax['M_cut'], flush=True)

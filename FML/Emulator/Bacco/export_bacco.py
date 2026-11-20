#!/usr/bin/env python3
"""Convert the baccoemu weight files to the files read by FML/Emulator/Bacco/Bacco.h.

baccoemu (R. Angulo, M. Zennaro, G. Arico, M. Pellejero Ibanez, P. Burger; MIT licence, see LICENSE in this folder)
downloads its trained networks from https://bacco.dipc.org at run time. Download the tarballs yourself (see README.md),
then

    python3 export_bacco.py WEIGHTS_DIR data

WEIGHTS_DIR holds the downloaded *.tar / *.zip files (or their extracted folders). Needs numpy and h5py only. Networks
whose tarball is missing are skipped.

Output, for every network NAME:
  data/NAME.bin          FML_BACCO_NN 1: the ReLU (Keras) networks. A text header (sizes, input and output scaling
                         with 17 digits) followed by the weights and biases as little-endian float32 (as in the
                         Keras files, so exact), layer by layer, weights (out x in, row-major) then biases
  data/NAME.txt          FML_BACCO_COSMOPOWER 1: the Burger et al. 2025 baryon boosts (CosmoPower activation), text
  data/NAME_details.txt  the parameter names and bounds, the k bins of the outputs and (bias expansion emulators)
                         the constant subtracted after exp()

The Keras networks: x (scaled to [0, 1] by the bounds for the nonlinear, baryon and nonlinear bias emulators, raw
otherwise) -> ReLU layers -> linear output -> + scaler_mean. The scaling is stored as the network's input and output
scaling, so the network's output is the emulated quantity before any exp().
"""
import os
import sys
import glob
import pickle
import tarfile
import zipfile
import numpy as np
import h5py

COSMO = ['omega_cold', 'sigma8_cold', 'omega_baryon', 'ns', 'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']
LIN = ['omega_cold', 'omega_baryon', 'hubble', 'neutrino_mass', 'w0', 'wa', 'expfactor']


def keras_jobs():
    """(output name, h5 file, details.pickle, keys, input scaled by the bounds?, field index into scaler_mean or None)"""
    jobs = []
    for name, keys, scaled in [
            ('cold_matter_linear_emu_1.0.2', LIN, False),
            ('total_matter_linear_emu_1.0.2', LIN, False),
            ('sigma8_emu_2.0.2', ['omega_cold', 'omega_baryon', 'ns', 'hubble', 'neutrino_mass', 'w0', 'wa'], False),
            ('smeared_bao_emu_1.0.2', COSMO, False),
            ('no_wiggles_emu_1.0.2', None, False),  # keys in details.pickle
            ('nonlinear_emu_1.0.3', COSMO, True),
            ('nonlinear_emu_2.0.1', COSMO, True),
            ('baryon_emu_1.0.2', ['omega_cold', 'sigma8_cold', 'omega_baryon', 'ns', 'hubble', 'neutrino_mass', 'w0',
                                  'wa', 'M_c', 'eta', 'beta', 'M1_z0_cen', 'theta_inn', 'M_inn', 'theta_out',
                                  'expfactor'], True)]:
        jobs.append((name, f'{name}/{name}.h5', f'{name}/details.pickle', keys, scaled, None))
    # Lagrangian bias expansion, real space (Zennaro et al. 2021; LPT: Arico et al. 2021)
    for n in range(15):
        f = f'lpt_emulator_v2.0.2/lpt_emu_field{n}'
        jobs.append((f'lbias_lpt_field{n}', f'{f}/lpt_emu_field{n}.h5', f'{f}/details.pickle', COSMO, False, None))
        f = f'lbias_real_1.0.1/lbias_emu_field{n}'
        jobs.append((f'lbias_nonlinear_field{n}', f'{f}/lbias_emu_field{n}.h5', 'lbias_real_1.0.1/details.pickle',
                     COSMO, True, n))
    # Redshift space (Pellejero Ibanez et al. 2023; LPT: velocileptors, Chen et al. 2021)
    for ell in (0, 2, 4):
        f = f'zspace_smeared_bao_emulator_v1.0.2/ell{ell}'
        jobs.append((f'lbias_rsd_smeared_bao_ell{ell}', f'{f}/ell{ell}.h5', f'{f}/details.pickle', COSMO, False, None))
        for n in range(15):
            f = f'velocileptor_emulator_v1.0.2/nfield{n}_ell{ell}'
            jobs.append((f'lbias_rsd_lpt_field{n}_ell{ell}', f'{f}/nfield{n}_ell{ell}.h5', f'{f}/details.pickle',
                         COSMO, False, None))
            f = f'lbias_emulator_RSD_1.0.2/field{n}_ell{ell}'
            jobs.append((f'lbias_rsd_nonlinear_field{n}_ell{ell}', f'{f}/field{n}_ell{ell}.h5',
                         f'lbias_emulator_RSD_1.0.2/ell{ell}_details.pickle', COSMO, True, n))
    return jobs


COSMOPOWER = {
    'powerspectra_ratio_bcm_emu_folded': 'baryon_boost_Burger2025_pk',
    'bispectra_ratio_bcm_emu_folded': 'baryon_boost_Burger2025_bk',
}


def g9(x):
    return '%.9g' % float(x)


def vec(name, v):
    """Scalings, bounds and activation parameters (float64 in baccoemu): 17 digits"""
    return name + ' ' + ' '.join('%.17g' % float(x) for x in np.ravel(v)) + '\n'


def extract(weights_dir, work):
    """Extract the tarballs / zip (if not already extracted) into work"""
    for f in sorted(glob.glob(os.path.join(weights_dir, '*.tar'))):
        name = os.path.basename(f)[:-4]
        if not os.path.isdir(os.path.join(work, name)):
            with tarfile.open(f) as tf:
                tf.extractall(work, filter='data')
    for f in sorted(glob.glob(os.path.join(weights_dir, '*.zip'))):
        with zipfile.ZipFile(f) as zf:
            zf.extractall(work)


def export_keras(work, name, h5, details, keys, scaled, field, outdir):
    h5, details = os.path.join(work, h5), os.path.join(work, details)
    if not (os.path.exists(h5) and os.path.exists(details)):
        return False
    with open(details, 'rb') as f:
        det = pickle.load(f)
    if keys is None:
        keys = list(det['keys'])
    bounds = np.array(det['bounds'], dtype=float)
    assert bounds.shape == (len(keys), 2), (name, bounds.shape)

    f = h5py.File(h5, 'r')
    mw = f['model_weights']
    layers = []
    for key in mw.attrs['layer_names']:
        key = key.decode() if isinstance(key, bytes) else key
        layers.append((np.array(mw[key][key]['kernel:0'], dtype='<f4'), np.array(mw[key][key]['bias:0'], dtype='<f4')))
    sizes = [layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers]
    assert sizes[0] == len(keys)

    if scaled:
        imean, istd = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    else:
        imean, istd = np.zeros(len(keys)), np.ones(len(keys))
    sm = det.get('scaler_mean')
    omean = np.zeros(sizes[-1]) if sm is None else np.array(sm if field is None else sm[field], dtype=float)
    assert omean.shape == (sizes[-1],), (name, omean.shape)
    assert det.get('scaler_std') is None or np.all(np.isnan(np.array(det['scaler_std'], dtype=float)))
    ostd = np.ones(sizes[-1])

    nw = sum(W.size + b.size for W, b in layers)
    with open(os.path.join(outdir, name + '.bin'), 'wb') as out:
        head = ('# baccoemu network %s (Keras, ReLU), converted by export_bacco.py. MIT licence: see LICENSE\n'
                % name)
        head += 'FML_BACCO_NN 1\nactivation relu\n'
        head += 'sizes %d %s\n' % (len(sizes), ' '.join(str(s) for s in sizes))
        head += vec('input_mean', imean) + vec('input_std', istd)
        head += vec('output_mean', omean) + vec('output_std', ostd)
        head += 'float32_le %d\n' % nw
        out.write(head.encode())
        for W, b in layers:
            out.write(np.ascontiguousarray(W.T).astype('<f4').tobytes())  # Keras kernel is (in, out)
            out.write(b.astype('<f4').tobytes())

    with open(os.path.join(outdir, name + '_details.txt'), 'w') as out:
        out.write('# baccoemu %s: parameters, bounds and output k bins (h/Mpc)\n' % name)
        out.write('FML_BACCO_DETAILS 1\n')
        out.write('keys %d %s\n' % (len(keys), ' '.join(keys)))
        out.write(vec('bounds', bounds))
        k = np.array(det['k'], dtype=float) if 'k' in det else np.zeros(0)
        out.write('k %d %s\n' % (len(k), ' '.join('%.17g' % x for x in k)))
        if 'subtract' in det:
            out.write('subtract %.17g\n' % float(det['subtract']))
    return True


def export_cosmopower(path, name, outdir):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    params = list(d['parameters'])
    nl = len(d['weights'])
    Ws = [np.array(d['weights']['layer_%d' % i]['w'], dtype=np.float32) for i in range(nl)]
    bs = [np.array(d['weights']['layer_%d' % i]['b'], dtype=np.float32) for i in range(nl)]
    hp = np.array(d['hyper_params'])
    sizes = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
    assert sizes[0] == len(params) and hp.shape == (nl - 1, 2, hp.shape[2])
    ranges = d['parameter_ranges']
    with open(os.path.join(outdir, name + '.txt'), 'w') as out:
        out.write('# baccoemu (Burger et al. 2025) network %s, CosmoPower activation (b + sigmoid(a x)(1 - b)) x,\n'
                  '# converted by export_bacco.py. MIT licence: see LICENSE\n' % name)
        out.write('FML_BACCO_COSMOPOWER 1\n')
        out.write('parameters %d %s\n' % (len(params), ' '.join(params)))
        out.write('sizes %d %s\n' % (len(sizes), ' '.join(str(s) for s in sizes)))
        out.write(vec('param_mean', d['param_train_mean']) + vec('param_std', d['param_train_std']))
        out.write(vec('feature_mean', np.atleast_1d(d['feature_train_mean'])))
        out.write(vec('feature_std', np.atleast_1d(d['feature_train_std'])))
        out.write('scaling %s %s\n' % ('%.17g' % d['scaling_division'], '%.17g' % d['scaling_subtraction']))
        out.write('ranges %d\n' % len(ranges))
        for key, (lo, hi) in ranges.items():
            out.write('%s %.17g %.17g\n' % (key, lo, hi))
        for l in range(nl):
            out.write('layer %d\n' % l)
            for o in range(Ws[l].shape[0]):  # (out, in) already
                out.write(' '.join(g9(x) for x in Ws[l][o]) + '\n')
            out.write(' '.join(g9(x) for x in bs[l]) + '\n')
            if l + 1 < nl:
                out.write(vec('alpha', hp[l, 0]) + vec('beta', hp[l, 1]))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    weights_dir, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    work = os.path.join(weights_dir, 'extracted')
    os.makedirs(work, exist_ok=True)
    extract(weights_dir, work)
    done, skipped = 0, []
    for job in keras_jobs():
        if export_keras(work, *job, outdir):
            done += 1
        else:
            skipped.append(job[0])
    for pkl, name in COSMOPOWER.items():
        hits = glob.glob(os.path.join(work, '**', pkl + '.pkl'), recursive=True)
        if not hits:
            skipped.append(name)
            continue
        export_cosmopower(hits[0], name, outdir)
        done += 1
    print('wrote %d networks to %s' % (done, outdir))
    if skipped:
        print('skipped (not downloaded): ' + ', '.join(skipped))


if __name__ == '__main__':
    main()

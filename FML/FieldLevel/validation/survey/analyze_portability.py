#!/usr/bin/env python3
"""Check bounded frontend replays; these runs are not posterior calibration."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import struct
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='fml-portability-mpl-'))
os.environ.setdefault('XDG_CACHE_HOME', tempfile.mkdtemp(prefix='fml-portability-cache-'))
import numpy as np
from analyze import metadata, snapshots


def relative(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite or mismatched replay arrays')
    return float(np.max(np.abs(a-b)/np.maximum(1, np.maximum(np.abs(a), np.abs(b)))))


def read(path):
    meta = metadata(path/'meta.tsv')
    if not meta.get('smoke') or not meta.get('divergence_free') or (path/'complete.txt').read_text().strip() != 'SMOKE':
        raise ValueError('Expected explicit smoke output')
    cells = np.concatenate([np.loadtxt(p, ndmin=2) for p in sorted(path.glob('cells.rank*.tsv'))])
    cells = cells[np.argsort(cells[:, 0])]
    n = int(meta['mesh'])
    if cells.shape != (n**3, 10) or not np.array_equal(cells[:,0], np.arange(n**3)):
        raise ValueError('Invalid global cell coverage')
    means = snapshots(path, n, int(meta['thin']), int(meta['chains']), int(meta['draws']))
    trace = np.loadtxt(path/'trace.tsv', ndmin=2)
    predictive = np.loadtxt(path/'predictive.tsv', ndmin=2)
    if trace.shape[0] != int(meta['chains'])*int(meta['draws']): raise ValueError('Incomplete trace')
    if not all(np.isfinite(x).all() for x in (cells, means, trace, predictive)): raise ValueError('Nonfinite smoke output')
    return cells, means, trace, predictive


def reader_checks():
    """Exercise real legacy snapshots plus malformed box coverage without sampling."""
    with tempfile.TemporaryDirectory(prefix='fml-snapshot-reader-') as tmp:
        root = Path(tmp); n = 2
        def integer(x): return struct.pack('<Q', x)
        def stream(tag, header, values):
            data = integer(len(tag))+tag.encode()+b''.join(integer(x) for x in header)
            return data+integer(0)+integer(0)+integer(len(values))+np.asarray(values, dtype='<f8').tobytes()
        values = np.arange(8.)
        legacy = stream('FML survey validation snapshots 1', [n,0,n,1], values)
        (root/'snapshots.rank0.bin').write_bytes(legacy)
        if not np.array_equal(snapshots(root,n,1,1,1), values[None,:]): raise ValueError('Legacy reader changed')
        box = stream('FML survey validation snapshots 2', [n,0,0,0,n,n,n,1], values)
        (root/'snapshots.rank0.bin').write_bytes(box)
        if not np.array_equal(snapshots(root,n,1,1,1), values[None,:]): raise ValueError('Box reader changed')
        # y-split boxes have noncontiguous IDs in the global row-major field.
        first = stream('FML survey validation snapshots 2', [n,0,0,0,n,1,n,1], values[[0,1,4,5]])
        second = stream('FML survey validation snapshots 2', [n,0,1,0,n,1,n,1], values[[2,3,6,7]])
        (root/'snapshots.rank0.bin').write_bytes(first)
        try: snapshots(root,n,1,1,1)
        except ValueError: pass
        else: raise ValueError('Missing box coverage accepted')
        (root/'snapshots.rank1.bin').write_bytes(second)
        if not np.array_equal(snapshots(root,n,1,1,1), values[None,:]): raise ValueError('y-split mapping changed')
        (root/'snapshots.rank0.bin').write_bytes(box)
        (root/'snapshots.rank1.bin').write_bytes(box)
        try: snapshots(root,n,1,1,1)
        except ValueError: pass
        else: raise ValueError('Overlapping boxes accepted')
        (root/'snapshots.rank1.bin').unlink()
        (root/'snapshots.rank0.bin').write_bytes(box[:-1])
        try: snapshots(root,n,1,1,1)
        except (ValueError, struct.error): pass
        else: raise ValueError('Truncated stream accepted')
    return 'legacy, box, y-split, missing coverage, overlap and truncation checks passed'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for root in sorted(args.input.glob('serial-*')):
        for case in root.glob('*_*/mock0'):
            baseline[(root.name[len('serial-'):], case.parent.name)] = read(case)
    if not baseline: raise ValueError('Missing serial reference outputs')
    results = []
    for root in sorted(args.input.glob('mpi*-*')):
        rank, engine = root.name.split('-',1)
        for case in sorted(root.glob('*_*/mock0')):
            key = (engine, case.parent.name)
            expected = baseline[key]; actual = read(case)
            errors = [relative(a,b) for a,b in zip(actual,expected)]
            if max(errors) > 2e-8: raise ValueError(f'Replay mismatch {case}: {errors}')
            results.append(dict(run=root.name, case=case.parent.name,
                                cells=errors[0], snapshots=errors[1], trace=errors[2], predictive=errors[3]))
    report = dict(comparisons=results, snapshot_reader=reader_checks(),
                  maximum_error=max((max(r[k] for k in ('cells','snapshots','trace','predictive')) for r in results), default=0),
                  interpretation='Bounded fixed-step replay; not convergence or posterior calibration')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    if results:
        labels = [r['run'].replace('survey_','')+' '+r['case'] for r in results]
        for key in ('cells','snapshots','trace','predictive'):
            ax.plot(np.maximum(1e-18,[r[key] for r in results]), marker='.', label=key)
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha='right', fontsize=7)
        ax.set_yscale('log');ax.set_ylabel('Maximum scaled replay difference')
        ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(.5, 1.01))
    fig.suptitle('Frontend slab/pencil replay — bounded smoke runs, not posterior calibration', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(args.output/'replay.png',dpi=160);fig.savefig(args.output/'replay.pdf');plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Run the bounded survey experiment, preserving commands and failure statuses."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi-bin-dir', type=Path)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--smoke', action='store_true', help='Bounded replay only; no calibration claim')
    parser.add_argument('--mpi-ranks', nargs='+', type=int, default=[1,2,4])
    args = parser.parse_args()
    if any(n < 1 for n in args.mpi_ranks): parser.error('MPI ranks must be positive')
    output = args.output.resolve()
    # A fresh directory prevents accidental append/overwrite of chain snapshots.
    output.mkdir(parents=True, exist_ok=False)
    binaries = args.bin_dir.resolve()
    records = []

    def run(name, command):
        command = list(map(str, command))
        print(f'Running {name}', flush=True)
        start = time.monotonic()
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(name=name, argv=command, exit_code=result.returncode,
                            seconds=time.monotonic() - start))
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(f'{name} failed; see {output / (name + ".log")}')

    source = Path(__file__).resolve().parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.iterdir() if p.suffix in ('.cpp', '.py')}
    hashes.update({str(binaries / name): hashlib.sha256((binaries / name).read_bytes()).hexdigest()
                   for name in ('survey_validate', 'survey_reference')})
    if args.mpi_bin_dir:
        for name in ('survey_validate', 'survey_reference'):
            p = args.mpi_bin_dir.resolve()/name
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (output / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    run('reference-check', [binaries / 'survey_reference', '--check-reference'])
    if args.smoke:
        def smoke_cases(prefix, folder, launcher):
            for engine in ('survey_validate', 'survey_reference'):
                for evolution in ('linear','2lpt'):
                    for kind in ('gaussian','poisson'):
                        run(f'{prefix}-{engine}-{evolution}-{kind}', launcher+[folder/engine,'--smoke',output/f'{prefix}-{engine}',kind,0,1,2,16,4,evolution])
        smoke_cases('serial',binaries,[])
        if args.mpi_bin_dir:
            launcher=shlex.split(args.mpirun);mpi=args.mpi_bin_dir.resolve()
            for ranks in args.mpi_ranks:
                run(f'reference-check-np{ranks}',launcher+['-np',ranks,mpi/'survey_reference','--check-reference'])
                smoke_cases(f'mpi{ranks}',mpi,launcher+['-np',ranks])
        run('portability-analysis',[sys.executable,source/'analyze_portability.py','--input',output,'--output',output/'figures'])
        return

    for kind in ('gaussian', 'poisson'):
        run(kind + '-full0', [binaries / 'survey_validate', output / 'main', kind, 0, 1, 4, 1000, 400])
        run(kind + '-full1-2', [binaries / 'survey_validate', output / 'main', kind, 1, 2, 2, 800, 400])
        # Mock 3 needed longer sampling in the all-voxel check; preserve that
        # measured refinement rather than repeating the known short run.
        batches = [(0, 12, 800)] if kind == 'gaussian' else [(0, 3, 800), (3, 1, 1600), (4, 8, 800)]
        for first, count, draws in batches:
            run(f'{kind}-reference-{first}', [binaries / 'survey_reference', output / 'reference_main', kind, first, count, 2, draws, 400])
    for name, exe in [('reference', 'survey_validate'), ('reference_active', 'survey_reference')]:
        run(name + '-linear', [binaries / exe, output / name, 'gaussian', 0, 1, 4, 1000, 400, 'linear'])
    run('convergence', [binaries / 'survey_validate', '--convergence', output / 'convergence'])
    mpi_args = []
    if args.mpi_bin_dir:
        launcher = shlex.split(args.mpirun)
        mpi_exe = args.mpi_bin_dir.resolve() / 'survey_reference'
        for ranks in args.mpi_ranks:
            run(f'reference-check-np{ranks}', launcher + ['-np', ranks, mpi_exe, '--check-reference'])
        for kind in ('gaussian', 'poisson'):
            run(kind + '-mpi2', launcher + ['-np', 2, mpi_exe, output / 'mpi2', kind, 0, 1, 2, 800, 400])
        mpi_args = ['--mpi', output / 'mpi2']
    run('analysis', [sys.executable, source / 'analyze.py', '--input', output / 'reference_main',
                    '--production', output / 'main', '--reference', output / 'reference_active/linear_gaussian/mock0',
                    '--full-linear', output / 'reference/linear_gaussian/mock0',
                    '--convergence', output / 'convergence', '--output', output / 'figures'] + mpi_args)
    print(f'Completed; plots: {output / "figures"}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run isolated processes; retain raw JSON/logs plus host/build/executable provenance."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys

CASES = ['fft_roundtrip', 'grid_copy', 'grid_clone', 'resample', 'migration', 'polyspectrum']


def execute(command, env, timeout, log):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        log.write_text(output)
        raise RuntimeError(f'timeout: {command}; see {log}')
    log.write_text(output)
    if process.returncode:
        raise RuntimeError(f'exit {process.returncode}: {command}; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory; existing runs are never overwritten')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--launcher', default='mpirun', help='MPI launcher plus optional arguments, parsed without a shell')
    parser.add_argument('--mesh', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--message-bytes', type=int, default=1048576)
    parser.add_argument('--cache-shells', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if any(n < 1 for n in args.ranks + args.threads) or args.repeat < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error('ranks, threads, repeat and timeout must be positive; warmup must be nonnegative')
    if args.mesh < 8 or any(args.mesh % n for n in args.ranks):
        parser.error('mesh must be >=8 and divisible by every requested rank count')
    exe = args.exe.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='false', OMP_WAIT_POLICY='passive')
    probe = output / 'probe.json'
    execute([str(exe), '--case', 'grid_clone', '--mesh', '8', '--repeat', '1', '--warmup', '0', '--output', str(probe)],
            env, args.timeout, output / 'probe.log')
    metadata = json.loads(probe.read_text())
    if not metadata['mpi_enabled'] and any(n != 1 for n in args.ranks):
        raise RuntimeError('serial executable cannot be launched as a multi-rank benchmark')
    if not metadata['omp_enabled'] and any(n != 1 for n in args.threads):
        raise RuntimeError('non-OpenMP executable requires --threads 1')
    manifest = {
        'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'arguments': vars(args) | {'exe': str(exe), 'output': str(output)},
        'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
        'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(),
        'hostname': platform.node(), 'logical_cpus': os.cpu_count(), 'python': sys.version,
        'driver_cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'environment': {k: env[k] for k in ('OMP_DYNAMIC', 'OMP_WAIT_POLICY', 'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_THREAD_LIMIT', 'OMP_SCHEDULE', 'OMP_MAX_ACTIVE_LEVELS') if k in env},
        'completed': False,
    }
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    save_manifest()
    with (output / 'results.jsonl').open('w') as rows:
        for ranks in dict.fromkeys(args.ranks):
            for threads in dict.fromkeys(args.threads):
                env['OMP_NUM_THREADS'] = str(threads)
                for case in dict.fromkeys(args.cases):
                    stem = f'{case}.r{ranks}.t{threads}'
                    report = output / (stem + '.json')
                    command = ([*shlex.split(args.launcher), '-np', str(ranks)] if metadata['mpi_enabled'] else []) + [
                        str(exe), '--case', case, '--mesh', str(args.mesh), '--repeat', str(args.repeat),
                        '--warmup', str(args.warmup), '--seed', str(args.seed), '--message-bytes', str(args.message_bytes),
                        '--cache-shells', str(args.cache_shells), '--output', str(report)]
                    (output / (stem + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
                    execute(command, env, args.timeout, output / (stem + '.log'))
                    result = json.loads(report.read_text())
                    if result['ranks'] != ranks or not result['validation_passed']:
                        raise RuntimeError(f'invalid report: {report}')
                    rows.write(json.dumps(result) + '\n')
                    rows.flush()
                    print(f'{stem}: {result["median_max_rank_seconds"]:.6g} s', flush=True)
    manifest['completed'] = True
    save_manifest()


if __name__ == '__main__':
    main()

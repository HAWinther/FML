"""Predeclared four-mock replication of the frozen same-model N16 protocol.

A failed mock is retained and the campaign continues. No replacements, retuning,
chain pooling across datasets, or extension until convergence are permitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['all', 'cases', 'analyze'], default='all')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    out = a.output.resolve()
    bins = a.bin_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.phase in ['all', 'cases']:
        if any(out.iterdir()):
            p.error('Use a fresh output directory; completed or failed mocks are never replaced')
        plan = dict(
            schema='FML-N16-repeated-mocks-v1',
            mocks=[dict(id=i, directory='mock'+str(i), ic_seed=291+100*i,
                        noise_seed=292+100*i) for i in range(4)],
            protocol=dict(n=16, m=16, steps=16, chains=4, warmup=64, draws=512,
                          leapfrog=6, epsilon=.25, target_acceptance=.85,
                          divergence_threshold=10, adapt_mass=False,
                          predictive_stride=32, hmc_seed0=12000,
                          scalar_seed0=23000, timeout_seconds=7200),
            gates=dict(rhat=1.05, ess_bulk=400, ess_tail=200),
            scope='Four predeclared fresh same-model N16 mocks; fixed A=0 and noise power=1e-4. Limited repeated-data recovery, not SBC.',
            historical_mocks_excluded=[dict(ic_seed=191, noise_seed=192)],
            failure_policy='Keep every failure; continue remaining declared mocks; no replacements, retuning or extra draws.',
            analysis_policy='Diagnose each dataset separately; no cross-mock chain pooling. Report all declared mocks, missing results and uncertainty on interval inclusion.',
            binaries={name: digest(bins/name) for name in ['pm_tsc_posterior', 'pm_tsc_joint', 'pm_tsc_diagnostics']},
            drivers={name: digest(here/name) for name in ['run_mocks.py', 'run_mock.py', 'analyze.py']})
        (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        records = []
        for mock in plan['mocks']:
            # Stop on provenance drift, rather than silently changing the experiment.
            for name, sha in plan['binaries'].items():
                if digest(bins/name) != sha:
                    raise RuntimeError('Binary changed during campaign: '+name)
            for name, sha in plan['drivers'].items():
                if digest(here/name) != sha:
                    raise RuntimeError('Driver changed during campaign: '+name)
            cmd = [sys.executable, str(here/'run_mock.py'), '--bin-dir', str(bins),
                   '--output', str(out/mock['directory']), '--ic-seed', str(mock['ic_seed']),
                   '--noise-seed', str(mock['noise_seed']), '--jobs', '4',
                   '--timeout-seconds', '7200', '--phase', 'all']
            record = dict(id=mock['id'], argv=cmd)
            records.append(record)
            start = time.monotonic()
            with (out/(mock['directory']+'.log')).open('w') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record['pid'] = child.pid
                (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
                try:
                    code = child.wait()
                except BaseException:
                    # run_mock catches KeyboardInterrupt and cleans its own native groups.
                    try:
                        os.killpg(child.pid, signal.SIGINT)
                        child.wait(timeout=20)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=10)
                    raise
            record.update(exit_code=code, seconds=time.monotonic()-start)
            (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
            print(mock['directory'], 'exit', code, flush=True)
    if a.phase in ['all', 'analyze']:
        return subprocess.call([sys.executable, str(here/'analyze_mocks.py'),
                                '--input', str(out), '--output', str(out/'analysis')])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

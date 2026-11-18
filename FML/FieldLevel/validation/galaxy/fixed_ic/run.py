"""Frozen, bounded fixed-IC field pilot. Outputs must be outside the library source tree."""
import argparse, concurrent.futures, csv, hashlib, json, os, subprocess, shutil
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    out, exe = a.output.resolve(), a.exe.resolve()
    library_root = root.parents[2]
    if out == library_root or library_root in out.parents or a.workers < 1:
        raise ValueError('Use an external output directory and positive worker count')
    plan_path = a.plan.resolve()
    plan = json.loads(plan_path.read_text())
    # This maintained driver intentionally implements one frozen fixture, not a general catalog/config reader.
    joint = plan.get('sampler','rescaled_bias') == 'joint_gaussian'
    if plan.get('sampler','rescaled_bias') not in ['rescaled_bias','joint_gaussian']:
        raise ValueError('Unsupported declared sampler')
    if plan['amplitude_step'] != .025 or plan['bias_sweeps'] != (0 if joint else 8) or plan['generator'] != [16,48,8]:
        raise ValueError('Native fixture/proposal configuration does not match this protocol')
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=sha(plan_path), executable_sha256=sha(exe))
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[k] != v for k,v in identity.items()):
            raise ValueError('Existing output belongs to a different executable or protocol')
    else:
        if any(out.iterdir()):
            raise ValueError('A fresh run needs an empty directory')
        manifest = dict(identity, plan=plan, status='running', completed=[])
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='passive')
    def command(args, log):
        with log.open('w') as f:
            subprocess.run([str(exe), *map(str,args)], stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
    if not (out/'mock.data.csv').exists():
        if plan.get('data_fixture'):
            fixture=(root/plan['data_fixture']).resolve()
            if root not in fixture.parents:raise ValueError('Fixture must be inside the driver directory')
            for ext in ['data.csv','problem.csv']:
                shutil.copyfile(Path(str(fixture)+'.'+ext),out/('mock.'+ext))
        else:
            command(['prepare', out/'mock'], out/'prepare.log')
    data = out/'mock.data.csv'
    if 'data_sha256' in manifest and manifest['data_sha256'] != sha(data):
        raise ValueError('Frozen data changed')
    if plan.get('data_sha256') and plan['data_sha256']!=sha(data):
        raise ValueError('Data does not match the declared frozen fixture')
    manifest['data_sha256'] = sha(data)
    with (out/'mock.problem.csv').open() as f:
        problem = {k:float(v) for k,v in next(csv.DictReader(f)).items()}
    expected = dict(boxsize=plan['boxsize'],initial_kf=plan['initial_cutoff_in_fundamental_modes'],
                    matter_kf=plan['matter_cutoff_in_fundamental_modes'],likelihood_kf=plan['likelihood_cutoff_in_fundamental_modes'],
                    noise_power=plan['noise_power'],n=plan['generator'][0],nd=plan['generator'][1],width=plan['generator'][2],
                    ic_seed=plan['ic_seed'],noise_seed=plan['noise_seed'])
    for j in range(5):
        expected.update({f'truth{j}':plan['truth'][j],f'lower{j}':plan['prior_lower'][j],f'upper{j}':plan['prior_upper'][j]})
    if any(abs(problem[k]-v)>1e-12 for k,v in expected.items()):
        raise ValueError('Native data/problem disagrees with the declared protocol')
    def save():
        temp = out/'manifest.tmp'
        temp.write_text(json.dumps(manifest,indent=2)+'\n'); temp.replace(manifest_path)
    save()
    jobs = []
    for n,nd,width in plan['cases']:
        case = out/f'n{n}-nd{nd}-w{width}'; case.mkdir(exist_ok=True)
        for j in range(plan['chains']):
            prefix = case/f'chain{j}'
            jobs.append((prefix,n,nd,width,plan['seeds'][j],plan['initial_alpha'][j]))
    def run(job):
        prefix,n,nd,width,seed,alpha = job
        chain, meta = Path(str(prefix)+'.chain.csv'), Path(str(prefix)+'.meta.csv')
        key = str(prefix.relative_to(out))
        if key in manifest['completed']:
            if sha(chain) != manifest['chain_sha256'][key]:
                raise ValueError('Completed chain changed')
            return key
        command(['chain-joint' if joint else 'chain',data,prefix,n,nd,width,seed,alpha,plan['warmup'],plan['draws']],Path(str(prefix)+'.log'))
        with meta.open() as f:
            m = {k:float(v) for k,v in next(csv.DictReader(f)).items()}
        expected_meta = dict(n=n,nd=nd,width=width,seed=seed,warmup=plan['warmup'],draws=plan['draws'],
                             amplitude_step=plan['amplitude_step'],bias_sweeps=plan['bias_sweeps'],joint_proposal=int(joint))
        if any(m[k]!=v for k,v in expected_meta.items()) or sum(1 for _ in chain.open())!=plan['draws']+1:
            raise ValueError('Incomplete or mismatched native chain')
        return key
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            key = future.result()
            manifest.setdefault('chain_sha256',{})[key] = sha(out/(key+'.chain.csv'))
            if key not in manifest['completed']: manifest['completed'].append(key)
            save(); print('Completed',key,flush=True)
    for n,nd,width in plan['cases']:
        case=out/f'n{n}-nd{nd}-w{width}'
        files=[case/f'chain{j}.chain.csv' for j in range(plan['chains'])]
        command(['diagnose',case/'diagnostics.csv',*files],case/'diagnose.log')
        arrays=[np.genfromtxt(f,names=True,delimiter=',') for f in files]
        means=[float(np.mean(np.concatenate([x[name] for x in arrays]))) for name in plan['parameter_order']]
        command(['predict',data,case/'mean',n,nd,width,*means],case/'predict.log')
    if sha(data)!=manifest['data_sha256']: raise ValueError('Frozen data changed during run')
    manifest['status']='complete'; save()

if __name__=='__main__':
    main()

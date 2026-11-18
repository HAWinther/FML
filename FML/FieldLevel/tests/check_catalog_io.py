"""Small end-to-end catalog example checks; provide already-built serial/MPI executables."""
import argparse,json,subprocess,shlex,sys,tempfile,os
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--serial',type=Path,required=True)
    p.add_argument('--mpi-slab',type=Path);p.add_argument('--mpi-pencil',type=Path)
    p.add_argument('--mpirun',default='mpirun');p.add_argument('--output',type=Path)
    a=p.parse_args();temp=None
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='passive')
    if a.output:out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    else:temp=tempfile.TemporaryDirectory(prefix='fml-catalog-');out=Path(temp.name)
    i=np.arange(59);x=np.column_stack([100*((.013+i*.38196601125)%1),100*((.71+i*.17320508075)%1),100*((.29+i*.27182818284)%1)])
    x[0]=[0,100,-100];x[1]=[50,50,50];x[2]=[100,0,100];x[i%3==0,0]-=200;x[i%5==0,1]+=300
    w=np.where(i%7==0,0.,.25+i*i)
    np.savetxt(out/'catalog.csv',np.column_stack([x,w]),delimiter=',',header='x,y,z,weight',comments='',fmt='%.17g')
    records=[];reference=None
    def execute(exe,label,np_=1,epsilon=1e-8,batch=13,path=None,expect=True):
        nonlocal reference
        prefix=out/label;cmd=([*shlex.split(a.mpirun),'-np',str(np_)] if np_>1 else [])+[str(exe.resolve()),str(path or out/'catalog.csv'),str(prefix),'16','100','.45','cube',str(batch),str(epsilon)]
        with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=60,env=env)
        if not expect:
            assert r.returncode!=0,label;records.append(dict(check=label,passed=True));return
        assert r.returncode==0,label
        m=json.loads(Path(str(prefix)+'.json').read_text());assert m['rows']==len(x)
        assert np.isclose(m['mode_variance'],np.sum(w*w)/sum(w)**2,rtol=2e-14,atol=0)
        parts=[np.atleast_1d(np.genfromtxt(str(prefix)+f'.rank{rank}.modes.csv',names=True,delimiter=',')) for rank in range(np_)]
        modes=np.concatenate(parts);order=np.lexsort((modes['nz'],modes['ny'],modes['nx']));modes=modes[order]
        n=np.column_stack([modes['nx'],modes['ny'],modes['nz']]);actual=modes['delta_re']+1j*modes['delta_im']
        direct=np.exp(-2j*np.pi*(n@x.T)/100)@w/sum(w)
        err=float(max(abs(actual-direct)));assert err<max(3*epsilon,2e-12),(label,err)
        if reference is None:reference=actual
        if epsilon==1e-8:assert np.allclose(actual,reference,rtol=0,atol=2e-12),label
        assert np.any(modes['k']>.45) # Cubic corners must survive the radial magnitude .45.
        records.append(dict(check=label,epsilon=epsilon,absolute_error=err,rows=m['rows'],fine_mesh=m['fine_mesh'],passed=True))
    execute(a.serial,'serial')
    execute(a.serial,'batch-one',batch=1)
    for kind,exe in [('slab',a.mpi_slab),('pencil',a.mpi_pencil)]:
        if exe:
            for np_ in [1,2,4]:execute(exe,f'{kind}-{np_}',np_)
    for epsilon in [1e-4,1e-6,1e-10]:execute(a.serial,f'tolerance-{epsilon}',epsilon=epsilon)
    for text,name in [('1,,2,3\n','missing-field'),('1,2,3,4,5\n','extra-field'),('1,2,3,-1\n','negative-weight')]:
        f=out/(name+'.csv');f.write_text(text);execute(a.serial,name,path=f,expect=False)
    try:import h5py
    except ImportError:records.append(dict(check='hdf5',skipped='h5py not installed'))
    else:
        wrapper=Path(__file__).resolve().parents[1]/'example/catalog_hdf5.py'
        for kind in ['compound','columns']:
            file=out/(kind+'.h5')
            with h5py.File(file,'w') as f:
                if kind=='compound':
                    data=np.empty(len(x),dtype=[(k,'f8') for k in ['x','y','z','weight']]);
                    for j,k in enumerate(['x','y','z']):data[k]=x[:,j]
                    data['weight']=w;f.create_dataset('galaxies',data=data)
                else:
                    g=f.create_group('galaxies')
                    for j,k in enumerate(['x','y','z']):g.create_dataset(k,data=x[:,j])
                    g.create_dataset('weight',data=w)
            prefix=out/kind
            cmd=[sys.executable,str(wrapper),'--input',str(file),'--exe',str(a.serial.resolve()),'--output-prefix',str(prefix),'--mesh','16','--boxsize','100','--cutoff','.45','--weight','weight','--batch-rows','7']
            with (out/(kind+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60,env=env)
            modes=np.genfromtxt(str(prefix)+'.rank0.modes.csv',names=True,delimiter=',');order=np.lexsort((modes['nz'],modes['ny'],modes['nx']));modes=modes[order]
            err=float(max(abs(modes['delta_re']+1j*modes['delta_im']-reference)));assert err<2e-12,(kind,err)
            records.append(dict(check='hdf5-'+kind,max_difference=err,passed=True))
        if a.mpi_pencil:
            prefix=out/'hdf5-pencils'
            cmd=[sys.executable,str(wrapper),'--input',str(out/'compound.h5'),'--exe',str(a.mpi_pencil.resolve()),'--output-prefix',str(prefix),'--mesh','16','--boxsize','100','--cutoff','.45','--weight','weight','--batch-rows','7','--np','4','--mpirun',a.mpirun]
            with (out/'hdf5-pencils.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60,env=env)
            pieces=[np.atleast_1d(np.genfromtxt(str(prefix)+f'.rank{r}.modes.csv',names=True,delimiter=',')) for r in range(4)]
            modes=np.concatenate(pieces);order=np.lexsort((modes['nz'],modes['ny'],modes['nx']));modes=modes[order]
            err=float(max(abs(modes['delta_re']+1j*modes['delta_im']-reference)));assert err<2e-12
            records.append(dict(check='hdf5-pencils-4',max_difference=err,passed=True))
    (out/'io-checks.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
    if temp:temp.cleanup()

if __name__=='__main__':main()

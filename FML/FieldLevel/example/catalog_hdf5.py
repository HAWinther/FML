"""Stream periodic HDF5 coordinates to catalog_to_modes, without an intermediate catalog file.
Requires NumPy/h5py; the C++ library has no HDF5 dependency. Units must match --boxsize.
"""
import argparse, shlex, subprocess, json
from pathlib import Path
import numpy as np
import h5py

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--mesh',type=int,required=True)
    p.add_argument('--boxsize',type=float,required=True);p.add_argument('--cutoff',type=float,required=True)
    p.add_argument('--geometry',choices=['cube','sphere'],default='cube')
    p.add_argument('--group',default='galaxies');p.add_argument('--columns',nargs=3,default=['x','y','z'])
    p.add_argument('--weight');p.add_argument('--batch-rows',type=int,default=100000)
    p.add_argument('--epsilon',type=float,default=1e-8);p.add_argument('--np',type=int,default=1)
    p.add_argument('--mpirun',default='mpirun')
    a=p.parse_args()
    if a.batch_rows<1 or a.np<1:p.error('batch-rows and np must be positive')
    names=[*a.columns]+([a.weight] if a.weight else [])
    with h5py.File(a.input,'r') as f:
        catalog=f[a.group]
        compound=isinstance(catalog,h5py.Dataset)
        if compound:
            if catalog.ndim!=1 or not catalog.dtype.names or any(n not in catalog.dtype.names for n in names):
                raise ValueError('Expected a one-dimensional compound dataset with the declared fields')
            count=len(catalog)
        else:
            if any(catalog[n].ndim!=1 for n in names) or len({len(catalog[n]) for n in names})!=1:
                raise ValueError('Coordinate/weight columns must be one-dimensional with equal lengths')
            count=len(catalog[names[0]])
        cmd=([*shlex.split(a.mpirun),'-np',str(a.np)] if a.np>1 else [])+[str(a.exe.resolve()),'-',str(a.output_prefix.resolve()),str(a.mesh),str(a.boxsize),str(a.cutoff),a.geometry,str(a.batch_rows),str(a.epsilon)]
        child=subprocess.Popen(cmd,stdin=subprocess.PIPE,text=True)
        try:
            for start in range(0,count,a.batch_rows):
                end=min(count,start+a.batch_rows)
                if compound:
                    block=catalog.fields(names)[start:end]
                    values=np.column_stack([block[n] for n in names])
                else:
                    values=np.column_stack([catalog[n][start:end] for n in names])
                np.savetxt(child.stdin,values,delimiter=',',fmt='%.17g')
            # Explicit framing avoids relying on MPI launchers to forward a pipe EOF promptly.
            child.stdin.write('#FML_CATALOG_END\n')
            child.stdin.close()
            if child.wait()!=0:raise RuntimeError('catalog_to_modes failed; inspect its diagnostic output')
            metadata=json.loads(Path(str(a.output_prefix.resolve())+'.json').read_text())
            if metadata['rows']!=count:raise RuntimeError('Native output did not consume every HDF5 row')
        except BaseException:
            try:child.stdin.close()
            except (BrokenPipeError,OSError):pass
            if child.poll() is None:child.terminate()
            child.wait()
            raise

if __name__=='__main__':main()

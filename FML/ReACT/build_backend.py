#!/usr/bin/env python3
"""Build pinned ReACT + SUNDIALS privately; no root or system installation required."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REACT_REV = 'a36dda02db6b537a4ddcc7bf044c446c10a58373'
SUNDIALS_REV = '881083722e6140c96c6381b196da6258b97862bc'


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def checkout(path, url, revision):
    if not path.exists():
        run('git', 'clone', '--no-checkout', '--filter=blob:none', url, path)
        run('git', '-C', path, 'checkout', '--detach', revision)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'{path}: expected {revision}, found {actual}; use a fresh build directory')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, default=HERE / '.backend')
    ap.add_argument('--build-dir', type=Path, default=HERE / '.build')
    ap.add_argument('--react-source', type=Path)
    ap.add_argument('--sundials-source', type=Path)
    ap.add_argument('--gsl-prefix', type=Path, default=Path.home() / 'local')
    ap.add_argument('--cc', default=os.environ.get('CC', 'clang' if sys.platform == 'darwin' else 'cc'))
    ap.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    ap.add_argument('-j', type=int, default=4)
    a = ap.parse_args()
    if a.j < 1:
        ap.error('-j must be positive')
    a.prefix, a.build_dir = a.prefix.resolve(), a.build_dir.resolve()
    a.build_dir.mkdir(parents=True, exist_ok=True)
    react = (a.react_source or a.build_dir / 'ACTio-ReACTio').resolve()
    sundials = (a.sundials_source or a.build_dir / 'sundials').resolve()
    checkout(react, 'https://github.com/nebblu/ACTio-ReACTio.git', REACT_REV)
    checkout(sundials, 'https://github.com/LLNL/sundials.git', SUNDIALS_REV)
    deps = a.build_dir / 'deps'
    sb = a.build_dir / 'sundials-build'
    run('cmake', '-S', sundials, '-B', sb, f'-DCMAKE_C_COMPILER={a.cc}',
        f'-DCMAKE_INSTALL_PREFIX={deps}', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_STATIC_LIBS=ON', '-DBUILD_ARKODE=OFF',
        '-DBUILD_CVODES=OFF', '-DBUILD_IDA=OFF', '-DBUILD_IDAS=OFF', '-DBUILD_KINSOL=OFF',
        '-DEXAMPLES_ENABLE_C=OFF', '-DEXAMPLES_ENABLE_CXX=OFF', '-DEXAMPLES_INSTALL=OFF')
    run('cmake', '--build', sb, '-j', a.j)
    run('cmake', '--install', sb)
    rb = a.build_dir / 'react-build'
    extra = ['-DCMAKE_OSX_SYSROOT='] if sys.platform == 'darwin' else []
    run('cmake', '-S', HERE, '-B', rb, f'-DREACT_SOURCE={react}', f'-DSUNDIALS_ROOT={deps}',
        f'-DGSL_ROOT_DIR={a.gsl_prefix.resolve()}', f'-DCMAKE_C_COMPILER={a.cc}',
        f'-DCMAKE_CXX_COMPILER={a.cxx}', f'-DCMAKE_INSTALL_PREFIX={a.prefix}',
        '-DCMAKE_BUILD_TYPE=Release', *extra)
    run('cmake', '--build', rb, '-j', a.j)
    run('cmake', '--install', rb)
    print(f'Backend installed. Build FML with USE_REACT=true REACT_LIB={a.prefix / "lib"}')


if __name__ == '__main__':
    main()

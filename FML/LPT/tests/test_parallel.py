#!/usr/bin/env python3
"""Cross-rank/thread/reference validation and fresh-process workspace measurements."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile

import eds_reference as ref
from test_eds_reference import error, norm


def run(command, threads):
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=environment, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise RuntimeError(f"MPI test exceeded 90 seconds: {command}")
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {command}\n{output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("double", "float"), default="double")
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--np-list", default="1 2 4")
    parser.add_argument("--exe", default="./test_parallel")
    args = parser.parse_args()
    tolerance = 3e-5 if args.precision == "float" else 3e-12
    launcher = shlex.split(args.launcher)
    ranks = [int(value) for value in args.np_list.split()]
    if not ranks or any(value not in (1, 2, 4) for value in ranks):
        raise ValueError("This reference suite supports NP_LIST containing 1, 2 and/or 4")
    executable = str(Path(args.exe).resolve())
    def command(count):
        return [*launcher, "-np", str(count), executable]

    for count in ranks:
        output = run(command(count), 2)
        print(f"ranks={count} threads=2: {output.splitlines()[-1]}", flush=True)

    positive = {(3, 0, 0): .1 + .02j, (-2, 1, 0): -.07 + .03j, (0, 1, 2): .06 - .01j}
    high = {**positive, **{tuple(-x for x in mode): value.conjugate() for mode, value in positive.items()}}
    checks = 0
    worst = 0
    with tempfile.TemporaryDirectory(prefix="fml-lpt-parallel-") as directory:
        root = Path(directory)
        for mesh, cutoff, delta in [(16, 2, ref.fixture()), (8, 3, high)]:
            oracle = ref.solve(delta, 5)
            expected = {n: [{mode: value for mode, value in component.items()
                            if mode[2] >= 0 and all(2*abs(x) < mesh for x in mode)} for component in field]
                        for n, field in oracle.items()}
            input_path = root / "input.txt"
            input_path.write_text("".join(f"{x} {y} {z} {value.real:.17g} {value.imag:.17g}\n"
                                          for (x, y, z), value in sorted(delta.items())))
            baseline = None
            for count in ranks:
                for threads in (1, 2):
                    for policy in ("cache", "recompute"):
                        prefix = root / "fields"
                        run([*command(count), "--export", str(input_path), str(prefix),
                             str(mesh), str(cutoff), "5", policy], threads)
                        actual = {n: [{}, {}, {}] for n in range(1, 6)}
                        for rank in range(count):
                            for line in Path(f"{prefix}.{rank}").read_text().splitlines():
                                n, x, y, z, axis, re, im = line.split()
                                field = actual[int(n)][int(axis)]
                                mode = (int(x), int(y), int(z))
                                if mode in field:
                                    raise AssertionError("Overlapping MPI output ownership")
                                field[mode] = complex(float(re), float(im))
                        if sum(len(c) for field in actual.values() for c in field) != mesh*mesh*(mesh//2+1)*15:
                            raise AssertionError("Incomplete MPI output ownership")
                        for n in range(1, 6):
                            difference = error(actual[n],expected[n])
                            if difference > tolerance*norm(expected[n])+1e-18:
                                raise AssertionError(f"Order {n} disagrees with sparse oracle")
                            worst = max(worst, difference/max(norm(expected[n]),1e-300))
                            if baseline and error(actual[n],baseline[n]) > tolerance*norm(baseline[n])+1e-18:
                                raise AssertionError("Rank/thread/workspace comparison failed")
                        if baseline is None:
                            baseline = actual
                        checks += 1
                        print(f"reference pass mesh={mesh} ranks={count} threads={threads} workspace={policy}", flush=True)
        measurements = []
        for count in ranks:
            for policy in ("cache", "recompute"):
                path = root / "benchmark.json"
                run([*command(count), "--benchmark", str(path), policy], 2)
                data = json.loads(path.read_text())
                if data["accounted_max_bytes"] > data["planned_max_bytes"]:
                    raise AssertionError("Benchmark exceeded its planned owned capacity")
                measurements.append(data)
                print("measurement " + json.dumps(data, sort_keys=True), flush=True)
        for count in ranks:
            cached = next(row for row in measurements if row["ranks"] == count and row["workspace"] == "cache")
            recomputed = next(row for row in measurements if row["ranks"] == count and row["workspace"] == "recompute")
            if recomputed["planned_max_bytes"] >= cached["planned_max_bytes"]:
                raise AssertionError("Order-8 recomputation did not reduce the owned memory bound")
            if recomputed["fft_count"] <= cached["fft_count"]:
                raise AssertionError("Recomputation transform cost was not reported")
    print(f"All {checks} rank/thread/workspace reference cases passed; worst relative coefficient error {worst:.3g}")


if __name__ == "__main__":
    main()

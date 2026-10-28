"""Optional isolated-Plummer comparison with an externally built ROCKSTAR executable.

Build `rockstar_fixture` in this directory in serial mode first. No reference code is downloaded or bundled.
Masses have different definitions; the script checks counts and Vmax, and reports rather than asserts mass equality.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("rockstar", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("rockstar_fixture"))
parser.add_argument("--reference-label", default="external ROCKSTAR")
args = parser.parse_args()
directory = args.output_directory.resolve()
directory.mkdir(parents=True, exist_ok=True)
with (directory / "fml.log").open("w") as log:
    subprocess.run([str(args.fixture.resolve()), str(directory)], check=True, stdout=log, stderr=subprocess.STDOUT)
config = directory / "rockstar.cfg"
config.write_text(f'''FILE_FORMAT = "ASCII"
PARTICLE_MASS = 20815247.025
BOX_SIZE = 1
TOTAL_PARTICLES = 4000
SCALE_NOW = 1
h0 = 0.7
Om = 0.3
Ol = 0.7
FORCE_RES = 0.0005
MASS_DEFINITION = "200b"
OUTPUT_FORMAT = "ASCII"
OUTBASE = "{directory}"
PARALLEL_IO = 0
FOF_LINKING_LENGTH = 0.28
MIN_HALO_OUTPUT_SIZE = 20
''')
with (directory / "rockstar.log").open("w") as log:
    subprocess.run([str(args.rockstar.resolve()), "-c", str(config), str(directory / "plummer.ascii")],
                   check=True, cwd=directory, stdout=log, stderr=subprocess.STDOUT)

def data_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]

fml = data_rows(directory / "fml_summary.txt")
reference = data_rows(directory / "halos_0.0.ascii")
if len(fml) != 1 or len(reference) != 1:
    raise RuntimeError(f"Expected one isolated halo: FML={len(fml)}, ROCKSTAR={len(reference)}")
f, r = list(map(float, fml[0])), list(map(float, reference[0]))
result = {"reference": args.reference_label, "particles": 4000,
          "fml_mass_exclusive_bound": f[1], "rockstar_mass_bound_200b": r[3],
          "fml_vmax": f[2], "rockstar_vmax": r[5],
          "vmax_fractional_difference": f[2] / r[5] - 1,
          "mass_fractional_difference_different_definitions": f[1] / r[3] - 1,
          "centre_separation_mpch": sum((f[3+d] - r[8+d])**2 for d in range(3))**.5}
(directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if abs(result["vmax_fractional_difference"]) > .05 or result["centre_separation_mpch"] > .005:
    raise RuntimeError("Controlled Plummer comparison exceeds the stated Vmax/centre tolerance")

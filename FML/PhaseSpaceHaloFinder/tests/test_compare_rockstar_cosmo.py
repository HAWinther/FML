"""End-to-end checks for integer ID preservation and sparse membership joins.

Run with Python, NumPy and SciPy; no external ROCKSTAR installation is needed.
"""
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = pathlib.Path(__file__).with_name("compare_rockstar_cosmo.py")
NO_HALO = 2**64 - 1


class ComparisonTests(unittest.TestCase):
    def run_script(self, *args):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_convert_preserves_uint64_ids(self):
        ids = np.array([2**53, 2**53 + 1, 2**63 + 7, NO_HALO - 1], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "snapshot"
            output = pathlib.Path(directory) / "particles"
            header = bytearray(256)
            struct.pack_into("6I", header, 0, 0, len(ids), 0, 0, 0, 0)
            struct.pack_into("6d", header, 24, 0, 1, 0, 0, 0, 0)
            struct.pack_into("d", header, 72, 1)  # scale factor
            struct.pack_into("d", header, 128, 1)  # boxsize
            struct.pack_into("d", header, 136, .3)
            with source.open("wb") as stream:
                for block in (header, np.full((len(ids), 3), .25, dtype=np.float32).tobytes(),
                              np.zeros((len(ids), 3), dtype=np.float32).tobytes(), ids.tobytes()):
                    stream.write(struct.pack("i", len(block)))
                    stream.write(block)
                    stream.write(struct.pack("i", len(block)))
            self.run_script("convert", source, output)
            rows = pathlib.Path(str(output) + ".0.txt").read_text().splitlines()
            self.assertEqual([int(row.split()[0]) for row in rows], list(map(int, ids)))

    def compare_fixture(self, directory, reference_ids):
        prefix = pathlib.Path(directory) / "fml"
        reference = pathlib.Path(directory) / "rockstar.particles"
        particle_ids = [2**53, 2**53 + 1, 2**63 + 7, NO_HALO - 1]
        halo_ids = [2**53 + 11, 2**63 + 9]
        pathlib.Path(str(prefix) + ".0.halos.txt").write_text("".join(
            f"{hid} 0 {NO_HALO} 20 20 .1 {vmax} .01 {x} .5 .5 0 0 0\n"
            for hid, vmax, x in zip(halo_ids, (100, 200), (.2, .8))))
        # Deliberately unsorted; an unassigned member must also map to -1.
        pathlib.Path(str(prefix) + ".0.members.txt").write_text("".join(
            f"{particle_ids[i]} {halo_ids[i % 2]} 0\n" for i in (3, 0, 2, 1)) +
            f"42 {NO_HALO} 0\n")
        lines = [f"0 0 0 0 0 0 {pid} {2**53 + i % 2}\n" for i, pid in enumerate(reference_ids)]
        lines.append("#Halo table begins\n")
        for i, (vmax, x) in enumerate(zip((100, 200), (.2, .8))):
            row = [i, 2**53 + i, 0, 20, 20, 100, vmax, 0, 0, x, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0]
            lines.append("#" + " ".join(map(str, row)) + "\n")
        reference.write_text("".join(lines))
        return self.run_script("compare", prefix, reference, "--particle-mass", 1, "--boxsize", 1)

    def test_sparse_uint64_memberships_and_integer_reference_halo_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.compare_fixture(directory, [2**53, 2**53 + 1, 2**63 + 7, NO_HALO - 1])
        self.assertIn("objects: FML 2 (hosts 2), ROCKSTAR 2 (hosts 2)", output)
        row = next(line.split() for line in output.splitlines() if line.strip().startswith("20-50"))
        self.assertEqual(row, ["20-50", "2", "1.000", "1.000", "|", "2", "1.000"])

    def test_missing_and_unassigned_ids_do_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            # Below, between, and above stored IDs, plus an explicitly unassigned member.
            output = self.compare_fixture(directory, [0, 2**53 + 2, NO_HALO, 42])
        row = next(line.split() for line in output.splitlines() if line.strip().startswith("20-50"))
        self.assertEqual(row, ["20-50", "2", "0.000", "nan", "|", "2", "0.000"])


if __name__ == "__main__":
    unittest.main()

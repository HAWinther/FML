#!/usr/bin/env python3
"""Stage-1 checks. Run with --library-exe to also compare existing FML 1-3LPT."""

import argparse
from itertools import permutations
from math import factorial
from pathlib import Path
import subprocess
import tempfile
import unittest

import eds_reference as ref


LIBRARY_EXE = None
MESH_EXE = None
PRECISION = "double"


def norm(vector):
    return max((abs(value) for component in vector for value in component.values()), default=0.0)


def error(left, right):
    return max((abs(a.get(mode, 0) - b.get(mode, 0))
                for a, b in zip(left, right) for mode in a.keys() | b.keys()), default=0.0)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = ref.fixture()
        cls.orders = ref.solve(cls.delta, 5)

    def assert_fields_close(self, actual, expected, tolerance=3e-12):
        self.assertLessEqual(error(actual, expected), tolerance * norm(expected) + 1e-18)

    def assert_mesh_close(self, actual, expected):
        tolerance = 3e-5 if PRECISION == "float" else 3e-12
        self.assertLessEqual(error(actual, expected), tolerance * norm(expected) + 1e-18)

    def test_published_kernels_through_fourth_order(self):
        for n in range(1, 5):
            with self.subTest(order=n):
                expected = ref.published_field(self.delta, n)
                self.assert_fields_close(self.orders[n], expected)

    def test_four_distinct_waves_symmetrization(self):
        # At this output mode only the four distinct waves contribute once each.
        # Thus the 4! ordered contributions cancel the field convention's 1/4!.
        waves = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 1))
        delta = dict.fromkeys(waves, 1.0)
        mode = ref.vector_sum(*waves)
        n = len(waves)
        expected = [0j, 0j, 0j]
        for ordered in permutations(waves):
            s, t = ref.published_kernel_parts(ordered)
            kernel = ref.vector_sum(ref.scaled(mode, s), ref.cross(mode, t))
            for i in range(3):
                expected[i] += 1j * kernel[i] / (2 * ref.pi * ref.dot(mode, mode) * factorial(n))
        actual = ref.solve(delta, n)[n]
        for i in range(3):
            self.assertAlmostEqual(actual[i].get(mode, 0), expected[i], delta=2e-16)

    def test_cofactor_equation_and_cauchy_invariants_through_fifth_order(self):
        matrix = ref.deformation(self.orders)
        for n in range(1, 6):
            with self.subTest(order=n):
                residual = ref.add((1, ref.acceleration_divergence(matrix, n)),
                                   (-1.5, ref.determinant_coefficient(matrix, n)))
                scale = norm(self.orders[n]) * (2 * ref.pi) * n * n
                self.assertLess(norm([residual]), 3e-12 * scale + 1e-18)
                # Independently assemble the antisymmetric part of F^T F',
                # including identity factors, instead of using the curl solver.
                for j, k in ((0, 1), (1, 2), (2, 0)):
                    terms = []
                    for i in range(3):
                        derivative_k = {s: ref.add((s, field)) for s, field in matrix[i][k].items() if s}
                        derivative_j = {s: ref.add((s, field)) for s, field in matrix[i][j].items() if s}
                        terms.append((1, ref.coefficient([matrix[i][j], derivative_k], n)))
                        terms.append((-1, ref.coefficient([matrix[i][k], derivative_j], n)))
                    self.assertLess(norm([ref.add(*terms)]), 3e-12 * scale + 1e-18)

    def test_helmholtz_signs(self):
        # An arbitrary vector mode has both a longitudinal and transverse part.
        original = [{(1, -2, 3): value} for value in (0.3 + 0.2j, -0.7j, -0.1)]
        tensor = ref.gradient(original)
        divergence = ref.add(*[(1, tensor[i][i]) for i in range(3)])
        curl = [ref.add((1, tensor[(i + 2) % 3][(i + 1) % 3]),
                        (-1, tensor[(i + 1) % 3][(i + 2) % 3])) for i in range(3)]
        self.assert_fields_close(ref.helmholtz(divergence, curl), original)

    def test_transverse_feedback_at_fourth_order(self):
        third = self.orders[3]
        tensor = ref.gradient(third)
        divergence = ref.add(*[(1, tensor[i][i]) for i in range(3)])
        longitudinal = ref.helmholtz(divergence, [{}, {}, {}])
        self.assertGreater(error(third, longitudinal), 1e-3 * norm(third))
        # Remove only the third-order transverse field and recompute the scalar
        # fourth-order source directly from the original equation.
        modified = {1: self.orders[1], 2: self.orders[2], 3: longitudinal}
        matrix = ref.deformation(modified)
        scalar_without = ref.add((1.5 / 16.5, ref.determinant_coefficient(matrix, 4)),
                                 (-1 / 16.5, ref.acceleration_divergence(matrix, 4)))
        fourth_tensor = ref.gradient(self.orders[4])
        scalar_full = ref.add(*[(1, fourth_tensor[i][i]) for i in range(3)])
        self.assertGreater(error([scalar_full], [scalar_without]), 1e-4 * norm([scalar_full]))

    def test_reality_zero_mode_and_amplitude_scaling(self):
        scaled = ref.solve(ref.add((-0.7, self.delta)), 5)
        for n in range(1, 6):
            with self.subTest(order=n):
                expected = [ref.add(((-0.7) ** n, component)) for component in self.orders[n]]
                self.assert_fields_close(scaled[n], expected)
                for component in self.orders[n]:
                    self.assertEqual(component.get(ref.ZERO, 0), 0)
                    for mode, value in component.items():
                        opposite = tuple(-x for x in mode)
                        self.assertAlmostEqual(value, component.get(opposite, 0).conjugate(), delta=1e-17)

    def test_plane_parallel_and_zero_input(self):
        delta = {(1, 2, -1): 0.1 + 0.04j, (-1, -2, 1): 0.1 - 0.04j}
        plane = ref.solve(delta, 5)
        for n in range(2, 6):
            self.assertLess(norm(plane[n]), 1e-17)
        self.assertTrue(all(norm(field) == 0 for field in ref.solve({}, 5).values()))
        with self.assertRaises(ValueError):
            ref.solve({ref.ZERO: 1}, 3)

    def test_existing_fml_fields(self):
        if not LIBRARY_EXE:
            self.skipTest("Pass --library-exe or use make run for the FML comparison")
        with tempfile.TemporaryDirectory(prefix="fml-eds-reference-") as directory:
            input_path = Path(directory) / "input.txt"
            output_path = Path(directory) / "fields.txt"
            input_path.write_text("".join(f"{x} {y} {z} {value.real:.17g} {value.imag:.17g}\n"
                                           for (x, y, z), value in sorted(self.delta.items())))
            for policy in ("none", "padding"):
                with self.subTest(policy=policy):
                    result = subprocess.run([LIBRARY_EXE, str(input_path), str(output_path), policy],
                                            text=True, capture_output=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    actual = {n: [{}, {}, {}] for n in range(1, 4)}
                    for line in output_path.read_text().splitlines():
                        order, x, y, z, axis, re, im = line.split()
                        actual[int(order)][int(axis)][(int(x), int(y), int(z))] = complex(float(re), float(im))
                    self.assertEqual(sum(len(c) for field in actual.values() for c in field), 16 * 16 * 9 * 9)
                    # FML stores the nonnegative final-axis half spectrum. Check
                    # every represented coefficient, including expected zeros.
                    for n in range(1, 4):
                        expected = [{mode: value for mode, value in component.items() if mode[2] >= 0}
                                    for component in self.orders[n]]
                        self.assert_mesh_close(actual[n], expected)

    def test_mesh_coefficients_through_fifth_order(self):
        if not MESH_EXE:
            self.skipTest("Pass --mesh-exe or use make run for the generic mesh comparison")
        positive = {(3, 0, 0): .1 + .02j, (-2, 1, 0): -.07 + .03j, (0, 1, 2): .06 - .01j}
        high_modes = {**positive, **{tuple(-x for x in mode): value.conjugate()
                                   for mode, value in positive.items()}}
        for mesh, cutoff, delta in [(16, 2, self.delta), (9, 2, self.delta), (8, 3, high_modes)]:
            with self.subTest(mesh=mesh, cutoff=cutoff):
                expected_orders = ref.solve(delta, 5)
                with tempfile.TemporaryDirectory(prefix="fml-eds-mesh-") as directory:
                    input_path, output_path = Path(directory) / "input.txt", Path(directory) / "output.txt"
                    input_path.write_text("".join(f"{x} {y} {z} {value.real:.17g} {value.imag:.17g}\n"
                                                   for (x, y, z), value in sorted(delta.items())))
                    result = subprocess.run([MESH_EXE, "--export", str(input_path), str(output_path),
                                             str(mesh), str(cutoff), "5"], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    actual = {n: [{}, {}, {}] for n in range(1, 6)}
                    for line in output_path.read_text().splitlines():
                        order, x, y, z, axis, re, im = line.split()
                        actual[int(order)][int(axis)][(int(x), int(y), int(z))] = complex(float(re), float(im))
                    self.assertEqual(sum(len(c) for field in actual.values() for c in field),
                                     mesh * mesh * (mesh // 2 + 1) * 15)
                    for n in range(1, 6):
                        expected = [{mode: value for mode, value in component.items()
                                     if mode[2] >= 0 and all(2 * abs(x) < mesh for x in mode)}
                                    for component in expected_orders[n]]
                        self.assert_mesh_close(actual[n], expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("double", "float"), default="double")
    parser.add_argument("--library-exe", type=lambda path: str(Path(path).resolve()))
    parser.add_argument("--mesh-exe", type=lambda path: str(Path(path).resolve()))
    args, unittest_args = parser.parse_known_args()
    PRECISION = args.precision
    LIBRARY_EXE = args.library_exe
    MESH_EXE = args.mesh_exe
    unittest.main(argv=[__file__, *unittest_args], verbosity=2)

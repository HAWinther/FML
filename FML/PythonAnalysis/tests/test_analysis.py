import gc
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from fml_analysis import Catalogue, uniform_queries


def distances(points, queries, box=0, exclude=None):
    delta = np.abs(queries[:, None, :] - points[None, :, :])
    if box:
        delta %= box
        delta = np.minimum(delta, box - delta)
    result = np.sum(delta**2, axis=2)
    if exclude is not None:
        for i, index in enumerate(exclude):
            if index != -1:
                result[i, index] = np.inf
    return result


class Analysis(unittest.TestCase):
    def test_brute_force(self):
        rng = np.random.default_rng(123)
        points = rng.uniform(-1, 2, (31, 3))
        points[1] = points[0]
        queries = np.vstack([points[:4], rng.uniform(-1, 2, (11, 3))])
        radii = np.array([0.0, 0.125, 0.25, 0.5, 2.0])
        exclude = np.array([0, -1, 2] + [-1] * (len(queries) - 3))
        for box in [0.0, 1.0]:
            with self.subTest(box=box):
                cat = Catalogue(points, boxsize=box)
                d2 = distances(points, queries, box, exclude)
                d = np.sqrt(np.sort(d2, axis=1))
                orders = [3, 1, 31, 32]
                expected = np.vstack([d[:, k - 1] if k <= 31 else np.full(len(queries), np.inf) for k in orders])
                assert_allclose(cat.knn_distances(queries, orders, exclude=exclude), expected, atol=1e-15)
                counts = np.array([(d2 <= r**2).sum(axis=1) for r in radii], dtype=np.uint64)
                assert_array_equal(cat.sphere_counts(queries, radii, exclude=exclude), counts)
                result = cat.knn_cdf(queries, orders, radii, exclude=exclude)
                hits = np.array([(counts >= k).sum(axis=1) for k in orders], dtype=np.uint64)
                assert_array_equal(result['hits'], hits)
                assert_allclose(result['cdf'], hits / len(queries))
                assert_array_equal(result['orders'], orders)
                self.assertEqual(result['queries'], len(queries))
                self.assertEqual(result['excluded_queries'], 2)
                self.assertEqual(result['hits'].dtype, np.uint64)
                self.assertEqual(cat.sphere_counts(queries, radii).dtype, np.uint64)
                self.assertEqual(cat.knn_distances(queries, [1]).dtype, np.float64)

    def test_summary(self):
        p = np.array([[0, 0, 0], [0, 0, 0], [0.125, 0, 0], [0.5, 0.5, 0.5]])
        q = np.array([[0, 0, 0], [0.25, 0, 0], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5]])
        radii, edges = [0.125, 0.25, 0.5], [-1, 0, 1, 10]
        cat = Catalogue(p, boxsize=1)
        counts = cat.sphere_counts(q, radii)
        result = cat.counts_in_cells(q, radii, histogram_limit=1, factorial_order=4, delta_edges=edges)
        self.assertEqual(set(result), {'cells', 'boxsize', 'excluded_queries', 'histogram_limit', 'delta_edges', 'uncorrected_clipping'})
        for r, n, cell in zip(radii, counts, result['cells']):
            histogram = np.bincount(np.minimum(n, 2).astype(int), minlength=3)
            assert_array_equal(cell['histogram'], histogram)
            assert_allclose(cell['probability'], histogram / len(q))
            self.assertAlmostEqual(cell['mean'], n.mean())
            self.assertAlmostEqual(cell['variance'], n.var())
            self.assertAlmostEqual(cell['poisson_subtracted_variance'], n.var() - n.mean())
            self.assertAlmostEqual(cell['normalized_connected_variance'], (n.var() - n.mean()) / n.mean()**2)
            volume = 4 * np.pi * r**3 / 3
            self.assertAlmostEqual(cell['volume'], volume)
            self.assertEqual(cell['reference_number_density'], len(p))
            factorial = np.ones(len(n))
            for order in range(5):
                if order:
                    factorial *= np.maximum(n.astype(float) - order + 1, 0)
                self.assertAlmostEqual(cell['factorial_moments'][order], factorial.mean())
                self.assertAlmostEqual(cell['normalized_factorial_moments'][order], factorial.mean() / n.mean()**order)
            delta = n / (len(p) * volume) - 1
            hits = np.histogram(delta[(delta >= edges[0]) & (delta < edges[-1])], edges)[0]
            assert_array_equal(cell['density_histogram'], hits)
            assert_allclose(cell['density_probability'], hits / len(q))
            assert_allclose(cell['density_pdf'], hits / len(q) / np.diff(edges))
            self.assertEqual(cell['density_underflow'], int((delta < edges[0]).sum()))
            self.assertEqual(cell['density_overflow'], int((delta >= edges[-1]).sum()))
            self.assertEqual(cell['density_undefined'], 0)
            self.assertAlmostEqual(cell['mean_delta'], delta.mean())
        excluded = cat.counts_in_cells(p, [0.125], exclude=np.arange(len(p)))
        self.assertEqual(excluded['excluded_queries'], len(p))
        self.assertFalse(result['uncorrected_clipping'])

    def test_ownership_and_conversion(self):
        original = np.arange(60, dtype=np.float32).reshape(20, 3)[::-2] / 64
        original.flags.writeable = False
        saved = original.astype(float).copy()
        cat = Catalogue(original, boxsize=1)
        del original
        gc.collect()
        q = saved[::2, ::-1]
        result = cat.knn_distances(q, [1])
        expected = np.sqrt(distances(saved, q, 1).min(axis=1))
        assert_allclose(result[0], expected)
        copy = result.copy()
        result[:] = 123
        assert_allclose(cat.knn_distances(q, [1])[0], expected)
        summary = cat.counts_in_cells(q, [0.25])
        cdf = cat.knn_cdf(q, [1], [0.25])
        del cat, summary
        gc.collect()
        assert_allclose(copy[0], expected)
        self.assertTrue(copy.flags.owndata)
        self.assertTrue(cdf['hits'].flags.owndata)
        cdf['hits'][:] = 7
        mutable = saved.copy()
        owned = Catalogue(mutable)
        before = owned.knn_distances(q, [1])
        mutable[:] = 1e4
        assert_array_equal(owned.knn_distances(q, [1]), before)
        with self.assertRaises(AttributeError):
            owned.boxsize = 2

    def test_empty_and_duplicates(self):
        q = np.empty((0, 3))
        cat = Catalogue([[0, 0, 0], [0, 0, 0]], boxsize=1)
        assert_array_equal(cat.sphere_counts([[0, 0, 0]], [0], exclude=[0]), [[1]])
        self.assertTrue(np.isinf(cat.knn_distances([[0, 0, 0]], np.array([min(2**63, np.iinfo(np.uintp).max)], dtype=np.uint64))).all())
        self.assertEqual(cat.knn_distances(q, [1, 2]).shape, (2, 0))
        self.assertEqual(cat.sphere_counts(q, [0, 1]).shape, (2, 0))
        self.assertTrue(np.isnan(cat.knn_cdf(q, [1], [0])['cdf']).all())
        self.assertTrue(np.isnan(cat.counts_in_cells(q, [0.25])['cells'][0]['mean']))
        empty = Catalogue(q, boxsize=1)
        result = empty.counts_in_cells([[0, 0, 0]], [0.25])['cells'][0]
        self.assertEqual(result['probability'][0], 1)
        self.assertTrue(np.isnan(result['normalized_connected_variance']))
        self.assertTrue(np.isinf(empty.knn_distances([[0, 0, 0]], [1])).all())
        assert_array_equal(empty.sphere_counts([[0, 0, 0]], [0]), [[0]])

    def test_windows_and_errors(self):
        p = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
        cat = Catalogue(p)
        with self.assertRaises(ValueError):
            cat.counts_in_cells(p, [0.1])
        full = cat.counts_in_cells(p, [0.1], window=[[0, 0, 0], [1, 1, 1]], number_density=4)
        self.assertEqual(full['cells'][0]['reference_number_density'], 4)
        clipped = cat.counts_in_cells([[0, 0, 0]], [0.1], number_density=2, allow_truncated_cells=True, delta_edges=[])
        self.assertTrue(clipped['uncorrected_clipping'])
        self.assertEqual(len(clipped['delta_edges']), 0)
        invalid = [lambda: Catalogue([1, 2, 3]), lambda: Catalogue([[1, 2]]),
                   lambda: Catalogue([[np.nan, 0, 0]]), lambda: Catalogue([[1e151, 0, 0]]), lambda: cat.knn_distances(p, np.array([[1]])),
                   lambda: Catalogue(p, boxsize=-1), lambda: cat.knn_distances(p, [0]),
                   lambda: cat.knn_distances(p, [1, 1]), lambda: cat.knn_distances(p, [-1]), lambda: cat.knn_distances(p, np.array([], dtype=int)),
                   lambda: cat.knn_distances(p, [1], exclude=[-2, -1]),
                   lambda: cat.knn_distances(p, [1], exclude=[2, -1]),
                   lambda: cat.knn_distances(p, [1], exclude=[-1]),
                   lambda: cat.knn_distances(p, [1], exclude=np.array([2**64-1, 0], dtype=np.uint64)),
                   lambda: cat.sphere_counts(p, [1, 0]), lambda: cat.sphere_counts(p, [np.inf]), lambda: cat.sphere_counts(p, []),
                   lambda: cat.counts_in_cells(p, [0]), lambda: cat.counts_in_cells(p, [0.1], factorial_order=17),
                   lambda: cat.counts_in_cells(p, [0.5], window=[[0, 0, 0], [1, 1, 1]]),
                   lambda: cat.knn_distances([[np.inf, 0, 0]], [1]),
                   lambda: Catalogue(p, boxsize=1).counts_in_cells(p, [0.6])]
        for action in invalid:
            with self.assertRaises(ValueError):
                action()
        for dtype in [bool, complex, str, object]:
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                Catalogue(np.array(p, dtype=dtype))
        with self.assertRaises(TypeError):
            cat.knn_distances(p, [1.0])
        with self.assertRaises(TypeError):
            cat.knn_distances(p, [1], exclude=[0.0, -1.0])
        assert_array_equal(cat.knn_distances(p, [1]), [[0, 0]])

    def test_concurrent_reads(self):
        points = np.random.default_rng(4).uniform(0, 1, (100, 3))
        cat = Catalogue(points, boxsize=1)
        expected = cat.knn_distances(points, [1, 3], exclude=np.arange(len(points)))
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: cat.knn_distances(points, [1, 3], exclude=np.arange(len(points))), range(9)))
        for result in results:
            assert_array_equal(result, expected)

    def test_query_stream(self):
        bounds = [[-1, -2, -3], [2, 3, 4]]
        all_queries = uniform_queries(19, bounds, seed=2, first_id=5, stream=3)
        split = np.vstack([uniform_queries(7, bounds, seed=2, first_id=5, stream=3),
                           uniform_queries(12, bounds, seed=2, first_id=12, stream=3)])
        assert_array_equal(all_queries, split)
        self.assertTrue(((all_queries >= bounds[0]) & (all_queries < bounds[1])).all())
        self.assertEqual(uniform_queries(0, bounds, seed=0).shape, (0, 3))
        with self.assertRaises(ValueError):
            uniform_queries(2, bounds, seed=0, first_id=2**64-1)
        with self.assertRaises(ValueError):
            uniform_queries(2, [[1, 1, 1], [0, 0, 0]], seed=0)


if __name__ == '__main__':
    unittest.main()

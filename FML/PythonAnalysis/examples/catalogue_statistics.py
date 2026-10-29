"""A small periodic-box workflow; coordinates and radii share the same length unit."""
import numpy as np
from fml_analysis import Catalogue, uniform_queries

positions = np.random.default_rng(12).uniform(0, 1, (500, 3))
catalogue = Catalogue(positions, boxsize=1.0)
queries = uniform_queries(1000, [[0, 0, 0], [1, 1, 1]], seed=17)
radii = [0.05, 0.1, 0.2]
cdf = catalogue.knn_cdf(queries, [1, 2, 4], radii)
summary = catalogue.counts_in_cells(queries, radii)
assert cdf["hits"].shape == (3, 3)
assert all(cell["queries"] == len(queries) for cell in summary["cells"])
print("Tracers:", len(catalogue), "Queries:", cdf["queries"])
print("Nearest-neighbor CDF:", cdf["cdf"][0])
for cell in summary["cells"]:
    print("R, mean N, P(N=0):", cell["radius"], cell["mean"], cell["probability"][0])

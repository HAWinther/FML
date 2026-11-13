# LatinHyperCube

Latin hypercube sampling (LHS) of the unit hypercube $[0,1)^d$, after
[`scipy.stats.qmc.LatinHypercube`](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.qmc.LatinHypercube.html)
(scipy 1.18). It is meant for designing experiments, for example choosing the cosmologies for an emulator or a
parameter scan. The module is one header, `LatinHyperCube.h`. It uses `KDTree` for the Lloyd optimization and needs
`Global.o` at link time, like the other FML modules.

```cpp
#include <FML/LatinHyperCube/LatinHyperCube.h>
using namespace FML::SAMPLING;

LatinHyperCube lhs(3);                         // d = 3, scrambled, strength 1, seed from std::random_device
Sample points = lhs.random(100);               // points[i][j]: coordinate j of point i, in [0,1)

LatinHyperCube opt(3, true, 1, LHSOptimization::random_cd, 1234);   // seeded, with the discrepancy optimization
Sample params = scale(opt.random(100), {0.1, 0.6, 0.55}, {0.5, 1.0, 0.85});   // to [l, u)

LatinHyperCube oa(4, true, 2, LHSOptimization::none, 1);  // strength 2: n must be p^2 with p prime, d <= p + 1
Sample s = oa.random(49);

double cd = centered_discrepancy(s);           // as scipy.stats.qmc.discrepancy(s) (method "CD")
```

## Interface and the scipy equivalents

| FML | scipy |
| --- | --- |
| `LatinHyperCube(d, scramble = true, strength = 1, optimization = none, seed = {}, options = {})` | `LatinHypercube(d, *, scramble=True, strength=1, optimization=None, rng=None)` |
| `LHSOptimization::none / random_cd / lloyd` | `optimization=None / "random-cd" / "lloyd"` |
| `random(n)` → `Sample` (`std::vector<std::vector<double>>`, n × d) | `random(n)` → ndarray (n, d) |
| `integers(l_bounds, u_bounds, n, endpoint)`, `integers(u_bounds, n, endpoint)` | `integers(l_bounds, u_bounds=None, n, endpoint)` |
| `reset()`, `fast_forward(n)` | `reset()`, `fast_forward(n)` |
| `scale(sample, l_bounds, u_bounds, reverse = false)` | `qmc.scale` |
| `centered_discrepancy(sample)` | `qmc.discrepancy(sample, method="CD")` |
| `get_seed()`, `get_num_generated()`, `get_dimension()`, … | `num_generated`, `d` |

How it works:

- **Strength 1.** Each dimension gets an independent random permutation of the n strata $[k/n, (k+1)/n)$, so
  each stratum of each dimension holds exactly one point. With `scramble = true` the point is uniform within its
  cell. With `scramble = false` it sits at the centre of the cell.
- **Strength 2** (Tang 1993). The sampler builds the Bose orthogonal array $OA(p^2, p+1, p, 2)$, whose rows
  $(a, b)$ are $(a, b, a+b, a+2b, \dots, a+(p-1)b) \bmod p$. It relabels the symbols of each column at random.
  Then the $p$ rows holding symbol $s$ in a column get the values $(s + \text{LHS}_p)/p$. The result is a Latin
  hypercube in which every 2D projection also has exactly one point in each of the $p \times p$ cells. As in scipy,
  $n$ must be $p^2$ with $p$ prime and $d \le p + 1$; other values throw `std::invalid_argument`.
- **random-cd.** The sampler picks a random dimension and two random points and swaps those two coordinates if
  that lowers the centered discrepancy. It stops after `random_cd_max_iterations` proposals (default 10000) or
  after `random_cd_max_no_improvement` proposals in a row without a gain (default 100); these are scipy's
  defaults. Each proposal is evaluated with an exact O(n d) update rather than a full O(n² d) recompute. The
  optimization keeps the LHS property but not the strength-2 orthogonality.
- **lloyd.** The sampler runs damped Lloyd iterations towards a centroidal Voronoi tessellation of the unit cube.
  It uses scipy's damping schedule (from 1.9 down to about 1.0), its rule that a point is only moved if it stays
  inside the cube, and its stopping rule (the change of the minimum L1 distance is below `lloyd_tolerance`, after
  at most `lloyd_max_iterations` = 10 iterations). It requires d ≥ 2. The points end up evenly spread: the minimum
  distance typically grows 2–3×. They are pulled away from the faces of the cube, however, and are no longer a
  Latin hypercube, so their centered discrepancy can be *higher* than before (see the example). Use `random_cd`
  if you want good 1D and 2D projections.

## Differences from scipy

- **Random numbers.** They come from `std::mt19937_64` with our own conversions to doubles, integers and
  permutations, so a given seed gives the same points with every compiler and standard library. They are not the
  same points numpy's generator gives for that seed. Without a seed, one is drawn from `std::random_device`, and
  `get_seed()` returns it, so the run can be repeated.
- **Lloyd centroids.** scipy uses the mean of the Voronoi vertices from Qhull. Here the centroid of each Voronoi
  cell, clipped to the unit cube, is estimated from `lloyd_points_per_sample` (default 200) uniform Monte Carlo
  points per sample point, assigned to the nearest point with a KD-tree. This needs no Qhull, works in any d and
  uses the true clipped centroid. The resulting points differ from scipy's, and the cost per iteration is
  O(200 n log n).
- **No `workers` argument.** The sampler is serial. Under MPI every task that uses the same seed gets the same
  points.

## Files

- `LatinHyperCube.h`: the sampler, `scale`, `centered_discrepancy` and the two optimizations (also usable
  directly as `optimize_random_cd` and `optimize_lloyd`).
- `example/`: a 25-point design for (Omega_m, sigma8, h), comparing the designs by their discrepancy.
- `tests/`: `make run`. The tests check the discrepancy against scipy's documented value, the swap update against a
  full recompute, the LHS and orthogonal-array properties, the seeds, `reset` and `fast_forward`, both
  optimizations, `scale` and `integers`.

References: [McKay, Beckman & Conover (1979)](https://doi.org/10.1080/00401706.1979.10489755);
[Tang (1993)](https://doi.org/10.1080/01621459.1993.10476423), *J. Am. Stat. Assoc.* 88, 1392; Hickernell (1998),
*Math. Comp.* 67, 299 (centered discrepancy); Jin, Chen & Sudjianto (2005), *J. Stat. Plan. Inference* 134, 268
(swap updates).

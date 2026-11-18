# Eulerian bias operators

`EulerianBias.h` provides `FML::FIELDLEVEL::EULERIAN::QuadraticBias<Grid>` for the
periodic galaxy [sigma8 project](LEFTFIELD_PLAN.md). It accepts an **evolved
Eulerian matter overdensity**, not initial density or advected tracer weights.
It works with FFTWGrid slabs and PencilGrid pencils, storing only local cells.
The reusable operator component is now composed with LPT matter and a fixed-noise
likelihood in [DistributedEulerianLPT](EULERIAN_LPT.md). The full catalog analysis
and posterior-sampling application remain in progress.

The four real-space columns are

1. `Density`: d = F delta_m, with a declared sharp matter band F.
2. `DensitySquared`: d^2 - mean(d^2), without a factorial 1/2.
3. `TidalSquared`: sum_ij K_ij^2 - mean(sum_ij K_ij^2), where
   K_ij(k) = (k_i k_j / k^2 - delta_ij/3) d(k).
4. `Laplacian`: Laplacian(d), whose multiplier is -|2 pi n / L|^2.

Off-diagonal tensor components have multiplicity two. Means are global volume
means. Lengths use the box's physical units (normally Mpc/h); the coefficient
of the Laplacian therefore has units (Mpc/h)^2. A matter-band cutoff is in h/Mpc
for that choice. This defines our convention; the published numerical prior on
the derivative coefficient is not yet assumed to match it.

`SharpKBand` accepts `BandGeometry::Sphere` or `Cube` with **strict** bounds
`|k| < cutoff` or `max_i |k_i| < cutoff`. DC and every Nyquist plane are excluded.
A conservative requirement `cutoff <= pi N/(2 L)` ensures that all quadratic
product modes fit below Nyquist. Increase the product mesh rather than relaxing
this guard. Products are not subsequently filtered by this component; the
likelihood mask is a separate operation. Likewise this matter band does not
choose the initial-field band or populate framework likelihood-mode lists.

```cpp
namespace E = FML::FIELDLEVEL::EULERIAN;
E::QuadraticBias<Grid> bias(product_grid, boxsize,
                            E::SharpKBand{matter_cutoff, E::BandGeometry::Cube});
auto work = bias.workspace(product_grid);
const auto & columns = bias.evaluate(local_evolved_matter, work);
// Form a tracer prediction and its real-cell derivatives with respect to columns.
auto matter_derivative = bias.pullback(column_derivatives, work);
```

Fields follow `product_grid.get_real_range()` without ghost cells. The returned
columns remain valid until the next evaluation or workspace destruction. The
workspace must belong to the same model, mesh and domain; keep the model alive
and at a stable address. All calls are collective on `FML::communicator`, with
identical configuration and call order. Rank-local malformed/nonfinite input is
rejected collectively. A failed evaluation invalidates its reverse tape.

`pullback` returns the transpose Jacobian under the **sum over real cells** dot
product. Cotangents include any normalization from the caller's likelihood.
There are no implicit cell-volume, FFT or Hermitian-weight factors. The reverse
uses the self-adjoint sharp band, tidal projectors and Laplacian; quadratic
cotangents are globally centered before differentiating products. A volume-mean
objective should supply its factor 1/N^3 itself.

Validation uses independently constructed two/three-wave fields, including an
oblique cubic corner mode absent from a sphere, high-frequency/DC/Nyquist
contamination, physical derivative units, global mean subtraction, and nonzero
per-column directional derivatives at two step sizes. Both supported grid
backends and pencil orientations are exercised at 1/2/4 ranks. These are small
operator/adjoint tests, not an amplitude-posterior or N-body accuracy validation.

```sh
make -C FML/FieldLevel/tests eulerian-bias USE_MPI=false
make -C FML/FieldLevel/tests eulerian-bias USE_MPI=true USE_OMP=true NP_LIST='1 2 4'
```

The analysis follows the second-order Eulerian operator basis of
[2405.02252v2, Sec. V.3](https://arxiv.org/html/2405.02252v2#S5.SS3).
The [project plan](LEFTFIELD_PLAN.md) distinguishes verified conventions from
unresolved catalog and reproduction inputs. At fixed normalized ICs, the
amplitude map is delta_L(alpha)=alpha delta_L,fid and
P_L(alpha)=alpha^2 P_L,fid; alpha=sigma8/sigma8_fid. The scalar prior is uniform
in alpha, not in log power. Free ICs will add a unit-Gaussian field prior in
whitened coordinates after the fixed-IC pipeline has been validated.

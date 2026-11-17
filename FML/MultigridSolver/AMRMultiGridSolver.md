# AMRMultiGridSolver

A Full Approximation Storage (FAS) multigrid solver for (non-linear) PDEs on an adaptively refined grid
(`AdaptiveMPIGrid`), with MPI. Header: `MultigridSolver/AMRMultiGridSolver.h`. Example and tests:
`examples/AMRPoissonExample.cpp` (`make amrpoisson && mpirun -np 4 ./amrpoisson [2|3]`).

```cpp
#include <FML/MultigridSolver/AMRMultiGridSolver.h>
using namespace FML::SOLVERS::AMRMULTIGRIDSOLVER;

AMRMultiGridSolver<3> solver(grid);             // any AdaptiveMPIGrid<3, T>, after adapt / load_balance
int id = solver.add_external_field(delta);       // per local leaf, restricted to all levels
AMREquation<3> eq = [id](const AMRMultiGridSolver<3> & sol, int level, IndexInt i) {
    auto [lap, dlap] = sol.get_laplacian(level, i);  // D^2 f and d(D^2 f)/df in the cell
    return std::pair<double, double>{lap - sol.get_external_field(id, level, i), dlap};  // {L, dL/df}, L(f) = 0
};
auto result = solver.solve(eq);                  // result.converged(), .iterations, .residual
const auto & phi = solver.get_solution();       // per local leaf of grid (same indices)
auto force = solver.get_solution_gradient();     // D phi per leaf
```

The equation is written as `L(f) = 0`, returning `L` and `dL/df` in a cell, as for `MultiGridSolver`. Helpers:
`get_field`, `get_laplacian`, `get_laplacian_components` (per axis), `get_hessian` (needs `set_gradient_field(true)`),
`get_gradient`, `get_external_field`, `get_position`, `get_cell_size`, `get_cell`.
Level 0 is the grid, higher levels are coarser. Settings: `set_epsilon`, `set_maxsteps`, `set_ngs_sweeps(pre, post,
first)`, `set_coarsest_solve(reduction, max_sweeps)`, `set_convergence_criterion_residual` (absolute residual instead
of the reduction relative to the initial residual), `set_initial_guess`, `set_boundary_value` (Dirichlet value on the
non-periodic faces of the box, default 0), `set_coarse_fine_gradient`, `set_subtract_mean_residual` (solve
`L(f) = <L(f)>` for equations that fix `f` only up to a constant on a periodic grid, as in `MultiGridSolver`),
`set_gradient_field`, `set_custom_update` (replace the Newton step of the sweeps by your own solve of the equation in
a cell, as `MultiGridSolver::set_custom_update`; the Laplacian is affine in the cell's value, `lap + dlap (f' - f)`),
`set_coarse_fine_limiter` (minmod-limit the coarse gradient of the coarse/fine ghost values, so the Laplacian has
non-negative neighbour weights and a positive field stays positive).

## Design

- **Hierarchy.** The levels are the grid truncated at every refinement level (`AdaptiveMPIGrid::truncated`): level 0
  is the grid (finest leaves at `lmax`), level 1 has every leaf at `lmax` replaced by its parent, and so on down to
  the base grid and further to uniform grids coarser than the base grid, ending at 4 cells per dimension (periodic)
  or 2 (otherwise). Every level is a 2:1 balanced grid with its own ghosts. Values move between neighbouring levels
  with `AdaptiveMPIGridTransfer` (volume-average restriction, gather for prolongation).
- **Relaxation (multi-level adaptive technique).** On each level only the finest cells are relaxed (red-black
  Newton-Gauss-Seidel; the ghost exchange after each colour overlaps the interior cells). The coarser cells of a level
  are the same unknowns as on the next level, where they are relaxed or corrected. The FAS source of a coarse level,
  `S_c = L_c(R f) + R(S - L(f))`, carries the effect of the refinement on the coarse cells, so the fixed point solves
  the composite equation on the leaves. Cost per V-cycle is proportional to the number of leaves.
- **Prolongation** of the correction `e = f_c - R f`: linear (with the gradient of `e`) into the children, injection
  into the cells that exist on both levels. The coarsest level is relaxed until its residual drops by 10^3.
- **Laplacian.** Flux-conservative, cell centred. Across a coarse/fine face the fine cell uses a ghost value at its
  own tangential position (coarse value plus the coarse cell's gradient times the offset; the gradients of the coarse
  side cells are exchanged before each sweep), and the coarse cell uses the sum of the fluxes through the fine
  sub-faces. The gradient terms cancel in that sum, so the volume integral of the Laplacian vanishes exactly on a
  periodic grid (a Poisson source must have zero volume-weighted mean). Without the gradient term the interface is
  first order (`set_coarse_fine_gradient(false)`).

- **Second derivatives** (`get_hessian`). The diagonal is the flux Laplacian split by axis. The mixed derivatives
  are the symmetrized derivatives of the gradient field across the faces; on a uniform grid that is exactly the
  `(f[+,+] - f[+,-] - f[-,+] + f[-,-]) / 4h^2` stencil. The gradient field is computed (and its ghosts exchanged)
  before each sweep, so the mixed terms are lagged within a sweep (red-black ordering stays valid; the result does
  not depend on the number of tasks).

## DGP (AMRDGPSolver.h)

`AMRDGPSolverCosmology<3>` is the AMR version of `DGPSolverCosmology` (`DGPSolver.h`): same parameters (`OmegaM`,
`rcH0`, `beta`, `H0Box`), the same equation and operator splitting (`w = 1/3`), box units and returned potential
(`0.5 phi`, so the total force is `D(PhiGR + PhiF)`), the negative-discriminant check, warm start
(`set_previous_solution`) and the relative criterion. Input and output are vectors per local leaf:

```cpp
#include <FML/MultigridSolver/AMRDGPSolver.h>
AMRDGPSolverCosmology<3> dgp(OmegaM, rcH0, beta, H0Box, verbose);
dgp.set_epsilon(1e-6);
dgp.set_relative_criterion(true);
std::vector<double> phi;
std::vector<std::array<double, 3>> force;   // optional: D(phi) per leaf
dgp.solve(a, grid, delta, phi, &force);
```

Tests (`examples/AMRDGPExample.cpp`, `make amrdgp`): a spherical compensated overdensity
`delta = A exp(-r^2/2s^2)(3 - r^2/s^2)`, for which `r^2 phi' + (2/alpha) r phi'^2 = OmegaM a M(r) / beta` gives the
exact force. With `A = 5, s = 0.05, rcH0 = 2, beta = 2.2` the force at `r = s/2` is 0.47 of the linear one (the
linear force is off by 31% rms). Radial force error (rms within 3s, in units of the peak force):

| | force error rms | max | V-cycles |
|---|---|---|---|
| uniform level 3 | 0.96 | 0.96 | 8 |
| uniform level 6 | 1.0e-2 | 1.7e-2 | 10 |
| AMR levels 3-6 (refined within 4s) | 1.3e-2 | 2.9e-2 | 9 |
| AMR levels 4-7 | 2.6e-3 | 4.2e-3 | 10 |

Second order (2.35); the residual drops by about 0.1 per V-cycle; identical on 1-4 tasks. On a uniform grid
(32^3) the solution equals that of `DGPSolverCosmology` to 7e-11 (relative). The parameters must satisfy
`OmegaM a / (beta alpha) < 1/3`, as DGP in a cosmology does, otherwise the discriminant goes negative where
`delta ~ -1` (and the solve stalls).

## Symmetron (AMRSymmetronSolver.h)

`AMRSymmetronSolverCosmology<NDIM>` is the AMR version of `SymmetronSolverCosmology` (`SymmetronSolver.h`): the same
generalised (N, M) potential (even M > N >= 2; (2, 4) is the original symmetron), parameters (`OmegaM`, `assb`, `beta`,
`L_mpch`, `H0Box`), equation in box units

    D^2 chi = norm [ chi^(N-1) ((a_ssb/a)^3 (1 + delta) - 1) + chi^(M-1) ],   norm = a^2 (H0 Box)^2 / ((M - N) xi^2)

and returned potential `PhiF = forcenorm chi^N` (total force `D(PhiGR + PhiF)`), with the relative criterion
(`epsilon * rms(norm (a_ssb/a)^3 chi_bar^(N-1) delta)`). Input and output per local leaf:

```cpp
#include <FML/MultigridSolver/AMRSymmetronSolver.h>
AMRSymmetronSolverCosmology<3> sym(OmegaM, assb, beta, L_mpch, H0Box, verbose);   // optional N, M
sym.set_epsilon(1e-6);
sym.set_relative_criterion(true);
std::vector<double> phi;
std::vector<std::array<double, 3>> force;   // optional: D(PhiF) = forcenorm N chi^(N-1) D(chi) per leaf
sym.solve(a, grid, delta, phi, &force);
sym.get_field();                            // chi; the next solve on the same leaves starts from it
                                            // (or set_previous_field(chi)), else from the background value
```

The equation is local (only the flux Laplacian), so the solver needs no gradient field and no mean subtraction (the
mass term fixes the constant). It starts from the background (or the previous field), so it follows the positive
branch; domain walls are only followed if they are in the initial field. Fiducial 4 + 4 Gauss-Seidel sweeps.

Tests (`examples/AMRSymmetronExample.cpp`, `make amrsymmetron`, `mpirun -np 4 ./amrsymmetron [quick]`): a compensated
spherical overdensity `delta = A exp(-r^2/2s^2)(3 - r^2/s^2)` (A = 6, s = 0.05, `delta >= -1`) in a 100 Mpc/h box with
L = 2 Mpc/h (range ~0.025 box), assb = 0.7, a = 1: the centre is screened (chi = 0.058 against the background 0.81).
Reference: the radial equation solved with Newton on 20000 radial cells. Errors within 4s ("quick", levels to 7):

| (2, 4) | leaves | chi rms error | radial force rms error (of the peak) | V-cycles |
|---|---|---|---|---|
| uniform level 4 | 4096 | 2.3e-2 | 6.9e-2 | 2 |
| uniform level 6 | 262144 | 9.1e-4 | 1.1e-2 | 5 |
| AMR levels 4-6 (refined within 0.2) | 13224 | 9.3e-4 | 1.1e-2 | 7 |
| AMR levels 5-7 | 103440 | 2.3e-4 | 3.1e-3 | 7 |

Second order (2.00 for chi, 1.89 for the force; (2, 6): 2.01, 1.94). The AMR solution is as accurate as the uniform
grid at its finest level with 20x fewer leaves. Identical on 1 and 3 tasks (12-digit checksums). On a uniform 64^3
grid it equals `SymmetronSolverCosmology` to 1.2e-12 (relative, max over the grid) on 1, 2 and 4 tasks. Works with
both the original and the modified `AMRMultiGridSolver.h` (only the public interface is used).

## Dilaton (AMRDilatonSolver.h)

`AMRDilatonSolverCosmology<NDIM>` is the AMR version of `DilatonSolverCosmology` (`DilatonSolver.h`, Brax et al. 2012):
the same parameters (`OmegaM`, `A2`, `beta0`, `r <= 3/2`, `xi`, `H0Box`), the equation for `w = phi / phi_bar(a)`

    D^2 w = norm w [ (1 + delta) a^-3 - T(ln w) ],   norm = 3 A2 OmegaM a^2 (H0 Box)^2

and the returned potential `PhiF = PhiF_bar(a) w^2` (total force `D(PhiGR + PhiF)`), with the relative criterion
(`epsilon * rms(norm a^-3 delta)`). Input and output per local leaf:

```cpp
#include <FML/MultigridSolver/AMRDilatonSolver.h>
AMRDilatonSolverCosmology<3> dil(OmegaM, A2, beta0, r, xi, H0Box, verbose);
dil.set_epsilon(1e-6);
dil.set_relative_criterion(true);
std::vector<double> phi;
std::vector<std::array<double, 3>> force;   // optional: D(PhiF) = 2 PhiF_bar w D(w) per leaf
dil.solve(a, grid, delta, phi, &force);
dil.get_field();                            // w; the next solve on the same leaves starts from it
                                            // (or set_previous_field(w)), else from the background w = 1
```

As in the uniform solver the multigrid works with `w` and the Gauss-Seidel update solves the equation in the cell for
`u = ln w` (a `set_custom_update` with `DilatonSolverCosmology::solve_cell_affine`, shared with the uniform solver):
monotone after dividing by `w`, so the root is unique and positive even where `w` spans many e-folds. That needs a
positive sum over the neighbours: the coarse/fine ghosts use the limited gradient (`set_coarse_fine_limiter`; with the
unlimited gradient a ghost next to a screened halo in an N-body run went negative and the solve stalled). The root is
bracketed by the minima of the effective potential for the extreme densities, widened by 1 in `u`. For a code with its
own `AMRMultiGridSolver` (AMRCode, partial solves) `get_equation(a, id_delta)` and `get_update(a, id_delta, ulo, uhi)`
give the equation and the update. Fiducial 4 + 4 sweeps.

Tests (`examples/AMRDilatonExample.cpp`, `make amrdilaton`, `mpirun -np 4 ./amrdilaton [quick]`): the compensated
spherical overdensity `delta = A exp(-r^2/2s^2)(3 - r^2/s^2)` (A = 6, s = 0.05) in a 100 Mpc/h box, A2 = 5e5,
xi = 1e-3 (range 0.03 box), a = 1: the centre is screened (w = 0.28 for r = 3/2, 0.16 for r = 1). Reference: the radial
equation with Newton on 20000 cells. Errors within 4s ("quick", levels to 7), r = 1:

| | leaves | w rms error | radial force rms error (of the peak) | V-cycles |
|---|---|---|---|---|
| uniform level 4 | 4096 | 5.1e-2 | 8.1e-2 | 2 |
| uniform level 6 | 262144 | 1.8e-3 | 1.1e-2 | 5 |
| AMR levels 4-6 (refined within 0.2) | 13224 | 1.9e-3 | 1.2e-2 | 7 |
| AMR levels 5-7 | 103440 | 5.0e-4 | 3.6e-3 | 7 |

Order 1.93 for `w`, 1.75 for the force (r = 3/2: 1.92, 1.73; the limiter changes the errors by < 5%, slightly
smaller). Identical on 1, 2 and 4 tasks (12-digit checksums). On a uniform 32^3 grid with A2 = 5e6 at a = 0.5 (ln w
from -0.03 to 10.1) `PhiF` equals that of `DilatonSolverCosmology` to 5e-14 (relative, max over the grid).

A first solve from w = 1 in a dense peak (1 + delta = 2e4) resolved at level 9 while another region goes to level 10
(models D4 and C4 at the redshifts where AMRCode starts them, 256 Mpc/h): the peak's leaves are relaxed on a coarse
multigrid level, where w drops to ~e^-14 or below in the first V-cycle. The coarse-grid correction used to bring such a
value back to the leaf as f + (f_c - Rf) = 1 + (e^-40 - 1), which is exactly 0 in double precision, and the solve
failed with NaN after one V-cycle (C4; it is what stopped the AMRCode runs of C4 and D4). The cells that exist on both
levels now get the coarse value itself (the same in exact arithmetic; the Poisson, DGP and symmetron examples give
identical output). Both converge in 4 V-cycles.

The cell solve (`DilatonSolverCosmology::solve_cell_affine`, also used by the uniform solver) is a safeguarded Newton
iteration that now bisects whenever a Newton step does not halve the step before it (as rtsafe). Before, a bisection
into the side where `c e^-u` dominates (u ~ -350 for the bracket [-700, ~0]) was followed by Newton steps of only ~+1
and the 200 iterations ran out ~150 e-folds from the root. It happened rarely, in cells next to dense haloes of later
N-body steps, but then the multigrid stopped converging (C4 at z ~ 3 in 256 Mpc/h; 64^3 in 64 Mpc/h at z = 1.4) and
the result depended on the number of tasks (the starting value of the cell differs slightly). The example checks the
cell solve for the inputs of such a cell. A cell without a root in the bracket now keeps its value instead of going to
the bound. The checksums of the other tests and test_multigrid are unchanged.

## Tests (examples/AMRPoissonExample.cpp)

`D^2 phi = delta` for a Gaussian `phi` (periodic) and `D^2 u - 500 u^3 = s` (Dirichlet, `u = u_exact` on the faces),
with the grid refined around the peak:

| 2D, refined within 5 sigma | rms error Poisson | rms error non-linear |
|---|---|---|
| uniform level 4 | 9.6e-3 | 2.2e-2 |
| uniform level 8 | 6.1e-5 | 6.9e-5 |
| AMR levels 4-8 | 6.2e-5 | 6.7e-5 |
| AMR levels 5-9 | 1.5e-5 | 1.7e-5 |

The AMR solution is as accurate as the uniform grid at its finest level and converges at second order (3D: same
picture). The residual drops by 0.12-0.13 per V-cycle for Poisson and 0.17-0.21 for the non-linear Dirichlet problem
(2 pre + 2 post sweeps). The solution is the same on 1, 2, 3, 4, 7 and 13 tasks (to round-off in the reductions),
including tasks that are empty on the coarse levels, without MPI, and clean under AddressSanitizer.

## Limitations / next steps

- The gradient used for the coarse/fine ghost ignores the tangential offset of a coarser neighbour of the coarse cell
  (three levels within two cells, only at corners of the refinement).
- No helper for a variable-coefficient operator `D(b(f) D f)` yet (f(R) in the sqrt(f_R) form needs it or a custom
  pointwise update, `set_custom_update`).
- DGP is not wired into SimulationCode (which uses uniform grids).
- Construction: about 3 s for 2M leaves in 3D on one core (was 15 s), versus 1 s per V-cycle. Most of it is
  now the face-neighbour lookups of the stencils (6 `find_leaf` per leaf and level) and making the truncated cells;
  `AMRDGPSolverCosmology` builds the hierarchy on every call (sharing one with a Poisson solve would halve it).
- No OpenMP.

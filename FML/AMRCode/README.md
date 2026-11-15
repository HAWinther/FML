# AMRCode

A simple adaptive mesh refinement (AMR) N-body code for LCDM, in the spirit of RAMSES, built on
`AdaptiveMPIGrid` (Hilbert-ordered linear octree, MPI) and `AMRMultiGridSolver` (FAS multigrid on the leaves).

```
make && mpirun -np 4 ./amrnbody parameterfile.txt
cd tests && make && mpirun -np 4 ./testamrnbody && mpirun -np 4 ./testpartial
```

The Makefiles are not in the repository yet. `Makefile`:

```make
USE_MPI       ?= true
USE_OMP       ?= false
USE_FFTW      ?= true
USE_MEMORYLOG ?= false
USE_GSL       ?= true
TARGETS := amrnbody
include ../Makefile.config
FML_MODULES := Global FileUtils GadgetUtils Spline ODESolver
amrnbody: Main.o Global.o FileUtils.o GadgetUtils.o Spline.o ODESolver.o
	$(LINK_EXE)
```

`tests/Makefile`:

```make
USE_MPI ?= true
TARGETS := testamrnbody testpartial
include ../../Makefile.config
FML_MODULES := Global
testamrnbody: TestAMRNBody.o Global.o
	$(LINK_EXE)
testpartial: TestPartialSolve.o Global.o
	$(LINK_EXE)
```

Needs MPI, FFTW (initial conditions and P(k) output) and GSL (FML's spline/ODE code used by the IC routines),
set in `../Makefile.config` / `Makefile.local` as for the rest of FML.

## What it does

1. **Initial conditions.** 1LPT/2LPT with `FML::NBODY::NBodyInitialConditions` from a linear P(k) at z = 0 (a
   CAMB-format file `k [h/Mpc]  P [(Mpc/h)^3]`, or Eisenstein & Hu no-wiggle normalized to `sigma8`). Velocities
   `v = a^2 E f_n Psi_n` with the 1LPT/2LPT growth rates of the flat LCDM background (`LCDM`, growth factors from
   an ODE; checked against CAMB to 5e-4).
2. **Grid.** Base level `levelmin` (usually `log2(Npart_1D)`), refined up to `levelmax`. A leaf is refined when it
   holds more than `n_refine` particles (cloud-in-cell mass, as RAMSES), and a family of leaves is merged when
   together they hold at most `n_refine`. Done once per coarse step, then load balanced (weight: 1 + particles);
   the particles follow the grid's Peano-Hilbert domain (`particles.set_domain(grid.get_domain())`).
3. **Gravity.** Cloud-in-cell assignment on the tree (each particle on the cells of its own leaf's level; a refined
   cell of the cloud splits its mass equally among its children, so mass is conserved exactly; mass falling in
   other tasks' cells is sent to them). `D^2 phi = delta` (box units) is solved with `AMRMultiGridSolver`. The force
   is the 4th order (5 point, as RAMSES) gradient of phi where the neighbours are at the same level, else 2nd order,
   interpolated back to the particles with the same CIC stencil.
4. **Time stepping.** Kick-drift-kick in `tau = log(a)` with block time steps: a particle in a leaf at level `l` has
   the step `dt_coarse / 2^(l - levelmin)` (or smaller, `cfl`: `|dx| < cfl * cellsize` per step). Coarse steps are
   uniform in `log(a)` (`nsteps`, plus the output times). Every particle is drifted at every fine sub-step, only the
   active ones are kicked. At a sub-step where only the bins of levels `>= L` are active, Poisson is solved only on
   the leaves at levels `>= L` (a *partial solve*); the coarser leaves keep the potential of their last solve and
   act as Dirichlet values, as in RAMSES.

Units: positions in [0,1), velocities `v = a^2 dx/dt / (H0 B)` (FML's convention), and
`dx/dtau = v / (a^2 E)`, `dv/dtau = -(3/2) OmegaM / (a E) D phi`, `D^2 phi = delta`.

Outputs: `pofk_<label>_z<z>.txt` (k, P(k) without shot-noise subtraction, linear P(k)) and optionally GADGET
snapshots, at `output_redshifts`. Every coarse step prints the leaves and particles per level and a timing
breakdown.

## Files

| File | |
|---|---|
| `AMRNBody.h` | The code: `AMRParticle`, `LCDM`, `assign_mass_to_leaves`, `interpolate_to_particle`, `compute_force_field`, `refine_on_particle_count`, `AMRNBodySimulation` |
| `Main.cpp` | Parameter file, power spectrum, initial conditions, time loop and outputs |
| `parameterfile.txt` | Example (64^3 or 32^3 in a 100 Mpc/h box) |
| `tests/TestAMRNBody.cpp` | Deposit, Gaussian blob force, Zel'dovich plane wave (with/without refinement and subcycling) |
| `tests/TestPartialSolve.cpp` | Partial solves of `AMRMultiGridSolver` against full solves |

The library parts (partial solves in `MultigridSolver/AMRMultiGridSolver.h`, the faster `adapt` and lookups in
`AdaptiveMPIGrid/AdaptiveMPIGrid.h`, the table-driven keys in `PeanoHilbertDomain/HilbertCurve.h`) are in the library.

## Tests

- **Deposit:** total mass conserved to 1e-15 (CIC) on a grid refined to level 9; a lattice gives delta = 0.
- **Gaussian blob** (2^18 particles, sigma = 0.02) against the exact periodic force `M(<r)/(4 pi r^2) - r/3`: rms
  radial force error 47-54% on the uniform level 5 grid, 2-5% with levels 5-7 and 1.3-1.6% with levels 5-9 (the
  particle noise).
- **Zel'dovich plane wave** from z = 49 to just before shell crossing (D b = 0.88): rms position error 2.3% with
  32^3 particles on level 5 and 1.0% with 64^3 on level 6 (the force on a plane wave is 0.999 of the exact one at
  the fundamental mode); refined to level 7 on the pancake 0.8-0.9%. Subcycling vs. every particle on the finest
  step: 0.15% of the amplitude. The result does not depend on the time step (40 vs 160 steps: 0.2%).
- **Partial solves:** recover the full solution to 1e-10 with the fixed cells unchanged (bitwise), 0.05-0.1
  residual reduction per V-cycle.
- **LCDM 64^3, 100 Mpc/h, levels 6-10, z = 49 -> 0:** large-scale growth follows linear theory (the growth factor
  agrees with CAMB to 5e-4); P(k) on 1 and 2 tasks agrees to 0.4% at all k; refinement adds the small-scale power
  missing from the uniform base grid (x1.35 at k = 1, x3.3 at k = 3 h/Mpc). With the base grid equal to the
  particle grid the intermediate scales lose a few % of growth (k = 0.3-0.5 h/Mpc at z = 3), as in a PM code with
  Nmesh = Npart; one level finer base grid removes it.

## Performance

The 64^3 LCDM run to z = 0 (42 coarse steps, up to 16 sub-steps at the end) takes 1m12s on 2 cores (8m27s for the
first version). At z = 0 a coarse step costs: refinement 0.8 s, solver setup 1.1 s, densities 1.0 s, solves 2.6 s
(of which the full solve 1.1 s), forces 0.25 s, kicks/drifts 0.2 s. What made it fast (several in the library):

- `AdaptiveMPIGrid::adapt`: 2:1 balance with a work list (only leaves next to a leaf being refined are checked):
  a no-op adapt went from 2.7 s to 0 for 300k leaves, refining from scratch 15.8 s -> 3.1 s. Same leaf sets
  (the grid tests give the same hashes on 1-5 tasks).
- `HilbertCurve`: keys from state tables (derived from the bitwise Skilling transform, so the same curve): 4-5x
  faster in 3D; `cell_key(x, level)` computes only the top levels for a whole cell.
- `AdaptiveMPIGrid::find_leaf_near(x, hint, level)`: searches the leaves around a hint first (face neighbours,
  CIC stencils).
- `AMRMultiGridSolver`: partial solves whose V-cycles only touch the unknowns (lists per level, selective
  transfers `restrict_average_selected`/`gather_from_coarse_selected`, stopping at the coarsest level with
  unknowns): a sub-step solving only level 10 costs ~1/100 of a full V-cycle. Full solves: a cell with the same
  equation on the next coarser level gets its FAS source copied instead of computing a residual and the coarse
  operator (S_c = S), and the prolongation only computes gradients where they are used (2.5x faster full solve;
  `AMRPoissonExample` gives the same output).
- The potential and the particle counts are carried through refinement and load balancing on the grid: the full
  solve starts from the last potential, and the counts of the refinement are the density of the full solve.
- Particles are sorted along the curve at each regrid and cache their leaf; sub-step densities only deposit the
  particles that can reach the solved levels (leaf at those levels, or touching a finer leaf: exact).

## Limitations / next steps

- **Partial solves next to large fixed regions.** On the coarse levels a cell mixing fixed and unknown leaves is
  held fixed, which moves the boundary of the unknown region by up to a coarse cell. For a large unknown region
  next to fixed cells the V-cycle can stagnate; the solver then stops early and the step is redone as a full solve
  (counted as "redone as full"; once in the refined Zel'dovich run, never in the LCDM run). Also, with less than
  10% of the box fixed a full solve is done directly. A coarse operator that knows where the boundary is (Guillet &
  Teyssier 2011, as RAMSES) would remove this.
- Every sub-step drifts all particles (Gadget-style) and passes once over all particles to find which to deposit.
  The full solve, the solver setup (truncated grids, faces, transfers) and the refinement are O(leaves) per coarse
  step and now dominate.
- The coarse step is fixed (`nsteps` in log a); there is no adaptive global step.
- With the base level equal to the particle grid, intermediate scales lose a few % of growth (see Tests).
- No OpenMP. Only flat LCDM (no massive neutrinos or modified gravity, though the solver has DGP).

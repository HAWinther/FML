# Forces, time stepping and COLA

## Forces

The code computes particle-mesh (PM) forces: the particles are assigned to a grid of size `force_nmesh` (NGP, CIC,
TSC, PCS or PQS), and the force is found with Fourier transforms. The Green's function and the gradient kernel are
options: the continuous kernels, and the kernels of Hamming, Hockney & Eastwood and GADGET; the fiducial gradient
kernel depends on the grid, see below. The force of the screened models is described in
[MODELS.md](MODELS.md#screening). Short-range forces on top of the PM force (P3M or TreePM, a common time step) are
in [SHORT_RANGE.md](SHORT_RANGE.md), with COLA in [COLA_SHORT_RANGE.md](COLA_SHORT_RANGE.md).

Force discretization uses per-simulation [ForceOptions](../NBody/ForceOptions.md), passed explicitly to the gravity
calculation, including early-time force switching.

## The force grid, the gradient kernel and the lattice offset

A force grid finer than the particle grid (force_nmesh = 2-4 x particle_Npart_1D) is the usual choice for
COLA. Then the regular particle lattice of the initial conditions shows up in the forces. With an integer
ratio m = force_nmesh / particle_Npart_1D the particles start exactly on the nodes of the force grid, where
the CIC kernel has a kink, so the assigned density responds to the *size* of the displacement: a rectified
signal at the lattice frequency, which is the grid Nyquist frequency for m = 2 and lies below it for m >= 3.
With the continuous gradient kernel (D = ik), which is large near Nyquist, this gives spurious forces: the
growth of individual large-scale modes is off by tens of percent and there is too much small-scale power.

The fiducial settings when force_nmesh > particle_Npart_1D therefore are:
- `ic_lattice_offset = "fiducial"`: the particle lattice is moved by half a force-grid cell, off the nodes,
  which removes the rectified signal for any assignment method (TSC and PCS are smooth at the nodes anyway).
- `force_gradient_kernel = "fiducial"`: a finite difference kernel, which vanishes at the Nyquist frequency:
  the four-point kernel (discrete_4pt) for force grids up to 3x the particle grid and the two-point kernel
  (discrete_2pt) for finer grids. (With the grid = particle grid the fiducial kernel is the continuous one.)

Tests with 64^3 particles, CIC: P(k) at z = 0 in 256 Mpc/h relative to runs with a 6x grid and TSC (with the
2pt and 4pt kernels, which agree to 0.2% / 0.5% at k = 0.2 / 0.4 h/Mpc), and the growth of the modes at
k = 0.028 / 0.053 h/Mpc relative to linear theory in 1000 Mpc/h (`tests/check_force_grid.py`):

| Force grid | Settings                     | P(k), k = 0.2 / 0.4 h/Mpc | Linear growth, k = 0.028 / 0.053 |
|------------|------------------------------|---------------------------|----------------------------------|
| 2x         | on nodes, continuous         | +10% / +14%               | 40-70% scatter                   |
| 2x         | on nodes, discrete_2pt       | -1.8% / -5.1%             | +0.3% / +1.0%                    |
| 2x         | offset, discrete_2pt         | -1.3% / -5.3%             | -1.3% / -4.3%                    |
| 2x         | offset, discrete_4pt (fid.)  | +0.9% / +0.8%             | -0.6% / -2.0%                    |
| 3x         | on nodes, discrete_2pt       | +0.7% / +0.8%             | +1.1% / +2.7% (7% scatter)       |
| 3x         | offset, discrete_2pt         | +0.6% / +0.9%             | -0.4% / -1.3%                    |
| 3x         | offset, discrete_4pt (fid.)  | -0.1% / -0.3%             | -0.4% / -1.4%                    |
| 4x         | offset, discrete_4pt         | -0.4% / -1.6%             | -0.9% / -3.2%                    |
| 4x         | offset, discrete_2pt (fid.)  | +0.3% / +0.3%             | -0.6% / -2.0%                    |

The remaining ~1-2% scatter between modes at k ~ 0.05 h/Mpc in the linear test is the discreteness of the
particles themselves (it is the same with TSC on a grid equal to the particle grid). Use
ic_lattice_offset = "none" and force_gradient_kernel = "discrete_2pt" for the previous behaviour. With the
continuous kernel a 2x grid is wrong (+10% at k = 0.2 h/Mpc) even with the offset. Deconvolving the
assignment window in the force amplifies the lattice noise and does not help.

## Accurate forces on a grid equal to the particle grid (experimental)

With force_nmesh = particle_Npart_1D the standard CIC force is smoothed twice by the assignment window and is
aliased, so P(k) is low on all but the largest scales. These options give a much more accurate force on such a grid.
They are *only* meant for force_nmesh = particle_Npart_1D (the code warns otherwise): on a finer grid the modes
between the particle and the grid Nyquist frequency are dominated by the discreteness of the particles, and removing
the smoothing there gives far too much small-scale power (+14% to +58% at k = 0.2 h/Mpc on a 2x grid, -9% on a 3x
grid), while the standard settings (CIC, lattice offset, discrete_4pt/2pt kernel) are already accurate there.

- `force_interlacing_order` = M > 1: the force is computed for M copies of the particles shifted by a fraction of a
  cell (2: (1/2,1/2,1/2); 4: half-cell face diagonals; 8: all half-cell corners; 64: quarter cells), interpolated to
  the shifted particles and averaged. This cancels the leading aliasing of the assignment and the interpolation.
- `force_deconvolve_window = true`: divide the force by W(k)^2 (undo the smoothing of the assignment and the
  interpolation). Unstable on its own (it amplifies the aliased power), but with interlacing it works. TSC or PCS
  with M = 2 plus deconvolution is the combination of List, Hahn & Rampf (2024). M = 4 or 8 gives the same as 2 for
  TSC/PCS (the remaining aliases are the even images, which only quarter-cell shifts cancel).
- `force_density_method = nufft`: the exact Fourier modes of the particles (non-uniform FFT, FML/NUFFT/NUFFT.h,
  accuracy force_nufft_epsilon) for |k_i| < k_Nyquist; the force is then also evaluated exactly at the particles
  (force_interpolation_method = fiducial = nufft). Using the NUFFT only for the density (with CIC interpolation)
  gives a self-force that pulls the particles towards the grid nodes and slows the growth of all modes (-1.8% in
  P(k) on the largest scales in the test below).
- `force_density_method = sheet / sheet_nufft`: spectral sheet interpolation (FML/Interpolation/SheetInterpolation.h,
  force_sheet_refinement^3 tracers per particle from the Fourier interpolated displacement field). It does not help
  much here and is mostly a library feature.
- `force_early_method = interlaced / nufft` with `force_early_until_redshift`: use interlacing (2 grids, TSC,
  deconvolution; change with force_early_interlacing_order / force_early_density_assignment_method) or the NUFFT
  only for steps at z > force_early_until_redshift, then the normal force.

Tests: P(k) at z = 0 relative to a converged reference (64^3 particles in 256 Mpc/h: runs with a 6x grid; 128^3:
runs with a 3x grid), 2LPT at z = 19 and 100 FastPM steps unless stated. The cost is the time spent in the
density, force and kick per step relative to CIC (128^3, optimized build, 2 MPI tasks).

| 1x grid, 64^3                   | k = 0.03 / 0.1 / 0.2 / 0.4 h/Mpc       | cost  |
|---------------------------------|-----------------------------------------|-------|
| CIC (default)                   | -0.6% / -4.3% / -12.8% / -26.7%         | 1     |
| CIC, interlacing 2              | -0.4% / -3.8% / -12.6% / -29.5%         | ~2    |
| TSC, interlacing 2, deconvolved | -0.1% / -0.5% / -0.2% / -0.3%           | 6.9   |
| PCS, interlacing 2, deconvolved | -0.1% / -0.4% / -0.2% / -0.2%           | ~11   |
| NUFFT                           | -0.1% / -0.2% / +0.3% / +0.5%           | 13    |
| early interlaced / NUFFT, z > 4 | -0.2% / -1.3% / -4.5% / -13%            | ~2-3  |
| early interlaced / NUFFT, z > 1 | -0.2% / -0.8% / -2.3% / -10%            |       |
| sheet R = 2 (CIC)               | -1.4% / -4.3% / -12.0% / -28.3%         |       |

| 1x grid, 128^3                  | k = 0.1 / 0.2 / 0.4 / 0.8 h/Mpc         |
|---------------------------------|-----------------------------------------|
| CIC (default)                   | -0.9% / -2.8% / -6.8% / -18%            |
| TSC, interlacing 2, deconvolved | -0.3% / -0.7% / -1.8% / -4.0%           |
| PCS, interlacing 2, deconvolved | -0.3% / -0.6% / -1.6% / -3.6%           |
| NUFFT                           | -0.1% / -0.4% / -1.1% / -2.8%           |
| early interlaced, z > 4         | -0.3% / -1.0% / -3.5% / -12%            |
| (2x grid, CIC fiducial)         | +0.0% / +0.1% / -0.1% / -1.4%           |

| Few steps / COLA, 1x grid, 64^3 (k = 0.03 / 0.2 / 0.4) | CIC                  | TSC, interlacing 2, deconv. |
|---------------------------------------------------------|----------------------|-----------------------------|
| COLA, 10 FastPM steps                                   | -0.6% / -12.9% / -26% | -0.2% / -0.5% / -1.0%       |
| BullFrog, 10 steps                                      | -0.5% / -12.1% / -26% | -0.2% / -0.4% / -0.6%       |
| BullFrog, 5 steps                                       | -0.5% / -11.2% / -25% | -0.2% / -0.8% / -1.3%       |
| BullFrog, 10 steps from z = 1e7 (see below)             |                      | -0.1% / -0.0% / +0.2%       |

So on a 1x grid interlacing + TSC + deconvolution gives close to the P(k) of a much finer grid for ~4x the total run
time of CIC (7x the time of the force), and the NUFFT is slightly better but twice as expensive again. The early
switch fixes the largest scales (the growth at early times when the particles are close to the lattice, where CIC is
worst) at little cost, but the small scales need the accurate force at late times too. For the NUFFT the nonlinear
P(k) is the same to 0.1% for force_nufft_epsilon = 1e-3 to 1e-6, but in a linear test with a tiny amplitude
eps = 1e-3 gives +5% and 1e-4 -0.2%, so the default is 1e-5. The NUFFT needs FFTs on a grid 2x finer than
force_nmesh.

## Starting from the Big Bang with BullFrog

A BullFrog step is exact for the second order (2LPT) growing mode, so the simulation can start at (almost) a = 0 with
the particles on the lattice and the Zel'dovich velocity: set `timestep_method = "BullFrog"`, `ic_LPT_order = 1` and a
very high `ic_initial_redshift` (e.g. 1e7). The first step then produces the 2LPT displacement itself (List, Hahn &
Rampf 2024). This needs an accurate force at early times, when the particles are still close to the lattice: with
CIC on a 1x grid the force is ~5% too weak there and a one-step start from z = 1e7 to z = 19 misses the linear
displacement by 5%, while with the NUFFT the linear displacement is right to 0.01% and the 2LPT displacement is
reproduced on large scales (to ~10% of the 2LPT correction at k < k_Ny/4, more at higher k where the discreteness of
the particles matters). P(k, z = 0) with N BullFrog steps relative to 2LPT at z = 19 + 100 FastPM steps with the same
force (64^3 particles, 256 Mpc/h, 1x grid):

| Start                         | N = 5, k = 0.1 / 0.2 / 0.4 | N = 10                  | N = 20                  |
|-------------------------------|----------------------------|-------------------------|-------------------------|
| 2LPT at z = 19, CIC           | +0.7% / +1.8% / +1.8%      | +0.4% / +0.9% / +1.0%   | +0.1% / +0.3% / +0.5%   |
| z = 1e7, Zel'dovich, CIC      | -1.6% / -4.1% / -6.4%      | -3.1% / -6.6% / -6.9%   | -4.8% / -9.2% / -8.7%   |
| 2LPT at z = 19, NUFFT         | -0.2% / -0.5% / -0.8%      | -0.1% / -0.1% / -0.2%   | -0.0% / -0.0% / -0.0%   |
| z = 1e7, Zel'dovich, NUFFT    | -0.3% / -0.7% / -1.1%      | -0.1% / -0.2% / -0.1%   | -0.0% / +0.0% / +0.2%   |

So with an accurate early-time force the start from the Big Bang converges to the same answer as 2LPT initial
conditions (with CIC it gets worse with more steps as more of them are taken while the force is inaccurate). With
interlacing + TSC + deconvolution 10 steps from z = 1e7 give P(k) within 0.2% of the 6x grid reference up to
k = 0.4 h/Mpc (table above; but see below for the interlaced force at early times). The growth factor tables start before the initial redshift (and before a_eq/10 when
there is radiation) with the growing mode of a matter + radiation universe (D1 = 1 + 3a/(2a_eq), Meszaros, and the
corresponding 2LPT and 3LPT growing modes), so the growth factors are right also deep in the radiation era.

Which force to use from the Big Bang: With interlacing + TSC + deconvolution on a 1x grid the Big Bang runs used to
converge to a P(k) that differs from the 2LPT z = 19 runs by up to 0.4% for k < 0.1 h/Mpc and 3% at k = 0.5 h/Mpc.
This comes from the interlaced force while the particles are still close to their lattice (z > ~19): it makes the
modes with k > ~k_Ny/8 grow too fast (at z = 19 a Big Bang run then has +0.6% / +2.3% / +8% / +11% more power at
k = 0.1 / 0.2 / 0.4 / 0.5 h/Mpc than the 2LPT initial conditions), and starting from the Big Bang gives this error
much longer to act. With the exact (NUFFT) force the two starts agree. P(k, z = 0) relative to 2LPT at z = 49 with
the NUFFT force and 100 BullFrog steps (64^3 particles, 256 Mpc/h, 1x grid, fixed amplitude; 80 BullFrog steps,
linear in a from z = 1e7 and uniform in D from z = 19; `force_early_method = nufft` uses the NUFFT for z > 10):

| Force, start                                   | k = 0.1 | 0.2    | 0.4    | 0.6    |
|------------------------------------------------|---------|--------|--------|--------|
| NUFFT, z = 1e7                                 | +0.01%  | +0.02% | +0.09% | +0.25% |
| NUFFT, 2LPT z = 19 (3LPT: the same)            | -0.02%  | -0.07% | -0.21% | -0.49% |
| interlaced, z = 1e7                            | +0.17%  | +0.92% | +2.32% | +1.70% |
| interlaced, z = 1e7, NUFFT for z > 10          | -0.27%  | -0.53% | -0.85% | -1.04% |
| interlaced, z = 1e7, NUFFT for z > 3           | -0.22%  | -0.42% | -0.53% | -0.56% |
| interlaced, 2LPT z = 19                        | -0.27%  | -0.54% | -1.00% | -1.76% |
| CIC, z = 1e7, NUFFT for z > 10                 | -2.64%  | -9.5%  | -22%   | -37%   |
| CIC, 2LPT z = 19                               | -4.1%   | -13%   | -27%   | -43%   |

So when starting from the Big Bang with a force other than the NUFFT, use `force_early_method = "nufft"` (with
`force_early_until_redshift` ~ 10): it costs little, as with steps linear in a only a few steps are at z > 10 (here
8.0 s against 8.4 s without it and 14.7 s with the NUFFT throughout), and the result is then that of 2LPT initial
conditions at z = 19 with the same force (the code warns otherwise). The rest of the difference to the NUFFT runs is
the error of the 1x force after z = 19 (see the force sections above); the NUFFT runs differ by < 0.1% for k < 0.3
h/Mpc between the starts (z = 1e7 is closest to the z = 49 start; a later start loses a little small-scale power).

## Time stepping

The integrator is a second-order leapfrog (a fourth-order Yoshida integrator is implemented for testing). The steps
are fixed before the run: either the total number of steps or the number between the output redshifts. The methods
(`timestep_method`) are:

- `Quinn`: standard leapfrog (Quinn et al. 1997).
- `Tassev`: modified leapfrog for COLA (Tassev et al. 2013).
- `FastPM`: leapfrog where the kick and drift factors come from the linear growth factor, so the linear growing mode is followed exactly whatever the step size (Feng et al. 2016, [arXiv:1603.00476](https://arxiv.org/abs/1603.00476)).
- `BullFrog`: drift-kick-drift in growth-factor time with the kick chosen so that 2LPT is followed after every step (Rampf, List & Hahn 2024, [arXiv:2409.19049](https://arxiv.org/abs/2409.19049)). It already does what COLA does, so it cannot be combined with COLA.

Without COLA, FastPM and BullFrog need far fewer steps than Quinn: with 10 steps from z = 19 they get P(k) at z = 0 to about 0.5% for k < 0.1 h/Mpc and 1-2% up to k = 0.5 h/Mpc, while Quinn is off by 7-12% (see `tests/check_timestepping.py`).

The steps are placed by `timestep_scalefactor_spacing`: `linear` (the default), `logarithmic`, `powerlaw` (uniform in a^`timestep_spacing_power`) or `growthfactor` (uniform in the linear growth factor D(a), as used for BullFrog by Rampf, List & Hahn 2024). Steps uniform in D reduce the time-stepping error of BullFrog by 15-40% for the same number of steps (except at 5 steps from z = 19 on small scales), and help FastPM a little; the default is unchanged. Max |P(k)/P_ref - 1| at z = 0 (64^3 particles, 256 Mpc/h, 1x grid, `tests/check_timestepping.py`; the Big Bang rows use interlacing + TSC + deconvolution and a 60-step Big Bang reference):

| Method, start                     | spacing      | 5 steps, k < 0.1 / 0.5 | 10 steps      | 20 steps      |
|-----------------------------------|--------------|------------------------|---------------|---------------|
| FastPM, 2LPT z = 19               | linear       | 1.3% / 4.9%            | 0.5% / 2.0%   | 0.2% / 0.7%   |
| FastPM, 2LPT z = 19               | growthfactor | 1.2% / 5.8%            | 0.4% / 2.0%   | 0.1% / 0.6%   |
| BullFrog, 2LPT z = 19             | linear       | 0.9% / 2.4%            | 0.4% / 1.3%   | 0.2% / 0.6%   |
| BullFrog, 2LPT z = 19             | growthfactor | 0.7% / 2.5%            | 0.3% / 1.1%   | 0.1% / 0.4%   |
| BullFrog, Zel'dovich z = 1e7      | linear       | 0.7% / 4.3%            | 0.4% / 2.6%   | 0.2% / 1.5%   |
| BullFrog, Zel'dovich z = 1e7      | growthfactor | 0.6% / 2.6%            | 0.3% / 1.8%   | 0.1% / 1.0%   |

Logarithmic spacing is a poor choice when starting at very high redshift (most steps are spent where nothing happens).

The Quinn, Tassev, FastPM and BullFrog coefficients come from the library's
[NBody/TimeStepping.h](../NBody/TimeStepping.h); see the [interface and growth conventions](../NBody/TimeStepping.md).
SimulationCode keeps its model adapters, output scheduling and step reporting.

## COLA

COLA can be switched on or off, with 1LPT or 2LPT (`simulation_cola_LPT_order`; the initial conditions can be 3LPT).
The initial displacement fields are stored in the particles when the initial conditions are made. COLA with
scale-dependent growth is also supported: it is slower, but as long as the force grid is larger than the particle
grid (as it usually is) the extra cost is moderate. It is used if `simulation_use_scaledependent_cola = true` (the
default) and the growth is scale-dependent (massive neutrinos, or a gravity model with k-dependent growth such as
f(R), the symmetron or a k-dependent mu). With scale-independent growth it would give the same result as standard
COLA at a much higher cost (4x the time and 40% more memory for 64^3 particles in GR), so standard COLA is used then
and the parameter is set to false (a message says so).

Some advice for COLA runs. Use as large a force grid as you can afford if you care about halos: their properties
depend on the small-scale force resolution. For halos use a slightly smaller FoF linking length, and calibrate the
masses and velocities against high-resolution simulations. For the matter power spectrum this matters much less;
make sure the scales you use are below about half the Nyquist frequency of the grid and of the particles. The number
of steps and their distribution matter for getting accurate results at the smallest cost; see e.g. Izard, Crocce &
Fosalba 2016 (arXiv:1509.04685) for settings.

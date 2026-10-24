# About the code

This is a simple Particle-Mesh (PM) N-body code for a wide range of model that is fast and very flexible. For models with complex dynamics (screened models) we provide several options from doing it exactly, to approximate but fast to just simulating linear theory equations. Every time-consuming operation is parallelized over MPI and OpenMP. It uses a slab-based parallelization so its not good for high resolution simulations, but perfect for fast approximate (COLA) simulations. Its also useful for testing different things or to simply use it as an analysis code to compute stuff from other simulations. This is part of a bigger library that can be found in the folder above.

You can find some documentation [on this page](https://fml.wintherscoming.no/nbodycode.php) or look at the source files.

# Running the code

```bash
make                                   # settings in ../Makefile.config (or ../Makefile.local)
mpirun -np 4 ./nbody parameterfile.lua
./nbody --parameters PARAMETERS.md     # write the reference of all parameters
```

All the parameters, with their type, default value, when they are used and the allowed values, are listed in
[PARAMETERS.md](PARAMETERS.md) (generated from `src/ParameterSchema.h`). `parameterfile.lua` is a commented
example. String parameters are checked against the allowed values, and names in the parameterfile that are not
parameters (e.g. misspelled ones) give a warning with a suggestion for what was meant.

The particle type is chosen automatically to use as little memory as possible: 56 bytes per particle without COLA,
80 bytes with scale-independent COLA and 104 bytes with scale-dependent COLA (in 3D, see `src/Particles.h`).

How the code is organised:

| File | Contents |
|---|---|
| `src/Main.cpp` | Reads the parameterfile, creates the models and runs the simulation |
| `src/ParameterSchema.h` | All the parameters (read, checked and documented from one table) |
| `src/Models.h` | The list of cosmologies and gravity models that can be selected |
| `src/Particles.h` | The particle types |
| `src/Simulation.h` | The simulation: initial conditions, time-stepping loop and output |
| `src/SimulationParameters.h` | The parameters of the simulation itself, grouped and validated |
| `src/TimeStepping.h` | Where the steps are and the size of the kick/drift steps |
| `src/InitialPowerSpectrum.h`, `src/InitialConditions.h`, `src/ExternalIC.h` | Making the initial conditions |
| `src/COLA.h` | The COLA method |
| `src/AnalyzeOutput.h` | On the fly analysis (P(k), multipoles, bispectrum, halos) and output |
| `src/Cosmology*.h`, `src/GravityModel*.h` | The models |
| `tests/` | Accuracy tests (comparison to MG-PICOLA), checks and regression tests (see `tests/README.md`) |

# Models

The two main concepts in the code is that of a Cosmology and a GravityModel. The former contains all of the background evolution quantities (Hubble function etc.) and the latter everything related to growth of perturbations: LPT growth factors and how to compute forces from the density field. The reason these things are separate is that it makes it possible to combine different thing while keeping the rest fixed. Adding a new model is as simple as making a new class that inherits from the base class Cosmology or GravityModel and implement the relevant functions, adding it to the list in `src/Models.h` and adding the parameters it reads to `src/ParameterSchema.h`.

The models that are currently implemented are

## Cosmology Models

* The Cosmology base class contains the stuff all cosmologies have - CDM, baryons, photons, neutrinos, a cosmological constant and primordial parameters (As, ns, kpivot). Initialization creates splines of boltzmann integrals (needed for e.g. exact treatment of neutrinos in the background).

* LCDM : Good old LCDM.

* w0waCDM : Chevallier-Polarski-Linder parametrization for the dark energy equation of state.

* JBD : Jordan-Brans-Dicke model which has a time-varying gravitational constant. Two free parameters: the JBD parameter wBD (GR recovered as it goes to infinity) and the value of Newtons constant today relative to Newtons actual constant (which is useful for e.g. testing how well cosmology is at constraining this).

* DGP : The self-accelerating DGP cosmology. One free parameter: the cross-over scale.

* Quintessence : a scalar field phi as the dark energy, with the potential V = V0 u(phi) (phi in units of the reduced Planck mass): `exponential` (u = exp(-lambda phi)), `inverse_power_law` (u = phi^-alpha, Ratra-Peebles, has a tracker solution), `pngb` (u = 1 + cos(phi/f), axion-like thawing) or `hilltop` (u = (1 - (phi/mu)^p)^2, flat top at phi = 0). The field starts frozen at `cosmology_quintessence_phi_ini` deep in the radiation era (optionally with a velocity `cosmology_quintessence_dphidlna_ini`) and V0 is found so that the dark energy density today is 1 - OmegaM - OmegaR - OmegaK. If the field oscillates around a minimum many times per e-fold it is replaced by a w = 0 fluid. The code stops with a message if no V0 works (the field rolls or oscillates too early). w(a) is in the column `w_phi` of `cosmology_<name>.txt`.

* EDE : axion-like early dark energy, V = m^2 f^2 (1 - cos(phi/f))^n, plus a cosmological constant. Parametrized by the peak fraction `cosmology_ede_fEDE` of the total density, the redshift of the peak `cosmology_ede_zc`, the initial angle `cosmology_ede_theta_ini` and `cosmology_ede_n` (m and f are found by shooting; takes ~10-20 s). Only the background is modified (the EDE perturbations are neglected), so start the simulation well after zc with initial conditions from a Boltzmann code that includes EDE.

* Tabulated : the background from a file (`cosmology_tabulated_filename`): `cosmology_tabulated_format = "H"` (columns z, H/H0, used as given), `"CLASS"` (a CLASS `_background.dat` file; the columns z and H [1/Mpc] are used, e.g. any hi_class background) or `"w"` (columns a, w(a) of the dark energy). Matter and radiation are those of the parameter file (they should agree with the table: the largest dark energy fraction above z = 10 is printed as a check). `input/class_background_w0wa.dat` is an example (CLASS, w0 = -0.9, wa = 0.1, the cosmology of the regression tests).

For Quintessence, EDE and Tabulated the dark energy is smooth (no perturbations) and the gravity model is GR (they can be combined with the other gravity models, but those use the background only through H(a)). The backgrounds were checked against an independent solution (scipy): H(a) agrees to < 1e-4 and w(a) to < 1e-5 for a < 1, and the EDE peak fraction and redshift to < 1e-3.

## Gravity Models

* The GravityModel base class contains an implementation of a fairly general set of LPT growth equation up to 3LPT (that works with most of the common modified gravity models for example and easy to add in parametrizations of this). All that models need to provide are the free functions appearing in this like the effective newtons constant etc. If you want a model that does not fit into this pattern then you can override this fiducial method and run your own solver.

* GR : Does what GR does in simulations; take the density field and computes the forces by solving the Poisson equation. All other methods must also remember to add the usual Newtonian force (just a 10 line routine to copy over).

* f(R) : The fiducial toy-model for modified graity; the Hu-Sawicky f(R) model. Two free parameters: n (power in f(R) function) and fR0 which determined the range of the fifth-force the extra scalar degree of freedom propagates. When fR0 goes to 0 we recover GR.

* DGP : The DGP model (a cubic galileon). Combined with a LCDM cosmology we get the normal branch and with a DGP cosmology we get the original DGP model. This is the fiducial toy-model for modifications of gravity with a Vainstein screening mechanism.

* JBD : Jordan-Brans-Dicke model. The gravity model corresponding to the cosmology with the same name (and currently requires this cosmology to work).

* Symmetron : The symmetron model. A f(R)-like modified gravity model - just to have a bit of variety.

# Forces

The code computes PM forces. Particles are binned to a grid (free choice of grid size and density assignment method: NGP, CIC, TSC, PCS, PQS, ...) and used fourier transforms to get the forces. The choice of kernel for this is also a free choice (the fiducial option is the "poor-mans Poisson solver" using the continuous Greens function, but other kernels like the ones in Hamming et al., Hockney & Eastwood and GADGET are also included). 

For the modified gravity models that has non-trivial non-linear evolution (a screening mechanism) we offer four ways of including this:

* Just simulate using the linearized field equations. Fast and agrees with linear theory on large scales.

* Use an approximate screening model (Winther & Ferreira 2014). This is almost as fast and is quite accurate (and can be tuned to be more accurate).

* Use the approximate screening model + the linear field equation to ensure that we get the correct linear evolution on large scales. This is probably the best option in general.

* Solve the exact field equation using a multigrid-solver. This is how high-resolution N-body codes like ECOSMOG, ISIS, MG-GADGET does it. This is very slow and if you want to solve it exactly you probably want to use a high-resolution code also, but its useful for testing. The multigrid-solver it also very easy to extend to other models if needed.

# Particles and matter content

The code deals with CDM (CDM+baryon) as particles and massive neutrinos in the form of a linear evolved grid. The particles are fully customizable and can contain whatever you want (its also possible to add dynamically allocated memory, though this is often not a good idea). The only restriction is that they must contain at least positions and velocity (of whatever type you want).

The particle container we use, MPIParticles, can hold any particle as long as it has a position (get\_pos) and the particles can be moved across tasks by simply calling communicate\_particles() and it will move any particle that has left the local domain. Another useful method is that we can swap Lagrangian and Eulerian positions (if your particle has both) and move the particles back to their original position (and compute things there; very useful for LPT related stuff).

If your particle has dynamically allocated memory then you must implement a function that appends to (and one that assigns from) a communication buffer and a method that returns the byte-size of the particle.

## CDM and baryons as separate particles

By default CDM and baryons are one set of particles ("cb"). With `particle_baryons = true` they are two sets (one MPIParticles container each, `particle_Npart_1D^3` of each) with masses in the ratio Omegab / OmegaCDM. Both feel the gravity of the total CDM+baryon density delta_cb = (OmegaCDM delta_cdm + Omegab delta_b) / OmegaCB (plus linear massive neutrinos), so the baryons are collisionless (no hydrodynamics). What this gets right is the difference between baryons and CDM left over from before decoupling (the baryons lag behind the CDM and catch up). The initial conditions are made from the CDM and baryon transfer functions at zini (`ic_type_of_input = transferinfofile`):

- delta_i(k) = delta_cb(k) T_i(k,zini) / T_cb(k,zini) from the same random field;
- 1LPT velocities with f_i = f_cb + dlog(T_i/T_cb)/dlna, where f_cb is the growth rate of the gravity model. The total CDM+baryon velocity is then exactly the growing mode, and the relative velocity comes from the transfer functions. (The growth rates computed directly from the CAMB transfer functions are inaccurate near the ends of the redshift range of the table: from z = 49 they gave 0.8% too much CDM+baryon power.)
- the 2LPT displacement of the total field for both species.

By default both species start from the same particle lattice (`particle_baryon_lattice = "same"`). Interleaved lattices offset by half a particle spacing (`"offset"`, the usual choice in tree codes) give a large spurious baryon-CDM difference on small scales in this PM code. Tests: 64^3 particles of each species in 256 Mpc/h, z_ini = 49, 2x force grid, P(k) at z = 0. The relative difference is 1 - sqrt(P_b/P_cdm), compared with CAMB 1 - T_b/T_cdm:

| k (h/Mpc)                                    | 0.025 | 0.05  | 0.1   | 0.2   | 0.3   | 0.4    |
|----------------------------------------------|-------|-------|-------|-------|-------|--------|
| CAMB 1 - T_b/T_cdm (%)                       | 0.54  | 0.55  | 0.64  | 0.44  | 0.52  | 0.49   |
| linear amplitude, same lattice (%)           | 0.51  | 0.53  | 0.60  | 0.45  | 0.55  | 0.55   |
| linear amplitude, offset lattices (%)        | 0.54  | 0.62  | 0.96  | 2.06  | 4.40  | 8.06   |
| P_cb (two species) / P_cb (one species) - 1, same lattice (%) | 0.00 | 0.00 | -0.01 | 0.00 | 0.00 | 0.00 |
| ... offset lattices, linear amplitude (%)    | -0.06 | -0.14 | -0.55 | -2.4  | -5.6  | -10.5  |
| ... offset lattices, normal amplitude, 1x grid, 20 BullFrog steps (%) | 0.08 | 0.13 | 0.21 | -0.31 | -2.0 | -2.8 |

With the same lattice the CDM+baryon P(k) is the same as that of one species, and the baryon-CDM difference follows linear theory on all scales. With 20 BullFrog steps and a 1x force grid at normal amplitude it is 0.51 / 0.54 / 0.60 / 0.58 / 0.64 / 0.69 % (the nonlinear evolution changes it a little on small scales). With offset lattices the difference grows spuriously for k > k_Ny/8 (worse with a 1x force grid). Time stepping: BullFrog with 20 steps gets the CDM+baryon P(k) at z = 0 to 0.04% for k < 0.1 h/Mpc (relative to 300 logarithmic Quinn steps), and FastPM with 20 steps to 0.3%.

Output: P(k) of CDM (`pofk_cdm_zX.txt`), baryons (`pofk_baryon_zX.txt`) and CDM+baryons (`pofk_zX.txt`, as before) with linear theory P_cb (T_i/T_cb)^2. The same files are written for every step. GADGET (binary and HDF5) files have the CDM as type 1 and the baryons as type 2 (a collisionless type: type 0 is SPH gas in GADGET, which many readers expect to have gas fields). Baryon IDs follow the CDM IDs.

Not implemented for two species yet (the code stops with a message): COLA, reading the IC from file, NUFFT / sheet densities, force interlacing and the early-time force, the N-body gauge, the lightcone, FoF, multipoles and the bispectrum. Use `ic_input_redshift = ic_initial_redshift` (the two species are made from the transfer functions at zini).

# COLA

The code has full support for COLA up to 3LPT, and can be turned off or on. The way it does this is that it stored the initial displacement fields in the particles when creating the IC. However it also supports COLA with scaledependent growth if you want to use that (its slower, but as long as the PM grid one uses for forces is larger than the number of particles - as we typically want it to be - this extra cost is not that bad). If the growth-factors are scaledependent (like with massive neutrinos or modified gravity) then the code will automatically detect this and use the scaledependent version unless you set a flag in the parameterfile. The LPT order we use for COLA is determined by what your particle contains: if it contains storage for only 1LPT then we will only use 1LPT and so on.

Some useful things about COLA simulations: you want to use as big a PM grid as possible to get high enough force-resolution on small scales if you care about halos. For locating halos you want to use a slightly lower linking length and you probably want to calibrate the masses and velocities to get a better fit to high-resolution simulations. If you only care about DM then this is of much less importance, you just want to make sure the grid you use is big enough as to have the largest scales smaller than ~ 1/2 the (grid and particle) nyquist frequency. Number of steps and time-step distribution is also important to get as accurate as possible results for the smallest cost. If you want tips for "best settings" see for example 1509.04685 (there are some other good papers on this that I will add when I remember them again).

# Initial conditions

The code used LPT to generate the particle positions and velocities from an initial density field. The code supports using either 1, 2 or 3LPT. The initial density field can be generated in fourier space or from a white noise field in real space and with a free choice of random number generator. We also have the same generator as used in the 2LPTic code (and in L-PICOLA, MG-PICOLA) to facilite tests with exactly the same IC.

We can also generate simple non-gaussian (fNL) initial conditions:

* Local

* Equilateral

* Orthogonal

When generating the initial density field we have the option of using the amplitude-fixed fields (Angulo and Pontzen 1603.05253 ; Francisco Villaescusa-Navarro et al. 1806.01871) and also easily invert the phases to do so-called paired-fixed simulation.

When computing the IC the code will check if the particle you use has quantities it supports and if so it will store data like: the initial Lagrangian position of the particles, unique ID and displacement fields up to 3LPT.

We also have the option to read in particles (GADGET binary or HDF5 format, but the library this builds on also contains a RAMSES reader) from file and run a simulation with this. For COLA simulations we also provide methods to reconstruct the LPT fields that are needed from the particle distribution in the external IC. This is mainly useful for testing (e.g. to compare to high-resolution simulations done with, say, RAMSES, GADGET or PKDGRAV).

# Input files

To generate the initial conditions we require either a (CDM+b) power-spectrum or a (CDM+b) transfer function in standard CAMB format (i.e. k in h/Mpc vs pofk in (Mpc/h)^3 or tofk in Mpc^2). For simulations with massive neutrinos we need the massive neutrino transfer functions for all redshifts (well we only need it + growth rates for the initial redshift as we can solve for the neutrino growth-factors) so for this you will have to provide a transfer-infofile. This is just a file listing the path to CAMB transfer function outputs (CLASS should be straight forward to add). The power-spectrum and transfer functions should ideally be at z = 0 for back-scaling to be as accurate as it can, however this is a small effect and you can also provide inputs at different redshifts (and specify the input redshift in the parameterfile).

The code can also read IC from file (GADGET format) and if you want to do a run with COLA we will reconstruct the LPT displacement fields corresponding to this IC. This is done naively by basically assuming the IC are 1LPT, but for IC generated at a not too low redshift its pretty good.

# Time-stepping

The code currently only does a second order leap-frog, but there are methods implemented for a 4th order Yoshida integrator if you want to test that. We work with a fixed set of steps determined ahead of time and you can freely choose how many steps in total or how many steps between output redshifts. The time-stepping options (`timestep_method`) are:

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
k = 0.4 h/Mpc (table above). The growth factor tables start before the initial redshift (and before a_eq/10 when
there is radiation) with the growing mode of a matter + radiation universe (D1 = 1 + 3a/(2a_eq), Meszaros, and the
corresponding 2LPT and 3LPT growing modes), so the growth factors are right also deep in the radiation era. NB: the
Big Bang runs converge to a P(k) that differs from the 2LPT z = 19 runs by up to 0.4% for k < 0.1 h/Mpc and 3% at
k = 0.5 h/Mpc (64^3, 1x grid, interlacing + TSC + deconvolution); which one is closer to the truth needs a test at
higher resolution.

## Radiation, neutrinos and GR: the N-body gauge

On very large scales (k < ~0.01 h/Mpc) a Newtonian simulation misses the effect of the radiation
(photons and massless neutrinos), and of the massive neutrinos, on the growth, and the meaning of the
particle positions is ambiguous. In the N-body gauge (Fidler et al. 2015, 2016) the particles follow the
Newtonian equations exactly to linear order in GR if the force is sourced by
rho_cb (delta_cb + delta_extra), where delta_extra(k,z) is a linear field that contains everything else
(photons, neutrinos and a metric term). With `force_nbody_gauge = true` we add this source to the density
field every time we compute the force, and the initial conditions are made from the N-body gauge transfer
functions (with the scale-dependent growth rate for the velocities).

The input is made with CLASS (`pip install classy`) for the cosmology in the parameterfile:

```bash
python3 scripts/make_nbody_gauge_transfer.py parameterfile.lua --outdir nbgauge_input
```

This writes the N-body gauge transfer functions (CAMB format, with a transfer info file) and the table
T_extra(k,z), checks the gauge transformation against CLASS, shows how big the effect is for your initial
redshift and prints the lines to put in the parameterfile. Only for LCDM with GR; the code checks that the
table was made for the same cosmology and that H(z) agrees.

For a start at z = 49 the effect on P(k) at z = 0 is about 2% at k = 1e-3 h/Mpc (radiation, without
massive neutrinos) and much less than 0.1% for k > 0.01 h/Mpc. `tests/check_nbody_gauge.py` checks that the
simulation follows the linear N-body gauge evolution from CLASS on the largest scales.


# On the fly analysis

The code can do a lot of calculations as we go ahead. First of all it outputs CDM+baryon and total matter power-spectrum for every step (`pofk_every_step`, on by default; since we have the density field in fourier space this is basically free). These are written as `pofk_<name>_cb_z<z>.txt` (and `_total_` with massive neutrinos or the N-body gauge) in the output folder, together with linear theory. They use the force grid and the force density assignment (with the window deconvolved, but no interlacing), so they are accurate on large scales and dominated by aliasing above about half the Nyquist frequency of the force grid; use `pofk` at the outputs for an accurate P(k). Shot noise is subtracted if `pofk_subtract_shotnoise = true`. With force interlacing the unshifted copy of the particles is used. On top of that whenever we output we can:

* Compute P(k) : Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions to easily be able to compare that.

* Compute P\_ell(k) multipoles (using the plane parallel approximation): Particles are put into redshift-space along the coordinate axes (one by one) and the multipoles are computed from this and we return the mean over the three axes. Options: compute up to any given ellmax, subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions (the Kaiser limit) to easily be able to compare that.

* Compute Friends-of-friends halos. Options: number of particles per halo and the linking-length. If your particle has fofid then the ID of the halo is stored in the particles if you want to do extra post-processing. You can also implement your own FoF class to determine what to bin up. The fiducial choice is just to add up positions and velocities (and velocity dispersion). The halos can span any number of tasks, but if you are sure they max span two tasks then you can choose to merge the halos in parallel to save some time. Also an option to set the maximum gridsize used for binning the particles in case memory associated with this is an issue. NB: this method does not change the particles, but it does sorts them so the order will be different than going in. We also output the mass-function (differential and total) for te halo catalouge (going to add the Tinker et al. prediction also at some point to make it easy to compare).

* Compute bispectrum (or any polyspectrum for that matter): Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). NB: this is a very expensive operation. We use the method of Scoccimarro and compute everything with fourier transforms. This means we need to do 2nbins FFTS and from these grids we will need to compute nbins^3 integrals (summing all the cells in the grid). Thus for large grid-sizes and a large number of bins this gets both memory and computationally expensive.

* Easy to add more analysis methods.

The code can also be set to store the results from the output for post-processing when the simulation is over if you want that. The code can also easily (small modification) be set to read data and analyse it and the library this is part of has many more algorithms one can use to compute things like real space correlation functions, tessellations, watershed (void finding) etc.

# Output

The fiducial fileformat for the particles is GADGET. If you want the standard gadget format (pos+vel+id) your particle must have an ID field (get/set_id) otherwise it will only output positions and velocities. With `output_fileformat = "HDF5"` the particles are written as GADGET HDF5 files (`gadget_zX.<task>.hdf5`, readable by e.g. h5py, yt, pynbody and GADGET-4). This needs the HDF5 library: compile with `make clean && make USE_HDF5=true` (set `HDF5_INCLUDE` and `HDF5_LIB` in `../Makefile.local` if HDF5 is not in `$(HOME)/local`). Reading particles (`ic_reconstruct_gadgetfilepath`) works for both binary and HDF5 GADGET files; the format is detected from the file names. (The old `output_fileformat = "FML"`, a raw dump of the particles that could only be read back with the same particle type and number of tasks, has been removed.)

The lightcone particles can be written as ascii (`plc_output_asciifile`), binary GADGET (`plc_output_gadgetfile`) or GADGET HDF5 files (`plc_output_hdf5file`, needs USE_HDF5): `lightcone_<name>/lightcone_cb_hdf5_step<N>[_batch<B>].<task>.hdf5`, one set per step. In all formats the positions are comoving in Mpc/h (not wrapped into the box) and the velocities are peculiar velocities in km/s; the header time is the middle of the step's crossing scale factors.

# External libraries: 

We require FFTW3, GSL and LUA. LUA is only for reading the parameterfiles.

# Caveats

Everything in this code is not properly tested so use it with care and make sure you test it. The code itself should be correct - well tested - but all the different models and options makes it likely some mistake is still there. There are also some things not implemented that in principle should be there like all the higher order LPT functions for models beyond LCDM (which should have very small effects though so its not a huge deal).

# Contributors

Many people have helped improve the code, here is a list of a few:

* Bartolomeo Fiorini

* Guilherme Brando

* Kazuya Koyama

* Daniela Saadeh

* Herman Slettmoen

* Renate Mauland-Hus

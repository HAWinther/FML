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

## Exact forces with NUFFTs and sheet interpolation (experimental)

`force_density_method` sets how the density field for the force is computed and `force_interpolation_method`
how the force is brought back to the particles:

- `assignment` (default): particles to grid with force_density_assignment_method (CIC, TSC, ...), and the same
  kernel for the interpolation.
- `nufft`: the exact Fourier modes of the particle distribution, computed with a non-uniform FFT
  (FML/NUFFT/NUFFT.h, accuracy force_nufft_epsilon), for all modes |k_i| < k_Nyquist of the force grid: no window
  function and no aliasing. The force is then also evaluated exactly at the particles with a NUFFT
  (force_interpolation_method = fiducial = nufft). It is important to use the NUFFT both ways: with the exact density
  and CIC interpolation a particle feels a force from itself that pulls it towards the nearest grid node, which slows
  down the growth of all modes (by ~1% in P(k) on the largest scales in the test below).
- `sheet` / `sheet_nufft`: spectral sheet interpolation (FML/Interpolation/SheetInterpolation.h). The displacement
  field x - q of the particles is Fourier interpolated to force_sheet_refinement^3 points per particle and these
  tracers are assigned to the grid with the assignment kernel (`sheet`) or with a NUFFT (`sheet_nufft`). This
  removes the discreteness of the particles but rings after shell crossing.

Tests with a force grid equal to the particle grid (64^3 particles; P(k) at z = 0 in 256 Mpc/h relative to a 6x grid
run, linear growth of the modes at k = 0.028 / 0.053 h/Mpc in 1000 Mpc/h; run time with 2 MPI tasks):

| Density / interpolation       | P(k), k = 0.1 / 0.2 / 0.4 h/Mpc | Linear growth (mean / scatter)  | Time  |
|-------------------------------|---------------------------------|---------------------------------|-------|
| CIC / CIC (default)           | -4.3% / -12.8% / -26.7%         | -3.2/4.4%, -8.4/11.6%           | 29 s  |
| TSC / TSC                     | -6.5% / -21% / -46%             | -5.2/1.3%, -16/4.3%             | 55 s  |
| sheet R = 2, CIC / CIC        | -4.3% / -12.0% / -28.3%         | -2.1/2.2%, -6.1/5.9%            | 126 s |
| NUFFT / CIC                   | -2.6% / -4.6% / -11.6%          | -0.1/0.04%, -0.2/0.08%          |  –    |
| NUFFT / NUFFT, eps = 1e-5     | -0.2% / +0.3% / +0.5%           | +0.07/0.05%, +0.07/0.08%        | 103 s |
| sheet R = 2 NUFFT / NUFFT, eps = 1e-3 | -0.1% / +0.6% / +1.3%   |                                 | > 2x NUFFT |

With NUFFTs both ways a 1x grid gives the P(k) of a 6x grid to < 1% up to half the particle Nyquist frequency
(the largest scales are -0.1%, compared to -0.6% for CIC and -1.8% for NUFFT density with CIC interpolation), for
about 3.5x the cost of CIC. Smaller force_nufft_epsilon is more accurate but slower (the kernel covers
(log10(1/eps)+1)^3 cells): the nonlinear P(k) is the same to 0.1% for eps = 1e-3 to 1e-6 (56 s, 65 s, 103 s,
102 s), but in a linear test with a tiny amplitude (sigma8 = 0.02) eps = 1e-3 gives a +5% error and 1e-4 -0.2%,
so the default is 1e-5. Sheet interpolation does not help much on a 1x grid. NUFFT and sheet_nufft need FFTs on a grid 2x finer than
force_nmesh.

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

For the time-step spacing there are three options: linear, logarithmic or a powerlaw (free power) spacing in the scalefactor.

# On the fly analysis

The code can do a lot of calculations as we go ahead. First of all it outputs CDM+baryon and total matter power-spectrum for every step (since we have the density field in fourier space this is basically free). On top of that whenever we output we can:

* Compute P(k) : Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions to easily be able to compare that.

* Compute P\_ell(k) multipoles (using the plane parallel approximation): Particles are put into redshift-space along the coordinate axes (one by one) and the multipoles are computed from this and we return the mean over the three axes. Options: compute up to any given ellmax, subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions (the Kaiser limit) to easily be able to compare that.

* Compute Friends-of-friends halos. Options: number of particles per halo and the linking-length. If your particle has fofid then the ID of the halo is stored in the particles if you want to do extra post-processing. You can also implement your own FoF class to determine what to bin up. The fiducial choice is just to add up positions and velocities (and velocity dispersion). The halos can span any number of tasks, but if you are sure they max span two tasks then you can choose to merge the halos in parallel to save some time. Also an option to set the maximum gridsize used for binning the particles in case memory associated with this is an issue. NB: this method does not change the particles, but it does sorts them so the order will be different than going in. We also output the mass-function (differential and total) for te halo catalouge (going to add the Tinker et al. prediction also at some point to make it easy to compare).

* Compute bispectrum (or any polyspectrum for that matter): Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignment method (NGP, CIC, TSC, PCS, PQS, ...). NB: this is a very expensive operation. We use the method of Scoccimarro and compute everything with fourier transforms. This means we need to do 2nbins FFTS and from these grids we will need to compute nbins^3 integrals (summing all the cells in the grid). Thus for large grid-sizes and a large number of bins this gets both memory and computationally expensive.

* Easy to add more analysis methods.

The code can also be set to store the results from the output for post-processing when the simulation is over if you want that. The code can also easily (small modification) be set to read data and analyse it and the library this is part of has many more algorithms one can use to compute things like real space correlation functions, tessellations, watershed (void finding) etc.

# Output

The fiducial fileformat for the particles is GADGET. If you want the standard gadget format (pos+vel+id) your particle must have an ID field (get/set_id) otherwise it will only output positions and velocities. With `output_fileformat = "HDF5"` the particles are written as GADGET HDF5 files (`gadget_zX.<task>.hdf5`, readable by e.g. h5py, yt, pynbody and GADGET-4). This needs the HDF5 library: compile with `make clean && make USE_HDF5=true` (set `HDF5_INCLUDE` and `HDF5_LIB` in `../Makefile.local` if HDF5 is not in `$(HOME)/local`). Reading particles (`ic_reconstruct_gadgetfilepath`) works for both binary and HDF5 GADGET files; the format is detected from the file names. The other option is to output in the internal format. This will dump all the data in the particles to disk and can easily be ready again by the library (see MPIParticles).

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

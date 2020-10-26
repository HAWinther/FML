# About the code

This is a simple Particle-Mesh (PM) N-body code for a wide range of model that is fast and very flexible. For models with complex dynamics (screened models) we provide several options from doing it exactly, to approximate but fast to just simulating linear theory equations. Every time-consuming operation is parallelized over MPI and OpenMP. It uses a slab-based parallelization so its not good for high resolution simulations, but perfect for fast approximate (COLA) simulations. Its also useful for testing different things or to simply use it as an analysis code to compute stuff from other simulations. This is part of a bigger library that can be found in the folder above.

You can find some documentation [on this page](https://fml.wintherscoming.no/nbodycode.php) or look at the source files.

# Models

The two main concepts in the code is that of a Cosmology and a GravityModel. The former contains all of the background evolution quantities (Hubble function etc.) and the latter everything related to growth of perturbations: LPT growth factors and how to compute forces from the density field. The reason these things are seperate is that it makes it possible to combine different thing while keeping the rest fixed. Adding a new model is as simple as making a new class that inherits from the base class Cosmology or GravityModel and implement the relevant functions.

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

* DGP : The DGP model (a cubic galileon). Combined with a LCDM cosmology we get the normal branch and with a DGP cosmology we get the original DGP model. This is the fiducial toy-model for modifications of gravity with a Vainstein screening mechnism.

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

The code deals with CDM (CDM+baryon) as particles and massive neutrinos in the form of a linear evolved grid. The particles are fully customizable and can contain whatever you want (its also possible to add dynamically allocated memory, though this is often not a good idea). The only restriction is that they must contain atleast positions and velocity (of whatever type you want).

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

We also have the option to read in particles (GADGET format, but the library this builds on also contains a RAMSES reader) from file and run a simulation with this. For COLA simulations we also provide methods to reconstruct the LPT fields that are needed from the particle distribution in the external IC. This is mainly useful for testing (e.g. to compare to high-resolution simulations done with, say, RAMSES, GADGET or PKDGRAV).

# Input files

To generate the initial conditions we require either a (CDM+b) power-spectrum or a (CDM+b) transfer function in standard CAMB format (i.e. k in h/Mpc vs pofk in (Mpc/h)^3 or tofk in Mpc^2). For simulations with massive neutrinos we need the massive neutrino transfer functions for all redshifts (well we only need it + growth rates for the initial redshift as we can solve for the neutrino growth-factors) so for this you will have to provide a transfer-infofile. This is just a file listing the path to CAMB transfer function outputs (CLASS should be straight forward to add). The power-spectrum and transfer functions should ideally be at z = 0 for back-scaling to be as accurate as it can, however this is a small effect and you can also provide inputs at different redshifts (and specify the input redshift in the parameterfile).

The code can also read IC from file (GADGET format) and if you want to do a run with COLA we will reconstruct the LPT displacement fields corresponding to this IC. This is done naively by basically assuming the IC are 1LPT, but for IC generated at a not too low redshift its pretty good.

# Time-stepping

The code currently only does a second order leap-frog, but there are methods implemented for a 4th order Yoshida integrator if you want to test that. We work with a fixed set of steps determined ahead of time and you can freely choose how many steps in total or how many steps between output redshifts. There are two main time-stepping options: we can either use Quinn et al. or we can use the method of Tassev et al. (mainly useful for COLA). For the time-step spacing there are three options: linear, logarithmic or a powerlaw (free power) spacing in the scalefactor.

# On the fly analysis

The code can do a lot of calculations as we go ahead. First of all it outputs CDM+baryon and total matter power-spectrum for every step (since we have the density field in fourier space this is basically free). On top of that whenever we output we can:

* Compute P(k) : Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignement method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions to easily be able to compare that.

* Compute P\_ell(k) multipoles (using the plane parallel approximation): Particles are put into redshift-space along the coordinate axes (one by one) and the multipoles are computed from this and we return the mean over the three axes. Options: compute up to any given ellmax, subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignement method (NGP, CIC, TSC, PCS, PQS, ...). Included in the output is also the linear theory predictions (the Kaiser limit) to easily be able to compare that.

* Compute Friends-of-friends halos. Options: number of particles per halo and the linking-length. If your particle has fofid then the ID of the halo is stored in the particles if you want to do extra post-processing. You can also implement your own FoF class to determine what to bin up. The fiducial choice is just to add up positions and velocities (and velocity dispersion). The halos can span any number of tasks, but if you are sure they max span two tasks then you can choose to merge the halos in parallel to save some time. Also an option to set the maximum gridsize used for binning the particles in case memory associated with this is an issue. NB: this method does not change the particles, but it does sorts them so the order will be different than going in. We also output the mass-function (differential and total) for te halo catalouge (going to add the Tinker et al. prediction also at some point to make it easy to compare).

* Compute bispectrum (or any polyspectrum for that matter): Options: subtract shot-noise, use interlacing for alias reduction (allowing us to use a smaller grid), free choice of density assignement method (NGP, CIC, TSC, PCS, PQS, ...). NB: this is a very expensive operation. We use the method of Scoccimarro and compute everything with fourier transforms. This means we need to do 2nbins FFTS and from these grids we will need to compute nbins^3 integrals (summing all the cells in the grid). Thus for large grid-sizes and a large number of bins this gets both memory and computationally expensive.

* Easy to add more analysis methods.

The code can also be set to store the results from the output for post-processing when the simulation is over if you want that. The code can also easily (small modification) be set to read data and analyse it and the library this is part of has many more algorithms one can use to compute things like real space correlation functions, tesselations, watershed (void finding) etc.

# Output

The fiducial fileformat for the particles is GADGET. If you want the standard gadget format (pos+vel+id) your particle must have an ID field (get/set_id) otherwise it will only ouput positions and velocities. The other option is to output in the internal format. This will dump all the data in the particles to disk and can easily be ready again by the library (see MPIParticles).

# External libraries: 

We require FFTW3, GSL and LUA. LUA is only for reading the parameterfiles.

# Caveats

Everything in this code is not properly tested so use it with care and make sure you test it. The code itself should be correct - well tested - but all the different models and options makes it likely some mistake is still there. There are also some things not implemented that in principle should be there like all the higher order LPT functions for models beyond LCDM (which should have very small effects though so its not a huge deal).

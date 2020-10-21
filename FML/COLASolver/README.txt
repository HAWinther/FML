This is a simple Particle-Mesh N-body code that is fast and very flexible. Parallelized over MPI and OpenMP.

CDM with the option of having massive neutrinos (linear). 

Including COLA up to 3LPT (free choice of the order). If gravity model has scaledependent growth then one can use scaledependent LPT for this or scale-independent if you want it to go faster at the expence of some accuracy. The way this is choosen is by what data and methods that are included in the particle you choose to use (which is fully customizable in type and what to include).

Comes with a few standard cosmologies (LCDM, w0waCDM, DGP, ...) and a few standard gravity models (GR, DGP, f(R), ...). Easy to add more.
All the background cosmology is seperated out in a class that one can inherit from and simply implement a few simple functions. Same goes with the gravity model which deals with LPT and how to compute forces. The fiducial implementation of LPT works for any Geff/G type of model.

Gaussian or non-gaussian (local, equilateral, orthogonal) initial density field (with or without amplitude fixing for pair-fixed simulations). Particles generated with 1, 2 or 3LPT.

Different time-stepping options, kernels for PM force, arbitrary density assigments methods (NGP, CIC, TSC, PCS, PQS, ...)

Can do a lot of on-the-fly analysis: P(k), RSD multipoles, FoF halo catalogues, bispectrum, ...

External libraries: FFTW3, GSL, LUA (only for reading the parameterfile).

Not properly tested.

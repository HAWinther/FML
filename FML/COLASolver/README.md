This is a simple Particle-Mesh N-body code that is fast and very flexible. Parallelized over MPI and OpenMP.

CDM with the option of having massive neutrinos (linear). 

Including COLA up to 3LPT (free choice of the order). If gravity model has scaledependent growth then one can use scaledependent LPT for this or scale-independent if you want it to go faster at the expence of some accuracy. The way this is choosen is by what data and methods that are included in the particle you choose to use (which is fully customizable in type and what to include).

The cosmology (background quantities; hubble function) and gravity model (LPT growth rates; forces) is treated as separate entities. Comes with a few standard cosmologies (LCDM, w0waCDM, DGP, Jordan-Brans-Dicke, ...) and a few standard gravity models (GR, DGP, f(R), JBD, Symmetron, ...). Easy to add more. All the background cosmology is in a class that one can inherit from and implement a few simple functions to get a new one that will work with the code. Same goes with the gravity model. The fiducial implementation of LPT in the gravity model works for any Geff/G type of model.

For the MG models with screening one has all the options of 1) simulating using linear field equations (no screening) 2) using fast approximate screening method or 3) solving the exact equations using a multigridsolver (this is of course slow).

Gaussian or non-gaussian (local, equilateral, orthogonal) initial density field (with or without amplitude fixing for pair-fixed simulations). Particles generated with 1, 2 or 3LPT.

Different time-stepping options, kernels for PM force, arbitrary density assigments methods (NGP, CIC, TSC, PCS, PQS, ...)

Can do a lot of on-the-fly analysis: P(k), RSD multipoles, FoF halo catalogues, bispectrum, ...

External libraries: FFTW3, GSL, LUA (only for reading the parameterfile).

Not properly tested (the code itself should be correct, but all the different models should be checked). Not implemented all the higher order LPT functions (which should have very small effects though) for all the models.

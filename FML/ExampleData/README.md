# ExampleData

Synthetic input data for the examples, so that they run without data files (`ExampleData.h`, header only):

- `clustered_box(npart, box, nvoids, seed)`: a toy clustered particle box. Half of the particles are in halos
  (Gaussian clouds of 20 - 2000 particles with a bulk velocity and a velocity dispersion), the rest a uniform
  background with spherical voids cut out. Positions, peculiar velocities (km/s), the voids and the number of halos.
- `redshift_space(data, redshift, OmegaM, los)`: the particles moved along an axis by their velocities.
- `survey_catalogue(...)`: galaxies (half in groups) and randoms in a patch of the sky as (DEC, RA, z), uniform in
  comoving volume for flat LCDM.
- `write_ramses_snapshot(folder, outputnr, particles, box, ...)`: the particles as a RAMSES snapshot.

Every task makes the same data (portable random numbers), so MPIParticles can be created with `Input::Replicated`.
The data is for trying out the methods (there are halos and voids to find, the power spectrum and correlation
function are those of a clustered field), not a cosmological simulation.

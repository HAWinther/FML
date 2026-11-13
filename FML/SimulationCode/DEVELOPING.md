# Developing SimulationCode

## Code layout

| File | Contents |
|---|---|
| `src/Main.cpp` | Reads the parameterfile, creates the models and runs the simulation |
| `src/ParameterSchema.h` | All the parameters (read, checked and documented from one table; the simulation, cosmology and gravity model settings from the `fields()` in `src/SimulationParameters.h`, `src/Cosmology.h` and `src/GravityModelSettings.h`) |
| `src/Models.h` | The list of cosmologies and gravity models that can be selected |
| `src/Particles.h` | The particle types |
| `src/Simulation.h` | The simulation class; its members are in `src/SimulationInit.h` (parameters, initial conditions), `src/SimulationStep.h` (time-stepping loop, output) and `src/SimulationDensity.h` (the density field of the force) |
| `src/SimulationParameters.h` | The parameters of the simulation itself, grouped and validated |
| `src/TimeStepping.h` | Where the steps are and the size of the kick/drift steps |
| `src/ICFromParameters.h`, `src/ExternalIC.h` | Making the initial conditions |
| `src/COLA.h` | The COLA method |
| `src/AnalyzeOutput.h`, `src/HaloFinding.h` | On the fly analysis (P(k), multipoles, bispectrum, hybrid bias spectra, halos) and output |
| `src/SnapshotOutput.h` | Writing the particles as GADGET / HDF5 snapshots |
| `src/Cosmology*.h`, `src/GravityModel*.h` | The models |
| `tests/` | Accuracy tests (comparison to MG-PICOLA), checks and regression tests (see `tests/README.md`) |

## Parameters

All the parameters, with their type, default value, when they are used and the allowed values, are listed in
[PARAMETERS.md](PARAMETERS.md) (generated from `src/ParameterSchema.h`). `parameterfile.lua` is a commented
example. String parameters are checked against the allowed values, and names in the parameterfile that are not
parameters (e.g. misspelled ones) give a warning with a suggestion for what was meant.

A new parameter of the simulation is a member of its settings struct in `src/SimulationParameters.h` and one line in
the `fields(f)` of that struct, e.g.

```cpp
f.optional("fof_linking_length", linking_length, 0.2, "The linking length in units of the mean particle separation");
```

(name, member, default, description, and the allowed values for a string; a check of the value as the last argument,
e.g. `check::positive()`; `f.if_true("fof", ...)` / `f.when(...)` for when it is used, `f.choice(...)` for a string read
into an enum, see `src/ParameterFields.h`). The cosmology
parameters are the `fields()` of `Cosmology::Settings` (`src/Cosmology.h`) and the gravity model parameters those of
`GravityModelSettings` (`src/GravityModelSettings.h`; `f.for_cosmology` / `f.for_gravity` for the parameters of one
model). The parameter table (`src/ParameterSchema.h`) and the reading both come from them. Then regenerate `PARAMETERS.md` with `./nbody --parameters PARAMETERS.md` (CI checks that it is up to
date).

## Particle types

By default the simulation is compiled for one particle type that can run every configuration (position, velocity,
ID, the 1LPT and 2LPT displacements for COLA, the Lagrangian position, a mass and a volume (interacting dark energy):
120 bytes per particle in 3D, see
`src/Particles.h`). Compile with `make clean && make USE_MINIMAL_PARTICLES=true` to compile it for all the particle
types and use the smallest one a run needs (56 bytes per particle without COLA, 80 bytes with scale-independent COLA
and 104 bytes with scale-dependent COLA, 16 more with particle masses (the mass-encoded baryon mode, interacting dark
energy); 7 times the compile time). The results are the same.
Scale-dependent COLA is only used when the growth is scale-dependent (see [COLA](EVOLUTION.md#cola)).

The particles are stored in `MPIParticles`, which holds any particle type with a position (`get_pos`). Particles
that leave a task's domain are moved by `communicate_particles()`. If the particle type has both Lagrangian and
Eulerian positions they can be swapped, to move the particles back to their initial positions (useful for LPT).
The particles are customizable; dynamically allocated memory in a particle is possible but usually not a good idea:
it then needs functions that append the particle to a communication buffer and assign it from one, and one that
returns its size in bytes.

## Adding a model

A new cosmology is a background model in `FML/Cosmology/Background` plus its mapping in `src/Cosmology.h`; a new
gravity model is a class that inherits from `GravityModel` (or `GravityModelScreened` for a screened model). Add
either to the list in `src/Models.h`. The parameters of a cosmology go in `Cosmology::Settings::fields`
(`src/Cosmology.h`, under `f.for_cosmology({"<model>"}, ...)`), those of a gravity model in
`GravityModelSettings` (`src/GravityModelSettings.h`, under `f.for_gravity`). The theory of a modified-gravity
model (G_eff(k, a)/G, the screening) belongs in `FML/Cosmology/Perturbations/QuasiStaticGravity_*.h`, shared with
the Einstein–Boltzmann solver; see [MODELS.md](MODELS.md#gravity-models).

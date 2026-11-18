# Quick start

Three short programs that show what working with FML looks like. Each does a real calculation, prints the result and
checks it, and they are built and run in CI, so they stay up to date with the library.

| Program | What it shows |
|---|---|
| [`pofk.cpp`](example/pofk.cpp) | Your own particle type, distributed over the MPI tasks, and its power spectrum. The catalogue is uniformly random, so P(k) must be the shot noise V/N. |
| [`pm_simulation.cpp`](example/pm_simulation.cpp) | A particle-mesh N-body simulation in about 60 lines: 2LPT initial conditions, a leapfrog in ln a and the power spectrum today. In an Einstein–de Sitter universe with P(k) ∝ k⁻² the largest scales must follow linear theory, P ∝ a²; on smaller scales the growth is nonlinear. |
| [`domains.cpp`](example/domains.cpp) | The same power spectrum with the particles on slabs, on pencils and on a Peano–Hilbert curve. The grid is made for the domain of the particles (`FFTWGrid` or `PencilGrid`); particles on the Peano–Hilbert curve are assigned to a slab grid without being moved. The three results agree to rounding. |

```sh
cd FML/QuickStart/example
make                        # needs FFTW; MPI and OpenMP optional (domains needs MPI)
make run NP_LIST="1 4"      # run them on 1 and 4 MPI tasks (each checks its result)
mpirun -np 4 ./pm_simulation
```

The programs are written for reading: the particle types are plain structs, every step is one call to the library,
and none of them contains any MPI code. What the calls do is described in the module READMEs and in the FML book
(`Documentation/`): [MPIParticles](../MPIParticles), [ComputePowerSpectra](../ComputePowerSpectra/README.md),
[NBody](../NBody), [PencilDomain](../PencilDomain/README.md) and [Global/Domain.h](../Global/Domain.h). For a full
simulation code see [SimulationCode](../SimulationCode/README.md).

# Field-level inference

Differentiable periodic linear, Zel’dovich/2LPT and particle-mesh models, Fourier
and Cartesian survey likelihoods, MAP/HMC inference and checkpoint/restart.
Distributed models support FFTWGrid slabs and PencilGrid pencils. Scientific
validation is bounded; consult the current limitations before interpreting results.

## Guides

- [Framework API and conventions](docs/FRAMEWORK.md)
- [Periodic galaxy / sigma8 project](docs/LEFTFIELD_PLAN.md), [Eulerian bias operators](docs/EULERIAN_BIAS.md),
  [composed LPT model](docs/EULERIAN_LPT.md), and [fixed-IC sampling](docs/FIXED_IC.md)
- [Periodic catalog / NUFFT observations](docs/PERIODIC_CATALOG.md)
- [MAP optimization](docs/MAP.md)
- [Particle-mesh evolution](docs/PARTICLE_MESH.md), [adjoint](docs/PM_ADJOINT.md),
  [posterior](docs/PM_POSTERIOR.md), [checkpoint/restart](docs/PM_CHECKPOINT.md)
- [Survey likelihoods](docs/SURVEY.md)
- [Known issues](docs/KNOWN_ISSUES.md) and [development plan](docs/FRAMEWORK_PLAN.md)
- [BAO prototype and historical development notes](docs/BAO.md)

## Organization

Public headers stay here to preserve existing `#include <FML/FieldLevel/...>` paths.
`example/` contains examples; `tests/` contains regression tests.
[validation/](validation/README.md) contains maintained scientific drivers and
small input fixtures. [Validation records](docs/VALIDATION.md) summarize the saved
studies. Large generated chains, predictions and exploratory programs are kept
in a verified external archive; see [archive access](docs/ARCHIVE.md).

```sh
make -C FML/FieldLevel/tests run
make -C FML/FieldLevel/example
```

See the framework guide for compiler/backend options and MPI launcher settings.
Write new scientific run output outside the library source tree.

[PSWF NUFFT matter deposition and position adjoint](docs/NUFFT_MATTER.md)
provides an explicit alternative to smooth LPT particle assignment.

[Compact latent/product meshes](docs/COMPACT_MESHES.md) separate sampled field
coordinates from LPT and nonlinear operator padding.

[Eulerian checkpoint/restart](docs/EULERIAN_RESTART.md) supports shared/compact
meshes, with a bounded HMC example and indexed draw output.

[Bounded numerical/resource screen](docs/NUMERICAL_SCREEN.md) records mesh/tolerance
errors and measured small-fixture costs, including the FL-7 diagnosis and its [validated correction](docs/EULERIAN_METRIC.md).

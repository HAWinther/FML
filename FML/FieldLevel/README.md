# Field-level inference

Differentiable periodic linear, Zel’dovich/2LPT and particle-mesh models, Fourier
and Cartesian survey likelihoods, MAP/HMC inference and checkpoint/restart.
Distributed models support FFTWGrid slabs and PencilGrid pencils. Scientific
validation is bounded; consult the current limitations before interpreting results.

## Guides

- [Framework API and conventions](docs/FRAMEWORK.md)
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

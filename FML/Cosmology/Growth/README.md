# Cosmology/Growth

The linear and LPT growth of structure on a `Background` (separate from the background: the growth depends on the
theory of gravity at the perturbation level). Header-only, namespace `FML::COSMOLOGY::GROWTH`.

```
LPTGrowthFactors.h   LPTGrowthFactors: D1, D2, D3a, D3b (and the rates dlnD/dx) as functions of x = ln a and, when
                     scale dependent, of k (koverH0 = c k / H0). The equations are given by LPTGrowthEquations:
                     E, dlnE/dx, the clustering matter density (from a Background), mu and the second-order (2LPT)
                     source (from a PERTURBATIONS::QuasiStaticGravity or any functions), per-order source factors and
                     the initial growth rate (massive neutrinos). Initial conditions: the growing mode of a matter +
                     radiation universe (series in a/a_eq), or EdS. history() gives LPT/GrowthHistory.h.
```

```cpp
#include <FML/Cosmology/Growth/LPTGrowthFactors.h>
using namespace FML::COSMOLOGY;
auto bg = BACKGROUND::make_background("lcdm", params);
GROWTH::LPTGrowthSettings s;
s.a_normalize = 1.0 / 50.0;                       // D1 = 1 at z = 49
GROWTH::LPTGrowthFactors growth(*bg, s);          // GR; pass a QuasiStaticGravityPtr for modified gravity
double D2 = growth.D2(std::log(0.5));             // ~ -3/7 D1^2
```

Used by SimulationCode's GravityModel (with its GeffOverG, source factors and neutrino transfer functions) and by
`LPT::compute_LPT_growth_factors_LCDM / _GeffGLCDM`. Not yet: SphericalCollapse, HaloModel,
HybridBiasOneLoopMG (their own growth ODEs).

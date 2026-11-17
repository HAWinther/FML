# Hybrid bias spectra in redshift space

The multipoles P_ab^l(k) of the cross spectra of the Lagrangian bias operators 1, delta_L, delta_L^2 - <delta_L^2>,
s^2 - <s^2>, nabla^2 delta_L advected to the redshift-space positions of the particles (s = x + v_n n, plane-parallel
along a box axis), and their predictions. The tracer multipoles are P_tt^l = sum_ab b_a b_b P_ab^l. Galaxy-specific
velocities (fingers of god) and noise are not in the templates: they enter the fit as counterterms k^2 mu^(2n) P and
stochastic terms. Separate from the real-space code for now (`HybridBiasSpectra.h` is unchanged).

| | Header | |
|---|---|---|
| Cross multipoles | `ComputePowerSpectrum.h` | `compute_cross_power_spectrum_multipoles_fourier(grid1, grid2, Pell, los)`; the auto version now uses the same code (identical results) |
| Measurement | `HybridBiasRedshiftSpace.h` | `compute_hybrid_bias_multipoles` (particles with velocities + linear field), `compute_zeldovich_hybrid_bias_multipoles` (the Zel'dovich field in redshift space: control variate, and an exact test), result `HybridBiasMultipoles<N>` (`get(a, b, ell)`, `scale`, `tracer_multipole`, `write`) |
| Zel'dovich prediction | `HybridBiasZeldovich.h` | `zeldovich_hybrid_bias_pkmu(P, W_op, W_disp, k, f, nu, ...)`, `zeldovich_hybrid_bias_multipoles(..., f, ellmax, ...)`: exact (all orders) for the Zel'dovich field, separate filters for operators and displacement |
| One-loop prediction | `HybridBiasOneLoopRSD.h` | `HybridBiasOneLoopRSD(k, P, options)`: `pkmu(f, nu, k)`, `multipoles(f, k, ellmax)` |

```cpp
#include <FML/ComputePowerSpectra/HybridBiasRedshiftSpace.h>
using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
HybridBiasOptions options;                      // as for the real-space spectra
HybridBiasMultipoles<3> m;
compute_hybrid_bias_multipoles<3>(part.get_particles_ptr(), part.get_npart(), part.get_npart_total(),
                                  delta_linear_fourier, Npart_1D, Nmesh, velocity_to_displacement, options, m);
m.scale(boxsize);                               // k in h/Mpc, P in (Mpc/h)^3
auto & P_delta_s2_quadrupole = m.get("delta", "s2", 2);
```

`velocity_to_displacement` converts a particle velocity to a shift in box units, as for
`compute_power_spectrum_multipoles`. By default the result is the mean over the three axes (`los_axes`); `ellmax`
(default 4) can be raised for a fit with a fingers-of-god damping (which mixes the multipoles).

## Predictions

- **Zel'dovich.** The redshift-space displacement is (1 + f n n) Psi, so the Zel'dovich integral is the real-space one
  with the displacement wavevector K = (1 + f n n) k and the phase e^{ik.q}; the angle of q around K is done
  analytically (J_0). The operators are Lagrangian (no redshift-space factor). Large scales: P_11 = (1 + f nu^2)^2 P,
  P_1d = (1 + f nu^2) P, P_1n = -k^2 (1 + f nu^2) P (times the Zel'dovich damping exp(-K^2 sigma^2)).
- **One loop.** The ten spectra without nabla^2 delta are the components of `LPTRedshiftSpace` (velocileptors'
  LPT_RSD) mapped to the basis as in real space. The five with nabla^2 delta = F delta_L (F = -k^2) are the delta_L
  terms with F on the operator leg (`LPTRedshiftSpace::operator_components`). (delta^2, n) and (s^2, n) keep the
  one-loop terms only (the real-space `HybridBiasOneLoop.h` keeps the full Zel'dovich expressions for these two).
  Input P(k) times W^2 of the field (one smooth filter for operators and displacement). Valid for k >~ 0.005 h/Mpc.

**Caveat (nabla^2 delta).** The spectra with nabla^2 delta are dominated by small scales: the loop terms renormalize
the k^2 P of the template by tens of percent (P_dn / (-k^2 P) = 1.5 at f = 0 with a 1 h/Mpc cutoff; real space agrees),
and the zero-lag term <nabla^2 delta Psi^(1) Psi^(2)>, which gets the redshift-space factors (1 + f mu^2)(1 + 2 f mu^2),
changes the sign of P_1n along the line of sight (f = 0.8, nu = 1). These terms depend on the cutoff of the operator
field, so compare with the measurement for the same cut / smoothing (`options.kcut`, `options.smoothing_scale`) and
treat the amplitude of the k^2 P part as a free parameter (it is degenerate with b_nabla and the counterterms).

## Tests (`tests/test_hybridbias_rsd.cpp`, 1, 2 and 4 tasks, ~2 min)

- Cross multipoles of a grid with itself = auto multipoles (exactly); for delta_2 = (1 + beta mu^2) delta_1,
  P_0 / P = 1 + beta/3 (1e-13), P_2 / P = 2 beta/3 (1.7%, the discrete mu of the modes).
- f = 0: the monopole of the Zel'dovich multipoles = the real-space spectra (5e-15). Particles at q + Psi with
  velocities Psi = the Zel'dovich field in redshift space (all 15 spectra, l = 0, 2, 4; 1e-13).
- The Zel'dovich field (64^3, 500 Mpc/h, f = 0.8) against the Zel'dovich prediction averaged over the modes of each
  bin: mean (deviation / sigma_Gaussian)^2 = 0.13 for the 15 spectra, l = 0, 2, 4, k < 0.15 h/Mpc.
- The Zel'dovich prediction at f -> 0 = the real-space one (1e-9); against the components of `LPTRedshiftSpace`
  without loops (hypergeometric series + FFTLog, an independent calculation) for (1,1), (1,d), (d,d), (1,d2), (1,s2),
  (1,n), (d,n), (n,n) at f = 0.8: 1.4e-3.
- One loop: f = 0 = the real-space one-loop prediction (1.2e-3 for the ten, 2.4e-3 for (1,n), (d,n), (n,n)); the
  operator machinery with F = 1 = the delta_L components (1.2e-3, f = 0 and 0.8).

Not done: comparison with N-body multipoles (the ingredients for an emulator: TODO in the project notes), wide-angle
terms for light cones, scale-dependent growth (f(k)).

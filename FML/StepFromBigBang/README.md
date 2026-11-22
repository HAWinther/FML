# StepFromBigBang

Initial conditions made by taking LPT-consistent time steps from (almost) the Big Bang, instead of displacing the
particles with nLPT at the initial redshift.

The particles start on the lattice with the Zel'dovich displacement and velocity at `a_start` ~ 0. A few steps move
them to `a_end`. Each step is exact for the LPT growing mode with any step size:

| Scheme | Forces | Exact for | Models |
|---|---|---|---|
| `bullfrog` | 1 per step | 2LPT | scale-independent growth (BullFrog: Rampf, List & Hahn 2024) |
| `lpt3` | 2 per step (+1) | 3LPT incl. the transverse term (first step); longitudinal 3LPT (later steps) | GR-like, scale-independent growth |
| `hybrid` | 1 per step | linear growth D(k, a) for every k, then 2LPT at a reference k | any G_eff(k, a), massive neutrinos |

The steps only use the force identity of the growing mode: the force at the exact LPT configuration is a known
combination of the LPT fields. So they follow the growing mode of the *particle system with the force that is used*,
not continuous LPT. Near the particle Nyquist frequency these differ: a reference started from continuous 3LPT had a
2.6% error there. There is also no aliasing from products of fields on a grid.

Header only:
- `StepFromBigBang.h`: the schemes;
- `Growth.h`: the growth history;
- `Forces.h`: ready-made force functions, the growth model of a `Background`, and the one-call
  `generate_particles_from_big_bang`.

Tests are in `tests/`. The derivations are in the note `lpt3step.pdf` (Documentation).

## Usage

The short way (`Forces.h`):

```cpp
#include <FML/StepFromBigBang/Forces.h>
using namespace FML::STEPFROMBIGBANG;

GrowthModel growth = growth_from_background(bg, growth_settings);           // bg must outlive it
auto force = make_force_nufft<3, P>(Npart_1D, kappa0_from_background(bg));  // the exact force (recommended)
Settings s;
s.scheme = "bullfrog";      // or "lpt3", "hybrid"
s.a_start = 1e-7;
s.a_end = 1.0 / 20.0;
s.nsteps = 1;
s.boxsize = 500.0;          // Mpc/h
FML::PARTICLE::MPIParticles<P> part;
generate_particles_from_big_bang<3, P>(part, Npart_1D, 1.5, delta_fourier, a_ref, growth, force, s);
// part: positions and velocities at a_end
```

`delta_fourier` is the linear density contrast at `a_ref`. With scale-dependent growth each k is moved with
D₁(k, a). For a modified-gravity model, pass μ(a, kBox) to the force (`mu_of_a_kBox(qs_mu, boxsize)`) and the theory
to `growth_from_background`.

**Your own force.** The force is any function with the signature below. It computes F = dv/dln a at the particles'
current positions at a, with an optional factor `filter(kBox)` on the density in Fourier space, and adds wᵢ F to the
targets. Add linear neutrinos, several species, screening and so on here; `add_to_targets` helps.

```cpp
auto force = [&](MPIParticles<P> & part, double a, const std::function<double(double)> & filter,
                 const std::vector<Target> & targets) { ... add_to_targets<NDIM>(p, F, targets); };
```

**Step by step.** `StepFromBigBang<NDIM, P> sfbb(growth, force, settings)`, then
`sfbb.make_start_particles(part, Npart_1D, buffer, delta_fourier, a_ref, lattice_shift)`, then `sfbb.run(part)`.

**Which force.** Use `make_force_nufft`, the exact force band-limited at the mesh's Nyquist frequency. Near the
lattice a particle-mesh force (`make_force_pm`) differs from it by 9-17% rms in a Zel'dovich configuration at half the
final growth. The causes are the assignment window, and on a finer grid the lattice's discreteness above the particle
Nyquist frequency. Over one step that is ~2× the beyond-Zel'dovich displacement; the mean force is right (projection
0.98-1.005). `make_force_pm` is for later times or for comparisons.

## Conventions

- **Positions** are in box units [0, 1).
- **Velocities** are the FML NBody convention, v = a² dx/dt / (H₀ B) = a² E dx/dln a.
- **Force:** the callback returns F = dv/dln a in these units. For a linear mode ψ it is κ(a, k) D₁(a, k) ψ, with
  κ = a² E (3/2) Ω_m(a) μ(a, k) s₁(a, k) (see `Growth.h`).
- **Growth factors:** `GrowthModel` uses EdS-normalised higher orders, r₂ = −7/3 D₂, r₃a = −3 D₃a, r₃b = 21/5 D₃b, and
  the transverse r₃c, which the module integrates. In EdS they are D₁² and D₁³.
- **Particle type:** needs `get_pos()` and `get_vel()`. `lpt3` also needs `get_sfbb_scratch()`: one more vector per
  particle, which moves with the particle under MPI. `set_id()` is used if present.

## The force decomposition used by `lpt3`

Write the displacement as x − q = D₁Ψ₁ + r₂Ψ₂ + r₃aΨ₃a + r₃bΨ₃b + r₃cΨ₃c. The force is κ(a) times a time-independent
functional F = F₁ + F₂ + F₃ + …:

- F₁[ψ] = ψ_L (the longitudinal part);
- F₂[Ψ₁, Ψ₁] = 7/3 Ψ₂;
- X ≡ 2F₂[Ψ₁, Ψ₂] = 9/5 Ψ₃b;
- Y ≡ F₃[Ψ₁, Ψ₁, Ψ₁] = 6Ψ₃a + 21/5 Ψ₃b + 7Ψ₃c.

The split of X and Y was measured with the exact particle force (residuals ∝ D, i.e. fourth order). It is the one
the growth equations of `LPTGrowthFactors` imply for D₃b. The transverse third-order force comes entirely from Y. The
first `lpt3` step chooses its two force times and weights so that the positions are exact for Ψ₂, Ψ₃a, Ψ₃b and Ψ₃c. In
EdS that gives s₁ = 7/54 D₁ and s₂ = 7/18 D₁.

## Tests (`tests/`, `make run`)

All tests use 64³ particles and the exact NUFFT force (tolerance 10⁻¹²). Errors are relative to the beyond-Zel'dovich
displacement, against 64 BullFrog steps.

- **EdS, smooth field (k < k_Ny/4), σ = 0.3:**

  | Forces | `bullfrog` | `lpt3` |
  |---|---|---|
  | 2 | 2.9×10⁻² | 2.4×10⁻² |
  | 4 | 1.1×10⁻² | 3.3×10⁻³ |
  | 8 | 3.7×10⁻³ | 1.3×10⁻³ |
  | 16 | 1.1×10⁻³ | 3.6×10⁻⁴ |

  These match the prototype. The reference agrees with 32 `lpt3` steps to 9×10⁻⁵.
- **ΛCDM to z = 19 (σ → 0):** `bullfrog` and `lpt3` are linear-exact. The distance from linear growth is ~10⁻⁶ at
  σ = 10⁻⁵, i.e. second order.
- **ΛCDM with radiation, start at z = 10⁶ (smooth field, σ = 0.3).** At the start r₂/D₁² is 0.012 of its EdS value.
  The errors are as in EdS: `bullfrog` 3.0×10⁻², 1.1×10⁻², 3.8×10⁻³ and `lpt3` 2.4×10⁻², 3.3×10⁻³, 1.3×10⁻³ at 2, 4,
  8 forces. The references agree to 9×10⁻⁵ and converge at second order. This needs the first BullFrog step to treat
  the start as the Zel'dovich state: in the radiation era D₁′ is small, so the growing mode's second-order velocity is
  not negligible next to D₁′Ψ₁. Treating the start as the full growing mode gave first-order convergence.
- **Scale-dependent toy model**, μ = 1 + k²/[3(k² + m²)] with the transition at k = 0.45 h/Mpc at z = 19:
  - the scale-dependent steps are linear-exact for every k (one step to z = 19: 2.5×10⁻¹⁰);
  - the hybrid's BullFrog part after z = 100 costs 0.2% (1 step) or 0.06% (2 steps), because it follows the
    growth of one reference k.

## Limitations

- `lpt3` needs scale-independent growth and the GR third-order force. Its later steps do not resolve the transverse
  third order; that needs the Lagrangian position, and the effect measured was below 4%.
- `lpt3` with growth other than EdS treats the longitudinal third order of the later steps as one field
  (¼ Ψ₃a + ¾ Ψ₃b). This is exact in EdS; elsewhere the error is set by how much D₃a/D₁³ and D₃b/D₁³ change over a step.
- `hybrid` follows one reference k after `a_switch`.
- There is no velocity friction (interacting dark energy with a drag) and no two-fluid start (CDM and baryons with
  different growth).

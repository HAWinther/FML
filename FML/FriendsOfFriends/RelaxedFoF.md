# Relaxed friends-of-friends

`RelaxedFoF.h` implements the relaxed FoF halo finder of Dai, Feng, Seljak & Singh (2019),
[arXiv:1908.05276](https://arxiv.org/abs/1908.05276), §2.1. It follows the paper's Algorithm 1 and FastPM's
implementation (`libfastpm/rfof.c` at
[15b6c4f](https://github.com/fastpm/fastpm/blob/15b6c4fd7502a81d99dd13f54fcc9cfa44be1331/src/lua-runtime-fastpm.lua#L236)).
In low-resolution simulations, small halos are too diffuse for the standard linking length of 0.2 and break up, so
the mass function is too low and the bias of small halos too high. Relaxed FoF fixes this in two ways:

- it uses a longer linking length for small halos than for large ones;
- it rejects "fake" halos: unbound clumps in dense regions that the longer linking length joins, recognised by
  their high velocity dispersion.

```cpp
#include <FML/FriendsOfFriends/RelaxedFoF.h>
FML::FOF::RelaxedFoFParameters p;           // FastPM's defaults; p.mode = Mode::paper for Algorithm 1
std::vector<size_t> halo_id;                // per particle: the halo ID or FML::FOF::no_FoF_ID
std::vector<FML::FOF::RelaxedFoFPass> passes;
auto halos = FML::FOF::RelaxedFriendsOfFriends<Particle, 3>(
    part, npart, p, z, E_of_z,
    mass_unit_msunh,      // halo mass (sum of get_mass) -> Msun/h
    velocity_unit_kms,    // particle velocity -> peculiar km/s
    true, buffer_over_box, &halo_id, &passes);
// halos[i]: the FoFHalo quantities (id, np, mass, pos, vel, vel_rms) and pass, bin, linking_length,
// mass_msunh, vdisp_kms, vdisp_ratio
```

In SimulationCode and SnapshotAnalyzer (through `FML/Analysis`, `find_halos`), set `fof = true` and
`fof_relaxed = true`; the parameters are the `fof_relaxed_*` entries in
[PARAMETERS.md](../SimulationCode/PARAMETERS.md), with the same names and defaults in both. The halo catalogue then also has the pass, bin, linking length
and σ_v/V_std of each halo: extra text columns, or the `RelaxedFoF` group of the HDF5 file.

## The algorithm

1. **Bins.** Halos are split into bins by their number of particles: N < 20, 40, 80, 160, 320, ∞
   (`bin_edges`, eq. 2.2).
2. **Linking length per bin**, in units of the mean particle separation of the whole simulation:
   - smallest halos: l₁(z) = l1 − a1/(1+z) (eq. 2.3);
   - largest halos: lₙ(z) = max(l6 − a2/(1+z), `linking_length_min`) (eq. 2.4);
   - in between: lᵢ = [(n−i) l₁ + (i−1) lₙ]/(n−1). This is eq. 2.5, which is printed with a typo in the paper;
     FastPM implements it this way.
3. **Velocity cut.** A halo is fake if σ_v ≥ r₀(z) V_std(M, z), where
   - V_std = V0 (E(z) M / 10¹⁵ M☉/h)^{1/3}, V0 = 1100 km/s (eq. 2.1, Evrard et al. 2008);
   - r₀(z) = b1 − b2 ln(1+z) (eq. 2.6);
   - σ_v is the **3D** velocity dispersion √(σ_x² + σ_y² + σ_z²) in peculiar km/s, as in FastPM;
     `velocity_dispersion_3d = false` uses the 1D dispersion instead;
   - M is the halo mass in M☉/h.
4. **Pass i** (i = 1..n) runs FoF with lᵢ on the particles that are still active:
   - a group of at least `nmin` particles is saved if it has fewer than Nᵢ particles and is not fake, and its
     particles are removed;
   - particles that are in no group are removed;
   - the rest (particles of larger groups and of fake groups) go on to the next pass.

   Each pass therefore only splits groups of the pass before, and every particle ends up in at most one halo.
5. **Modes.**
   - `fastpm` (the default) stops after the n passes and drops the particles of fake halos from the last pass. This
     is FastPM's implementation, and FastPM's parameters were fitted with it.
   - `paper` adds the `while` loop of Algorithm 1: after the n passes, the linking length is multiplied by
     `shrink_factor` = 0.9 per pass, and every group that passes the velocity cut is saved, of any size. It stops
     when no particles are left or the linking length is below `shrink_linking_length_min`.

## Parameters

The defaults are FastPM's (`rfof_*` in its `lua-runtime-fastpm.lua`):

| | l1 | a1 | l6 | a2 | linking_length_min | b1 | b2 | nmin |
|---|---|---|---|---|---|---|---|---|
| default | 0.25 | 0.012 | 0.24 | 0.06 | 0.2 | 7.02 | 6.025 | 8 |

FastPM says they are fitted to Illustris (TNG300-2-Dark, 1250³ particles in 205 Mpc/h) for FastPM with a mean
separation of 0.3 Mpc/h (625³ particles) and 40 steps, with an initial-condition mesh twice as fine as the particle
grid. The paper also gives the fitting procedure (matching the halo mass function and bias), but not the fitted
values. The parameters depend on the resolution, the number of time steps and the N-body scheme (FastPM's PM versus
COLA), so for other setups they should be refitted against a high-resolution simulation. With the defaults the
relaxation is modest. The linking length of the smallest halos is 0.238 at z = 0 and 0.246 at z = 2, and that of
the largest is 0.20 and 0.22.

**The defaults at high redshift.** r₀(z) = 7.02 − 6.025 ln(1+z) is 7.0 at z = 0, 2.8 at z = 1 and 0.40 at z = 2. It
is ≤ 0 for z > 2.2, where every halo is rejected. The parameters were fitted at z ≤ 2 (the paper's outputs). The
code warns when r₀ ≤ 0, and `r0_min` sets a floor; the default is no floor, as in FastPM.

## Calibration for coarse PM/COLA runs (`pm_calibrated`)

The FastPM defaults do not carry over to coarser runs. The settings below were fitted against Gadget-4 TreePM
from the same initial conditions, using FoF b = 0.2 halos of at least 20 particles. The runs were SimulationCode COLA
with a force grid twice the particle grid, CIC and 40 steps (linear in a), from 2LPT at z = 19. There were two
resolutions, each with two paired-and-fixed phases:

- 256³ particles in 250 Mpc/h: a mean separation of Δ = 0.98 Mpc/h, m_p = 8.3×10¹⁰ M☉/h;
- 512³ particles in 1000 Mpc/h: Δ = 1.95 Mpc/h, m_p = 6.7×10¹¹ M☉/h.

Without relaxation, FoF b = 0.2 on these runs misses 19–24% of the halos above 20 particles and 9–10% above 100, and
the 20–50 particle halos are 11–14% too light. The PM halos are too diffuse. A single longer linking length does not
fix this: b = 0.25 gives the right counts at 20–50 particles, but +15 to +60% above 300.

The fit was over a grid of (l1, l6), without the velocity cut, minimising the rms of the counts above 20–1000
particles and of the median masses of matched halos. It gives:

| | value |
|---|---|
| bins | 18, `smooth_bin_edges()` = 20·2^(k/4), k = 0..16 (20, 24, 28, …, 269, 320) |
| l6, a2, linking_length_min | 0.20, 0, 0.1 (the large halos; constant in z at both resolutions) |
| l1, a1 | l1 = 0.22 at Δ = 1 Mpc/h and 0.20 at Δ = 2 Mpc/h, a1 = −0.04, i.e. l₁(z) = l1 + 0.04/(1+z) |
| velocity_rejection | false |

`RelaxedFoFParameters::pm_calibrated(Delta)` returns these settings, with l1 linear in log₂Δ. The accuracy is 2–3%
(rms) in the counts and the matched masses at z = 0–1, and the halo bias at fixed number density is within
0.5–3.5% of Gadget-4. Note:

- **The sign of a1 is opposite to FastPM's.** The linking length of the small halos grows towards z = 0.
- **Only two resolutions were fitted.** With them, the dependence on the particle spacing cannot be told apart from
  the dependence on the particle mass.
- **PGD needs other settings.** On particles moved by potential gradient descent (FML/PGD), the large halos need l6
  smaller by about 0.005–0.01. PGD does not recover the small halos, so find the halos before applying PGD.

**Why 18 bins.** The linking length jumps at each bin edge. With FastPM's 6 bins (edges 20, 40, 80, 160, 320) this
gives too few halos just below each edge and too many just above it: a ±10–30% sawtooth in dn/dM at 1–2 Mpc/h.
The 18 bins have the same end points and the same linear-in-log-N interpolation, but steps of 1/17 instead of 1/5
of l₁ − lₙ. This reduces the sawtooth to about ±5% at the same cost, since each pass only splits the groups of the
previous one. With the same l1 and l6, 18 bins give 3–5% more halos above 20 particles and 4–8% fewer above 300 than
6 bins, so the end points above were refitted with 18 bins.

## Implementation notes

- **Particles.** The particles are neither reordered nor changed; FriendsOfFriends sorts them in place. The halo ID
  is stored with `set_fofid` if the particle type has it. Each pass links a copy of the active particles
  (position, velocity, mass and index).
- **FoF options used.** The passes call `FriendsOfFriends` with two options added for this:
  - `mean_particle_separation`, so that the linking length of a subset is in units of the separation of the whole
    simulation;
  - support for tasks without particles: later passes leave tasks empty. All tasks take the same path, and only an
    all-empty input returns early.

  `FoF_verbose = false` silences the FoF output of the passes. RelaxedFoF prints one line per pass instead; set
  `params.verbose = false` to turn that off as well.
- **MPI.** The slab decomposition and buffer are those of `FriendsOfFriends`, and a halo is kept by the task where
  its center is. The decisions (saved or not) for the groups a task keeps are sent to the left and right neighbour
  tasks. The buffer reaches only that far, so the neighbours can remove the halo's particles in their domains.
- **Halo IDs** are unique over the tasks but not contiguous: the FoF ID of the pass plus an offset per pass.
- **Masses.** σ_v and the mass use the FoFHalo quantities: the mass-weighted mean velocity, and the mass as the sum
  of `get_mass` (1 per particle without it) times `mass_unit_msunh`. FastPM uses the number of particles times the
  particle mass, which is the same for equal-mass particles.
- **SimulationCode.** With CDM and baryons as separate particles, the linking lengths are in units of the
  separation of one species, and `nmin` and the bin edges count particles of one species. This is the same
  convention as standard FoF there.

## Tests

`tests/test_relaxed_fof.cpp` runs with `make run` on 1, 2 and 4 MPI tasks. It checks:

- **The halos against an independent implementation.** `tests/make_rfof_reference.py` uses SciPy's KD-tree FoF on
  a catalogue of 9016 particles: a uniform background, clumps of 8–600 particles from compact to diffuse, fake
  (too fast) clumps, and cold cores in hot envelopes that only the shrinking passes of paper mode recover. For
  fastpm mode at z = 0, 1 and 1.5 and paper mode at z = 0 and 1.5, the members, pass, bin and size of every halo
  must be the same.
- **Standard FoF.** One bin without the velocity cut is standard FoF, and `FriendsOfFriends` gives the
  reference's groups.
- **The FoF options.** FoF of a subset (x > 0.75) with the mean separation of all the particles; on 2 and 4 tasks,
  1 and 3 of the tasks have no particles.
- **The details.** The linking lengths, r₀, the checks of the settings, and the IDs stored in the particles.
- **The PM calibration.** `smooth_bin_edges` (18 bins; one edge per doubling gives FastPM's edges) and the
  settings and linking lengths of `pm_calibrated`.

The test data are drawn so that no pair distance or velocity ratio is within 10⁻¹⁰ (relative) of a threshold. With
the exchange between neighbour tasks disabled, the 2- and 4-task runs fail. The test data and the SciPy code are
regenerated with `python3 make_rfof_reference.py`, which needs numpy and scipy.

**A SimulationCode run** (COLA, 128³ particles, 64 Mpc/h, 20 steps; mean separation 0.5 Mpc/h, so the defaults are
not tuned for it):

- relaxed FoF finds about 20% more halos than standard FoF with the same nmin = 8, mostly below 100 particles;
- real halos have σ_v/V_std ≈ 0.7–1.2 at all masses;
- at z = 0, 8% of the 8–20 particle groups are rejected as fake (σ_v > 7 V_std), and fewer than 0.1% of the
  groups above 100 particles are.

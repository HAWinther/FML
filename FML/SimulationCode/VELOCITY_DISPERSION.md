# Redshift space: the halo velocity-dispersion correction (not implemented yet)

Coarse PM/COLA runs get the redshift-space power spectrum wrong on small scales, because their halos are too cold.
This note records what was found in the PM-settings and halo studies (backlog item 59 and follow-ups, November 2026):
which correction works, which does not, and how to add it to SimulationCode. **The correction is not in the code
yet.**

## The problem

The runs were SimulationCode COLA with a force grid twice the particle grid, CIC and 40 steps (linear in a), from 2LPT
at z = 19. There were two resolutions, each with two paired-and-fixed phases:

- 256³ particles in 250 Mpc/h: a mean particle separation of Δ = 0.98 Mpc/h;
- 512³ particles in 1000 Mpc/h: Δ = 1.95 Mpc/h.

The reference is Gadget-4 TreePM from the same initial conditions, using its redshift-space multipoles P₀, P₂, P₄.
These are the mean over the three axes, with PCS, interlacing and shot noise subtracted.

- **The Kaiser limit is right.** At k = 0.05 h/Mpc, P₀ and P₂ of the PM runs agree with Gadget-4 to 0.5 and 2%.
  So the large-scale velocities, and the velocity units of the snapshots, are fine.
- **On small scales the PM run has too much power in redshift space,** although it has too little in real space:

  | | 250 Mpc/h | 1000 Mpc/h |
  |---|---|---|
  | P₀ at k = 0.3 h/Mpc | +8 to +11% | +12 to +17% |
  | P₀ at k = 1 h/Mpc | +26 to +28% | +43 to +57% |
  | P₂ at k = 0.3–0.5, relative to P₀ | +15 to +25% | +15 to +25% |

  The fingers of god are too weak.
- **The halos are too cold.** The table gives the 1D velocity dispersion of the PM halos (FoF b = 0.2) divided by
  the Evrard et al. (2008) relation σ_DM = 1082.9 km/s (E(z) M/10¹⁵ M☉/h)^0.3361, with M the FoF mass:

  | | 20–50 p | 50–100 p | 100–300 p | 300–1000 p | > 1000 p |
  |---|---|---|---|---|---|
  | z = 1 | 0.46 | 0.53 | 0.59 | 0.66 | 0.72 |
  | z = 0 | 0.37 | 0.42 | 0.48 | 0.55 | 0.61 |

  The ratios are the same in both boxes, with relaxed FoF, and with or without PGD on the positions. The internal
  motions are not resolved by the PM force (force cell 0.5–1 Mpc/h).

## What does not work

- **Potential gradient descent on the positions** (FML/PGD) makes P₀ in redshift space worse: +44% at k = 2 h/Mpc
  at 250 Mpc/h. It adds small-scale clustering but no velocity dispersion.
- **A PGD-like velocity kick also fails.** The kick was v → v + γ S_v / (velocity-to-displacement factor), with S_v
  the PGD displacement for α = 1 (filters k_l = 1.5 h/Mpc and k_s = 4, 6 or 10 h/Mpc). For |γ| ≤ 0.02 it changes the
  multipoles by less than 1%. The best |γ| = 0.2–0.4 acts as a dispersion, because the sign does not matter. Even then,
  P₀ at 250 Mpc/h stays +6 to +15% at k = 1–3 and P₂ +7 to +17% of P₀. It also starts to damage the Kaiser limit
  (−1.4% at k = 0.1). A coherent displacement towards the potential minima has the wrong shape for the missing random
  motions.

## What works: raise the dispersion of the halos

Find the halos, with FoF or relaxed FoF, before any PGD. Then give the particles of each halo an extra isotropic
Gaussian velocity with

    sigma_add^2 = max[ (c sigma_Evrard(M, z))^2 - sigma_PM^2, 0 ]   (1D)

Here σ_PM is the halo's own 1D dispersion about its mean velocity, and M is its FoF mass. Field particles are not
changed, so the Kaiser limit is kept.

**At z = 0,** with the position PGD on and the halos of at least 20 particles, a single c works at both resolutions:

| (rms, 0.05 < k ≤ k_Ny/2) | P₀ error | P₂ error (of P₀) |
|---|---|---|
| 250 Mpc/h, PM | 15.4% | 9.8% |
| 250 Mpc/h, c = 0.80 | **1.5%** | **1.0%** |
| 1000 Mpc/h, PM | 17.9% | 17.4% |
| 1000 Mpc/h, c = 0.84 | **1.3%** | **3.9%** |

- At 250 Mpc/h with c = 0.8, P₀ is within about 1% of Gadget-4 up to k ≈ 0.6 h/Mpc.
- FoF b = 0.2 and relaxed FoF give nearly the same result. Relaxed FoF finds about 30% more halos (5.9M against
  5.3M particles at 250 Mpc/h, z = 0) and damps slightly more above k ≈ 1.
- The result is steep in c, because most halos already have 0.4–0.6 of the target, so little is added until c passes
  about 0.65. At 250 Mpc/h, z = 0: c = 0.7 leaves P₀ +10% at k = 0.5–1, and c = 0.9 gives −5 to −8%.
- A mass dependence c = c₀ (N/100)^s with s = 0.1 or 0.2 does not help: P₀ gets slightly better, P₂ worse.
- Halos down to 10 particles help a little (1 point at 250 Mpc/h). At 1000 Mpc/h, z = 1 they help clearly
  (P₀ at k = 1: +24% instead of +34%).

## The redshift dependence and the halo definition

These fits used the position PGD calibrated to Gadget-4 (α = 0.0042/a at Δ = 1 Mpc/h, α = 0.0187–0.0234 at
2 Mpc/h, with k_l = 1.5 and k_s = 6/Δ). Using the EuclidEmulator2-calibrated PGD instead changes the numbers by less
than 1 point. Three halo definitions were tried: FoF b = 0.2 with at least 20 particles, FoF with at least 10, and
relaxed FoF with at least 20 (the `RelaxedFoFParameters::pm_calibrated` settings). The c grid was 0.7–1.1 at z = 0,
0.5 and 1, with both phases. The table gives the rms of the P₀ error and of the P₂ error (relative to P₀) for
0.05 < k ≤ k_Ny/2, at the best c on that grid:

| | z = 1 | z = 0.5 | z = 0 |
|---|---|---|---|
| 250 Mpc/h, PM + PGD (no correction) | 17.1 / 11.9% | 17.5 / 12.3% | 17.8 / 13.2% |
| FoF ≥ 20 | c = 0.9: 5.0 / 5.4% | c = 0.9: 2.0 / 3.9% | c = 0.8: 1.4 / 1.2% |
| FoF ≥ 10 | c = 0.9: 4.2 / 4.1% | c = 0.9: 1.6 / 3.2% | c = 0.8: 1.2 / 1.0% |
| relaxed FoF ≥ 20 | c = 0.9: 3.2 / 3.9% | c = 0.9: **0.8 / 3.0%** | c = 0.8: **0.7 / 0.5%** |
| 1000 Mpc/h, PM + PGD (no correction) | 16.7 / 21.3% | 18.6 / 18.2% | 19.3 / 16.0% |
| FoF ≥ 20 | c = 1.0: 6.6 / 13.5% | c = 0.9: 4.9 / 8.5% | c = 0.8: 2.2 / 3.7% |
| FoF ≥ 10 | c = 1.0: 4.4 / 9.9% | c = 0.9: 3.6 / 6.2% | c = 0.8: 1.7 / 2.7% |
| relaxed FoF ≥ 20 | c = 1.0: 5.2 / 12.2% | c = 0.9: 3.2 / 6.8% | c = 0.8: **0.8 / 2.0%** |

- **The best c is nearly the same at both resolutions and for all three halo definitions:** 0.8 at z = 0, 0.9 at
  z = 0.5 and 0.95–1.0 at z = 1. That is c(a) ≈ 1.1 − 0.3a, or c = 0.8 + 0.3 z/(1+z).
- **Relaxed FoF is the best choice at z ≤ 0.5.** At z = 0 it gets P₀ and P₂ to within 1% up to k_Ny/2 at 1 Mpc/h,
  and within 1 and 2% at 2 Mpc/h.
- **At 2 Mpc/h and z ≥ 0.5, adding the halos of 10–19 particles helps.** Particles in small groups also need the
  correction there.
- **What is left: 2 Mpc/h at z ≥ 0.5, and k ≳ 1 h/Mpc at 1 Mpc/h and z = 1.** In these cases no single c fits P₀ and
  P₂ together. At 1000 Mpc/h with c = 1.0, z = 1, P₀ is +5 to +8% at k = 0.5 and +25 to +30% at k = 1, and P₂ is
  +13 to +16% of P₀ at k = 0.5. Part of the missing velocity dispersion there is in particles outside the resolved
  halos, which this correction does not touch.

## Implementation in SimulationCode

The halos are already found at each output (`fof`, `compute_fof_halos`), with the particles' halo IDs (`set_fofid`).
The correction would be an output-time option, like the PGD output:

1. Sum the members' velocities, squared velocities and counts per halo. With MPI, a halo's particles live on more than
   one task. The halo IDs are global but not contiguous, so either reduce the sums by ID across neighbour tasks (as
   RelaxedFoF exchanges its decisions) or attach the halo's σ_PM and mass to the halo catalogue and send them back to
   the member particles.
2. For each member particle, add σ_add times a unit normal vector. Use a reproducible random stream per particle ID,
   so the result does not depend on the number of tasks.
3. Apply it only to a copy of the velocities used for the redshift-space outputs (`pofk_multipole`, snapshots), as
   PGD does with a copy of the positions, so the time stepping is not affected.

Parameters would be the switch, c(a) (default 1.1 − 0.3a), the halo definition (default: the relaxed FoF of the
`fof` output) and the minimum halo size.

The study tools (`runs/pgd/src/Disp.cpp`, `pgd_disp`; `Rsd.cpp`, `pgd_rsd`) do this on one MPI task on Gadget
snapshots. They are kept with the study runs and are not part of the library.

# Dataset-203 deterministic mixing diagnosis

Audit complete. All source records are unchanged. The original dataset remains
**FAIL**: 11/12 coordinate gates fail, maximum Rhat 1.2647, minimum bulk ESS 11.78,
with zero post-warmup divergences. This audit does not reclassify warmup or discard
unconverged records.

## Integration and transport are separate problems

Original cold field mean acceptance probabilities are 0.9735,0.9843,0.9837,0.9762.
Together with zero divergences and the sampled-state derivative controls, there
is no observed integration instability explaining this dataset's severe separation.
High local acceptance does not certify mixing.

The cold-end exchange bottleneck spans beta 1/0.5 and 0.5/0.25. In ensemble 1,
the latter edge accepts 186/4096 attempts (0.04541); reconstructed conditional mean
acceptance is 0.04524. Median log acceptance ratio is -8.288. Ensemble 0 accepts
154/4096 (0.03760). Other ensembles have their weakest edge at 1/0.5. The pooled
minimum rate is 0.05139 and passes the original 0.05 pooled gate; these individual
rates are diagnostic, not a newly introduced failure rule.

Independent qualified-journey reconstruction matches all native phase/trip counters:

| Ensemble | Cold-to-actual-prior-refresh legs | Completed return legs | Unfinished qualified legs |
|---|---:|---:|---:|
| 0 | 10 | 6 | 4 |
| 1 | 4 | **0** | 4 |
| 2 | 14 | 6 | 8 |
| 3 | 12 | 5 | 7 |

Thus ensemble 1 did reach the refreshed prior, but no qualifying journey returned
to cold during the retained window. This is a conservative transport diagnostic;
labels are refreshed configurations, and lack of a round trip alone is not a proof
of a biased marginal. The severe field Rhat/ESS failures provide separate evidence.
The original sin001 quarter means illustrate the nonstationarity/separation:
ensemble 0 changes from about -1.01 to +1.18, while ensemble 1 remains near -1.25
to -1.07. All retained quarters remain in the verdict.

## Selected geometry paths

Two declared endpoint sets produce 18 paths and two rescaling controls, 2020 points
total. Bias profiling exactly minimizes a convex box-constrained four-coefficient
likelihood; it is not bias marginalization. At draw 1024, all nine inspected routes
have profiled point-potential barriers of 37.1–61.6 above their higher endpoint.
At draw 2047 the routes toward ensemble 0 have barriers 44.4–66.0; several routes
toward ensembles 2/3 have no barrier above the higher endpoint, while spherical
routes have barriers about 16.0/2.0. There is no universal disconnected-mode claim.

For the same inspected paths, the maximum profiled barrier decreases from 66.0
at beta 1 to 32.9 at 0.5, 16.3 at 0.25, 8.0 at 0.125 and 3.85 at 0.0625. This uses
white_prior+beta*Q_profile, preserving the untempered Gaussian prior. It describes
point potentials, not temperature overlap or free-energy/volume barriers.

Profiler reference checks pass; maximum KKT residual is below 3.9e-16, profile/full
target discrepancy below 3.3e-14, and maximum endpoint derivative discrepancy below
9.6e-9 across 64 comparisons. Rescaling controls retain their unchanged likelihood
quadratic. All recorded pre-pass energies reconstruct and match native counters.

## Implication and next experiment

The evidence supports investigating cold-end temperature overlap and exploration
between separated field configurations. A useful next controlled ladder experiment
would add intermediate temperatures between 1/0.5 and 0.5/0.25 (for example their
geometric midpoints), while preserving the cold target and accounting for extra
replica cost. No such ladder experiment has been run, and more round trips alone
would not establish convergence. Smaller integration steps address dataset 204's
instability; they cannot be assumed to resolve this dataset's mixing problem.

[Method and commands](../../../../validation/galaxy/mixing_audit/README.md), plan.json,
summary.json, supplement.json and PROVENANCE.json preserve all outcomes. The full
per-record manifest, native logs, paths and inspected figure stay externally at
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-mixing-audit/`.
The figure is mixing-audit.png. No posterior or calibration success is claimed.

## Supplement: local curvature and trajectory-period check

A separately declared post-hoc diagnostic evaluates 64x64 mass-scaled conditional
field Hessians at twelve saved states, using two finite-difference scales. This
dense calculation is limited to the 64-coordinate fixture; no production dense
mass is introduced. Maximum relative antisymmetry is 1.12e-9 and the maximum
between-step matrix difference is 8.07e-10, passing the declared consistency checks.

No inspected positive stable eigendirection lies within the declared 0.1-radian
threshold of a nonzero full leapfrog oscillation period. Thus this check finds no
evidence for that specific resonance explanation at these states. It cannot rule
out resonance elsewhere or treat a nonlinear trajectory as a fixed oscillator.

Positive curvatures include very small values around 0.004 in mass-scaled units;
several points have negative local curvatures, consistent with a non-Gaussian
landscape. Weak directions and poor temperature flow remain more direct concerns
than integration instability here. Forty-six eigenvalues remain at the prior-only
value one in every inspected matrix, leaving eighteen likelihood-active directions
for this fixture. Full-white jump metrics therefore include substantial uninformed
motion. Better support-aware moves and nonlinear curvature-aware preconditioning
are possible follow-ups, but were not implemented as remedies in this audit.

curvature-plan.json and curvature-summary.json retain all consistency checks and
eigenvalues; the raw matrices stay in the external artifact folder. No additional
posterior draws or revised convergence verdicts are produced.

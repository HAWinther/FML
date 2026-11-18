# Bounded numerical convergence and resource screen

The [maintained driver](../validation/galaxy/numerics/README.md) freezes one
synthetic field, data vector, physical cutoffs and three amplitude states while
varying compact, product/particle and deposition meshes and PSWF tolerance.
[Recorded results](validation/studies/numerical_screen/RESULTS.md) separate
numerical distances, local Fisher proxies, actual process peak RSS and runtime.
The plan and gates were saved before running; all cases and their failures are
retained, with per-case and aggregate time caps. Completed cases can be resumed
without repeating native work.

The default compact N8 / product N12 / deposition N24, epsilon=1e-8 passes the
frozen screen on this moderately nonlinear Gaussian-field mock. Its maximum
noise-weighted model distance is 0.00123 against a N24/N48, epsilon=1e-12
reference, versus the declared 0.1 screen budget. It takes about 0.037 s per
warm forward-plus-gradient call and peaks at 9.8 MiB in this single-process run.
This is a candidate for the next **small synthetic** pilot, not a real-catalog
production resolution recommendation. The local Fisher amplitude-shift proxy
is not a posterior-mean shift or a posterior-SD validation.

Tightening epsilon at fixed product/particle mesh barely changes the residual
floor. Joint product/particle refinement to N16 reduces the distance to about
8.3e-6. This does not establish independent particle-lattice and LPT-product
convergence, because those roles still share a parameter.

At fixed N16 product/N32 deposition, compact N8, compact N12 and shared N16 have
512, 1728 and 4096 sampler field coordinates. Their observed peaks are 17.7,
19.1 and 16.1 MiB, respectively. These single-run process peaks include setup,
assessment and a short HMC block. They establish **no total-memory saving** from
compact storage on this fixture; padded/sparse workspaces and transfer overhead
remain. The previous per-field-vector storage accounting is unchanged. Wider
resource and RSS sampling is needed before a production memory claim.

MPI predictions agree with serial to roundoff on 1/2/4 slab and pencil ranks.
Four ranks provide only about 1.6--1.7 times the one-rank forward/gradient speed
on this small fixture. MPI tests use two OpenMP threads per rank; compare them
to their own one-rank build. Timings have three repeats, RSS has one measurement
per process, and no large-box runtime or scaling extrapolation is made.

## Sampling readiness finding: field metric (FL-7, now resolved)

The short two-draw HMC smoke is a finite/integration check, not a mixing test.
The pre-fix screen led to the finding that `DistributedEulerianLPTModelT::apply_field_metric`
reuses the unbiased matter parameter vector. That vector sets b1=1, all other
legacy coefficients to zero and the legacy log-noise coordinate to zero. The
inherited metric therefore uses unit noise and omits the actual Eulerian linear
and Laplacian response, even when the posterior uses fixed noise power 1000.

This positive frozen metric does not change the Metropolis-corrected target.
It can give poor field exploration and makes a tiny-step acceptance rate an
inadequate readiness signal. Relative to the linearized surrogate
`1 + (b_delta-b_laplacian*k^2)^2 P(k)/P_noise`, the inherited precision is about
394--462 times larger on this fixture's active modes. This ratio is an analytic
surrogate comparison, not the nonlinear posterior Hessian or a measured mixing
slowdown. The full issue and acceptance criteria are recorded as FL-7 in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

The [FL-7 correction and separate field-only probe](EULERIAN_METRIC.md) now
implement and validate the posterior-aware metric and restart identity. The probe
shows improved exploration but does not pass every production mixing gate.
Next declare a larger joint field/scalar budget and posterior/refinement diagnostics. Do not treat the passing numerical screen as
completed posterior calibration, a measured amplitude-mean convergence gate or
readiness for the public galaxy catalog.

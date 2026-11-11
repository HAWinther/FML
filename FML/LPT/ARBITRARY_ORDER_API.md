# Arbitrary-order EdS LPT numerical and API contract

Contract for backlog item 55, 11 November 2026. Stages 3-4 implement the
collected, streaming, planning and evaluation entry points in `ArbitraryOrder.h`
for serial and supported distributed slabs. Double precision is validated
through order 5. Multi-rank validation and execution use collective recovery
points; empty slabs are not supported by the current `FFTWGrid` contract. The
stage-1 equations and executable reference are described in
[`ARBITRARY_ORDER_DERIVATION.md`](ARBITRARY_ORDER_DERIVATION.md).

## Mathematical problem and normalization

Compute the 3D periodic, single-stream, pressureless, growing-mode EdS solution
from a reference-epoch linear density field. Coordinates q and displacements
use a unit box. A caller multiplies by its comoving box length to obtain comoving
length units. The input is a normalized `FFTWGrid<3>` half spectrum with
f(q) = sum_m f_m exp(2 pi i m.q). Forward FFT normalization is FML's existing
normalization; spectral remapping introduces no volume factor.

For g = a/a_ref,

    x(q,a) = q + sum(n=1..P) g^n Psi_n(q)
    dPsi/dln(a) = sum(n=1..P) n g^n Psi_n(q)

There is no factorial in this field expansion. All EdS coefficients are included
in Psi_n, and each field is homogeneous of degree n in the input. The seed is
Psi_1(k) = i k delta_ref(k)/k^2. The full vector, including its transverse
contribution from order 3 onward, participates in subsequent orders.

The numerical target is the Fourier projection of this continuum recursion for
the accepted band-limited input onto the original mesh's strictly sub-Nyquist
cube. Every returned field uses that same output projection. Internal fields
retain their full order-dependent bandwidth until subsequent sources are done.

## Public interface

The implementation is in `FML/LPT/ArbitraryOrder.h`, under
`FML::COSMOLOGY::LPT`. Use the existing `FFTWGrid<3>`, configured scalar precision,
Fourier derivative conventions and communicator. The report outline below is
abbreviated; its complete member declarations are in the header.

```cpp
using EdSLPTVector = std::array<FML::GRID::FFTWGrid<3>, 3>;

struct EdSLPTOptions {
    int max_order = 3;
    std::array<int, 3> input_mode_limit{-1, -1, -1};
    std::size_t max_owned_bytes_per_rank = 0;
    std::size_t fft_workspace_allowance_bytes_per_rank = 0;
    std::size_t max_remap_message_bytes = 8 * 1024 * 1024;
};

enum class EdSLPTOutput { RetainOrders, StreamOrders };

struct EdSLPTReport {
    // See ArbitraryOrder.h for the full report definition.
};

struct EdSLPTEvaluation {
    EdSLPTVector displacement_fourier;
    EdSLPTVector ddisplacement_dln_a_fourier;
};

struct EdSLPTResult {
    std::vector<EdSLPTVector> orders_fourier;
    EdSLPTReport report;
    EdSLPTEvaluation evaluate(double growth_ratio,
                             std::size_t max_owned_bytes_per_rank) const;
};

using EdSLPTOrderSink = std::function<void(int order,
                                         const EdSLPTVector& field_fourier)>;

EdSLPTReport plan_eds_lpt(const FML::GRID::FFTWGrid<3>& delta_ref,
                         const EdSLPTOptions& options,
                         EdSLPTOutput output);

EdSLPTResult compute_eds_lpt(const FML::GRID::FFTWGrid<3>& delta_ref,
                            const EdSLPTOptions& options);

EdSLPTReport for_each_eds_lpt_order(const FML::GRID::FFTWGrid<3>& delta_ref,
                                   const EdSLPTOptions& options,
                                   const EdSLPTOrderSink& sink);
```

The maximum order must be positive. Each support component must be explicitly
set in [0, floor((N-1)/2)]. A nonzero owned-memory budget is required. The defaults
therefore require the caller to choose support and memory before execution.
`input_mode_limit = {0,0,0}` is a valid zero-field request after mean validation.
There is no order-specific branch or artificial order-5 algorithmic ceiling.
Orders above the tested range are recorded as beyond current validation in the
report. Requests must still fit all integer, layout and memory bounds.

The initial implementation offers the conservative full-bandwidth product policy
only. A reduced-band or aliased policy requires its own later contract and tests.

## Input checks and accepted seed

`plan_eds_lpt` and both computation functions are collective on
`FML::communicator`. They require an allocated Fourier-space input, FFTW support,
and identical mesh, order, support, budget, message limit and output mode on all
ranks. Validate the actual local slab geometry against FFTW's layout. The current
`FFTWGrid` constructor requires N divisible by the rank count, so the first
implementation enforces that condition. Input ghost slices are allowed, but
only physical Fourier coefficients are read.

Every physical stored coefficient must be finite. Define A as the global maximum
absolute input coefficient and tau = 64 epsilon A for the configured precision.
For A = 0 use tau = 0. Check that

- The zero mode has absolute value <= tau.
- Every mode outside |m_i| <= K_i has absolute value <= tau. This includes all
  even-mesh Nyquist planes, regardless of the caller's support request.
- On the stored m_z = 0 and even-mesh m_z = N/2 boundary planes, coefficients
  obey Hermitian symmetry within absolute tolerance tau, including real-valued
  self-conjugate coefficients. Interior half-spectrum modes specify their
  conjugates implicitly. Boundary checks use bounded distributed exchange.

Reject a violation above tolerance. On an internal copy, remove the zero mode
and off-support coefficients and average each retained boundary pair with its
complex conjugate. This defines the accepted seed unambiguously; preserve the
caller's input. Record the maximum corrections, A and tau in the report.
Fields exactly satisfying the constraints receive zero correction. A requested
support limit is a promise about the supplied seed; these checks prevent a
significant implicit change of the physical input.

## Work mesh and derivative rules

For requested order P and K = max(K_i), choose

    L = lcm(2, number_of_ranks)
    M = L * ceil(max(N, 2*P*K + 1) / L)

using checked integer arithmetic before multiplication or rounding. M is even,
M >= N and M > 2 P K. Query the actual FFTW local allocation extent, including
any transform scratch stored in its grid allocation. Reject sizes exceeding
the grid, index, allocation or transport representation. The serial case uses
number_of_ranks = 1. Supported nonempty slabs follow the current FFTWGrid
contract; empty-slab support would require a separate backend change.

At order n, all physical input-generated modes satisfy |m_i| <= n K_i. The work
mesh can represent every intermediate and every product through P below its
Nyquist planes. Fourier differentiation, Poisson and Helmholtz inversion use
the full work wavevector. The zero-mode displacement is set to zero. Check
source-mean compatibility and source-curl transversality with an
amplitude/precision-scaled residual, reporting the worst normalized residual.
Use the absolute-contribution envelope before cancellation to scale these
checks, so a nearly vanishing physical source does not imply a zero tolerance.
Include summation lengths and FFT roundoff in the error bound. Stage 3 must
document the concrete bound alongside the implemented contraction schedule and
test it against the sparse reference. A failed compatibility check terminates
the calculation.

Crop output only after constructing each complete order. Keep modes satisfying
2 abs(m_i) < N for every axis. Output Nyquist planes and the mean are exactly
zero. Returned grids use the original N and slab geometry, Fourier status and
zero ghost slices. A caller needing real fields or interpolation halos performs
those conversions explicitly with the existing grid utilities.

## Ownership, streaming and evaluation

`compute_eds_lpt` returns exactly P vectors in increasing order.
`orders_fourier[n-1]` is Psi_n at g = 1. The result owns its grids and its copied
metadata, and remains usable after the input is released. Collecting outputs
adds three original-mesh grids per retained order to the memory requirement.

`for_each_eds_lpt_order` invokes a nonempty sink once per order on every rank,
in increasing order, with the local part of the cropped Fourier vector. The
reference is valid only for that invocation. A caller retaining data must copy
it and account for those allocations. Sinks must perform local operations only;
they must not invoke collectives or re-enter the solver. After each invocation,
synchronize failure status before any subsequent FFT/remapping collective.
An exception in any sink fails the computation on all ranks. Earlier delivered
orders and external sink side effects remain observable, so retries begin a
new stream at order 1. Internal lower-order state remains necessary in this mode.

`evaluate(g, budget)` is collective and requires the same finite g > 0 on every
rank. It returns the two Fourier vector sums defined above, with zero ghosts.
The evaluator budget includes retained order fields plus the six new sum grids
and evaluator scratch, even though retained fields already exist. Validate the
budget and every g^n and n g^n weight before allocating. Finite-weight overflow
or a nonfinite accumulated coefficient fails collectively. Underflow follows
the configured floating-point arithmetic and is documented. The result exposes
dPsi/dln(a); physical velocity additionally needs a H and a length conversion.

## Memory and execution report

`plan_eds_lpt` validates input and computes a schedule without allocating large
work/output grids or constructing FFT plans. It may use bounded metadata and
Hermitian-check buffers. Include those buffers in the planner's budget check
before allocating them. The compute functions repeat preflight, so a report
does not authorize later execution with changed data or configuration.

Reports must include

- Input/output/work widths, support, requested order, validated order range,
  precision, rank/thread configuration and product policy.
- The actual local FFTW allocation extent for each grid layout, per-rank
  input payload, and per-rank output and retained internal payload.
- Peak solver-owned bytes from the concrete allocation schedule, including
  retained capacities, derivative/source scratch, remapping geometry and buffers,
  metadata and temporary allocation overlap. Report the local peak, maximum
  local peak, and conservative sum of per-rank peaks.
- The user-supplied opaque FFT/MPI allowance, listed separately, and estimated
  input-plus-owned-plus-allowance totals. Allocator bookkeeping and external
  library allocations are estimates rather than an RSS guarantee.
- Planned FFT and remapping counts. After execution, record measured counts,
  maximum owned capacity, input corrections and compatibility residuals.

The owned-memory budget applies to all incremental memory controlled by this
operation, including retained returned outputs. Borrowed input and caller-owned
sink copies are reported/excluded explicitly. It is independent of the optional
opaque FFT allowance. Use checked arithmetic for both local and aggregate counts.
Reject when any rank's owned schedule exceeds the common per-rank budget.

The planner follows the allocation schedule used by the engine. Preflight
must conservatively bound vector capacities and remapping
construction transients, rather than account for grid elements alone. If an
existing helper cannot provide that bound before allocating, add the smallest
necessary metadata/counting path or use a conservative checked bound. Compare
actual capacities with planned capacities at allocation checkpoints. Report
deviations as internal errors before the next collective.

## Collective failure and resource lifetime

All participants enter the same preflight phases. Local validation errors are
reduced to one deterministic common error before any dependent collective.
Serial failures throw standard exceptions; recoverable MPI validation, sink,
and explicitly wrapped allocation errors reach synchronized failure points on
all ranks. Signed extents are checked before conversions. Hermitian checks
exchange only bounded chunks of required boundary rows; Fourier remapping
reuses its distributed plan and bounded transport buffers.

Grid construction, remapping metadata and remapping transport allocation are
coordinated before dependent collectives. FFTW's own distributed plan creation
and failures outside those coordinated phases remain backend-level operations;
the implementation uses the library MPI abort path when FFTW cannot create a
plan. The test harness injects rank-local failures into the recoverable paths
and checks that every rank returns from the failed operation.

Failed collected calculations return no partial result. Temporary owned buffers
are released on exit; the caller's input retains its coefficients, status and
halo storage. Planning/execution/evaluation are separate operations and do not
depend on hidden global solver state. Concurrent calls on the same communicator
are outside the initial execution contract.

## Examples for implementation and tests

An N = 16 input with K = (1,2,2), P = 5 and one rank requires M = 22. Four ranks
round that work width to 24. Returned vectors still have N = 16. An N = 256
full-support input with K = (127,127,127), P = 5 and one rank needs M = 1272;
one double-precision serial work grid then occupies approximately 16.49 GB
(15.36 GiB) before the solver's multiple retained/scratch grids.

The double-precision sparse fixtures use a relative comparison
threshold 3e-12 times the maximum reference coefficient plus 1e-18. Independent
float and larger-order tests must set justified precision/amplitude-scaled
tolerances. Do not reuse the double threshold as a universal accuracy claim.

The stage-2 acceptance gate is this explicit written contract. Stage-3 tests
exercise collected and streamed agreement, output lifetime, evaluator weights,
budget boundaries, invalid support, significant zero/off-band/non-Hermitian modes,
roundoff correction reporting, overflow and input preservation, in addition to
the independent order-4/5 reference checks. Stage 4 covers cross-rank mismatches,
per-rank validation/allocation/sink failures, distributed output agreement,
workspace alternatives and execution under MPI/OpenMP. Measured rank/thread and
memory results are recorded in `ARBITRARY_ORDER_PLAN.md`.

## Implementation and measured checks

The engine offers two workspace policies. `CacheGradients` retains nine real
displacement-gradient grids for each order needed by a later source, up to
9(P-1) work grids. `RecomputeGradients` retains displacement vectors and rebuilds
up to three gradient tensors through an LRU dependency cache. Pair/triple contractions
evaluate the compact recursion directly. Each higher order uses four source
grids, three displacement grids, and potentially one planner-backup grid. The
conservative owned bound additionally includes one accepted original-mesh input
copy, three cropped outputs per retained order (or three in streaming mode),
gradient/output vector metadata and one remapping-plan construction bound.
Phase maxima are summed conservatively, so the bound can exceed the actual
simultaneous live allocation.

`accounted_owned_peak_bytes` tracks actual grid capacities at allocation
checkpoints together with the reserved metadata bound and a possible FFTW
planning backup. It is a conservative capacity account, not measured process RSS.
Remapping reserves its known row-count bound to avoid geometrical
vector-growth transients. Actual plan/vector capacities are checked against
the assumptions used by preflight. The grid helper's allocation query is checked
against the overflow-validated local storage calculation for the active slab
layout.

The cache schedule performs 13(P-1) transforms: nine inverse derivative
transforms per reusable lower order and four forward source transforms per
higher order. The recompute schedule reports the derivative transform count from
its three-entry dependency cache. Both perform 1+3P spectral remappings. Executed
counts must equal planned counts. A non-ESTIMATE FFTW planner can copy one grid
while planning, and that payload is covered separately from opaque plan storage.

For compatibility checks, the envelope E is the sum over contractions and source
components of their maximum absolute-contribution envelopes before cancellation.
The absolute tolerance is 256 epsilon P^2 (1+log2(M)) E. Compare source zero modes
directly and compare |k dot curl_source|/|k| at nonzero k. An identically zero
envelope requires exact zero residuals; nonfinite sources, envelopes, tolerances
or displacements fail. The report records the largest residual/tolerance ratio.
This roundoff allowance is validated on the sparse fixtures, rather than a
bound on physical perturbative truncation error.

Run `make -C FML/LPT/tests run` with the shared FFTW path settings. The stage-3
checks exercise all collected/streamed orders, evaluator sums and finite
differences, input/halo preservation and output lifetime, exact budget acceptance
and rejection one byte below the bound, invalid data/options, correction
reporting, and zero input. A generic order-6 zero-field request is accepted and
marked beyond the validated range. The independent sparse reference checks all
stored coefficients through order 5 on N=16 and N=9 meshes, plus an N=8,
K=3 high-frequency fixture with nonlinear intermediate modes outside the final
output band. The existing low-order and resampling regression suites also run.

For a small input already stored in Fourier space and supported within two
integer modes per axis, the entry points are used as follows.

```cpp
EdSLPTOptions options;
options.max_order = 5;
options.input_mode_limit = {2, 2, 2};
options.max_owned_bytes_per_rank = 512ULL * 1024 * 1024;
auto plan = plan_eds_lpt(delta_ref, options, EdSLPTOutput::RetainOrders);
auto fields = compute_eds_lpt(delta_ref, options);
auto at_half_reference_a = fields.evaluate(0.5, options.max_owned_bytes_per_rank);
```

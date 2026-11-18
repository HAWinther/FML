# Fixed-IC amplitude inference

`FixedIC.h` provides two reusable components for a field model that is linear in
its bias coefficients at fixed amplitude/ICs. The first application is the
[bounded Eulerian LPT driver](../validation/galaxy/fixed_ic/README.md).

`FixedNoiseLinearBiasLikelihood<N>` stores exact sufficient statistics for

```
U(b) = 1/2 sum_k weight_k [ |sum_j b_j O_j(k) - d(k)|^2 / C + ln C ].
```

Its constructor receives distributed half-complex operator columns, data,
Hermitian weights, fixed variance C=P_noise/V and a reference bias vector. It
reduces only 1+N+N*N numbers, then evaluates the likelihood and bias gradient in
O(N*N) replicated work without a new forward model, FFT or reduction. Statistics
are centered on the supplied reference coefficients to reduce cancellation.
The cache contains no scalar priors; callers add the declared priors separately.
It does not profile or analytically marginalize any bias coefficient. A
nonidentifiable/zero column is allowed; a positive-definite Gram matrix is not
required. N is bounded to 1..64; this is not a dense field metric.

`DistributedEulerianLPTModelT::bias_modes(work)` exports the four Fourier columns
from the latest successful forward evaluation. It preserves that evaluation's
reverse tape. At fixed amplitude, changing bias values therefore does not
require recomputing LPT evolution or operator fields.

`sample_fixed_ic_amplitude<N>` samples one positive amplitude and N bounded bias
coefficients. The callback `provider(alpha)` returns their linear likelihood
cache for the **same fixed IC realization and data**. Bias coefficients have
bounded uniform or bounded Gaussian priors. Their conditional negative log
densities are convex quadratics, so a connected-slice kernel can start with the
full finite support and shrink without a tuning width. This kernel must not be
reused for a nonlinear/multimodal amplitude conditional.

For the amplitude update, the sampler uses a Gaussian random walk reflected at
the amplitude bounds in coordinates

```
beta_j = alpha^(power_j) b_j.
```

The proposed physical bias is b_j'=(alpha/alpha')^(power_j) b_j. The reflected
amplitude proposal is symmetric in these coordinates. Its negative log target
is the physical likelihood plus all scalar-prior penalties **plus**
`sum(power_j) ln(alpha)`, because db/d beta=alpha^(-sum power_j). This Jacobian is
included in the Metropolis ratio. The default application uses powers 1,2,2,1
for density, squared density, tidal squared and Laplacian coefficients. These
powers improve the leading amplitude/bias degeneracy; they do not assume that
the nonlinear fields scale exactly that way. Every amplitude proposal is checked
with the actual forward-model likelihood, with no amplitude interpolation.
Out-of-prior proposals and numerical `UnrepresentableState` proposals are
rejected; malformed model contracts propagate as errors.

The field itself is never updated or reweighted by this sampler. A unit-Gaussian
penalty for a fixed white realization would be an additive constant; no
amplitude-dependent field determinant is added in whitened coordinates. Priors
are specified in physical alpha/b coordinates, not uniformly in beta or log
amplitude. All retained draws, including rejected-amplitude repeats, are kept.
Warmup is discarded; proposal scales and sweep counts stay fixed. Replicated
controls/RNG streams make the same collective forward calls on every MPI rank.
The provider must return the same replicated cache on all tasks. Use a stable
model and workspace captured by the callback.

The sampler stores only scalar draws and a small cache, but has no mid-chain
checkpoint or adaptation state. A provider with additional free spectrum/noise
parameters needs a different blocking contract. Free-IC HMC and large-catalog
reconstruction remain separate work; this bounded fixed-IC kernel does not
establish their mixing.

Validation compares cached values and all bias derivatives against the full
Eulerian target; independent direct residual sums test MPI reduction. A separate
2D quadrature comparison tests the joint amplitude/bias posterior and the
coordinate Jacobian. Seeded replay, numerical-proposal rejection and 1/2/4-rank
RNG/actual-model chain comparisons are included. Run:

```sh
make -C FML/FieldLevel/tests fixed-ic eulerian-lpt
make -C FML/FieldLevel/tests fixed-ic eulerian-lpt USE_MPI=true NP_LIST='1 2 4'
```

## Joint amplitude/bias proposal

`sample_fixed_ic_joint_amplitude<N>` updates amplitude and the entire bias vector
in physical coordinates. It uses the same reflected amplitude random walk, then
makes **one** Gaussian bias draw from the proposed amplitude's cache. A draw
outside any original prior bound is rejected, without clipping or repeated draws.
The finite prior domain and all prior penalties remain in the statistical target.

`cache.gaussian_proposal(priors)` builds this Gaussian from the small likelihood
Gram matrix. For a Gaussian prior it uses the prior's mean and standard deviation;
for a bounded uniform prior it uses the interval midpoint and width/sqrt(12) as
**proposal** Gaussian scales. A dimensionless diagonal ridge, at least one in
these scaled coordinates, makes the proposal proper for unidentified directions.
At extreme precision the ridge grows with roundoff scale. This ridge is never
added to the posterior likelihood or prior. No conditional normalization over the
finite box is evaluated or assumed.

The implemented negative-log acceptance difference is

```
Delta = U(alpha',b') - U(alpha,b)
        + log q(b' | alpha') - log q(b | alpha).
```

The reflected amplitude proposal is symmetric and cancels. Both q densities are
normalized **unbounded** Gaussians in physical b coordinates, including their
amplitude-dependent precision determinants. There is no alpha^power transport
Jacobian in this proposal. No analytic marginalization, profiling, Gaussian prior
replacement or truncation-normalization approximation is used. One-shot support
rejection accounts for the original finite prior domain in the joint kernel.

The proposal's dense algebra is limited to the small N-by-N bias matrix. Bias
coordinate slice refreshes are optional (`bias_sweeps=0` is valid here); each
accepted joint move already refreshes the bias block. The original rescaled-bias
kernel remains available with its original behavior. The pure joint kernel is
tested against independent joint-posterior quadrature, including a completely
unidentified coefficient that must retain its bounded uniform prior despite the
Gaussian proposal regularization. Correlated proposal density/covariance and
Gaussian-prior scaled geometry have separate analytic-reference checks.

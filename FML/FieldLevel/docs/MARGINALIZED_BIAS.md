# Analytic bias marginalization (Eulerian model)

`MarginalizedBias.h` integrates the four Eulerian bias coefficients out of the
fixed-noise Gaussian likelihood exactly. The tracer is linear in the biases,
d_pred = sum_a b_a C_a(s, theta), so with unbounded Gaussian priors
b_a ~ N(mu_a, sigma_a^2) (the model's own bias priors; bounded or uniform priors are
rejected) the marginal is Gaussian:

    -ln p(d | s, theta) = U(b_hat) + P(b_hat) + 1/2 ln det A + const,
    A = G + diag(1/sigma^2),   G_ab = sum_k w_k Re(conj C_a,k C_b,k) / v.

The gradient needs one forward and one adjoint: by the envelope theorem the first two
terms differentiate at fixed b = b_hat, and the log-determinant gives a second
cotangent on the columns. Both go through `DistributedEulerianLPTModelT::pullback_columns`,
which reverses the four Fourier columns with one cotangent each.

```cpp
#include <FML/FieldLevel/MarginalizedBias.h>
FF::BiasMarginalizedPosterior<Grid> posterior(model, data, noise_power);
double U = posterior.evaluate(white, theta, work, grad_white, grad_theta);
auto conditional = posterior.conditional_bias(white, theta, work); // mean, covariance, Cholesky
auto result = FF::sample_bias_marginalized(posterior, white, theta, hmc_options, sink);
```

`theta` holds only the spectrum parameters (here the sigma8 ratio). The sampler uses
the Eulerian field metric frozen at the start, with the bias at its conditional mean.
Exact bias draws for each sample follow from `conditional_bias`. Collective on the
model's communicator; tested on slabs and pencils with 1/2/4 ranks
(`tests/test_marginalized_bias.cpp`: finite-difference gradients, conditional mean
and covariance against the joint target, marginal differences).

On the joint-pilot target the marginal sampler mixes within a mode much faster, but
the posterior is multimodal and no sampler tested so far moves between modes; see
[the comparison](validation/studies/reparametrization/RESULTS.md).

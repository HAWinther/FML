# Proposed next kernel: amplitude/field interweaving

Not implemented or run in this campaign. The blocked comparison retains its failed
gates. Its chain-specific scalar plots motivate testing joint amplitude/field moves;
they do not establish which source of poor exploration dominates.

For the amplitude-only spectrum `P_L = alpha^2 P_ref`, the physical linear initial
field depends on `alpha * white`. A candidate transition proposes a reflected,
symmetric random walk from alpha to alpha_new and rescales the whitened field:

```
r = alpha / alpha_new
white_new = r * white
log_accept = -U(white_new, alpha_new, bias) + U(white, alpha, bias) + N * log(r)
```

Here U is the complete existing physical-coordinate posterior and N is the total
number of real white coordinates actually rescaled. This is the Jacobian of the
invertible amplitude/field proposal, not a new target term. Biases remain fixed.
Using the complete posterior avoids assuming that numerical forward outputs are
perfectly unchanged. Bounds and the amplitude prior remain part of the target.
A nonsymmetric amplitude proposal would additionally require its reverse/forward
proposal-density ratio.

Rescaling every stored real coordinate first makes N unambiguous (64 in this
fixture); prior-only directions also rescale and must enter the prior/Jacobian.
An active-only version could be more efficient, but must use an orthogonal
projection and its independent real dimension, including Hermitian constraints.
Counting stored complex coefficients or observed modes would be incorrect.

This proposal is intended to move along the amplitude/initial-field dependence
rather than alternate short conditional moves. In exact amplitude-only LPT the
rescaling should preserve the physical initial density and hence evolved positions;
that must be checked in the actual sampled/product mesh implementation. It need
not resolve separated bias/field configurations by itself.

Before a new campaign:

1. Verify the Jacobian sign, reciprocal transformation, prior-only directions and
   Gaussian/uniform amplitude priors against a small analytic marginal/quadrature
   target. A likelihood-neutral move still needs the prior and Jacobian factors.
2. Compare predictions before/after scaling for shared/compact 1LPT/2LPT grids;
   retain the full-posterior Metropolis ratio even when agreement is excellent.
3. Persist the new component seed/outer schedule in the checkpoint identity and
   test fresh-process exact replay. A parallel production implementation needs
   global reductions for N and prior differences, plus collective validation.
4. Freeze a separate protocol before sampling, including kernel schedule, step,
   budget and unchanged convergence gates. Keep all existing primary/replay draws
   separate. Compare amplitude mixing and total cost; no post-run extension.

No new Monte Carlo run, retuning or rescaling transition has been added to the
recorded blocked campaign.

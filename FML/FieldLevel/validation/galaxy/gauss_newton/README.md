# Frozen nonlinear field-metric feasibility probe

This selected-dataset diagnostic tests one state-independent conditional-field
preconditioner before any full-chain use. It reads the twelve frozen dataset-203
states (four ensembles × draws 0/1024/2047) from the earlier stability study.

For each state, construct the observation Jacobian using model pullbacks, scaled
by sqrt(Hermitian weight / mode variance). Independently check its action with
four field directions and two finite-difference steps, and reconstruct the actual
posterior field gradient from the rows and residuals. For each ensemble, average
J^T W J / variance equally over its three reference states. A small-reference SVD
retains positive excess-precision eigenpairs under the declared threshold/cap.

The applied mass is M_beta=I+beta U diag(lambda) U^T. Its powers use
I+U diag[(1+beta lambda)^exponent-1] U^T. Storage/action are O(rank × dimension);
prior-only directions retain unit mass. The operator does not depend on the
current field or scalars. No Riemannian metric terms, changed posterior measure,
field-prior determinant or adaptive metric are introduced. Gauss-Newton omits
residual-weighted second derivatives and is proposal geometry, not the true Hessian.

The exact construction is explicitly limited to this 64-coordinate serial fixture,
with rank cap 32; observed rank is eighteen. It is not a production distributed
Jacobian builder or sketch. Production construction/communication must be designed
separately if a frozen chain comparison justifies it.

plan.json fixes all thresholds before execution. Compare the original frozen
zero-field metric and this nonlinear metric using eight common standardized
momentum seeds at every frame: 192 actual native HMC trajectories, both step
0.075 ×16, integration time 1.2. Count actual field gradients and measure accepted
squared jumps in six informative Fourier projections. Full-white movement includes
46 uninformed directions and is not used as the efficiency criterion.

Independent controls cover dense inverses, square-root composition, 48 analytic
harmonic HMC trajectories, unchanged prior-only action and invalid metrics.
The local screen requires no candidate divergences, mean acceptance >=0.8,
minimum frame mean >=0.5, geometric-mean projection jump/call ratio >1 and every
projection ratio >=0.5. These are local feasibility gates, not chain ESS gates.
The same twelve states build and probe the candidate: this is in-sample exploratory
evidence, not independent validation of mixing or calibration.

```sh
make gn_build gn_probe gn_test
./gn_test > /absolute/unit.log
python3 run.py prepare /absolute/empty-study --source /absolute/stability/inputs/mock203.txt --builder /absolute/gn_build --probe /absolute/gn_probe --unit /absolute/gn_test --unit-log /absolute/unit.log
python3 run.py construct /absolute/study
python3 run.py probe /absolute/study
python3 run.py analyze /absolute/study
```

Original input/record/checkpoint identities, protocol, binaries, source and generated
metric files are frozen. Failed controls or trajectories are preserved. Large raw
Jacobians, builds and logs stay outside the library. See [results](../../../docs/validation/studies/gauss_newton_probe/RESULTS.md)
and the separate [full-chain comparison](../metric_comparison/README.md).

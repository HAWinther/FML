# Multigrid termination status

`MultiGridSolver::solve(equation, criterion)` returns a `SolveResult`. Existing ordinary calls can ignore the return value. The same result is available afterwards from `get_solve_result()`; `is_converged()` is now a pure query that reports successful convergence and never invokes the user callback.

```cpp
auto result = solver.solve(equation, criterion);
if (result.status == FML::SOLVERS::MULTIGRIDSOLVER::SolveStatus::Converged) {
    // Use solver.get_grid() or solver.get_y().
} else {
    // Handle exhaustion or numerical failure before using the current iterate.
}
```

`SolveResult` contains:

| Field | Meaning |
|---|---|
| `status` | `NotStarted`, `Converged`, `MaxIterations` or `NonFinite` |
| `residual` | Final finest-grid RMS residual, matching `get_residual()` |
| `initial_residual` | Finest-grid RMS residual after the initial fine-grid sweeps, before the first V-cycle |
| `iterations` | Completed V-cycles, matching `get_number_of_vcycles()`; initial fine-grid sweeps do not count |
| `converged()` | True only for `Converged` |

Before the first solve, the result is `NotStarted`, the cycle count is zero and residual diagnostics are NaN. Each solve resets the result and numerical-failure flag. The solution grid retains the current iterate on failure; there is no rollback. Exceptions from user callbacks still propagate under the existing callback contract.

## Termination rules

At driver checkpoints (after initial fine-grid sweeps and after each V-cycle), nonfinite residuals take precedence over convergence and the iteration budget. The solver remembers a nonfinite RMS residual encountered on any grid level, even when a later finest-grid residual is finite. Such a failure is reported at the next checkpoint as `NonFinite`; it does not wait for the maximum iteration count.

For finite residuals, convergence means the supplied criterion returned true on every MPI rank. This collective decision prevents rank-dependent criteria from splitting the V-cycle communication path. Criteria must be invoked collectively and return normally on every rank. A successful criterion takes precedence over the iteration budget, so convergence on the final allowed cycle remains success. The effective V-cycle budget is the collective minimum of the configured rank budgets, without changing their stored settings. Otherwise, reaching the budget returns `MaxIterations` and `is_converged()` stays false.

A zero V-cycle budget still performs the configured initial fine-grid sweeps and checks for initial convergence. An exactly zero initial residual satisfies the built-in relative-error criterion, as well as the absolute-residual criterion, with a positive tolerance. Other relative-error behavior and the multigrid updates remain unchanged.

The cosmological wrapper classes retain their existing solve APIs and ordinary calls to the generic solver. This result API is exposed by `MultiGridSolver` itself.

## Tests

`tests/test_multigrid.cpp` checks initial convergence, exact exhaustion counts, success on the iteration boundary, solver reuse, pure status queries, NaN/infinity on one rank, failure during a V-cycle, sticky coarse-grid failure, rank-dependent callbacks and differing rank budgets. The existing Poisson test also verifies the returned residual history and cycle count. Analytic derivative, Poisson, DGP-type and FFTW-enabled f(R)/Symmetron/DGP regressions exercise existing callers.

# Dataset-203 mixing audit

This post-hoc deterministic investigation reads the completed pilot. No posterior
chain is added, extended, reweighted or excluded. Ensemble 1 was selected because
it has zero qualified cold/prior-refresh/cold round trips; this selection is
exploratory and is retained explicitly in plan.json.

The primary audit uses all 2048 records from all four ensembles to reconstruct
pre-pass energy differences for every attempted exchange. Reversing accepted swaps
and replaying them must recover the recorded energy ordering; attempt/accept
counters must equal native summaries. Rates are reported per ensemble and edge.
The original 0.05 acceptance gate is pooled; a low individual rate is diagnostic,
not a new per-ensemble gate.

At draws 1024 and 2047, three fixed paths connect ensemble 1 to each other ensemble:
linear white coordinates, linear amplitude-weighted coordinates and a spherical
white-direction path. Each has 101 points. The existing convex bounded-bias
profiler evaluates the full target and the exact minimum over four linear bias
coefficients. Optional observed-mode input supplies this dataset's actual data;
the profiler's old no-data-file behavior remains unchanged. Rescaling controls,
three independent profiler references, KKT conditions, full-target agreement and
32 endpoint derivative comparisons are checked for each 1010-point audit.

```sh
python3 prepare.py /absolute/pilot/mock203 /absolute/audit --exe /absolute/geometry_probe
/absolute/geometry_probe /absolute/audit/points1024.csv /absolute/audit/results1024.csv /absolute/audit/data.txt
/absolute/geometry_probe /absolute/audit/points2047.csv /absolute/audit/results2047.csv /absolute/audit/data.txt
python3 analyze.py /absolute/audit
python3 supplement.py /absolute/audit
```

Preparation hashes endpoints, all transport records, source summaries, implementation
and geometry inputs before evaluating paths. Source files are never modified.
Full per-record hashes stay in the external artifact folder, not the library.

The post-hoc supplement reconstructs qualified walker legs and native phase/trip
counters. Labels measure refreshed-state transport, not persistence of the same
physical configuration. It also evaluates U_beta=white_prior+beta*Q_profile on the
same paths. Positive beta scaling does not change the bounded-uniform bias optimum.
Only the likelihood is tempered; multiplying the entire posterior potential by
beta would be incorrect.

These are selected point-potential paths. Their barriers are not free energies,
proofs of globally disconnected modes or marginalized posterior masses. Profiling
is not integration over bias parameters. A low barrier on one route likewise does
not establish easy typical-set mixing. No refined temperature ladder has been run.
See [results](../../../docs/validation/studies/mixing_audit/RESULTS.md).

Curvature.cpp and curvature.py provide a separate, dense **64-coordinate-only**
local diagnostic using the frozen stability-probe input for dataset 203. They
check Hessian symmetry/step sensitivity and local harmonic angles. The matrix is
M^-1/2 H M^-1/2 with the original frozen mass; negative curvature and nonlinearity
limit harmonic interpretations. No production dense preconditioner or new chains
are introduced.

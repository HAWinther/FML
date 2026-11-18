# Bounded fixed-IC amplitude pilot

This driver infers alpha=sigma8/sigma8_fid and marginalizes four Eulerian bias
coefficients on a synthetic periodic Gaussian tracer **field**. It does not yet
read a galaxy catalog, infer absolute sigma8, calibrate coverage, or reproduce
the challenge. The data generator uses the same model family on a finer mesh.

The frozen [protocol](plan.json) specifies the known physical IC realization,
spectrum, cutoffs, Poisson-like fixed power, bias priors, seeds, three numerical
configurations and run budgets. The initial realization is represented with
1/sqrt(N^3) normalized white Fourier coefficients so its physical linear field
stays fixed when the mesh changes. There are 56 active real initial degrees of
freedom and 124 real likelihood-mode degrees of freedom in this small fixture.
The final configuration refines several numerical settings jointly; this is a
bounded comparison rather than a complete independent resolution scan.

Build outside the library source directory, or use `make` here and keep all run
outputs elsewhere. Set the usual FML compiler/FFTW variables as needed:

```sh
make -C FML/FieldLevel/validation/galaxy/fixed_ic
python3 FML/FieldLevel/validation/galaxy/fixed_ic/run.py \
  --exe /absolute/path/to/galaxy_fixed_ic --output /external/run/directory --workers 4
python3 FML/FieldLevel/validation/galaxy/fixed_ic/analyze.py --input /external/run/directory
```

The campaign runner uses a **serial** executable and launches independent chains.
The native driver also builds with MPI (slabs) or `USE_MPI=true
USE_PENCIL_GRID=true` (pencils); run individual chains collectively using the
native commands for backend comparisons. The four-rank small fixture can have
ranks with no retained modes, which are valid. Native commands:

```text
galaxy_fixed_ic prepare PREFIX
galaxy_fixed_ic chain DATA.csv PREFIX N ND WIDTH SEED ALPHA_START WARMUP DRAWS
galaxy_fixed_ic predict DATA.csv PREFIX N ND WIDTH ALPHA B1 B2 BK BLAP
galaxy_fixed_ic diagnose OUTPUT.csv CHAIN1.csv CHAIN2.csv [...]
```

The input schema and reference spectrum are intentionally frozen in `Main.cpp`;
this is a scientific driver, not the general catalog/data adapter. Prepare emits
native problem metadata, which the runner checks against the protocol. Each
completed chain records the proposal settings and output counts. The runner pins
executable/protocol/data hashes and checks completed-chain hashes on reuse. It
can reuse **completed chains** after interruption; it cannot resume an unfinished
chain, and the bounded sampler has no trajectory checkpoint yet. Rerun with the
same executable/output to finish missing chains. No old completed chain is
silently replaced under changed inputs.

Outputs include all retained scalar draws, native rank-normalized Rhat/bulk/tail
ESS, acceptance/runtime metadata, model modes at the pooled posterior-mean
parameters and five PNG plots. Mean-parameter predictions are not posterior
predictive draws. All failed cases remain in the summary. The analysis reports
mesh shifts relative to the finest-mesh posterior SD and an approximate paired
length-64 batch Monte Carlo error, using the common chain seeds. Neither that
estimate nor this one-mock result is a rigorous numerical error/coverage bound.
There is no automatic chain extension, thinning or failed-case removal.

The reusable sampler/cache and its Jacobian are documented in
[the fixed-IC guide](../../../docs/FIXED_IC.md). The full staged analysis remains
in [the project plan](../../../docs/LEFTFIELD_PLAN.md).

## Separate joint-proposal campaign

`plan_joint.json` declares a second campaign on the **exact original** small data
fixture in `data/`, with new chain seeds and the original target, draw budget and
mixing gates. It replaces the rescaled amplitude/conditional-bias sweeps with a
joint Gaussian bias proposal (zero extra conditional sweeps). The original
`plan.json` and failed-mixing campaign remain unchanged.

```sh
python3 FML/FieldLevel/validation/galaxy/fixed_ic/run.py \
  --exe /absolute/path/to/galaxy_fixed_ic --plan FML/FieldLevel/validation/galaxy/fixed_ic/plan_joint.json \
  --output /external/joint-run --workers 4
python3 FML/FieldLevel/validation/galaxy/fixed_ic/analyze.py --input /external/joint-run
python3 FML/FieldLevel/validation/galaxy/fixed_ic/compare.py \
  --reference /external/original-run --trial /external/joint-run
```

The native `chain-joint` command has the same arguments as `chain`. Its metadata
records the sampler kind, sweep count and number of support rejections. The
comparison checks that targets, data, budgets and mixing gates agree, keeps the
two campaigns separate, and compares amplitude ESS per forward evaluation. Chain
wall times are elapsed wall seconds, not measured CPU time.

# TSC posterior validation

Active-field/amplitude/noise validation for a bounded periodic N16 PM problem.
This is a scientific validation driver, not a general galaxy-analysis frontend.

```sh
make
python3 run_mock.py --bin-dir "$PWD" --output /path/to/fresh/independent --ic-seed 191 --noise-seed 192 --jobs 4
python3 run_mocks.py --bin-dir "$PWD" --output /path/to/fresh/repeated
```

`run_mocks.py` predeclares four datasets and a fixed four-chain 512-draw protocol.
These opt-in campaigns take hours; the cleanup checks do not rerun them.
`analyze_mocks.py` also understands the explicit scope amendment in the archived
three-mock study. A skipped queued dataset is separate from a missing/failed run.

`run.py` retains the original reference-data protocol; its two small frozen input
files live in [data/](data/README.md), with original hashes in plan.json. Its
`--scheme preconditioned` option constructs a fresh fixed metric before sampling.
All analysis tools accept explicit input/output directories. Keep generated chains
and predictions outside the source tree. The pilot does not checkpoint midway
through a chain; interrupted unfinished chains must be restarted.

Conditional and metric kernel checks can be built with
`make pm_tsc_conditionals pm_tsc_metric_test`. The conditional quadrature reader
is `check_conditionals.py`; supply its binary and an external output directory.

[Recorded studies](../../../docs/VALIDATION.md) preserve failed and passing cases;
[full historical records](../../../docs/ARCHIVE.md) are external to the library.

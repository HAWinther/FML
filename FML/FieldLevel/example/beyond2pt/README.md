# beyond2pt Box-1 application

This is the observed-data entry point for the real periodic galaxy catalog.
It downloads or verifies the pinned public Box-1 HDF5 file, streams coordinates
through the existing `catalog_to_modes` adapter and saves Fourier density modes.
It then produces a galaxy-map/power-spectrum figure and an input-readiness report.
No Markov chains are launched here.

The catalog is pinned by repository commit, raw-file SHA256, Git blob SHA1 and
byte length in [plan.json](plan.json). The native executable must be a **serial**
`catalog_to_modes` build. NumPy, h5py and matplotlib are Python dependencies; the
C++ catalog adapter has no HDF5 dependency. For example:

```sh
python3 FML/FieldLevel/example/beyond2pt/prepare.py \
  --exe /absolute/build/catalog_to_modes \
  --output /absolute/external/box1
```

Add `--input /absolute/mock_lcdm_real-space_01.h5` to reuse a verified download.
A completed extraction can be reused only with matching inputs and output hashes.
Incomplete native outputs are retained for inspection; retry in a fresh directory.
Source edits, catalog changes or a changed executable require a fresh output root.
Build outside the library and keep all catalog/mode/figure products there.
The current application invokes the adapter directly, using one process. This
application workflow does not add MPI orchestration; the underlying catalog
adapter already supports slab and pencil layouts.

Two observation meshes (96 and 128) extract the same cubic band through
0.12 h/Mpc. Reports retain the nested 0.08/0.10/0.12 h/Mpc masks. These are cuts
on **observed data**, not completed EFT-cutoff fits or choices for the matter/IC
filters. Hermitian multiplicities count the full real degrees of freedom;
zero and Nyquist modes are excluded. Eight direct Fourier sums use every galaxy
and independently check the transform sign, coordinate origin and normalization.
The mesh difference is measured in Poisson mode-noise units. It is an ingestion
check, not a parameter-bias or forward-model accuracy bound.

`observed-power.csv` and `catalog-overview.png` are descriptive. The plot's dashed
curve subtracts Poisson power from a summary statistic only. The saved density
modes used for a future likelihood remain unchanged. The plotted spherical shells
are complete; cubic corner modes are retained separately in the likelihood data.

## Input dependency for the fixed-IC fit

The [challenge paper](https://arxiv.org/html/2405.02252v2), sections IV.4 and V.3,
reports that the organizers separately supplied the true Box-1 initial conditions
to the LEFTfield team. The inspected public repository tree contains catalogs,
but no matching initial field or reference linear-spectrum file. We have not
established another public release of those inputs.

The actual fit requires the matching phases, their Fourier layout and
normalization, the non-amplitude background cosmology, and the reference spectrum
with its species/epoch/growth/sigma8 conventions. The intermediate filters and
higher-derivative bias convention must also be matched before calling the result
a reproduction. These unresolved dependencies are emitted in
`INPUT_REQUIREMENTS.json`; `summary.json` explicitly records
`fixed_ic_fit_ready=false` and `posterior_computed=false`.

Random phases and phases fitted to the observed catalog cannot stand in for the
known simulation initial conditions. An independent known-IC simulation would
be a different scientific application, and must be labeled accordingly.

The next fitting deliverable remains the conditional amplitude likelihood and
bias-marginalized amplitude posterior, followed by model-mesh and EFT-cutoff
comparisons on the **same** catalog and IC realization. The existing synthetic
fixed-IC driver and its frozen toy spectrum are not used for this catalog.

[Project status](../../docs/BEYOND2PT_PLAN.md) ·
[Box-1 preparation results](../../docs/beyond2pt/BOX1_RESULTS.md)

# Periodic catalog observation validation

Recorded 18 November 2026. Overall adapter checks pass; [machine-readable results](summary.json).

A controlled 59-row weighted periodic catalog is streamed to an N16 Fourier grid.
All usable modes are compared to independent direct particle sums, excluding DC
and the exactly-zero Nyquist planes. Tests include periodic images, boundary
positions, nonzero origins, increasing weight scales, and zero weights.

| Requested NUFFT tolerance | Largest absolute mode error |
| --- | --- |
| 1e-4 | 5.76865e-5 |
| 1e-6 | 5.50124e-7 |
| 1e-8 | 5.54676e-9 |
| 1e-10 | 8.36265e-11 |

The selected fine mesh is N32. These are measured fixture errors, not a universal
bound on every catalog or mode.

Serial native tests pass 18 checks. MPI tests pass 36, 57 and 76 checks at
1, 2 and 4 ranks respectively, covering slabs and all pencil process-grid
orientations at those rank counts. Builds use double precision, bounds checks,
Clang 17, FFTW and Open MPI; MPI runs use two OpenMP threads per rank.
Checks cover the weighted Poisson normalization, invariance under common weight
scales 1e150 and 1e-150, one-record MPI chunks, arbitrary/root-only input,
reset/failure states, layout validation and collective input/capacity errors.
Exported modes and noise also enter an actual Eulerian inference target.

All 17 file-interface checks pass: CSV conversion in serial and slab/pencil
1/2/4-rank builds, one-row versus 13-row streaming batches, tolerance controls,
cubic corner modes, malformed rows/weights, two HDF5 schemas and a complete
HDF5-to-four-rank-pencil pipeline. HDF5 and CSV differ by at most 3.37662e-16.
The optional h5py cases ran; none was skipped. MPI stdin streams use an explicit
end marker and the HDF5 wrapper verifies the final native row count; relying
only on pipe EOF stalled in this launcher environment.

The adapter retains plans/kernel/ownership and accumulates batches before one
final FFT. Its storage comprises a fine grid, local real accumulator, coarse
output grid and capped routing buffers. This is a design accounting, not a
production memory or throughput benchmark.

This validates fixed observed-catalog ingestion. It does not replace LPT matter
deposition with NUFFT, supply its position adjoint, validate production-rank
scaling or demonstrate a new amplitude posterior. No public challenge catalog
was downloaded. Catalog-specific modeling/convergence and free-IC inference
remain open. Raw fixtures, logs and accuracy plot are kept outside the library
under the workspace's `outputs/fieldlevel-catalog/`.

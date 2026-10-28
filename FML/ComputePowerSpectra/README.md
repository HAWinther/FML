# Fourier shell polyspectra

`PolyspectrumBinning::dealias` defaults to `true`. The estimator counts genuinely
closed integer-wavevector configurations rather than closure modulo the FFT mesh.
When the configured shell range could wrap an ORDER-fold product into its zero
mode, the shell fields and geometry masks are evaluated on a padded mesh. Their
normalization uses the work mesh volume. Low-frequency ranges that cannot wrap
keep the original FFT width.

For nonlinear orders (ORDER > 2), even-mesh Nyquist planes are excluded from both
field and geometry selections: their signed Fourier wavevector is ambiguous.
Strictly sub-Nyquist modes are retained, including all represented modes of odd
meshes. Ordinary power spectra keep their full even/odd r2c multiplicities.

Set `bins.dealias = false` explicitly to reproduce the previous modulo-grid
estimator, including its Nyquist convention. This affects physical results at
high wavenumbers. The fourth-order function remains a raw shell-averaged fourth
moment; disconnected Gaussian contributions are not subtracted.

Padding is expensive: at full radial Nyquist range, each 3D shell grid grows by
approximately 3.375 times for a bispectrum and 8 times for a fourth moment. Grids
are rounded to an even width divisible by the MPI task count. Shell work grids
have no particle-deposition halos. A bounded shell-cache algorithm is not part
of this correctness fix.

Geometry counts are cached automatically after successful computation. Internal
cache identities include the source mesh width, exact shell edges and the
closure policy. Changing any of these, or calling `reset`, invalidates reuse.
For external counts use `set_bincount(counts, source_mesh)` to record the mesh
identity. The legacy `set_bincount(counts)` overload trusts the caller and binds
the counts to the mesh of their first use; it cannot verify which mesh originally
produced an unlabelled vector. Imported counts must use the same bin edges and
closure convention as the receiving object.

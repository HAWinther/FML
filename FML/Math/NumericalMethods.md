# Shared numerical helpers

Include `FML/Math/Grids.h` for grids alone or `FML/Math/NumericalMethods.h` for the numerical helpers.
`Math.h` also exposes them. Link `NumericalMethods.o` for quadrature, Brent root finding and the
spherical Bessel routines. Define `USE_GSL` and link GSL for `integrate`, `find_root` and scalar
`spherical_bessel_j`. Grids, `trapz` and the array recurrence work without GSL or cosmology.

## Names and existing APIs

| Function | Convention |
| --- | --- |
| `FML::MATH::linspace(a,b,n)` | Existing linear grid; n <= 0 gives empty, n = 1 gives {a}. |
| `FML::MATH::logspace(a,b,n,base)` | Existing grid between base^a and base^b. |
| `FML::MATH::geomspace(a,b,n)` | Positive physical endpoints a and b; n >= 2. |
| `FML::MATH::spherical_bessel_j(ell,x)` | Former cosmology scalar algorithm, retaining its high-order cutoff. |
| `FML::MATH::spherical_bessel_j_array(lmax,x,out)` | Former cosmology upward/Miller recurrence; x >= 0, lmax >= 0 and a buffer of lmax+1 entries. |

The older `j_ell` and vector-returning `j_ell_array` remain unchanged. The scalar routines have different
small-argument accuracy policies, so they are deliberately not substituted for one another. The new
array API accepts caller-owned storage; it does not silently replace the older recurrence.

`integrate` uses adaptive GSL quadrature; `find_root` uses Brent's method rather than the existing
bisection/Newton methods. Their convergence warnings and return-value policy are retained from cosmology.
Linking the GSL helpers disables GSL's process-wide abort handler at startup, as the original implementation did.

## Other relocated helpers

- `FML::FILEUTILS::fml_file` lives in `FileUtils/DataPath.h` (also exposed by `FileUtils.h`). It resolves
  `FML_DIR`, then the compiled `FML_ROOT_DIR`, then a relative path. Define `FML_ROOT_DIR` consistently
  in every translation unit that includes it.
- `FML::UTILS::Timer` lives in `Timing/Timer.h`. It is a simple stopwatch; the existing `Timings` class
  continues to accumulate timings by label. `Timer::print(false)` suppresses printing.
- `FML::INTERPOLATION::SPLINE::spline2d_type` lives in `Spline/Spline.h`.

Cosmology's `Common.h` retains its aliases, strict linear-grid check, physical-endpoint `Utils::logspace`,
and local `Utils::verbose` output policy. Include `Cosmology/Common.h` or the individual shared module
headers; the former numerics header and folder have been removed. Programs using `Cosmology.mk`
automatically link the shared numerical object.

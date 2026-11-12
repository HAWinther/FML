# Follow-up work

- Calibrate shell and collapsed estimators on independent Gaussian and nonlinear cosmological box ensembles across volume, resolution, redshift and bin width; compare with an established reference implementation.
- Measure finite-mode plug-in bias, reference-covariance uncertainty and signal-to-noise in those ensembles. Do not treat spatial cells as independent rows for an IID bias correction.
- Add validated connected particle-noise subtraction before interpreting tracer statistics as continuous-field trispectra.
- Profile memory, FFT count and MPI scaling at production meshes; add bounded geometry/shell caches if worthwhile.
- Add explicit weighted/overlapping-shell Gaussian contractions, cross-field/tracer power covariance and multipoles as separately tested extensions.
- General diagonal/angle-resolved trispectra, survey-window/edge corrections and super-sample terms remain outside this periodic first scope.

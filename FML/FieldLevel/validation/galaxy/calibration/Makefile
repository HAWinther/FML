USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := calibration_diagnostics
include ../../../../Makefile.config
calibration_diagnostics: Diagnostics.o
	$(LINK_EXE)

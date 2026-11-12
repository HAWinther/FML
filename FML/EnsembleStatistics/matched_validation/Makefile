USE_MPI ?= true
USE_OMP ?= false
USE_FFTW := true
# NBody.h includes ODESolver's GSL headers; this EdS driver uses analytic time integrals.
USE_GSL := true
TARGETS := matched_pm
include ../../Makefile.config
FML_MODULES := Global
matched_pm: MatchedPM.o Global.o
	$(LINK_EXE)

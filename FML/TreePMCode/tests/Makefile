# Tests of TreePMCode: testtreepm (blob force, Zel'dovich plane wave, two fluids, GADGET-2 time steps)
USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := testtreepm
include ../../Makefile.config
FML_MODULES := Global Spline ODESolver Math
testtreepm: TestTreePM.o Global.o Spline.o ODESolver.o NumericalMethods.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif

.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do \
	  for t in blob zeldovich twofluid gadget; do echo "== Running testtreepm $$t with $$np task(s)"; OMP_NUM_THREADS=1 $(RUN) ./testtreepm $$t; done; \
	done

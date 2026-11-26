USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_short_range test_yukawa
include ../../Makefile.config
FML_MODULES := Global Spline ODESolver
EXTRA_CLEAN += benchmark_clustered_force

test_short_range: test_short_range.o Global.o
	$(LINK_EXE)
test_yukawa: test_yukawa.o Global.o
	$(LINK_EXE)
# Longer clustered-force experiment (CLUSTERED_FORCE.md); not part of 'make run'. Needs FFTW (and MPI for --mpi)
benchmark_clustered_force: benchmark_clustered_force.o Global.o Spline.o ODESolver.o
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
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_short_range; \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_yukawa; \
	done

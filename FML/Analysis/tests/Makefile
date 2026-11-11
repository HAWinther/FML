USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := test_analysis
include ../../Makefile.config
FML_MODULES := Global

test_analysis: test_analysis.o test_analysis_odr.o Global.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
.PHONY: run
run: $(TARGETS)
ifeq ($(USE_MPI),true)
	@set -e; for np in $(NP_LIST); do OMP_NUM_THREADS=2 $(MPIRUN) -np $$np ./test_analysis; done
else
	./test_analysis
endif

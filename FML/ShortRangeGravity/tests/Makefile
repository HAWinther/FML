USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_short_range
include ../../Makefile.config
FML_MODULES := Global

test_short_range: test_short_range.o Global.o
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
run: test_short_range
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_short_range; \
	done

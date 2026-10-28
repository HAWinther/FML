USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_spatial
include ../../Makefile.config
FML_MODULES := Global
test_spatial: test_spatial.o Global.o
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
run: test_spatial
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_spatial; \
	done

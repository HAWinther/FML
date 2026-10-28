USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := test_spherical
include ../../Makefile.config
FML_MODULES := Global
test_spherical: test_spherical.o Global.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4 8
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif
.PHONY: run
run: test_spherical
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_spherical; \
	done

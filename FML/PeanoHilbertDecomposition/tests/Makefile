USE_MPI ?= true
USE_OMP ?= false
TARGETS := test_peano_hilbert
include ../../Makefile.config

test_peano_hilbert: test_peano_hilbert.o
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
run: test_peano_hilbert
	@set -e; for np in $(NP_LIST); do \
	  $(RUN) ./test_peano_hilbert; \
	done

USE_MPI ?= false
USE_OMP ?= false
TARGETS := test_baryonification
include ../../Makefile.config
FML_MODULES := Global

test_baryonification: test_baryonification.o test_baryonification_odr.o Global.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
.PHONY: run
run: $(TARGETS)
ifeq ($(USE_MPI),true)
	@set -e; for np in $(NP_LIST); do $(MPIRUN) -np $$np ./test_baryonification; done
else
	./test_baryonification
endif

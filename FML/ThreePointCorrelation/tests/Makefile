USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := test_threepoint
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_threepoint: test_threepoint.o Global.o
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
run: test_threepoint
	@set -e; for np in $(NP_LIST); do $(RUN) ./test_threepoint; done

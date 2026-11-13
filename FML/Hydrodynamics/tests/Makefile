USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_hydro
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_hydro: test_hydro.o Global.o
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
run: test_hydro
	@set -e; for np in $(NP_LIST); do OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} OMP_WAIT_POLICY=passive $(RUN) ./test_hydro; done

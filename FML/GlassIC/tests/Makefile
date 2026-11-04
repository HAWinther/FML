USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := test_glass_ic
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_glass_ic: test_glass_ic.o Global.o
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
run: test_glass_ic
	@set -e; for np in $(NP_LIST); do OMP_NUM_THREADS=2 $(RUN) ./test_glass_ic; done

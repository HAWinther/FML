USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := test_weak_lensing
include ../../Makefile.config
FML_MODULES := Global
# Explicit local object rule prevents VPATH reuse of differently configured objects.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_weak_lensing: test_weak_lensing.o Global.o
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
run: test_weak_lensing
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_weak_lensing; \
	done

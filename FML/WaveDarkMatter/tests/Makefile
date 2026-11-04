USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := test_wave_dm test_soliton
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_wave_dm: test_wave_dm.o Global.o
	$(LINK_EXE)
test_soliton: test_soliton.o Global.o
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
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do OMP_NUM_THREADS=2 $(RUN) ./test_wave_dm; OMP_NUM_THREADS=2 $(RUN) ./test_soliton; done

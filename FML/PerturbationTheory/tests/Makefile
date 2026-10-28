USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := test_direct
include ../../Makefile.config
ifeq ($(USE_FFTW),true)
TARGETS += test_perturbation
all: test_perturbation
endif
test_direct: test_direct.o Global.o
	$(LINK_EXE)
FML_MODULES := Global FFTLog
test_perturbation: test_perturbation.o Global.o FFTLog.o
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
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_direct; \
	  $(if $(filter true,$(USE_FFTW)),OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_perturbation,:); \
	done

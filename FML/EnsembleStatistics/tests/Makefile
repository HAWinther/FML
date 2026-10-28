USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_control
include ../../Makefile.config

ifeq ($(USE_FFTW),true)
TARGETS += test_spectra
all: test_spectra
endif
FML_MODULES := Global
test_control: test_control.o Global.o
	$(LINK_EXE)
test_spectra: test_spectra.o Global.o
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
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_control; \
	  $(if $(filter true,$(USE_FFTW)),OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_spectra,:); \
	done

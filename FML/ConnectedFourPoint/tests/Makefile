USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := test_four_point
ifeq ($(USE_FFTW),true)
TARGETS += test_four_point_fourier
endif
include ../../Makefile.config
FML_MODULES := Global
test_four_point: test_four_point.o Global.o
	$(LINK_EXE)
test_four_point_fourier: test_four_point_fourier.o Global.o
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
	@set -e; for np in $(NP_LIST); do $(foreach test,$(TARGETS),$(RUN) ./$(test);) done

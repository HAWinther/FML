USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_short_range test_yukawa
include ../../Makefile.config
FML_MODULES := Global

test_short_range: test_short_range.o Global.o
	$(LINK_EXE)
test_yukawa: test_yukawa.o Global.o
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
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_short_range; \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_yukawa; \
	done

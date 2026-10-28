USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := test_workflow
include ../../Makefile.config
FML_MODULES := Global
test_workflow: test_workflow.o Global.o
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
run: test_workflow
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_workflow; \
	done

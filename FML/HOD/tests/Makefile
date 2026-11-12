USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_hod test_assembly_bias
include ../../Makefile.config
FML_MODULES := Global
test_hod: test_hod.o Global.o
	$(LINK_EXE)
test_assembly_bias: test_assembly_bias.o Global.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
REFERENCE_LAUNCHER = --launcher "$(MPIRUN)"
else
NP_LIST = 1
RUN =
endif
.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_hod; \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_assembly_bias; \
	done
	$(PYTHON) regression/assembly_reference.py --exe ./test_assembly_bias $(REFERENCE_LAUNCHER)

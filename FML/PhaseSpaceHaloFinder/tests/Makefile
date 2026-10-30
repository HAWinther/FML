USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_phase_space_halos
include ../../Makefile.config
FML_MODULES := Global
test_phase_space_halos: test_phase_space_halos.o Global.o
	$(LINK_EXE)
rockstar_fixture: rockstar_fixture.o Global.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif
.PHONY: run run-python
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_phase_space_halos; \
	done

# Optional harness checks; requires NumPy and SciPy, but no ROCKSTAR binary.
run-python:
	$(PYTHON) test_compare_rockstar_cosmo.py

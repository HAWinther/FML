USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := test_visualization
include ../../Makefile.config
FML_MODULES := Global PencilDomain

# PencilGrid (the pencil part of the test) needs MPI
ifeq ($(USE_MPI),true)
test_visualization: test_visualization.o PencilGrid.o Global.o
	$(LINK_EXE)
else
test_visualization: test_visualization.o Global.o
	$(LINK_EXE)
endif

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
PYTHON ?= python3
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif
# The PNG and npy files are read back with Python's standard library when python3 is there
.PHONY: run
run: test_visualization
	@set -e; for np in $(NP_LIST); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) ./test_visualization; \
	  if command -v $(PYTHON) >/dev/null 2>&1; then $(PYTHON) check_png.py; else echo "No $(PYTHON): PNG check skipped"; fi; \
	done
	@rm -f test_pattern.* test_grey.* test_orientation.*

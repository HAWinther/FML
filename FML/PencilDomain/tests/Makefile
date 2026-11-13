USE_MPI ?= true
USE_FFTW ?= true
USE_OMP ?= false
TARGETS := test_pencil_grid
EXTRA_CLEAN += benchmark_pencil_grid

include ../../Makefile.config

FML_MODULES := Global

test_pencil_grid: test_pencil_grid.o PencilGrid.o Global.o
	$(LINK_EXE)

# FFT timings against FFTWGrid (not run by `make run`): mpirun -np P ./benchmark_pencil_grid [Nmesh ...]
benchmark_pencil_grid: benchmark_pencil_grid.o PencilGrid.o Global.o
	$(LINK_EXE)

PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)

Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4

.PHONY: run
run: test_pencil_grid
	@set -e; for np in $(NP_LIST); do \
	  echo "PencilGrid MPI validation: $$np rank(s)"; \
	  $(MPIRUN) -np $$np ./test_pencil_grid; \
	done

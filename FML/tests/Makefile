#===================================================================
# FML: tests
#
# make run   (or 'make test' in the top folder) builds and runs the
# regression tests. With MPI they are run for each task count in NP_LIST
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_GSL             ?= true

TARGETS := test_core test_grid_tools test_fof test_so test_fnl test_kdtree

include ../Makefile.config

FML_MODULES := Global Spline ODESolver

TEST_OBJS = test_core.o Global.o
ifeq ($(USE_GSL),true)
TEST_OBJS += Spline.o
endif

test_core: $(TEST_OBJS)
	$(LINK_EXE)

test_grid_tools: test_grid_tools.o Global.o
	$(LINK_EXE)

test_fof: test_fof.o Global.o
	$(LINK_EXE)

test_so: test_so.o Global.o
	$(LINK_EXE)

test_kdtree: test_kdtree.o test_kdtree_odr.o Global.o
	$(LINK_EXE)

# NonLocalGaussianRandomField.h includes the cosmology helpers that use splines and the ODE solver with GSL
FNL_OBJS = test_fnl.o Global.o
ifeq ($(USE_GSL),true)
FNL_OBJS += Spline.o ODESolver.o
endif

test_fnl: $(FNL_OBJS)
	$(LINK_EXE)

MPIRUN  ?= mpirun
NP_LIST ?= 1 2 4

ifeq ($(USE_MPI),true)
RUN  = $(MPIRUN) -np $$np
RUN1 = $(MPIRUN) -np 1
else
NP_LIST = 1
RUN  =
RUN1 =
endif

# Each run must pass; the two "must abort" runs must fail with the expected message
.PHONY: run
run: test_core test_grid_tools test_fof test_so test_fnl test_kdtree
	@set -e; for np in $(NP_LIST); do \
	  echo "== Running test_core with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_core; \
	  echo "== Running test_grid_tools with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_grid_tools; \
	  echo "== Running test_fof with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_fof; \
	  echo "== Running test_so with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_so; \
	  echo "== Running test_fnl with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_fnl; \
	  echo "== Running test_kdtree with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_kdtree; \
	done
	@echo "== Checking that bad input aborts with a clear message"
	@! $(RUN1) ./test_core alloc > alloc.log 2>&1 && grep -q "Reached allocation limit" alloc.log && echo "[PASS] allocation limit aborts"
ifeq ($(USE_MPI),true)
	@! $(MPIRUN) -np 2 ./test_core unwrapped > unwrapped.log 2>&1 && grep -q "outside \[0,1)" unwrapped.log && echo "[PASS] unwrapped particle aborts"
endif
	@rm -f alloc.log unwrapped.log

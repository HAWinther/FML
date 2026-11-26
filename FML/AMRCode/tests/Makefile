# Tests of AMRCode: testamrnbody (deposit, blob, zeldovich) and testpartial (partial multigrid solves)
USE_MPI ?= true
USE_OMP ?= false
USE_GSL ?= true
TARGETS := testamrnbody testpartial
include ../../Makefile.config
FML_MODULES := Global Spline ODESolver Math
# Spline, ODESolver, Math: the coupled quintessence background (Cosmology/Background)
testamrnbody: TestAMRNBody.o Global.o Spline.o ODESolver.o NumericalMethods.o
	$(LINK_EXE)
testpartial: TestPartialSolve.o Global.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif

# make run: deposit, blob, coupled quintessence and two fluids (seconds) and the partial solves. The Zel'dovich test takes ~4 min per task count:
# make run-zeldovich runs it on 4 tasks (nightly)
.PHONY: run run-zeldovich
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do \
	  for t in deposit blob cq twofluid; do echo "== Running testamrnbody $$t with $$np task(s)"; $(RUN) ./testamrnbody $$t; done; \
	  echo "== Running testpartial with $$np task(s)"; $(RUN) ./testpartial; \
	done

run-zeldovich: testamrnbody
	@echo "== Running testamrnbody zeldovich with 4 tasks"
	@$(MPIRUN) -np 4 ./testamrnbody zeldovich

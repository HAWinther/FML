#===================================================================
# FML: QuickStart/example
#
#   make                       build the three quick-start examples
#   make run [NP_LIST="1 4"]   build and run them (each checks its result)
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false
#===================================================================

USE_MPI             ?= true
USE_FFTW            ?= true
USE_OMP             ?= true
USE_GSL             ?= false

TARGETS := pofk pm_simulation
# Pencils need MPI (and FFTW)
ifeq ($(USE_MPI),true)
TARGETS += domains
endif

include ../../Makefile.config

FML_MODULES := Global PencilDomain

pofk: pofk.o Global.o
	$(LINK_EXE)

pm_simulation: pm_simulation.o Global.o
	$(LINK_EXE)

domains: domains.o Global.o PencilGrid.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif

.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do for t in $(TARGETS); do \
	  echo "== $$t on $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./$$t > $$t.log 2>&1 || { cat $$t.log; echo "== FAILED: $$t on $$np task(s)"; exit 1; }; \
	  tail -1 $$t.log; \
	done; done

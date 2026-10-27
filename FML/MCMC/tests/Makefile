#===================================================================
# FML: MCMC/tests
#
#   make       build every test_*.cpp
#   make run   build and run them
#===================================================================

USE_OMP             ?= true
USE_MPI             ?= false

TARGETS := $(patsubst %.cpp,%,$(wildcard test_*.cpp))

include ../../Makefile.config

# OpenMPI/MPICH C++ bindings are not used (and give warnings)
ifeq ($(USE_MPI),true)
OPTIONS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif

FML_MODULES := MCMC Global
MCMC_OBJS   := Posterior.o Sampler.o Minimizer.o Parallel.o Global.o

$(TARGETS): %: %.o $(MCMC_OBJS)
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	@status=0; for t in $(TARGETS); do echo "=== $$t ==="; ./$$t || status=1; done; exit $$status

#===================================================================
# FML: CosmologyFitting/tests
#
#   make       build every test_*.cpp
#   make run   build and run them (from the folder above, where they
#              find external/)
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := $(patsubst %.cpp,%,$(wildcard test_*.cpp))

include ../../Makefile.config

# OpenMPI/MPICH C++ bindings are not used (and give warnings)
ifeq ($(USE_MPI),true)
OPTIONS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif
include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk

FML_MODULES := $(COSMO_MODULES) $(COSMO_FML_MODULES) MCMC Global CosmologyFitting
MCMC_OBJS   := Posterior.o Sampler.o Minimizer.o Parallel.o Global.o
FIT_OBJS    := Theory.o Likelihoods.o SampledParameterMap.o

$(TARGETS): %: %.o $(FIT_OBJS) $(MCMC_OBJS) $(COSMO_OBJS)
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	@status=0; for t in $(TARGETS); do echo "=== $$t ==="; (cd .. && ./tests/$$t) || status=1; done; exit $$status

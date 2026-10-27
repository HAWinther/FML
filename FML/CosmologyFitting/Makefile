#===================================================================
# FML: CosmologyFitting
#
#   make                 ./mcmc (./mcmc params/mcmc_lcdm.ini)
#   make USE_MPI=true    ./mcmc with MPI (chains and minimizer starts
#                        over processes; make clean first when switching)
#   make test            build and run the tests in tests/
#
# Compiler, library paths and how the options below are used are set
# in ../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line.
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := mcmc

include ../Makefile.config

# OpenMPI/MPICH C++ bindings are not used (and give warnings)
ifeq ($(USE_MPI),true)
OPTIONS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif
include ../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk

FML_MODULES := $(COSMO_MODULES) $(COSMO_FML_MODULES) MCMC Global CosmologyFitting
MCMC_OBJS   := Posterior.o Sampler.o Minimizer.o Parallel.o Global.o
FIT_OBJS    := Theory.o Likelihoods.o SampledParameterMap.o

mcmc: Main.o $(FIT_OBJS) $(MCMC_OBJS) $(COSMO_OBJS)
	$(LINK_EXE)

.PHONY: test
test:
	@$(MAKE) --no-print-directory -C tests run

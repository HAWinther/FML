#===================================================================
# FML: MCMC/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=true
#===================================================================

USE_OMP             ?= true
USE_MPI             ?= false

TARGETS := mcmc

include ../../Makefile.config

# OpenMPI/MPICH C++ bindings are not used (and give warnings)
ifeq ($(USE_MPI),true)
OPTIONS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif

FML_MODULES := MCMC Global

mcmc: Main.o Posterior.o Sampler.o Minimizer.o Parallel.o Global.o
	$(LINK_EXE)

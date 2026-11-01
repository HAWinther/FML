# Tests of FML/Perturbations. They link the Einstein-Boltzmann solver, whose species and gravity perturbations are
# the reference. Run with: make run
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_perturbations
include ../../Makefile.config
include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk

FML_MODULES := Global $(COSMO_MODULES) $(COSMO_FML_MODULES)
OBJS := Global.o $(COSMO_OBJS)

test_perturbations: test_perturbations.o $(OBJS)
	$(LINK_EXE)

.PHONY: run
run: test_perturbations
	./test_perturbations

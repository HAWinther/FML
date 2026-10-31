# Tests of FML/Background. They link the Einstein-Boltzmann solver (its background is the reference) and ParameterMap
# (SimulationCode's JBD, quintessence and EDE cosmologies are references too). The CLASS comparison of test_background
# uses the reference data of TransferProviders/tests (CLASS=/path/to/class ./make_test_data.sh there); without them it
# prints SKIP. Run with: make run [DATA=dir]
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_background test_models test_scalar_fields
include ../../Makefile.config
include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk

FML_MODULES := Global ParameterMap $(COSMO_MODULES) $(COSMO_FML_MODULES)
OBJS := Global.o $(COSMO_OBJS)

test_background: test_background.o $(OBJS)
	$(LINK_EXE)

test_models: test_models.o ParameterMap.o $(OBJS)
	$(LINK_EXE)

test_scalar_fields: test_scalar_fields.o ParameterMap.o $(OBJS)
	$(LINK_EXE)

DATA ?= ../../TransferProviders/tests/data
.PHONY: run
run: test_background test_models test_scalar_fields
	./test_background $(DATA)
	./test_models
	./test_scalar_fields

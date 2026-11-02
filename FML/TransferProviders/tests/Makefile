# Tests of FML/TransferProviders against reference runs of CLASS and CAMB. The reference data are not part of FML:
# make them with  CLASS=/path/to/class ./make_test_data.sh  (needs the CLASS executable and pip install camb).
# Without them "make run" prints SKIP.
#
# The test of source = internal (FML's Einstein-Boltzmann solver) links the Cosmology modules.
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_transfer_providers
include ../../Makefile.config

include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk
FML_MODULES := Global $(COSMO_MODULES) $(COSMO_FML_MODULES)
OBJS := test_transfer_providers.o Global.o $(COSMO_OBJS)

test_transfer_providers: $(OBJS)
	$(LINK_EXE)

DATA ?= data
.PHONY: run
run: test_transfer_providers
	./test_transfer_providers $(DATA)

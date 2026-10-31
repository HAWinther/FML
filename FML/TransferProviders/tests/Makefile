# Tests of FML/TransferProviders against reference runs of CLASS and CAMB. The reference data are not part of FML:
# make them with  CLASS=/path/to/class ./make_test_data.sh  (needs the CLASS executable and pip install camb).
# Without them "make run" prints SKIP.
#
# USE_INTERNAL_COSMOLOGY=true (default) also tests source = internal (FML's Einstein-Boltzmann solver), which links
# the Cosmology modules; false builds without it.
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
USE_INTERNAL_COSMOLOGY ?= true
TARGETS := test_transfer_providers
include ../../Makefile.config

FML_MODULES := Global Spline
OBJS := test_transfer_providers.o Global.o Spline.o
ifeq ($(USE_INTERNAL_COSMOLOGY),true)
include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk
FML_MODULES += $(COSMO_MODULES) $(COSMO_FML_MODULES)
OBJS := test_transfer_providers.o Global.o $(COSMO_OBJS)
OPTIONS += -DUSE_INTERNAL_COSMOLOGY
endif

test_transfer_providers: $(OBJS)
	$(LINK_EXE)

DATA ?= data
.PHONY: run
run: test_transfer_providers
	./test_transfer_providers $(DATA)

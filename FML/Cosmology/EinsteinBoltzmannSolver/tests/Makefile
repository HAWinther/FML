#===================================================================
# FML: Cosmology/EinsteinBoltzmannSolver/tests
#
#   make       build every test_*.cpp
#   make run   build and run them (from the folder above, where they
#              find params/, external/ and tests/reference/)
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := $(patsubst %.cpp,%,$(wildcard test_*.cpp))

include ../../../Makefile.config
include ../Cosmology.mk

FML_MODULES := $(COSMO_MODULES) $(COSMO_FML_MODULES)

$(TARGETS): %: %.o $(COSMO_OBJS)
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	@status=0; for t in $(TARGETS); do echo "=== $$t ==="; (cd .. && ./tests/$$t) || status=1; done; exit $$status

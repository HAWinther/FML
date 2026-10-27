USE_MPI ?= false
USE_OMP ?= false
TARGETS := test_baryonification_parameters
include ../../Makefile.config
FML_MODULES := Global

test_baryonification_parameters: test_baryonification_parameters.o Global.o
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	./test_baryonification_parameters

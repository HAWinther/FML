USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := test_field_level test_field_level_safety test_framework test_framework_resolution test_framework_mpi
include ../../Makefile.config
FML_MODULES := Global
# Do not reuse a VPATH object compiled with another module's MPI/precision flags.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_field_level: test_field_level.o Global.o
	$(LINK_EXE)
test_field_level_safety: test_field_level_safety.o Global.o
	$(LINK_EXE)
test_framework: test_framework.o Global.o
	$(LINK_EXE)
test_framework_resolution: test_framework_resolution.o Global.o
	$(LINK_EXE)
test_framework_mpi: test_framework_mpi.o Global.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
SAFETY_RUN = $(MPIRUN) -np $$np
SINGLE_RUN = $(MPIRUN) -np 1
else
NP_LIST = 1
SAFETY_RUN =
SINGLE_RUN =
endif
.PHONY: run safety framework framework-mpi
safety: test_field_level_safety
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_field_level_safety; done
framework: test_framework test_framework_resolution
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework; done
	$(SINGLE_RUN) ./test_framework_resolution
framework-mpi: test_framework_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_mpi; done
run: $(TARGETS) safety framework framework-mpi
	$(SINGLE_RUN) ./test_field_level

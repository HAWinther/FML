USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := test_weak_lensing test_cosmic_shear
ifeq ($(USE_MPI),true)
ifeq ($(USE_FFTW),true)
# The flat-sky shear on pencils (PencilGrid<2>)
TARGETS += test_weak_lensing_pencil
PENCIL_TESTS = ./test_weak_lensing_pencil
endif
endif
include ../../Makefile.config
FML_MODULES := Global
# Explicit local object rule prevents VPATH reuse of differently configured objects.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_weak_lensing: test_weak_lensing.o Global.o
	$(LINK_EXE)
test_cosmic_shear: test_cosmic_shear.o Global.o
	$(LINK_EXE)
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_weak_lensing_pencil.o: test_weak_lensing.cpp
	$(CC) $(DEPFLAGS) -DWL_PENCIL -c -o $@ $< $(OPTIONS) $(INC)
test_weak_lensing_pencil: test_weak_lensing_pencil.o Global.o PencilGrid.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif
.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do for t in ./test_weak_lensing $(PENCIL_TESTS) $$( [ $$np = 1 ] && echo ./test_cosmic_shear ); do \
	  OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive $(RUN) $$t; \
	done; done

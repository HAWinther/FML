USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := integrator_test
include ../../../../Makefile.config
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
integrator_test: Test.o Global.o
	$(LINK_EXE)

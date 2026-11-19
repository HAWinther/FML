USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := gn_build gn_probe gn_test
include ../../../../Makefile.config
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
gn_build: Build.o Global.o
	$(LINK_EXE)
gn_probe: Probe.o Global.o
	$(LINK_EXE)
gn_test: Test.o Global.o
	$(LINK_EXE)

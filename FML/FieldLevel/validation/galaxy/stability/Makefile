USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := stability_probe
include ../../../../Makefile.config
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
stability_probe: Main.o Global.o
	$(LINK_EXE)
stability_test: Test.o Global.o
	$(LINK_EXE)

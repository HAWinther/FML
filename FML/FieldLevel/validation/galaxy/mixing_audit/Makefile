USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := curvature_probe
include ../../../../Makefile.config
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
curvature_probe: Curvature.o Global.o
	$(LINK_EXE)

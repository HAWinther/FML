USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := sph_example
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
sph_example: Main.o Global.o
	$(LINK_EXE)

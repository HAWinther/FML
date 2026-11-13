USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := hydro_example
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
hydro_example: Main.o Global.o
	$(LINK_EXE)

USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := reparam_study
include ../../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
reparam_study: Main.o Global.o
	$(LINK_EXE)

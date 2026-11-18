USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := refinement_predict
include ../../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
refinement_predict: Main.o Global.o
	$(LINK_EXE)

USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := weak_lensing_example
include ../../Makefile.config
FML_MODULES := Global
# Explicit local object rule prevents VPATH reuse of differently configured objects.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
weak_lensing_example: Main.o Global.o
	$(LINK_EXE)

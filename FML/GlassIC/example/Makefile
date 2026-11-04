USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := glass_ic_example
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
glass_ic_example: Main.o Global.o
	$(LINK_EXE)

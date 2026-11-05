USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := threepoint
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
threepoint: Main.o Global.o
	$(LINK_EXE)

USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := example
include ../../Makefile.config
FML_MODULES := Global
example: Main.o Global.o
	$(LINK_EXE)

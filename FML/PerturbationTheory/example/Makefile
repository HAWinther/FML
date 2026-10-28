USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := example
include ../../Makefile.config
FML_MODULES := Global FFTLog
example: Main.o Global.o FFTLog.o
	$(LINK_EXE)

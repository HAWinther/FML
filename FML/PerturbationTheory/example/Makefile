USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := example correlation_example
include ../../Makefile.config
FML_MODULES := Global FFTLog
example: Main.o Global.o FFTLog.o
	$(LINK_EXE)

correlation_example: Correlation.o Global.o
	$(LINK_EXE)

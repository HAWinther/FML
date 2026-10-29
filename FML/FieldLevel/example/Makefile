USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := fieldlevel
include ../../Makefile.config
FML_MODULES := Global
fieldlevel: Main.o Global.o
	$(LINK_EXE)

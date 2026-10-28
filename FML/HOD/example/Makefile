USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := mock_statistics
include ../../Makefile.config
FML_MODULES := Global
mock_statistics: Main.o Global.o
	$(LINK_EXE)

USE_MPI ?= true
USE_OMP ?= false
USE_FFTW := true
TARGETS := four_point_example
include ../../Makefile.config
FML_MODULES := Global
four_point_example: Main.o Global.o
	$(LINK_EXE)

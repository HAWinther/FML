USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := survey_mock
include ../../Makefile.config
FML_MODULES := Global
survey_mock: Main.o Global.o
	$(LINK_EXE)

USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := survey_workflow
include ../../Makefile.config
FML_MODULES := Global
survey_workflow: Main.o Global.o
	$(LINK_EXE)

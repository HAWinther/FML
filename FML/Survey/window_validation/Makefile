USE_MPI ?= true
USE_OMP ?= false
USE_FFTW := true
TARGETS := survey_window_catalog survey_window_matrix
include ../../Makefile.config
FML_MODULES := Global
survey_window_catalog: WindowCatalog.o Global.o
	$(LINK_EXE)
survey_window_matrix: WindowMatrix.o Global.o
	$(LINK_EXE)

USE_MPI ?= true
USE_OMP ?= false
USE_FFTW := true
TARGETS := survey_power_catalog
include ../../Makefile.config
FML_MODULES := Global
survey_power_catalog: PowerCatalog.o Global.o
	$(LINK_EXE)

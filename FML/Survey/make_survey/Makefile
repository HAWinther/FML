#===================================================================
# make_survey: survey mocks from simulation output (SurveyMocks.h)
#   make                      ascii and GADGET input
#   make USE_HDF5=true        also COLASolver HDF5 lightcones and HDF5 output
#   make USE_HEALPIX=true     also Healpix FITS footprints
#===================================================================
USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= false
USE_LUA ?= true
TARGETS := make_survey
include ../../Makefile.config
FML_MODULES := Global GadgetUtils
make_survey: Main.o Global.o GadgetUtils.o
	$(LINK_EXE)

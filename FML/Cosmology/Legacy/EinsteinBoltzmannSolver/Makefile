#===================================================================
# FML: Cosmology/Legacy/EinsteinBoltzmannSolver
#
# Compiler, library paths and how the options below are used are set
# in ../../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

# FFTW is optional (needed for correlation functions via FFTLog)
USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false
USE_FFTW            ?= false

TARGETS := CMFB

include ../../../Makefile.config

FML_MODULES := Spline ODESolver ParameterMap Math Global FFTLog Cosmology/Legacy/BackgroundCosmology Cosmology/Legacy/RecombinationHistory Cosmology/Legacy/LinearPerturbations Cosmology/Legacy/LinearPowerSpectra

CMFB_OBJS = Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o RecombinationHistory.o Math.o Perturbations.o PowerSpectrum.o Global.o
ifeq ($(USE_FFTW),true)
CMFB_OBJS += FFTLog.o
endif

CMFB: $(CMFB_OBJS)
	$(LINK_EXE)

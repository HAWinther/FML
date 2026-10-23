#===================================================================
# FML: NBody/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_FFTW_THREADS    ?= true
USE_MEMORYLOG       ?= true
USE_GSL             ?= true
USE_DEBUG           ?= false

# Extra defines added when USE_DEBUG = true
DEBUG_OPTIONS     = -DBOUNDSCHECK_FFTWGRID -DDEBUG_LPT

TARGETS := nbody

include ../../Makefile.config

OPTIONS     += -DMIN_BYTES_TO_LOG=0
FML_MODULES := Global Spline ODESolver

nbody: Main.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)

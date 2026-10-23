#===================================================================
# FML: Smoothing/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_DEBUG           ?= false

# Extra defines added when USE_DEBUG = true
DEBUG_OPTIONS     = -DDEBUG_FFTWGRID

TARGETS := gridop

include ../../Makefile.config

FML_MODULES := Global

gridop: Main.o Global.o
	$(LINK_EXE)

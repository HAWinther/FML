#===================================================================
# FML: TidalTensor/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_FFTW            ?= true
USE_GSL             ?= true
USE_MPI             ?= false
USE_OMP             ?= false
USE_DEBUG           ?= false

# Extra defines added when USE_DEBUG = true
DEBUG_OPTIONS     = -DDEBUG_FFTWGRID

TARGETS := test

include ../../Makefile.config

FML_MODULES := Global FileUtils

test: Main.o Global.o FileUtils.o
	$(LINK_EXE)

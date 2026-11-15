#===================================================================
# FML: Visualization/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_DEBUG           ?= false

TARGETS := visualization

include ../../Makefile.config

FML_MODULES := Global

visualization: Main.o Global.o
	$(LINK_EXE)

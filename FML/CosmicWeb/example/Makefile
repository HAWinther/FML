#===================================================================
# FML: CosmicWeb/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_FFTW            ?= true
USE_OMP             ?= false
USE_DEBUG           ?= false

TARGETS := cosmicweb

include ../../Makefile.config

FML_MODULES := Global

cosmicweb: Main.o Global.o
	$(LINK_EXE)

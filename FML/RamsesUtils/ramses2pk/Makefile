#===================================================================
# FML: RamsesUtils/ramses2pk
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

CXXSTD           ?= -std=c++20

USE_MPI             ?= true
USE_FFTW            ?= true
USE_OMP             ?= false

TARGETS := ramses2pk

include ../../Makefile.config

FML_MODULES := Global

ramses2pk: Main.o Global.o
	$(LINK_EXE)

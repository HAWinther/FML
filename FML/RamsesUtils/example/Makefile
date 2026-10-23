#===================================================================
# FML: RamsesUtils/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= false

TARGETS := ramses

include ../../Makefile.config

FML_MODULES := Global

ramses: Main.o Global.o
	$(LINK_EXE)

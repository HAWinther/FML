#===================================================================
# FML: Emulator/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_GSL             ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := ee2

include ../../Makefile.config

FML_MODULES :=

ee2: Main.o
	$(LINK_EXE)

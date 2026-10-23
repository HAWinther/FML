#===================================================================
# FML: ODESolver/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_GSL             ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := odesolver

include ../../Makefile.config

FML_MODULES := Global ODESolver

odesolver: Main.o ODESolver.o Global.o
	$(LINK_EXE)

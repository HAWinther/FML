#===================================================================
# FML: Spline/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := spline

include ../../Makefile.config

FML_MODULES := Spline Global

spline: Main.o Spline.o Global.o
	$(LINK_EXE)

#===================================================================
# FML: Math/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_GSL             ?= true

TARGETS := math

include ../../Makefile.config

FML_MODULES := Math Spline ODESolver Global

math: Main.o Math.o Spline.o ODESolver.o Global.o
	$(LINK_EXE)

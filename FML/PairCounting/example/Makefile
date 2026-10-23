#===================================================================
# FML: PairCounting/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := paircount

include ../../Makefile.config

FML_MODULES := Global Spline ODESolver FileUtils

paircount: Main.o Spline.o ODESolver.o Global.o FileUtils.o
	$(LINK_EXE)

#===================================================================
# FML: Cosmology/BackgroundCosmology/example
#
# Compiler, library paths and how the options below are used are set
# in ../../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true

TARGETS := cosmo

include ../../../Makefile.config

FML_MODULES := Cosmology/BackgroundCosmology Spline ODESolver ParameterMap Math Global

cosmo: Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o Math.o Global.o
	$(LINK_EXE)

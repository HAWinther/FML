#===================================================================
# FML: Cosmology/LinearPerturbations/example
#
# Compiler, library paths and how the options below are used are set
# in ../../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true

TARGETS := pert

include ../../../Makefile.config

FML_MODULES := Spline ODESolver ParameterMap Math Global Cosmology/BackgroundCosmology Cosmology/RecombinationHistory Cosmology/LinearPerturbations

pert: Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o RecombinationHistory.o Math.o Perturbations.o Global.o
	$(LINK_EXE)

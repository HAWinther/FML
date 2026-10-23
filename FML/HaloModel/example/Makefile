#===================================================================
# FML: HaloModel/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := halomodel

include ../../Makefile.config

OPTIONS     += -DHMCODETESTING
FML_MODULES := Global ODESolver Math Spline HaloModel SphericalCollapse FileUtils

halomodel: Main.o Global.o Math.o Spline.o ODESolver.o SphericalCollapse.o SphericalCollapseModel.o Halomodel.o FileUtils.o
	$(LINK_EXE)

#===================================================================
# FML: SphericalCollapse/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_GSL             ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := sphericalcollapse

include ../../Makefile.config

FML_MODULES := Global ODESolver Math Spline SphericalCollapse

sphericalcollapse: Main.o Global.o Math.o Spline.o ODESolver.o SphericalCollapse.o SphericalCollapseModel.o
	$(LINK_EXE)

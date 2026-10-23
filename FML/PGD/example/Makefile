#===================================================================
# FML: PGD/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_MEMORYLOG       ?= true
USE_GSL             ?= true

TARGETS := pgd

include ../../Makefile.config

FML_MODULES := Global GadgetUtils FileUtils ODESolver Spline

pgd: Main.o Global.o GadgetUtils.o ODESolver.o Spline.o
	$(LINK_EXE)

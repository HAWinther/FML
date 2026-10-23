#===================================================================
# FML: FFTLog/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_FFTW            ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := fftlog

include ../../Makefile.config

FML_MODULES := Global FFTLog

fftlog: Main.o FFTLog.o Global.o
	$(LINK_EXE)

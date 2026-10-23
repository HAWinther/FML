#===================================================================
# FML: FileUtils/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := fileutils

include ../../Makefile.config

FML_MODULES := FileUtils

fileutils: Main.o FileUtils.o
	$(LINK_EXE)

#===================================================================
# FML: Plotting/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

# Requires python with matplotlib (set PYTHON=... to choose the interpreter)
USE_PYTHON          ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := matplotlib

include ../../Makefile.config

OPTIONS     += -DWITHOUT_NUMPY
FML_MODULES := Global

matplotlib: Main.o Global.o
	$(LINK_EXE)

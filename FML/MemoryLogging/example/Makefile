#===================================================================
# FML: MemoryLogging/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MEMORYLOG       ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := memory

include ../../Makefile.config

OPTIONS     += -DMIN_BYTES_TO_LOG=0
FML_MODULES := Global

memory: Main.o Global.o
	$(LINK_EXE)

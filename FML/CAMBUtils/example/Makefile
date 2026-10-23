#===================================================================
# FML: CAMBUtils/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_FFTW_THREADS    ?= true
USE_MEMORYLOG       ?= true
USE_GSL             ?= true
USE_LUA             ?= true

TARGETS := cambreader

include ../../Makefile.config

OPTIONS     += -DMIN_BYTES_TO_LOG=0
FML_MODULES := Global Spline

cambreader: Main.o Global.o Spline.o
	$(LINK_EXE)

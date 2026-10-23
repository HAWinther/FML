#===================================================================
# FML: COLASolver
#
# Compiler, library paths and how the options below are used are set
# in ../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

# Number of dimensions of space
NDIM_SPACE       ?= 3

# GSL is required, LUA is required to read parameterfiles,
# HEALPIX is required to make weak lensing maps (CHUNKPIX: store them as chunks)
USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_FFTW_THREADS    ?= true
USE_MEMORYLOG       ?= true
USE_GSL             ?= true
USE_LUA             ?= true
USE_HEALPIX         ?= false
USE_CHUNKPIX        ?= false

TARGETS := nbody

include ../Makefile.config

OPTIONS     += -DNDIM_SPACE=$(NDIM_SPACE)
FML_MODULES := Global Spline ODESolver GadgetUtils ParameterMap FileUtils Math
EXTRA_VPATH := src

nbody: Main.o Global.o Spline.o ODESolver.o GadgetUtils.o ParameterMap.o FileUtils.o Math.o
	$(LINK_EXE)

# Rebuild if build options change
Main.o: Makefile

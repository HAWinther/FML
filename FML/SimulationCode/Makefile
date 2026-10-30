#===================================================================
# FML: SimulationCode
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

USE_INTERNAL_COSMOLOGY ?= false

TARGETS := nbody

include ../Makefile.config

OPTIONS     += -DNDIM_SPACE=$(NDIM_SPACE)
FML_MODULES := Global Spline ODESolver GadgetUtils ParameterMap FileUtils Math
EXTRA_VPATH := src

# The simple P(k)/table build does not compile or link the Einstein-Boltzmann solver.
INTERNAL_COSMO_OBJS :=
ifeq ($(USE_INTERNAL_COSMOLOGY),true)
include ../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk
OPTIONS += -DUSE_INTERNAL_COSMOLOGY
FML_MODULES += $(filter-out Cosmology/EinsteinBoltzmannSolver,$(COSMO_MODULES)) $(COSMO_FML_MODULES) InitialConditions
vpath ParameterFile.cpp $(FML_ROOT)/Cosmology/EinsteinBoltzmannSolver
INTERNAL_COSMO_OBJS := $(filter-out Spline.o ODESolver.o,$(COSMO_OBJS)) InternalTransferLoader.o
endif

# The simulation is compiled once per particle type, each in its own file (use make -j to compile them in parallel)
RUN_SIMULATION_OBJS := RunSimulation_COLAScaleDepWithMass.o RunSimulation_COLAWithMass.o RunSimulation_BasicWithMass.o RunSimulation_COLAScaleDep.o RunSimulation_COLA.o RunSimulation_BasicQ.o RunSimulation_Basic.o

nbody: $(INTERNAL_COSMO_OBJS) Main.o $(RUN_SIMULATION_OBJS) BaryonificationOutput.o Global.o Spline.o ODESolver.o GadgetUtils.o ParameterMap.o FileUtils.o Math.o
	$(LINK_EXE)

# Rebuild if build options change
Main.o $(RUN_SIMULATION_OBJS): Makefile

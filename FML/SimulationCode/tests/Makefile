USE_MPI ?= false
USE_OMP ?= false
# test_cola_frame uses the cosmology, gravity models and COLA of the solver (FFTW grids and the GSL ODE solver)
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_baryonification_parameters test_cola_frame test_models test_lightcone_crossing
include ../../Makefile.config
OPTIONS += -DNDIM_SPACE=3
FML_MODULES := Global Spline ODESolver ParameterMap FileUtils Math
# TransferInput.cpp (the transfer functions the gravity models read) is in ../src
EXTRA_VPATH := ../src

# TransferInput.o links FML's Einstein-Boltzmann solver (ic_type_of_input = internal), as in ../Makefile
include ../../Cosmology/EinsteinBoltzmannSolver/Cosmology.mk
FML_MODULES += $(filter-out Cosmology/EinsteinBoltzmannSolver,$(COSMO_MODULES)) $(COSMO_FML_MODULES)
vpath ParameterFile.cpp $(FML_ROOT)/Cosmology/EinsteinBoltzmannSolver
vpath EinsteinBoltzmannSolver.cpp $(FML_ROOT)/Cosmology/EinsteinBoltzmannSolver
INTERNAL_COSMO_OBJS := $(filter-out Spline.o ODESolver.o,$(COSMO_OBJS))

test_baryonification_parameters: test_baryonification_parameters.o Global.o
	$(LINK_EXE)

test_cola_frame: test_cola_frame.o TransferInput.o $(INTERNAL_COSMO_OBJS) Global.o Spline.o ODESolver.o ParameterMap.o FileUtils.o Math.o NumericalMethods.o
	$(LINK_EXE)

test_lightcone_crossing: test_lightcone_crossing.o
	$(LINK_EXE)

test_models: test_models.o TransferInput.o $(INTERNAL_COSMO_OBJS) Global.o Spline.o ODESolver.o ParameterMap.o FileUtils.o Math.o NumericalMethods.o
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	./test_baryonification_parameters
	./test_cola_frame
	./test_models
	./test_lightcone_crossing

USE_MPI ?= false
USE_OMP ?= false
# test_cola_frame uses the cosmology, gravity models and COLA of the solver (FFTW grids and the GSL ODE solver)
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_baryonification_parameters test_cola_frame test_models
include ../../Makefile.config
OPTIONS += -DNDIM_SPACE=3
FML_MODULES := Global Spline ODESolver ParameterMap FileUtils Math
# TransferInput.cpp (the transfer functions the gravity models read) is in ../src
EXTRA_VPATH := ../src

test_baryonification_parameters: test_baryonification_parameters.o Global.o
	$(LINK_EXE)

test_cola_frame: test_cola_frame.o TransferInput.o Global.o Spline.o ODESolver.o ParameterMap.o FileUtils.o Math.o NumericalMethods.o
	$(LINK_EXE)

test_models: test_models.o TransferInput.o Global.o Spline.o ODESolver.o ParameterMap.o FileUtils.o Math.o NumericalMethods.o
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	./test_baryonification_parameters
	./test_cola_frame
	./test_models

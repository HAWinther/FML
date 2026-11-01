# Tests of Cosmology/Background (header only; they do not link the Einstein-Boltzmann solver). The references are the
# backgrounds of the solver's old model classes (stored in data/, see SolverReference.h) and of SimulationCode's former
# JBD, quintessence and EDE cosmologies (data/simcode_*.txt). The CLASS comparison of test_background uses the reference data of
# TransferProviders/tests (CLASS=/path/to/class ./make_test_data.sh there); without them it prints SKIP.
# Run with: make run [DATA=dir]
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_background test_models test_scalar_fields
include ../../../Makefile.config

# Errors in splines and ODEs are exceptions (as in the solver); the stored references of the solver
OPTIONS += -DSPLINE_THROW_ON_ERROR -DODESOLVER_THROW_ON_ERROR
OPTIONS += -DREFERENCE_DATA_DIR=\"$(FML_ROOT)/Cosmology/Background/tests/data\"

FML_MODULES := Global Spline ODESolver Math
OBJS := Global.o Spline.o ODESolver.o NumericalMethods.o

test_background: test_background.o $(OBJS)
	$(LINK_EXE)

test_models: test_models.o $(OBJS)
	$(LINK_EXE)

test_scalar_fields: test_scalar_fields.o $(OBJS)
	$(LINK_EXE)

DATA ?= ../../../TransferProviders/tests/data
.PHONY: run
run: test_background test_models test_scalar_fields
	./test_background $(DATA)
	./test_models
	./test_scalar_fields

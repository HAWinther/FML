# Tests of Cosmology/Perturbations (header only; they do not link the Einstein-Boltzmann solver). The reference is the
# solver's old species and gravity perturbations, stored in data/solver_perturbations.txt (see the test).
# Run with: make run
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_perturbations test_quasi_static
include ../../../Makefile.config

# Errors in splines and ODEs are exceptions (as in the solver); the stored references of the solver
OPTIONS += -DSPLINE_THROW_ON_ERROR -DODESOLVER_THROW_ON_ERROR
OPTIONS += -DREFERENCE_DATA_DIR=\"$(FML_ROOT)/Cosmology/Perturbations/tests/data\"

FML_MODULES := Global Spline ODESolver Math
OBJS := Global.o Spline.o ODESolver.o NumericalMethods.o

test_perturbations: test_perturbations.o $(OBJS)
	$(LINK_EXE)

test_quasi_static: test_quasi_static.o $(OBJS)
	$(LINK_EXE)

.PHONY: run
run: test_perturbations test_quasi_static
	./test_perturbations
	./test_quasi_static

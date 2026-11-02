# Tests of Cosmology/Growth (header only). Run with: make run
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
USE_GSL ?= true
TARGETS := test_growth
include ../../../Makefile.config
OPTIONS += -DSPLINE_THROW_ON_ERROR -DODESOLVER_THROW_ON_ERROR
FML_MODULES := Global Spline ODESolver Math
OBJS := Global.o Spline.o ODESolver.o NumericalMethods.o

test_growth: test_growth.o $(OBJS)
	$(LINK_EXE)

.PHONY: run
run: test_growth
	./test_growth


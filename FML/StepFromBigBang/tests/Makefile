# Tests of StepFromBigBang (header only). Run with: make run
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := test_sfbb
include ../../Makefile.config
OPTIONS += -DSPLINE_THROW_ON_ERROR -DODESOLVER_THROW_ON_ERROR
FML_MODULES := Global Spline ODESolver Math
OBJS := Global.o Spline.o ODESolver.o NumericalMethods.o

test_sfbb: test_sfbb.o $(OBJS)
	$(LINK_EXE)

.PHONY: run
run: test_sfbb
	./test_sfbb eds 64 0.25 0.1
	./test_sfbb eds 64 0.25 0.3
	./test_sfbb lcdm 64 1e-5
	./test_sfbb lcdm 64 0.3
	RADIATION=1 KCUT=0.25 ./test_sfbb lcdm 64 0.3
	./test_sfbb mg 64 1e-5
	./test_sfbb api 64

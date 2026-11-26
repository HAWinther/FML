# TreePMCode: the TreePM N-body code (treepm). Build options and library paths are in ../Makefile.config (and your
# ../Makefile.local). Run: mpirun -np 4 ./treepm parameterfile.txt
USE_MPI       ?= true
USE_OMP       ?= true
USE_FFTW      ?= true
USE_MEMORYLOG ?= false
USE_GSL       ?= true
TARGETS := treepm
include ../Makefile.config
FML_MODULES := Global FileUtils GadgetUtils Spline ODESolver Math
treepm: Main.o Global.o FileUtils.o GadgetUtils.o Spline.o ODESolver.o NumericalMethods.o
	$(LINK_EXE)

# AMRCode: the AMR N-body code (amrnbody). Build options and library paths are in ../Makefile.config (and your
# ../Makefile.local). Run: mpirun -np 4 ./amrnbody parameterfile.txt
USE_MPI       ?= true
USE_OMP       ?= false
USE_FFTW      ?= true
USE_MEMORYLOG ?= false
USE_GSL       ?= true
TARGETS := amrnbody
include ../Makefile.config
FML_MODULES := Global FileUtils GadgetUtils Spline ODESolver
amrnbody: Main.o Global.o FileUtils.o GadgetUtils.o Spline.o ODESolver.o
	$(LINK_EXE)

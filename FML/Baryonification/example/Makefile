USE_MPI ?= false
USE_OMP ?= false
TARGETS := baryonify
include ../../Makefile.config
FML_MODULES := Global

baryonify: Main.o Global.o
	$(LINK_EXE)

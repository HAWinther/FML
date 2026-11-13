USE_MPI ?= true
USE_OMP ?= false
TARGETS := peano_hilbert_example
include ../Makefile.config

FML_MODULES := Global

peano_hilbert_example: Main.o Global.o
	$(LINK_EXE)

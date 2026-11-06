USE_MPI ?= false
USE_OMP ?= false
USE_GSL ?= true
TARGETS := example
include ../../Makefile.config
FML_MODULES := AxionHMcode
example: example.o AxionHMcode.o
	$(LINK_EXE)

#===================================================================
# FML: GadgetUtils/example
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= false

TARGETS := gadget

include ../../Makefile.config

FML_MODULES := Global GadgetUtils FileUtils
EXTRA_CLEAN := test testhdf5 output_testhdf5

gadget: Main.o Global.o GadgetUtils.o
	$(LINK_EXE)

# Not built by default (make <name>)
test: Test.o Global.o GadgetUtils.o FileUtils.o
	$(LINK_EXE)

# Test of the HDF5 reader/writer (make testhdf5 USE_HDF5=true)
testhdf5: TestHDF5.o Global.o GadgetUtils.o
	$(LINK_EXE)

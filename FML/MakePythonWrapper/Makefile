#===================================================================
# FML: MakePythonWrapper
#
# Example of how to make a python module (_MyLibrary.so) with SWIG.
# Compiler, library paths and how the options below are used are set
# in ../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make PYTHON=python3.12
#===================================================================

USE_PYTHON          ?= true
USE_MPI             ?= false
USE_OMP             ?= false

TARGETS := test

include ../Makefile.config

PYTHONLIBNAME = _MyLibrary.so
SWIGFILE      = InterfaceFile.i
SWIGWRAPPER   = InterfaceFile_wrap.cxx
SWIGWRAPPERO  = InterfaceFile_wrap.o

CXXFLAGS    += -fPIC
EXTRA_CLEAN := $(SWIGWRAPPER) $(PYTHONLIBNAME) MyLibrary.py *.pyc __pycache__

OBJS = Main.o

# Run swig and generate a library that can be called from python
.PHONY: test
test: $(OBJS)
	swig -c++ -python $(filter -D%,$(OPTIONS)) $(SWIGFILE)
	$(CC) -c $(SWIGWRAPPER) $(OPTIONS) $(INC)
	$(CC) -shared $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $(PYTHONLIBNAME) $(LIB) $(LINK)

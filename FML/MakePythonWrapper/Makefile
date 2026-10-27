#===================================================================
# FML: MakePythonWrapper
#
# Example of how to make a python module (_MyLibrary.so) with SWIG.
# Compiler, library paths and how the options below are used are set
# in ../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make PYTHON=python3.12
#
# make        builds _MyLibrary.so (and MyLibrary.py)
# make run    builds it and runs test.py with the same $(PYTHON)
#
# The module is built with the headers of $(PYTHON) and its numpy
# and has to be imported by that same python. As for any extension
# module it is NOT linked with libpython: the interpreter that imports
# it provides the python symbols (linking a libpython, e.g. from
# python3-config, can pull in a different python than the headers)
#===================================================================

USE_PYTHON          ?= true
USE_MPI             ?= false
USE_OMP             ?= false

PYTHONLIBNAME = _MyLibrary.so
SWIGFILE      = InterfaceFile.i
SWIGWRAPPER   = InterfaceFile_wrap.cxx
SWIGWRAPPERO  = InterfaceFile_wrap.o

TARGETS := $(PYTHONLIBNAME)

include ../Makefile.config

# Position independent code for the shared library (added after Makefile.config so the default CXXFLAGS are kept)
CXXFLAGS    += -fPIC
EXTRA_CLEAN := $(SWIGWRAPPER) $(PYTHONLIBNAME) MyLibrary.py *.pyc __pycache__

# Extension module: no libpython (see above). On macOS the python symbols are resolved when the module is loaded
ifeq ($(shell uname -s),Darwin)
SHAREDFLAGS = -bundle -undefined dynamic_lookup
else
SHAREDFLAGS = -shared
endif
LINK_MODULE = $(filter-out $(PYTHON_LINK),$(LINK))

OBJS = Main.o

.PHONY: test run check_python
test: $(PYTHONLIBNAME)

check_python:
	@test -n "$(PYTHON_INCLUDE)" -a -f "$(PYTHON_INCLUDE)/Python.h" || \
	  (echo "Python.h not found for PYTHON = $(PYTHON) (PYTHON_INCLUDE = $(PYTHON_INCLUDE))"; exit 1)
	@test -n "$(NUMPY_INCLUDE)" || (echo "numpy not found for PYTHON = $(PYTHON)"; exit 1)

# Run swig (makes the wrapper and MyLibrary.py)
$(SWIGWRAPPER): $(SWIGFILE) numpy.i
	swig -c++ -python $(filter -D%,$(OPTIONS)) -o $(SWIGWRAPPER) $(SWIGFILE)

$(SWIGWRAPPERO): $(SWIGWRAPPER) check_python
	$(CC) -c -o $@ $(SWIGWRAPPER) $(OPTIONS) $(INC)

# The library that can be called from python
$(PYTHONLIBNAME): $(OBJS) $(SWIGWRAPPERO)
	$(CC) $(SHAREDFLAGS) $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $@ $(LIB) $(LINK_MODULE)

run: $(PYTHONLIBNAME)
	$(PYTHON) test.py

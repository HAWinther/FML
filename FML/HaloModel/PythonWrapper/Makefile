#===================================================================
# FML: HaloModel/PythonWrapper
#
# Builds the python module _halomodel (pybind11) used by
# HaloModelWrapper.py. Needs pybind11 (pip install pybind11).
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make PYTHON=python3.12
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := halomodel

include ../../Makefile.config

PYBIND11_INCLUDE ?= $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
PYTHON_SUFFIX    ?= $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo .so)
PYTHONLIBNAME     = _halomodel$(PYTHON_SUFFIX)

OPTIONS     += -DDONT_SHOW_INIT_INFO
CXXFLAGS    += -fPIC
INC         += $(PYBIND11_INCLUDE)
FML_MODULES := Global ODESolver Math Spline FileUtils SphericalCollapse HaloModel
EXTRA_CLEAN := _halomodel*.so *.pyc __pycache__

OBJS = bindings.o ODESolver.o Global.o Math.o Spline.o FileUtils.o Halomodel.o HaloModelTools.o SphericalCollapse.o SphericalCollapseModel.o

.PHONY: halomodel
halomodel: $(OBJS)
	@if [ -z "$(PYBIND11_INCLUDE)" ]; then echo "pybind11 not found for $(PYTHON) (pip install pybind11)"; exit 1; fi
	$(CC) -shared $(OPTIONS) $(OBJS) -o $(PYTHONLIBNAME) $(LIB) $(LINK)

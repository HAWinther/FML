#===================================================================
# FML: HaloModel/PythonWrapper
#
# Builds a python module (_HaloModelCXX.so + HaloModelCXX.py) with SWIG.
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make PYTHON=python3.12
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_PYTHON          ?= true
USE_MPI             ?= false

TARGETS := halomodel

include ../../Makefile.config

PYTHONLIBNAME = _HaloModelCXX.so
PYWRAPFILE    = HaloModelCXX.py
SWIGFILE      = InterfaceFile.i
SWIGWRAPPER   = InterfaceFile_wrap.cxx
SWIGWRAPPERO  = InterfaceFile_wrap.o

OPTIONS     += -DDONT_SHOW_INIT_INFO
CXXFLAGS    += -fPIC
FML_MODULES := Global ODESolver Math Spline FileUtils SphericalCollapse HaloModel
EXTRA_CLEAN := $(SWIGWRAPPER) $(PYTHONLIBNAME) $(PYWRAPFILE) *.pyc __pycache__

OBJS = Wrapper.o ODESolver.o Global.o Math.o Spline.o FileUtils.o Halomodel.o SphericalCollapse.o SphericalCollapseModel.o

# Run swig and generate a library that can be called from python
.PHONY: halomodel
halomodel: $(OBJS)
	swig -c++ -python $(filter -D%,$(OPTIONS)) $(SWIGFILE)
	$(CC) -c $(SWIGWRAPPER) $(OPTIONS) $(INC)
	$(CC) -shared $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $(PYTHONLIBNAME) $(LIB) $(LINK)

# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# The name of the library and the interface file
#===================================================

PYTHONLIBNAME = _HaloModelCXX.so
PYWRAPFILE    = HaloModelCXX.py
SWIGFILE      = InterfaceFile.i
SWIGWRAPPER   = InterfaceFile_wrap.cxx
SWIGWRAPPERO  = InterfaceFile_wrap.o

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -Wall 

#===================================================
# Options
#===================================================

FML_INCLUDE    = $(HOME)/local/FML/
PYTHON_INCLUDE = /opt/local/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/
PYTHON_LIB     = /opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/
NUMPY_INCLUDE  = /opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/numpy/core/include
PYTHON_LINK    = -lpython3.8 -ldl -framework CoreFoundation

#===================================================
# Compile up all library defines from options above
#===================================================

FML_INCLUDE    = $(HOME)/local/FML
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = -DDONT_SHOW_INIT_INFO

INC  += -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
LIB  += -L$(PYTHON_LIB) 
LINK += $(PYTHON_LINK)

OPTIONS += -DUSE_OMP
CC      += -fopenmp

OPTIONS += -DUSE_GSL
INC     += -I$(GSL_INCLUDE)
LIB     += -L$(GSL_LIB)
LINK    += $(GSL_LINK)

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/Math/:$(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/FileUtils/:$(FML_INCLUDE)/FML/SphericalCollapse/:$(FML_INCLUDE)/FML/HaloModel/
OBJS = Wrapper.o ODESolver.o Global.o Math.o Spline.o FileUtils.o Halomodel.o SphericalCollapse.o SphericalCollapseModel.o

TARGETS := halomodel
all: $(TARGETS)
.PHONY: all clean

# Run swig and generate a library that can be called from python
halomodel: $(OBJS)
	swig -c++ -python $(OPTIONS) $(SWIGFILE)
	$(CC) -fPIC -c $(SWIGWRAPPER) $(INC)
	$(CC) -shared $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $(PYTHONLIBNAME) $(INC) $(LIB) $(LINK)
	#$(CC) -fPIC -shared $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $(PYTHONLIBNAME) $(INC) $(LIB) $(LINK)

clean:
	rm -rf $(TARGETS) *.o $(SWIGWRAPPER) $(PYTHONLIBNAME) $(PYWRAPFILE) *.pyc __pycache__

%.o: %.cpp 
	${CC} -fPIC -c -o $@ $< $(OPTIONS) $(INC) 


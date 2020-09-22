# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# The name of the library and the interface file
#===================================================

PYTHONLIBNAME = _MyLibrary.so
SWIGFILE      = InterfaceFile.i
# These are generated
SWIGWRAPPER   = InterfaceFile_wrap.cxx
SWIGWRAPPERO  = InterfaceFile_wrap.o

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++-mp-7 -std=c++1z -Wall 

#===================================================
# Options
#===================================================

# Required
USE_PYTHON       = true

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = /Users/hansw/local/FML

# Python and numpy includes
PYTHON_INCLUDE = /anaconda2/include/python2.7 
PYTHON_LIB     = /anaconda2/lib
NUMPY_INCLUDE  = /anaconda2/lib/python2.7/site-packages/numpy/core/include
PYTHON_LINK    = -lpython2.7 -ldl -framework CoreFoundation

# For Python 3
#PYTHON_INCLUDE = /astro/local/anaconda3/include/python3.6m
#PYTHON_LIB     = /astro/local/anaconda3/lib
#NUMPY_INCLUDE  = /astro/local/anaconda3/lib/python3.6/site-packages/numpy/core/include
#PYTHON_LINK    = -lpython3.6m -ldl

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = 

ifeq ($(USE_PYTHON),true)
INC  += -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
LIB  += -L$(PYTHON_LIB) 
LINK += $(PYTHON_LINK)
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH := 
FILE1 = Main.o
OBJS = $(FILE1)

TARGETS := test
all: $(TARGETS)
.PHONY: all clean

# Compile all the object files as a shared library
$(FILE1) : Main.cpp
	$(CC) $(OPTIONS) -fPIC -c $< -o $@ $(INC)

# Run swig and generate a library that can be called from python
test: $(OBJS)
	swig -c++ -python $(OPTIONS) $(SWIGFILE)
	$(CC) -fPIC -c $(SWIGWRAPPER) $(INC)
	$(CC) -fPIC -shared $(OPTIONS) $(OBJS) $(SWIGWRAPPERO) -o $(PYTHONLIBNAME) $(INC) $(LIB) $(LINK)

clean:
	rm -rf $(TARGETS) *.o $(SWIGWRAPPER) $(PYTHONLIBNAME) *.pyc

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


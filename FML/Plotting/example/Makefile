# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -std=c++1z -O3 -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = false
# Use OpenMP threads
USE_OMP          = false
# Only needeed if you want plotting with matplotlib from C++
USE_PYTHON       = true
# Use LUA (LuaFileParser)
USE_LUA          = false
# Check for bad memory accesses
USE_SANITIZER    = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = /Users/hansw/local/FML

# Python : only needed if USE_PYTHON = true
PYTHON_INCLUDE = /anaconda2/include/python2.7 
NUMPY_INCLUDE  = /anaconda2/lib/python2.7/site-packages/numpy/core/include
PYTHON_LINK    = -lpython2.7 -ldl -framework CoreFoundation

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = 

ifeq ($(USE_MPI),true)
CC       = $(MPICC)
OPTIONS += -DUSE_MPI
endif

ifeq ($(USE_OMP),true)
OPTIONS += -DUSE_OMP
CC      += -fopenmp
endif

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address
endif

ifeq ($(USE_PYTHON),true)
OPTIONS += -DWITHOUT_NUMPY
INC  += -I$(PYTHON_INCLUDE)
INC  += -I$(NUMPY_INCLUDE)
LINK += $(PYTHON_LINK)
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/ 
OBJS = Main.o Global.o

TARGETS := matplotlib
all: $(TARGETS)
.PHONY: all clean


clean:
	rm -rf $(TARGETS) *.o

matplotlib: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 

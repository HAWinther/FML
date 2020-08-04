# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++-mp-7 -std=c++1z
MPICC   = mpicxx-openmpi-gcc7 -std=c++1z

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = false
# Use OpenMP threads
USE_OMP          = false
# Check for bad memory accesses
USE_SANITIZER    = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = /Users/hansw/local

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

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/ParameterMap/
OBJS = Main.o ParameterMap.o Global.o

TARGETS := parametermap
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

parametermap: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


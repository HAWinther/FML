# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use OpenMP threads
USE_OMP          = true
# Use the GSL library (Spline, ODESolver) Required here!
USE_GSL          = true
# Check for bad memory accesses
USE_SANITIZER    = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# GSL 
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = 

ifeq ($(USE_OMP),true)
OPTIONS += -DUSE_OMP
CC      += -fopenmp
endif

ifeq ($(USE_GSL),true)
OPTIONS += -DUSE_GSL
INC     += -I$(GSL_INCLUDE)
LIB     += -L$(GSL_LIB)
LINK    += $(GSL_LINK)
endif

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address
endif

TARGETS := cosmo
all: $(TARGETS)
.PHONY: all clean

#===================================================
# Object files to be compiled
#===================================================

VPATH := ../:$(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/ParameterMap/:$(FML_INCLUDE)/FML/Math/:$(FML_INCLUDE)/FML/Global/
OBJS = Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o Math.o Global.o

clean:
	rm -rf $(TARGETS) *.o

cosmo: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 

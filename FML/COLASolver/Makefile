# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set C++17 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -std=c++1z -O3 -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use MPI 
USE_MPI          = true
# Use OpenMP threads
USE_OMP          = true
# Use the FFTW library (required)
USE_FFTW         = true
# Use threads in FFTW
USE_FFTW_THREADS = true
# Log allocations in the library
USE_MEMORYLOG    = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Use GSL (required)
USE_GSL          = true
# Use LUA (required to use parameterfile)
# One can edit ReadParameters.h to do without it
USE_LUA          = true

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# FFTW
FFTW_INCLUDE   = $(HOME)/local/include
FFTW_LIB       = $(HOME)/local/lib
FFTW_LINK      = -lfftw3
FFTW_MPI_LINK  = -lfftw3_mpi
FFTW_OMP_LINK  = -lfftw3_threads

# GSL
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

# LUA
LUA_INCLUDE    = $(HOME)/local/include
LUA_LIB        = $(HOME)/local/lib
LUA_LINK       = -llua -ldl

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

ifeq ($(USE_FFTW),true)
OPTIONS += -DUSE_FFTW
INC     += -I$(FFTW_INCLUDE)
LIB     += -L$(FFTW_LIB)
ifeq ($(USE_MPI),true)
LINK    += $(FFTW_MPI_LINK)
endif
ifeq ($(USE_OMP),true)
ifeq ($(USE_FFTW_THREADS),true)
OPTIONS += -DUSE_FFTW_THREADS
LINK    += $(FFTW_OMP_LINK)
endif
endif
LINK    += $(FFTW_LINK)
endif

ifeq ($(USE_MEMORYLOG),true)
OPTIONS += -DMEMORY_LOGGING
endif

ifeq ($(USE_GSL),true)
OPTIONS += -DUSE_GSL
INC     += -I$(GSL_INCLUDE)
LIB     += -L$(GSL_LIB)
LINK    += $(GSL_LINK)
endif

ifeq ($(USE_LUA),true)
OPTIONS += -DUSE_LUA
INC     += -I$(LUA_INCLUDE)
LIB     += -L$(LUA_LIB)
LINK    += $(LUA_LINK)
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/GadgetUtils/:$(FML_INCLUDE)/FML/ParameterMap/:$(FML_INCLUDE)/FML/FileUtils/:src/
OBJS = Main.o Global.o Spline.o ODESolver.o GadgetUtils.o ParameterMap.o FileUtils.o

TARGETS := nbody
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

nbody: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

Main.o: Makefile src/Main.cpp src/Cosmology.h src/GravityModel.h src/ReadParameters.h src/AnalyzeOutput.h src/Simulation.h
Global.o: $(FML_INCLUDE)/FML/Global/Global.cpp $(FML_INCLUDE)/FML/Global/Global.h
Spline.o: $(FML_INCLUDE)/FML/Spline/Spline.cpp $(FML_INCLUDE)/FML/Spline/Spline.h
ODESolver.o: $(FML_INCLUDE)/FML/ODESolver/ODESolver.cpp $(FML_INCLUDE)/FML/ODESolver/ODESolver.h
GadgetUtils.o: $(FML_INCLUDE)/FML/GadgetUtils/GadgetUtils.cpp $(FML_INCLUDE)/FML/GadgetUtils/GadgetUtils.h
ParameterMap.o: $(FML_INCLUDE)/FML/ParameterMap/ParameterMap.cpp $(FML_INCLUDE)/FML/ParameterMap/ParameterMap.h
FileUtils.o: $(FML_INCLUDE)/FML/FileUtils/FileUtils.cpp $(FML_INCLUDE)/FML/FileUtils/FileUtils.h

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


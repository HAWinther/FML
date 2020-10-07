# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -O3 -std=c++1z -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = true
# Use OpenMP threads
USE_OMP          = true
# Use the GSL library (Spline, ODESolver) Required here!
USE_GSL          = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Use the FFTW library
USE_FFTW         = true
# Use threads in FFTW
USE_FFTW_THREADS = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# GSL 
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

# FFTW : only needed if USE_FFTW = true
FFTW_INCLUDE   = $(HOME)/local/include
FFTW_LIB       = $(HOME)/local/lib
FFTW_LINK      = -lfftw3
FFTW_MPI_LINK  = -lfftw3_mpi
FFTW_OMP_LINK  = -lfftw3_threads

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

ifeq ($(USE_GSL),true)
OPTIONS += -DUSE_GSL
INC     += -I$(GSL_INCLUDE)
LIB     += -L$(GSL_LIB)
LINK    += $(GSL_LINK)
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

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address
endif

TARGETS := CMFB
all: $(TARGETS)
.PHONY: all clean

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/ParameterMap/:$(FML_INCLUDE)/FML/Cosmology/BackgroundCosmology/:$(FML_INCLUDE)/FML/Math/:$(FML_INCLUDE)/FML/Cosmology/LinearPerturbations/:$(FML_INCLUDE)/FML/Cosmology/RecombinationHistory/::$(FML_INCLUDE)/FML/Cosmology/LinearPowerSpectra/:$(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/FFTLog/
OBJS = Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o RecombinationHistory.o Math.o Perturbations.o PowerSpectrum.o Global.o
ifeq ($(USE_FFTW),true)
  OBJS += FFTLog.o
endif

clean:
	rm -rf $(TARGETS) *.o

CMFB: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 

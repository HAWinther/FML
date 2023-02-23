# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -g -Wall -Wextra -march=native
MPICC   = mpicxx-mpich-gcc11 -std=c++1z -O3 -g -Wall -Wextra -march=native

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
USE_FFTW_THREADS = false
# Log allocations in the library
USE_MEMORYLOG    = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Use GSL (required)
USE_GSL          = true

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# GSL : only needed if USE_GSL = true
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

# FFTW
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

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address
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

ifeq ($(USE_MEMORYLOG),true)
OPTIONS += -DMEMORY_LOGGING
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH = $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/GadgetUtils/:$(FML_INCLUDE)/FML/FileUtils/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/Spline/
OBJS = Main.o Global.o GadgetUtils.o ODESolver.o Spline.o

TARGETS := pgd
all: $(TARGETS)
.PHONY: all clean


clean:
	rm -rf $(TARGETS) *.o

pgd: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


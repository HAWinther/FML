# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++-mp-7 -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx-openmpi-gcc7 -std=c++1z -O3 -g -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = true
# Use OpenMP threads
USE_OMP          = true
# Log allocations in the library
USE_MEMORYLOG    = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Use the FFTW library (needed for Reconstruction.cpp)
USE_FFTW         = true

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = /Users/hansw/local/FML

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     = 
LINK    =
OPTIONS =

# FFTW : only needed if USE_FFTW = true
FFTW_INCLUDE   = $(HOME)/local/include
FFTW_LIB       = $(HOME)/local/lib
FFTW_LINK      = -lfftw3
FFTW_MPI_LINK  = -lfftw3_mpi
FFTW_OMP_LINK  = -lfftw3_threads

ifeq ($(USE_MPI),true)
CC       = $(MPICC)
OPTIONS += -DUSE_MPI
endif

ifeq ($(USE_OMP),true)
OPTIONS += -DUSE_OMP
CC      += -fopenmp
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

ifeq ($(USE_MEMORYLOG),true)
OPTIONS += -DMEMORY_LOGGING
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH = $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/MultigridSolver/:$(FML_INCLUDE)/FML/FileUtils/
OBJS      = GenericExample.o    Global.o
OBJS_CONT = ContinuityExample.o Global.o
OBJS_POIS = PoissonExample.o    Global.o
OBJS_FOFR = FofrExample.o       Global.o
OBJS_SYM  = SymmetronExample.o  Global.o 
OBJS_DGP  = dgpExample.o        Global.o
OBJS_REC  = Reconstruction.o    Global.o FileUtils.o
OBJS_TEST = Test.o              Global.o FileUtils.o

TARGETS := mgsolver
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

mgsolver: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

contsolver: $(OBJS_CONT)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

poissonsolver: $(OBJS_POIS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

fofrsolver: $(OBJS_FOFR)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

recon: $(OBJS_REC)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

test: $(OBJS_TEST)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

symm: $(OBJS_SYM)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

dgp: $(OBJS_DGP)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


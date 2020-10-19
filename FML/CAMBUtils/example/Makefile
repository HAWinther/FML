# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -std=c++1z -O3 -Wall -Wextra -march=native -ffast-math

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = true
# Use OpenMP threads
USE_OMP          = true
# Use the FFTW library
USE_FFTW         = true
# Use threads in FFTW
USE_FFTW_THREADS = true
# Log allocations in the library
USE_MEMORYLOG    = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Print more info as the code runs and do some more checks
USE_DEBUG        = false
# Use GSL
USE_GSL          = true
# Use LUA (LuaFileParser)
USE_LUA          = true

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# FFTW : only needed if USE_FFTW = true
FFTW_INCLUDE   = $(HOME)/local/include
FFTW_LIB       = $(HOME)/local/lib
FFTW_LINK      = -lfftw3
FFTW_MPI_LINK  = -lfftw3_mpi
FFTW_OMP_LINK  = -lfftw3_threads

# GSL : only needed if USE_GSL = true
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

# LUA : only needed if USE_LUA = true
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

ifeq ($(USE_DEBUG),true)
OPTIONS += 
endif

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
OPTIONS += -DMEMORY_LOGGING -DMIN_BYTES_TO_LOG=0
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

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/Spline/
OBJS = Main.o Global.o Spline.o 

TARGETS := cambreader
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

cambreader: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


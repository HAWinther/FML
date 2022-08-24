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
USE_MPI          = true
# Use OpenMP threads
USE_OMP          = false
# Use the CGAL library (Triangulation)
USE_CGAL         = true
# Check for bad memory accesses
USE_SANITIZER    = false
# Print more info as the code runs and do some more checks
USE_DEBUG        = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# CGAL : only needed if USE_CGAL = true
CGAL_INCLUDE   = /opt/local/include
CGAL_LIB       = /opt/local/lib
CGAL_LINK      =  -lgmp -lm 

# For older version of CGAL when it was a shared library
# instead of just headerfiles
#CGAL_LINK      = -lCGAL -lgmp -lm 

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = 

ifeq ($(USE_DEBUG),true)
OPTIONS += -DDEBUG_TESSELATION
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

ifeq ($(USE_CGAL),true)
OPTIONS += -DUSE_CGAL
INC     += -I$(CGAL_INCLUDE)
LIB     += -L$(CGAL_LIB)
LINK    += $(CGAL_LINK)
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/FileUtils/
OBJS = Main.o Global.o FileUtils.o

TARGETS := voronoi
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

voronoi: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 

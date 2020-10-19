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
# Use LUA (LuaFileParser)
USE_LUA          = true
# Check for bad memory accesses
USE_SANITIZER    = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

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

ifeq ($(USE_LUA),true)
OPTIONS += -DUSE_LUA
INC     += -I$(LUA_INCLUDE)
LIB     += -L$(LUA_LIB)
LINK    += $(LUA_LINK)
endif

#===================================================
# Object files to be compiled
#===================================================

VPATH := $(FML_INCLUDE)/FML/Global/
OBJS = Main.o Global.o

TARGETS := lua
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

lua: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


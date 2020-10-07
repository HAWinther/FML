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
# Check for bad memory accesses
USE_SANITIZER    = false

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

VPATH = $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/GadgetUtils/:$(FML_INCLUDE)/FML/FileUtils/
OBJS = Main.o Global.o GadgetUtils.o 
OBJS_TEST = Test.o Global.o GadgetUtils.o FileUtils.o

TARGETS := gadget
all: $(TARGETS)
.PHONY: all clean


clean:
	rm -rf $(TARGETS) *.o

gadget: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

test: $(OBJS_TEST)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


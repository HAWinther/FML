# Hans A. Winther (2015) (hans.a.winther@gmail.com)

SHELL := /bin/bash

# Include and library paths
FML_INCLUDE = /Users/hansw/local/
INC  = -I$(HOME)/local/include -I$(FML_INCLUDE)
LIBS = -L$(HOME)/local/lib
LINK = -lgsl -lgslcblas

CC = g++-mp-7 -O3 -fopenmp
OPTIONS = -std=c++17 -DUSE_GSL -DUSE_OMP

TARGETS := power
all: $(TARGETS)
.PHONY: all clean

# OBJECT FILES
VPATH := $(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/ParameterMap/:$(FML_INCLUDE)/FML/Cosmology/BackgroundCosmology/:$(FML_INCLUDE)/FML/Math/:$(FML_INCLUDE)/FML/Cosmology/LinearPerturbations/:$(FML_INCLUDE)/FML/Cosmology/RecombinationHistory/::$(FML_INCLUDE)/FML/Cosmology/LinearPowerSpectra/
OBJS = Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o RecombinationHistory.o Math.o Perturbations.o PowerSpectrum.o

clean:
	rm -rf $(TARGETS) *.o

power: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIBS) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


# Hans A. Winther (2015) (hans.a.winther@gmail.com)

SHELL := /bin/bash

# Include and library paths
FML_INCLUDE = /Users/hansw/local/
INC  = -I$(HOME)/local/include -I$(FML_INCLUDE)
LIBS = -L$(HOME)/local/lib
LINK = -lgsl -lgslcblas

CC = g++-mp-7
OPTIONS = -std=c++17

TARGETS := cosmo
all: $(TARGETS)
.PHONY: all clean

# OBJECT FILES
VPATH := ../:$(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/ParameterMap/:$(FML_INCLUDE)/FML/Math/
OBJS = Main.o BackgroundCosmology.o Spline.o ODESolver.o ParameterMap.o Math.o

clean:
	rm -rf $(TARGETS) *.o

cosmo: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIBS) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 


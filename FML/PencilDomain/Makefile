USE_MPI ?= true
USE_FFTW ?= true
USE_OMP ?= false
TARGETS := example_pencil

include ../Makefile.config

FML_MODULES := Global

all: example_pencil

example_pencil: Main.o PencilGrid.o Global.o
	$(LINK_EXE)

Main.o: example/Main.cpp PencilGrid.h
	$(CC) $(OPTIONS) $(INC) -c $< -o $@

PencilGrid.o: PencilGrid.cpp PencilGrid.h
	$(CC) $(OPTIONS) $(INC) -c $< -o $@

Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) -c -o $@ $< $(OPTIONS) $(INC)

.PHONY: all clean

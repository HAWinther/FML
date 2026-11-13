# Hans A. Winther (hans.a.winther@gmail.com)

SHELL := /bin/bash

CC      = g++ -std=c++17 -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -std=c++17 -O3 -Wall -Wextra -march=native

USE_MPI       = true
USE_SANITIZER = false

# Path to folder containing FML/ (default: the repository this example is in)
FML_INCLUDE ?= $(abspath $(CURDIR)/../../..)

INC     = -I$(FML_INCLUDE)
OPTIONS =

ifeq ($(USE_MPI),true)
CC       = $(MPICC)
OPTIONS += -DUSE_MPI
endif

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address -g
endif

VPATH = $(FML_INCLUDE)/FML/Global/
HEADERS = $(FML_INCLUDE)/FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h $(FML_INCLUDE)/FML/PeanoHilbertDomain/HilbertCurve.h

TARGETS := amrtest amrexample
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

amrtest: Test.o Global.o
	${CC} -o $@ $^ $(OPTIONS)

amrexample: Main.o Global.o
	${CC} -o $@ $^ $(OPTIONS)

%.o: %.cpp $(HEADERS)
	${CC} -c -o $@ $< $(OPTIONS) $(INC)

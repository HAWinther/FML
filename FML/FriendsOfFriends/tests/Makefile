#===================================================================
# FML: FriendsOfFriends/tests
#
#   make       build the tests
#   make run   build and run them on each number of MPI tasks in NP_LIST
#   python3 make_rfof_reference.py regenerates the reference data (numpy, scipy)
#===================================================================

USE_MPI ?= true
USE_OMP ?= false
USE_FFTW ?= false
TARGETS := test_relaxed_fof

include ../../Makefile.config

# OpenMPI/MPICH C++ bindings are not used (and give warnings)
ifeq ($(USE_MPI),true)
OPTIONS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
endif

FML_MODULES := Global

Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)

test_relaxed_fof: test_relaxed_fof.o Global.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
else
NP_LIST = 1
RUN =
endif

.PHONY: run
run: test_relaxed_fof
	@set -e; for np in $(NP_LIST); do $(RUN) ./test_relaxed_fof; done

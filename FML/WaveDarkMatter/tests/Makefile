USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= false
TARGETS := test_wave_dm test_soliton
ifeq ($(USE_MPI),true)
# The same tests on pencils (CPencilGrid)
TARGETS += test_wave_dm_pencil test_soliton_pencil
endif
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_wave_dm: test_wave_dm.o Global.o
	$(LINK_EXE)
test_soliton: test_soliton.o Global.o
	$(LINK_EXE)
%_pencil.o: %.cpp
	$(CC) $(DEPFLAGS) -DWAVE_PENCIL -c -o $@ $< $(OPTIONS) $(INC)
test_wave_dm_pencil: test_wave_dm_pencil.o Global.o PencilGrid.o
	$(LINK_EXE)
test_soliton_pencil: test_soliton_pencil.o Global.o PencilGrid.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
RUN_NP = $(MPIRUN) -np
PENCIL_TESTS = ./test_wave_dm_pencil ./test_soliton_pencil
else
NP_LIST = 1
RUN =
RUN_NP = true ||
PENCIL_TESTS =
endif
.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do for t in ./test_wave_dm ./test_soliton $(PENCIL_TESTS); do OMP_NUM_THREADS=2 $(RUN) $$t; done; done
ifeq ($(USE_MPI),true)
	@# Checkpoints are independent of the decomposition: slabs on 2 tasks -> pencils on 4 and back
	@set -e; rm -f wave_cross.*; $(RUN_NP) 2 ./test_wave_dm --write wave_cross; $(RUN_NP) 4 ./test_wave_dm_pencil --read wave_cross; \
	 rm -f wave_cross.*; $(RUN_NP) 4 ./test_wave_dm_pencil --write wave_cross; $(RUN_NP) 1 ./test_wave_dm --read wave_cross; rm -f wave_cross.*
endif

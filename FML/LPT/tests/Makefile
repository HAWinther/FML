USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := compare_low_orders test_arbitrary_order test_parallel
include ../../Makefile.config
FML_MODULES := Global

compare_low_orders: compare_low_orders.o Global.o
	$(LINK_EXE)

test_arbitrary_order: test_arbitrary_order.o Global.o
	$(LINK_EXE)

test_parallel.o: OPTIONS += -DFML_EDS_LPT_TEST_ALLOCATIONS -DFML_FOURIER_RESAMPLING_TEST_ALLOCATIONS
test_parallel: test_parallel.o Global.o
	$(LINK_EXE)

MPIRUN ?= mpirun
NP_LIST ?= 1 2 4

.PHONY: run
run: $(TARGETS)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./test_arbitrary_order
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(PYTHON) test_eds_reference.py --library-exe ./compare_low_orders --mesh-exe ./test_arbitrary_order
ifeq ($(USE_MPI),true)
	$(PYTHON) test_parallel.py --launcher "$(MPIRUN)" --np-list "$(NP_LIST)"
else
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} ./test_parallel
endif

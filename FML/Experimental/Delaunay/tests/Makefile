# Tests of the native Delaunay triangulation and periodic Voronoi tessellation.
# make run builds and runs them (test_delaunay on one task, test_delaunay_mpi for each task count in NP_LIST);
# make compare_cgal (needs CGAL) compares with CGAL's periodic triangulations
USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_delaunay test_delaunay_mpi
include ../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_delaunay: test_delaunay.o Global.o
	$(LINK_EXE)
test_delaunay_mpi: test_delaunay_mpi.o Global.o
	$(LINK_EXE)

# Comparison with CGAL (edge sets and timings): make compare_cgal CGAL_INCLUDE=... (header only CGAL >= 5)
compare_cgal.o: OPTIONS += -DUSE_CGAL
compare_cgal.o: INC += -I$(CGAL_INCLUDE)
compare_cgal: compare_cgal.o Global.o
	$(CC) -o $@ $^ $(OPTIONS) $(LIB) -L$(CGAL_LIB) $(LINK) -lgmp -lmpfr
EXTRA_CLEAN += compare_cgal

# The native tessellation against the library's MPIPeriodicDelaunay (volumes, neighbours, DTFE densities):
# make compare_library [CGAL2D=true] CGAL_INCLUDE=...
ifeq ($(CGAL2D),true)
compare_library.o: OPTIONS += -DCGAL_NDIM=2
endif
compare_library.o: OPTIONS += -DUSE_CGAL
compare_library.o: INC += -I$(CGAL_INCLUDE)
compare_library: compare_library.o Global.o
	$(CC) -o $@ $^ $(OPTIONS) $(LIB) -L$(CGAL_LIB) $(LINK) -lgmp -lmpfr
EXTRA_CLEAN += compare_library

MPIRUN ?= mpirun
NP_LIST ?= 1 2 3 4
ifeq ($(USE_MPI),true)
RUN = $(MPIRUN) -np $$np
RUN1 = $(MPIRUN) -np 1
else
NP_LIST = 1
RUN =
RUN1 =
endif

.PHONY: run
run: test_delaunay test_delaunay_mpi
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} OMP_WAIT_POLICY=passive $(RUN1) ./test_delaunay
	@set -e; for np in $(NP_LIST); do echo "== test_delaunay_mpi on $$np task(s)"; \
	  OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} OMP_WAIT_POLICY=passive $(RUN) ./test_delaunay_mpi; done

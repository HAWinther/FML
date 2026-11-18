# Tests of the native Delaunay triangulation and periodic Voronoi tessellation (serial + OpenMP).
# make run builds and runs them; make compare_cgal (needs CGAL) compares with CGAL's periodic triangulations
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= false
TARGETS := test_delaunay
include ../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_delaunay: test_delaunay.o Global.o
	$(LINK_EXE)

# Comparison with CGAL (edge sets and timings): make compare_cgal CGAL_INCLUDE=... (header only CGAL >= 5)
compare_cgal.o: OPTIONS += -DUSE_CGAL
compare_cgal.o: INC += -I$(CGAL_INCLUDE)
compare_cgal: compare_cgal.o Global.o
	$(CC) -o $@ $^ $(OPTIONS) $(LIB) -L$(CGAL_LIB) $(LINK) -lgmp -lmpfr
EXTRA_CLEAN += compare_cgal

.PHONY: run
run: test_delaunay
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} OMP_WAIT_POLICY=passive ./test_delaunay

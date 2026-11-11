# End-to-end checks of snapshot_analyzer on a synthetic Gadget snapshot (check_snapshot_analyzer.py, needs numpy;
# h5py for the HDF5 input check). make run builds ../snapshot_analyzer (with the USE_* options given here; -B since
# make does not track them) and runs the checks with each task count in NP_LIST. With USE_CGAL=true it also checks
# ZOBOV and DTFE, with USE_HDF5=true HDF5 input
USE_MPI ?= true
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
PYTHON ?= python3

ifeq ($(USE_MPI),true)
RUNNER = --mpirun "$(MPIRUN)" --np "$(NP_LIST)"
else
RUNNER = --mpirun "" --np 1
endif

.PHONY: run clean
run:
	$(MAKE) -C .. -B snapshot_analyzer USE_MPI=$(USE_MPI)
	OMP_NUM_THREADS=2 $(PYTHON) check_snapshot_analyzer.py ../snapshot_analyzer $(RUNNER) --workdir check_output

clean:
	rm -rf check_output

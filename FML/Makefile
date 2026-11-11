#===================================================================
# FML top-level Makefile
#
#   make examples   Build every example (keeps going, lists failures)
#   make test       Build and run the regression tests in tests/
#   make benchmark  Build opt-in instrumented performance cases
#   make benchmark-check  Check counters and benchmark reports
#   make python-analysis   Install the small analysis package into $(PYTHON)
#   make test-python-analysis  Test the installed analysis package
#   make clean      Clean all examples and tests (also the module tests */tests)
#
# Build options and library paths are in Makefile.config. Put your
# machine specific settings in Makefile.local (not tracked by git).
# Options given here are passed on, e.g. make examples USE_MPI=false
#===================================================================

# Examples that only need MPI/OpenMP/FFTW/GSL
EXAMPLE_DIRS := $(sort $(dir $(wildcard */example/Makefile */examples/Makefile */*/example/Makefile */*/*/example/Makefile))) \
                Cosmology/EinsteinBoltzmannSolver/ CosmologyFitting/ \
                RamsesUtils/ramses2pk/ SimulationCode/

# Examples that need extra libraries (built with make examples-extra)
#   LuaFileParser: LUA   Triangulation: CGAL   VoidFinders: CGAL + FFTW   Plotting: python + matplotlib
EXTRA_DIRS   := LuaFileParser/example/ Triangulation/example/ VoidFinders/example/ Plotting/example/
# Python wrappers (built with make python): MakePythonWrapper (SWIG), HaloModel (pybind11)
PYTHON_DIRS  := MakePythonWrapper/ HaloModel/PythonWrapper/

EXAMPLE_DIRS := $(filter-out $(EXTRA_DIRS),$(EXAMPLE_DIRS))

# The unit tests of the modules (cleaned with make clean; tests/ is the main test suite)
MODULE_TEST_DIRS := $(sort $(dir $(wildcard */tests/Makefile */*/tests/Makefile)))

.PHONY: help examples examples-extra python python-analysis test-python-analysis benchmark benchmark-check test clean

help:
	@sed -n '2,12p' Makefile

define build_dirs
	@failed=""; for d in $(1); do \
	  echo "== $$d"; $(MAKE) --no-print-directory -C $$d || failed="$$failed $$d"; \
	done; \
	if [ -n "$$failed" ]; then echo "== FAILED:$$failed"; exit 1; else echo "== All built"; fi
endef

examples:
	$(call build_dirs,$(EXAMPLE_DIRS))

examples-extra:
	$(call build_dirs,$(EXTRA_DIRS))

python:
	$(call build_dirs,$(PYTHON_DIRS))

PYTHON ?= python3
python-analysis:
	$(PYTHON) -m pip install ./PythonAnalysis

test-python-analysis:
	$(PYTHON) -m unittest discover -s PythonAnalysis/tests -v

benchmark:
	@$(MAKE) --no-print-directory -C Performance all

benchmark-check:
	@$(MAKE) --no-print-directory -C Performance check

test:
	@$(MAKE) --no-print-directory -C tests run

clean:
	@for d in $(EXAMPLE_DIRS) $(EXTRA_DIRS) $(PYTHON_DIRS) $(MODULE_TEST_DIRS) Performance/ tests/; do $(MAKE) --no-print-directory -C $$d clean > /dev/null; done

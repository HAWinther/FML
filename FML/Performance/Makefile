# Opt-in, separate objects: never mix counter-enabled and ordinary FML objects.
USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_FFTW_THREADS ?= false
USE_GSL ?= false
USE_LUA ?= false
TARGETS := fml_benchmark test_counters test_counters_disabled
include ../Makefile.config
PYTHON ?= python3
OPTIONS += -DFML_ENABLE_PERFORMANCE_COUNTERS
FML_MODULES := Global
export FML_BENCH_ROOT := $(FML_ROOT)
export FML_BENCH_CXX := $(CC)
export FML_BENCH_OPTIONS := $(OPTIONS)
export FML_BENCH_INCLUDES := $(INC)
export FML_BENCH_LIBRARIES := $(LIB)
export FML_BENCH_LINKS := $(LINK)
.PHONY: FORCE check
BuildInfo.h: FORCE build_info.py Main.cpp Counters.h Storage.h
	$(PYTHON) build_info.py $@
Main.o: BuildInfo.h
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
fml_benchmark: Main.o Global.o
	$(LINK_EXE)
test_counters: TestCounters.o Global.o
	$(LINK_EXE)
DisabledTestCounters.o: TestCounters.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) -UFML_ENABLE_PERFORMANCE_COUNTERS $(INC)
DisabledGlobal.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) -UFML_ENABLE_PERFORMANCE_COUNTERS $(INC)
test_counters_disabled: DisabledTestCounters.o DisabledGlobal.o
	$(LINK_EXE)
check: all
	./test_counters
	./test_counters_disabled
	$(PYTHON) test_reports.py --exe ./fml_benchmark
EXTRA_CLEAN += BuildInfo.h

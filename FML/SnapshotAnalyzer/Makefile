# Standalone snapshot analysis executable. Requires FFTW and Lua.
USE_MPI          ?= true
USE_OMP          ?= true
USE_FFTW         ?= true
USE_FFTW_THREADS ?= true
USE_MEMORYLOG    ?= false
USE_LUA          ?= true

TARGETS := snapshot_analyzer
FML_MODULES := Global GadgetUtils
EXTRA_VPATH := src

include ../Makefile.config

snapshot_analyzer: Main.o Global.o GadgetUtils.o
	$(LINK_EXE)

Main.o: Makefile

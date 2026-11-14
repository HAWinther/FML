# Standalone snapshot analysis executable. Requires FFTW and Lua.
USE_MPI          ?= true
USE_OMP          ?= true
USE_FFTW         ?= true
USE_FFTW_THREADS ?= true
USE_MEMORYLOG    ?= false
USE_LUA          ?= true
# The grids are PencilGrids (pencils) instead of FFTWGrids (slabs); FML_PENCILS="px py" in the environment
# chooses the process grid
USE_PENCIL_GRID  ?= false

TARGETS := snapshot_analyzer
FML_MODULES := Global GadgetUtils
EXTRA_VPATH := src

include ../Makefile.config

ifeq ($(USE_PENCIL_GRID),true)
OPTIONS     += -DUSE_PENCIL_GRID -DFML_ANALYSIS_PENCIL_GRID
FML_MODULES += PencilDomain
VPATH       += $(FML_ROOT)/PencilDomain
PENCIL_OBJS := PencilGrid.o
endif

# The objects depend on the grid type (rebuilt when USE_PENCIL_GRID changes)
GRID_FLAVOUR := $(if $(filter true,$(USE_PENCIL_GRID)),pencil,slab)
$(shell echo $(GRID_FLAVOUR) | cmp -s - .grid_flavour || echo $(GRID_FLAVOUR) > .grid_flavour)

snapshot_analyzer: Main.o Global.o GadgetUtils.o $(PENCIL_OBJS)
	$(LINK_EXE)

Main.o Global.o GadgetUtils.o $(PENCIL_OBJS): .grid_flavour

Main.o: Makefile

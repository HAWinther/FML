USE_PENCIL_GRID ?= false
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := eulerian_checkpoint catalog_to_modes particle_mesh_checkpoint survey_checkpoint_mpi survey_mpi fieldlevel calibration framework framework_mpi checkpoint_mpi particle_mesh particle_mesh_adjoint particle_mesh_map
ifeq ($(USE_PENCIL_GRID),true)
ifneq ($(USE_MPI),true)
$(error USE_PENCIL_GRID=true requires USE_MPI=true)
endif
OPTIONS += -DFML_FIELDLEVEL_PENCIL_GRID
PENCIL_OBJECT := PencilGrid.o
endif
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
fieldlevel: Main.o Global.o
	$(LINK_EXE)
calibration: Calibration.o Global.o
	$(LINK_EXE)

framework: Framework.o Global.o
	$(LINK_EXE)

framework_mpi: FrameworkMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

particle_mesh: ParticleMesh.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
particle_mesh_adjoint: ParticleMeshAdjoint.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
particle_mesh_map: ParticleMeshMAP.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

checkpoint_mpi: CheckpointMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

survey_mpi: SurveyMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

survey_checkpoint_mpi: SurveyCheckpointMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)

particle_mesh_checkpoint: ParticleMeshCheckpoint.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

catalog_to_modes: Catalog.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

eulerian_checkpoint: EulerianCheckpoint.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

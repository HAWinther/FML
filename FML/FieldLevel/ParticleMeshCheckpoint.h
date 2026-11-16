#ifndef FML_FIELD_LEVEL_PARTICLE_MESH_CHECKPOINT_H
#define FML_FIELD_LEVEL_PARTICLE_MESH_CHECKPOINT_H
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/ParticleMeshPosterior.h>

namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT {
    // Same-layout, trajectory-boundary inference restart. No PM workspace or
    // mid-evolution particle checkpoint is implied. All methods are local-only.
    template <class Grid>
    struct ForwardIdentity<PeriodicPMModelT<Grid>> {
        using Forward = PeriodicPMModelT<Grid>;
        static void describe_grid(Codec::Encoder & e, const Forward & model) {
            DETAIL::describe_grid_domain<Grid>(e, model.domain());
            e.text("FML PM observation layout 1");
            const auto layout = model.observation_layout();
            for (int x : {layout.global_mesh, layout.local_start, layout.local_width}) e.integer(x);
            e.integer(layout.global_cells);
            e.integer(layout.owned_cells);
            for (int d = 0; d < 3; ++d) {
                e.integer(layout.local_starts[d]);
                e.integer(layout.local_counts[d]);
            }
        }
        static void append(Codec::Encoder & e, const Forward & model) {
            // Bump the version if any fixed numerical convention below changes.
            // Opaque spectrum callbacks and external physics still require the
            // caller's explicit versioned CheckpointOptions::problem_id.
            const auto & settings = model.dynamics().settings();
            const bool compensated = settings.force_window != PMForceWindow::None;
            const bool tsc = settings.assignment == PMAssignment::TSC;
            const bool observed = model.observation_response_configured();
            // Preserve tag-1 bytes for the unchanged default numerical law.
            e.text(tsc ? "FML periodic PM matter Fourier target 4" : observed ? "FML periodic PM matter Fourier target 3" :
                   compensated ? "FML periodic PM matter Fourier target 2" : "FML periodic PM matter Fourier target 1");
            if (observed || compensated || tsc) e.integer(static_cast<int>(settings.force_window));
#ifdef USE_MPI
            e.integer(1);
#else
            e.integer(0);
#endif
#ifdef USE_OMP
            e.integer(1);
#else
            e.integer(0);
#endif
            e.text("Flat matter+Lambda; x/L; p=a^2 dx/d(H0 t); growing spectrum at a=1");
            e.text(tsc ? "ZA growing lattice IC; equal masses; TSC deposition/gather; density contrast" :
                         "ZA growing lattice IC; equal masses; CIC deposition/gather; density contrast");
            e.text(tsc ? (compensated ? "Continuous i*k/k^2 force; DC/component Nyquist zero; inverse squared TSC window" :
                                    "Continuous i*k/k^2 force; DC/component Nyquist zero; no TSC deconvolution") :
                   compensated ? "Continuous i*k/k^2 force; DC/component Nyquist zero; inverse squared CIC window" :
                               "Continuous i*k/k^2 force; DC/component Nyquist zero; no CIC deconvolution");
            e.text("KDK; arithmetic scale-factor midpoint; Simpson 128 background integrals");
            e.text("Euclidean discrete adjoint; binary midpoint/prefix replay; identity white metric");
            e.real(settings.background.omega_m);
            e.vector(settings.scale_factors);
            // Conservative numerical compatibility: replay budgets can change
            // floating-point operation order even when the mathematical target agrees.
            e.integer(model.adjoint_options().max_checkpoints);
            if (observed) {
                e.text("Fixed real Fourier signal response; noise after response; raw reconstruction fields");
                e.integer(model.reference_cic_mesh());
                e.vector(model.observation_response());
            }
        }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT
#endif

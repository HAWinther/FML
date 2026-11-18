#ifndef FML_FIELD_LEVEL_EULERIAN_CHECKPOINT_H
#define FML_FIELD_LEVEL_EULERIAN_CHECKPOINT_H
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/DistributedEulerianLPT.h>

namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT {
    template <class Grid>
    struct ConfigurationIdentity<DistributedEulerianLPTModelT<Grid>> {
        using Forward = DistributedEulerianLPTModelT<Grid>;
        static void assignment(Codec::Encoder & e, const Forward & model) {
            const auto & c = model.configuration();
            e.text("FML Eulerian LPT configuration 1");
            e.integer(int(c.matter_assignment));
            e.real(c.nufft_epsilon);
        }
        static void meshes(Codec::Encoder & e, const Forward & model) {
            const auto & c = model.configuration();
            for (int x :
                 {c.mesh, c.product_mesh, model.product_mesh(), c.density_mesh, c.kernel_width, int(c.evolution)})
                e.integer(x);
        }
        static void physics(Codec::Encoder & e, const Forward & model) {
            const auto & c = model.configuration();
            e.real(c.boxsize);
            e.real(c.initial_cutoff);
            e.real(c.matter_band.cutoff);
            e.integer(int(c.matter_band.geometry));
            e.real(c.likelihood_band.cutoff);
            e.integer(int(c.likelihood_band.geometry));
        }
    };
    template <class Grid>
    struct ForwardIdentity<DistributedEulerianLPTModelT<Grid>> {
        using Forward = DistributedEulerianLPTModelT<Grid>;
        static void describe_grid(Codec::Encoder & e, const Forward & model) {
            // Grid/domain, all mesh sizes, rank and local field/mode layout determine
            // each numerical decomposition. No grid/FFT allocation in this local codec.
            DETAIL::describe_grid_domain<Grid>(e, model.domain());
        }
        static void append(Codec::Encoder & e, const Forward &) {
            e.text("FML Eulerian density,centered-delta2,centered-K2,physical-laplacian 1");
            e.text("FML isometric compact-white lift and Euclidean transpose 1");
            e.text("FML Eulerian posterior-noise linear-response field metric 1");
            // Spectrum callbacks and external physics must be versioned by problem_id.
        }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT
#endif

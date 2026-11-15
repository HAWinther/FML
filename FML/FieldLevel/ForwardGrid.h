#ifndef FML_FIELD_LEVEL_FORWARD_GRID_H
#define FML_FIELD_LEVEL_FORWARD_GRID_H
#include <FML/FFTWGrid/GridInterface.h>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace FML::FIELDLEVEL::FRAMEWORK::DETAIL {
    // Local metadata only: safe inside checkpoint identity's collective_try phase.
    template <class Grid, class Encoder>
    void describe_grid_domain(Encoder & e, const FML::Domain & domain) {
        e.text(typeid(Grid).name());
        if (domain.is_slab()) {
            e.text("FML slab domain 1");
            const auto & slab = domain.template get<FML::SlabDomain>();
            e.integer(slab.ntasks());
            for (int t = 0; t < slab.ntasks(); ++t) {
                e.real(slab.x_min(t));
                e.real(slab.x_max(t));
            }
        } else {
            e.text("FML pencil domain 1");
            const auto & pencil = domain.template get<FML::PencilDomain>();
            for (int parts : pencil.process_grid())
                e.integer(parts);
            e.vector(pencil.x_edges());
            e.vector(pencil.y_edges());
        }
    }
    // Compatibility for the serial/reference forward and existing slab-only custom forwards.
    // Distributed forwards expose their backend and a copied, replicated model domain.
    template <class Forward, class = void>
    struct ForwardGrid {
        using Grid = FML::GRID::FFTWGrid<3>;
        static Grid make(const Forward & model) { return Grid(model.configuration().mesh); }
        template <class Encoder>
        static void describe(Encoder & e, const Forward &) {
            e.text(typeid(Grid).name());
            e.text("FML default FFTW layout 1");
        }
    };
    template <class Forward>
    struct ForwardGrid<Forward,
                       std::void_t<typename Forward::Grid, decltype(std::declval<const Forward &>().domain())>> {
        using Grid = typename Forward::Grid;
        static Grid make(const Forward & model) {
            return FML::GRID::make_grid<Grid>(model.configuration().mesh, 0, 0, model.domain());
        }
        // No collectives: checkpoint identity is constructed inside a collective_try local phase.
        template <class Encoder>
        static void describe(Encoder & e, const Forward & model) {
            describe_grid_domain<Grid>(e, model.domain());
        }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK::DETAIL
#endif

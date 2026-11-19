#ifndef FML_VALIDATION_INITIAL_SUPPORT_RESCALING_H
#define FML_VALIDATION_INITIAL_SUPPORT_RESCALING_H
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/ForwardGrid.h>
#include <FML/FieldLevel/validation/galaxy/interweave/Rescaling.h>
namespace ACTIVE_SUPPORT {
    namespace FF = FML::FIELDLEVEL::FRAMEWORK;
    using Field = FML::FIELDLEVEL::Field;
    // Exact spherical initial-band contract of DistributedEulerianLPTModelT.
    // The sampled grid and domain, rather than product/observation modes or a fitted
    // Jacobian rank, determine this orthogonal projector and its GLOBAL real rank.
    template <class Model>
    class InitialSupport {
        const Model & model_;
        using Factory = FF::DETAIL::ForwardGrid<Model>;
        typename Factory::Grid grid_;
        uint64_t rank_ = 0;
        bool active(FML::IndexIntType index) const {
            const auto n = grid_.get_fourier_integer_wavevector_from_index(index);
            const double k = 2 * M_PI / model_.configuration().boxsize *
                             std::sqrt(double(n[0]) * n[0] + double(n[1]) * n[1] + double(n[2]) * n[2]);
            return k > 0 && k < model_.configuration().initial_cutoff;
        }

      public:
        explicit InitialSupport(const Model & model) : model_(model), grid_(Factory::make(model)) {
            static_assert(std::is_same_v<Model, FF::DistributedEulerianLPTModelT<typename Factory::Grid>>,
                          "Initial support requires the Eulerian LPT spherical initial filter contract");
            double local = 0;
            for (auto i : grid_.get_fourier_range())
                if (active(i)) {
                    const auto n = grid_.get_fourier_integer_wavevector_from_index(i);
                    local += FML::GRID::fourier_mode_multiplicity(n[2], model_.configuration().mesh);
                }
            const double global = model_.sum(local), cells = double(model_.layout().global_cells);
            if (!(cells <= 9007199254740992.) || global < 0 || global > cells || global != std::floor(global))
                throw std::invalid_argument("Unrepresentable global initial-support rank");
            rank_ = uint64_t(global);
            model_.controls({double(model_.configuration().mesh),
                             model_.configuration().boxsize,
                             model_.configuration().initial_cutoff,
                             double(rank_)});
        }
        uint64_t rank() const { return rank_; }
        void controls(const std::vector<double> & p, size_t index, double alpha_new) const {
            bool ok =
                p.size() == model_.parameters().size() && index < p.size() && std::isfinite(alpha_new) && alpha_new > 0;
            if (index < p.size())
                ok &= std::isfinite(p[index]) && p[index] > 0;
            for (double x : p)
                ok &= std::isfinite(x);
            if (model_.maximum(ok ? 0. : 1.))
                throw std::invalid_argument("Invalid support-rescaling scalars");
            model_.controls(p);
            model_.controls({double(index), alpha_new});
        }
        void validate(const Field & input) const {
            bool valid = input.size() == model_.cells();
            for (double x : input)
                valid &= std::isfinite(x);
            if (model_.maximum(valid ? 0. : 1.))
                throw std::invalid_argument("Invalid support-projector input");
        }
        template <class Function>
        void local(Function && function) const {
            FML::COMMUNICATION::collective_try([&] { function(); }, "[InitialSupport] Allocate/transform proposal");
        }
        Field project(const Field & input) {
            validate(input);
            Field output;
            local([&] { output.resize(input.size()); });
            size_t q = 0;
            grid_.set_grid_status_real(true);
            for (auto i : grid_.get_real_range())
                grid_.set_real_from_index(i, input[q++]);
            grid_.fftw_r2c();
            for (auto i : grid_.get_fourier_range())
                if (!active(i))
                    grid_.set_fourier_from_index(i, FML::GRID::ComplexType(0));
            grid_.fftw_c2r();
            q = 0;
            for (auto i : grid_.get_real_range())
                output[q++] = grid_.get_real_from_index(i);
            validate(output);
            return output;
        }
    };
    template <class Support>
    RESCALING::Proposal propose(Support & support,
                                const Field & white,
                                const std::vector<double> & parameters,
                                size_t amplitude_index,
                                double alpha_new) {
        support.controls(parameters, amplitude_index, alpha_new);
        const double alpha = parameters[amplitude_index], ratio = alpha / alpha_new;
        const auto projected = support.project(white);
        if (!(ratio > 0) || !std::isfinite(ratio))
            throw std::invalid_argument("Unrepresentable support rescaling");
        RESCALING::Proposal result;
        support.local([&] {
            result = {white, parameters, double(support.rank()) * (std::log(alpha) - std::log(alpha_new))};
            for (size_t j = 0; j < white.size(); ++j) {
                result.white[j] += (ratio - 1) * projected[j];
                if (!std::isfinite(result.white[j]))
                    throw std::invalid_argument("Nonfinite support proposal");
            }
            if (!std::isfinite(result.log_jacobian))
                throw std::invalid_argument("Unrepresentable support Jacobian");
            result.parameters[amplitude_index] = alpha_new;
        });
        return result;
    }
} // namespace ACTIVE_SUPPORT
#endif

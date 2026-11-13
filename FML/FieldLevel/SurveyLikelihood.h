#ifndef FML_FIELD_LEVEL_SURVEY_LIKELIHOOD_H
#define FML_FIELD_LEVEL_SURVEY_LIKELIHOOD_H
#include <FML/FieldLevel/DistributedLPT.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Cartesian cells on the forward model's FFT mesh. Origin labels FFT node
    // (0,0,0) in physical Mpc/h; callbacks can embed angular/radial footprints.
    struct SurveyWindow {
        std::array<double, 3> origin{};
        std::vector<unsigned char> observed;
        Field selection; // Fixed completeness/response in [0,1], not a noise weight.
    };
    struct SurveyVoxel {
        bool observed = true;
        double selection = 1;
    };
    template <class Forward, class Callback>
    SurveyWindow make_survey_window(const Forward & model, std::array<double, 3> origin, Callback callback) {
        bool finite_origin = true;
        for (double x : origin)
            finite_origin &= std::isfinite(x);
        if (model.maximum(finite_origin ? 0 : 1) != 0)
            throw std::invalid_argument("[Survey window] Nonfinite origin");
        model.controls({origin[0], origin[1], origin[2]});
        SurveyWindow result;
        FML::COMMUNICATION::collective_try(
            [&] {
                result.origin = origin;
                result.observed.resize(model.cells());
                result.selection.resize(model.cells());
                const int N = model.configuration().mesh;
                const double dx = model.configuration().boxsize / N;
                for (size_t q = 0; q < model.cells(); ++q) {
                    const std::array<double, 3> position{origin[0] +
                                                             dx * (model.layout().local_start + q / (size_t(N) * N)),
                                                         origin[1] + dx * ((q / N) % N),
                                                         origin[2] + dx * (q % N)};
                    const auto voxel = callback(position);
                    result.observed[q] = voxel.observed;
                    result.selection[q] = voxel.selection;
                }
            },
            "Survey window callback/allocation");
        return result;
    }
    template <class Forward>
    SurveyWindow full_survey_window(const Forward & model) {
        return make_survey_window(model, {0, 0, 0}, [](auto) { return SurveyVoxel{}; });
    }
    // One mutable FFT workspace per observation/chain. Collective on the model's
    // communicator; no global field gather. Cotangents use full-Hermitian weights.
    template <class Forward>
    class CartesianSurveyObservation {
        const Forward * model_;
        SurveyWindow window_;
        mutable FML::GRID::FFTWGrid<3> grid_;

      public:
        CartesianSurveyObservation(const Forward & model, SurveyWindow window)
            : model_(&model), window_(std::move(window)), grid_(model.configuration().mesh) {
            bool finite_origin = true;
            for (double x : window_.origin)
                finite_origin &= std::isfinite(x);
            require(finite_origin, "[Survey observation] Nonfinite origin");
            model.controls({window_.origin[0], window_.origin[1], window_.origin[2]});
            bool ok = window_.observed.size() == model.cells() && window_.selection.size() == model.cells();
            if (ok)
                for (size_t q = 0; q < model.cells(); ++q)
                    ok &= window_.observed[q] <= 1 && std::isfinite(window_.selection[q]) &&
                          window_.selection[q] >= 0 && window_.selection[q] <= 1;
            require(ok, "[Survey observation] Invalid local mask/selection");
        }
        CartesianSurveyObservation(CartesianSurveyObservation &&) = default;
        CartesianSurveyObservation(const CartesianSurveyObservation &) = delete;
        const Forward & model() const { return *model_; }
        const SurveyWindow & window() const { return window_; }
        double cell_volume() const { return model_->volume() / model_->layout().global_cells; }
        void require(bool ok, const char * message) const {
            if (model_->maximum(ok ? 0 : 1) != 0)
                throw std::invalid_argument(message);
        }
        void representable(bool ok) const {
            if (model_->maximum(ok ? 0 : 1) != 0)
                throw UnrepresentableState("[Survey likelihood] Unrepresentable proposal");
        }
        Field synthesize(const Modes & prediction) const {
            bool valid = prediction.size() == model_->modes().size();
            for (auto z : prediction)
                valid &= std::isfinite(z.real()) && std::isfinite(z.imag());
            require(valid, "[Survey observation] Invalid local prediction dimensions/values");
            // Validate Hermitian planes using this observation's existing FFT plan.
            // Do not allocate a second forward-model workspace on every evaluation.
            grid_.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t i = 0; i < prediction.size(); ++i)
                grid_.set_fourier_from_index(model_->modes()[i].fourier_index, FML::GRID::ComplexType(prediction[i]));
            grid_.set_grid_status_real(false);
            grid_.fftw_c2r();
            grid_.fftw_r2c();
            for (size_t i = 0; i < prediction.size(); ++i)
                valid &= std::abs(Complex(grid_.get_fourier_from_index(model_->modes()[i].fourier_index)) -
                                  prediction[i]) <= 1e-10 * std::max(1., std::abs(prediction[i]));
            require(valid, "[Survey observation] Prediction must preserve Hermitian symmetry");
            grid_.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t i = 0; i < prediction.size(); ++i)
                if (model_->modes()[i].k < model_->configuration().likelihood_cutoff)
                    grid_.set_fourier_from_index(model_->modes()[i].fourier_index,
                                                 FML::GRID::ComplexType(prediction[i]));
            grid_.set_grid_status_real(false);
            grid_.fftw_c2r();
            Field result;
            FML::COMMUNICATION::collective_try(
                [&] {
                    result.resize(model_->cells());
                    const int N = model_->configuration().mesh;
                    for (size_t q = 0; q < result.size(); ++q)
                        result[q] = grid_.get_real_from_index(
                            grid_.get_index_real({int(q / (size_t(N) * N)), int(q / N % N), int(q % N)}));
                },
                "Survey synthesis buffer");
            bool ok = true;
            for (double x : result)
                ok &= std::isfinite(x);
            representable(ok);
            return result;
        }
        Modes adjoint(const Field & cotangent) const {
            require(cotangent.size() == model_->cells(), "[Survey observation] Cotangent dimension mismatch");
            bool ok = true;
            for (double x : cotangent)
                ok &= std::isfinite(x);
            representable(ok);
            const int N = model_->configuration().mesh;
            for (size_t q = 0; q < cotangent.size(); ++q)
                grid_.set_real_from_index(grid_.get_index_real({int(q / (size_t(N) * N)), int(q / N % N), int(q % N)}),
                                          cotangent[q]);
            grid_.set_grid_status_real(true);
            grid_.fftw_r2c();
            Modes result;
            FML::COMMUNICATION::collective_try(
                [&] {
                    result.assign(model_->modes().size(), 0);
                    for (size_t i = 0; i < result.size(); ++i)
                        if (model_->modes()[i].k < model_->configuration().likelihood_cutoff)
                            result[i] = double(model_->layout().global_cells) *
                                        Complex(grid_.get_fourier_from_index(model_->modes()[i].fourier_index));
                },
                "Survey adjoint buffer");
            bool finite = true;
            for (auto z : result)
                finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
            representable(finite);
            return result;
        }
    };
    // d_i = selection_i delta_i + epsilon_i on observed Cartesian cells.
    // Var(epsilon_i)=exp(log_noise_power)/Vcell * relative_variance_i.
    template <class Forward>
    class SurveyGaussianLikelihood {
        CartesianSurveyObservation<Forward> observation_;
        Field data_, relative_variance_;
        mutable Field bar_;

      public:
        SurveyGaussianLikelihood(const Forward & model, SurveyWindow window, Field data, Field relative_variance)
            : observation_(model, std::move(window)), data_(std::move(data)),
              relative_variance_(std::move(relative_variance)) {
            model.controls({10});
            bool ok = data_.size() == model.cells() && relative_variance_.size() == model.cells();
            if (ok)
                for (size_t q = 0; q < model.cells(); ++q)
                    if (observation_.window().observed[q])
                        ok &= std::isfinite(data_[q]) && relative_variance_[q] > 0 &&
                              std::isfinite(relative_variance_[q]);
            observation_.require(ok, "[Survey Gaussian] Invalid data/relative variance");
            FML::COMMUNICATION::collective_try([&] { bar_.resize(model.cells()); }, "Survey Gaussian scratch");
        }
        const auto & observation() const { return observation_; }
        Field expected_field(const Modes & prediction) const {
            auto field = observation_.synthesize(prediction);
            const auto & win = observation_.window();
            for (size_t q = 0; q < field.size(); ++q)
                field[q] = win.observed[q] ? win.selection[q] * field[q] : 0;
            return field;
        }
        double evaluate(const Modes & prediction, double log_noise, Modes & gradient, double & noise_gradient) const {
            observation_.representable(std::isfinite(log_noise));
            observation_.model().controls({log_noise});
            const auto mean = expected_field(prediction);
            const auto & win = observation_.window();
            auto & bar = bar_;
            std::fill(bar.begin(), bar.end(), 0);
            double U = 0;
            noise_gradient = 0;
            bool ok = true;
            const double base = std::exp(log_noise) / observation_.cell_volume();
            for (size_t q = 0; q < mean.size(); ++q)
                if (win.observed[q]) {
                    const double variance = base * relative_variance_[q], r = mean[q] - data_[q],
                                 chi = r * r / variance;
                    U += .5 * (chi + std::log(2 * M_PI) + std::log(variance));
                    noise_gradient += .5 * (1 - chi);
                    bar[q] = win.selection[q] * r / variance;
                    ok &= variance > 0 && std::isfinite(variance) && std::isfinite(bar[q]);
                }
            observation_.representable(ok && std::isfinite(U) && std::isfinite(noise_gradient));
            gradient = observation_.adjoint(bar);
            return U; // Local U/noise derivative; posterior reduces once.
        }
    };
    // Explicit positive intensity: lambda_i=exposure_i selection_i exp(delta_i-offset).
    // Exposure is an expected count per cell at unit link. No clipping of delta,
    // no implicit variance renormalization and no Gaussian-noise dependence.
    template <class Forward>
    class SurveyPoissonLikelihood {
        CartesianSurveyObservation<Forward> observation_;
        std::vector<std::uint64_t> counts_;
        Field exposure_, log_factorial_;
        mutable Field bar_;
        double offset_;

      public:
        SurveyPoissonLikelihood(const Forward & model,
                                SurveyWindow window,
                                std::vector<std::uint64_t> counts,
                                Field exposure,
                                double offset = 0)
            : observation_(model, std::move(window)), counts_(std::move(counts)), exposure_(std::move(exposure)),
              offset_(offset) {
            observation_.require(std::isfinite(offset), "[Survey Poisson] Nonfinite offset");
            model.controls({20, offset});
            bool ok = counts_.size() == model.cells() && exposure_.size() == model.cells();
            if (ok)
                for (size_t q = 0; q < model.cells(); ++q)
                    if (observation_.window().observed[q])
                        ok &= std::isfinite(exposure_[q]) && exposure_[q] >= 0 && counts_[q] <= 9007199254740991ULL &&
                              ((exposure_[q] > 0 && observation_.window().selection[q] > 0) || counts_[q] == 0);
            observation_.require(ok, "[Survey Poisson] Invalid counts/exposure or positive count at zero rate");
            FML::COMMUNICATION::collective_try(
                [&] {
                    log_factorial_.resize(counts_.size());
                    bar_.resize(counts_.size());
                    for (size_t q = 0; q < counts_.size(); ++q)
                        log_factorial_[q] = std::lgamma(double(counts_[q]) + 1);
                },
                "Survey Poisson normalization");
        }
        const auto & observation() const { return observation_; }
        Field expected_counts(const Modes & prediction) const {
            auto delta = observation_.synthesize(prediction);
            const auto & win = observation_.window();
            bool ok = true;
            for (size_t q = 0; q < delta.size(); ++q) {
                delta[q] = win.observed[q] && win.selection[q] > 0 && exposure_[q] > 0 ?
                               std::exp(std::log(exposure_[q]) + std::log(win.selection[q]) + delta[q] - offset_) :
                               0;
                ok &= std::isfinite(delta[q]);
            }
            observation_.representable(ok);
            return delta;
        }
        double evaluate(const Modes & prediction, double, Modes & gradient, double & noise_gradient) const {
            const auto delta = observation_.synthesize(prediction);
            const auto & win = observation_.window();
            auto & bar = bar_;
            std::fill(bar.begin(), bar.end(), 0);
            double U = 0;
            bool ok = true;
            noise_gradient = 0;
            for (size_t q = 0; q < delta.size(); ++q)
                if (win.observed[q] && exposure_[q] > 0 && win.selection[q] > 0) {
                    const double log_rate = std::log(exposure_[q]) + std::log(win.selection[q]) + delta[q] - offset_;
                    const double rate = std::exp(log_rate), n = double(counts_[q]);
                    U += rate - n * log_rate + log_factorial_[q];
                    bar[q] = rate - n;
                    ok &= std::isfinite(rate) && std::isfinite(bar[q]);
                }
            observation_.representable(ok && std::isfinite(U));
            gradient = observation_.adjoint(bar);
            return U;
        }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif

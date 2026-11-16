#ifndef FML_FIELD_LEVEL_PARTICLE_MESH_POSTERIOR_H
#define FML_FIELD_LEVEL_PARTICLE_MESH_POSTERIOR_H
#include <FML/FieldLevel/ParticleMeshAdjoint.h>
#include <FML/FieldLevel/ForwardGrid.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Fixed signal response; noise is added afterwards. The CIC reference is
    // a window convention only, not a new data layout or a spatial regridding.
    struct PMObservationSettings {
        FourierResponse response;
        int reference_cic_mesh{0}; // 0: raw selected assignment window; otherwise map to this coarser CIC window.
    };
    // Periodic matter-only Fourier adapter. Particle/white layout is on mesh N;
    // observation modes and grid indices are on density_mesh M, which may differ.
    template <class GridType = FML::GRID::FFTWGrid<3>>
    class PeriodicPMModelT {
      public:
        using Grid = GridType;
        using Dynamics = ParticleMeshForwardT<Grid>;
        using Observation = DistributedLinearModelT<Grid>;
        using Reconstruction = typename Dynamics::Result;
      private:
        Configuration config_;
        Dynamics dynamics_;
        Observation observation_;
        std::vector<Parameter> parameters_;
        PMAdjointOptions reverse_;
        std::vector<double> observation_response_;
        int reference_cic_mesh_{};
        bool observation_response_configured_{};
        static Configuration initial_configuration(Configuration c) {
            // Likelihood support belongs to the force/observation mesh, not the
            // IC generator. This also permits observed nonlinear modes above N's Nyquist.
            c.likelihood_cutoff = c.initial_cutoff;
            return c;
        }
        static Configuration observation_configuration(Configuration c) {
            c.mesh = c.density_mesh;
            c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff;
            c.evolution = Evolution::Linear; c.growth_rate = 0;
            return c;
        }
        void require(bool ok, const char * text) const {
            if (maximum(ok ? 0 : 1) != 0) throw std::invalid_argument(text);
        }
        static bool valid_parameter(const Parameter & p) {
            return !p.name.empty() && std::isfinite(p.initial) && std::isfinite(p.mean) &&
                std::isfinite(p.sigma) && p.sigma > 0 && p.lower < p.upper &&
                p.initial >= p.lower && p.initial <= p.upper &&
                (p.prior == Prior::Gaussian || (p.prior == Prior::Uniform && std::isfinite(p.lower) &&
                  std::isfinite(p.upper) && std::isfinite(p.upper - p.lower)));
        }
        std::vector<double> spectrum_parameters(const std::vector<double> & p) const {
            return {p.begin(), p.end() - 1};
        }
      public:
        struct Workspace {
            friend class PeriodicPMModelT;
          private:
            const PeriodicPMModelT * owner;
            typename Dynamics::Workspace pm;
            Grid grid;
            Modes prediction, cotangent;
            std::vector<double> parameters;
            PMAdjointReport report;
            bool valid{false};
            explicit Workspace(const PeriodicPMModelT * m)
                : owner(m), pm(m->dynamics_.workspace()),
                  grid(FML::GRID::make_grid<Grid>(m->config_.density_mesh, 0, 0, m->domain())) {}
          public:
            Workspace(Workspace &&) = default;
            Workspace & operator=(Workspace &&) = default;
            Workspace(const Workspace &) = delete;
            Modes & cotangent_buffer() { return cotangent; }
            const PMAdjointReport & adjoint_report() const { return report; }
        };
        PeriodicPMModelT(Configuration c, Spectrum spectrum, PMSettings settings = {},
                        Parameter noise = {"log_noise_power", std::log(1000.), std::log(1000.), 2., -30., 40.},
                        PMAdjointOptions reverse = {}, const FML::Domain & domain = FML::domain,
                        PMObservationSettings observed = {})
            : config_(c), dynamics_(initial_configuration(c), std::move(spectrum), std::move(settings), domain),
              observation_(observation_configuration(c), Spectrum::fixed([](double) { return 1.; }), {}, {}, domain),
              reverse_(reverse) {
            require(noise.name == "log_noise_power" && valid_parameter(noise), "[Periodic PM] Invalid noise prior");
            require(reverse_.max_checkpoints <= 32, "[Periodic PM] Invalid reverse checkpoint budget");
            controls({double(reverse_.max_checkpoints), noise.initial, noise.mean, noise.sigma, double(noise.prior),
                      std::isfinite(noise.lower) ? noise.lower : 0., std::isfinite(noise.upper) ? noise.upper : 0.,
                      double(std::isfinite(noise.lower)), double(std::isfinite(noise.upper))});
            FML::COMMUNICATION::collective_try([&] {
                parameters_ = dynamics_.parameters(); parameters_.push_back(std::move(noise));
            }, "Periodic PM parameters");
            controls({double(observed.reference_cic_mesh)});
            require(observed.reference_cic_mesh == 0 ||
                        (observed.reference_cic_mesh >= 4 && observed.reference_cic_mesh % 2 == 0 &&
                         observed.reference_cic_mesh <= c.density_mesh &&
                         c.likelihood_cutoff < M_PI * observed.reference_cic_mesh / c.boxsize),
                    "[Periodic PM] CIC reference must be an even coarser mesh supporting the observed band");
            int custom = !observed.response.multiplier.empty();
            FML::MaxOverTasks(&custom);
            require(!custom || observed.response.multiplier.size() == modes().size(),
                    "[Periodic PM] Custom response needs one value per owned mode on every rank");
            reference_cic_mesh_ = observed.reference_cic_mesh;
            observation_response_configured_ = custom || reference_cic_mesh_ != 0;
            FML::COMMUNICATION::collective_try([&] {
                observation_response_ = std::move(observed.response.multiplier);
                if (!custom) observation_response_.assign(modes().size(), 1.);
                if (reference_cic_mesh_) for (size_t i = 0; i < modes().size(); ++i) {
                    double coarse = 1., fine = 1.;
                    for (int k : modes()[i].n) if (k) {
                        const double x = M_PI * k / reference_cic_mesh_, y = M_PI * k / c.density_mesh;
                        coarse *= std::pow(std::sin(x) / x, 2);
                        fine *= std::pow(std::sin(y) / y, 2);
                        if (dynamics_.settings().assignment == PMAssignment::TSC) fine *= std::sin(y) / y;
                    }
                    observation_response_[i] *= coarse / fine;
                }
            }, "Periodic PM observation response");
            bool finite = true; for (double x : observation_response_) finite &= std::isfinite(x);
            require(finite, "[Periodic PM] Nonfinite observation response");
            if (observation_response_configured_) {
                Modes response_modes;
                FML::COMMUNICATION::collective_try([&] {
                    response_modes.assign(observation_response_.begin(), observation_response_.end());
                }, "Periodic PM response validation storage");
                observation_.validate_modes(response_modes);
            }
        }
        const Configuration & configuration() const { return config_; }
        const Dynamics & dynamics() const { return dynamics_; }
        const PMAdjointOptions & adjoint_options() const { return reverse_; }
        const auto & observation_response() const { return observation_response_; }
        bool observation_response_configured() const { return observation_response_configured_; }
        int reference_cic_mesh() const { return reference_cic_mesh_; }
        const auto & domain() const { return dynamics_.domain(); }
        const auto & modes() const { return observation_.modes(); }
        const auto & parameters() const { return parameters_; }
        auto layout() const { return dynamics_.layout(); }
        auto observation_layout() const { return observation_.layout(); }
        size_t cells() const { return dynamics_.cells(); }
        double volume() const { return observation_.volume(); }
        double sum(double x) const { return dynamics_.sum(x); }
        double maximum(double x) const { return dynamics_.maximum(x); }
        void controls(const std::vector<double> & values) const {
            bool finite = true; for (double x : values) finite &= std::isfinite(x);
            require(finite, "[Periodic PM] Nonfinite replicated controls"); observation_.controls(values);
        }
        void validate_state(const Field & s, const std::vector<double> & p) const {
            require(s.size() == cells() && p.size() == parameters_.size(), "[Periodic PM] State dimension mismatch");
        }
        bool in_support(const std::vector<double> & p) const {
            require(p.size() == parameters_.size(), "[Periodic PM] Parameter dimension mismatch");
            bool valid = true;
            for (size_t j = 0; j < p.size(); ++j)
                valid &= std::isfinite(p[j]) && p[j] >= parameters_[j].lower && p[j] <= parameters_[j].upper;
            if (maximum(valid ? 0 : 1) != 0) return false;
            controls(p); return true;
        }
        void validate_modes(const Modes & data) const { observation_.validate_modes(data); }
        Field white_noise(std::uint64_t seed) const { return dynamics_.white_noise(seed); }
        std::vector<double> initial_parameters() const {
            std::vector<double> p; for (const auto & parameter : parameters_) p.push_back(parameter.initial); return p;
        }
        Workspace workspace() const { return Workspace(this); }
        const Modes & predict(const Field & white, const std::vector<double> & p, Workspace & w) const {
            require(w.owner == this, "[Periodic PM] Foreign/moved workspace"); w.valid = false;
            validate_state(white, p); require(in_support(p), "[Periodic PM] Parameters outside support");
            const auto & r = dynamics_.evolve(white, spectrum_parameters(p), w.pm);
            size_t q = 0;
            for (auto i : w.grid.get_real_range()) w.grid.set_real_from_index(i, r.matter_density[q++]);
            w.grid.set_grid_status_real(true); w.grid.fftw_r2c();
            FML::COMMUNICATION::collective_try([&] {
                w.prediction.resize(modes().size()); w.parameters = p;
                for (size_t i = 0; i < modes().size(); ++i)
                    w.prediction[i] = w.grid.get_fourier_from_index(modes()[i].fourier_index);
                if (observation_response_configured_)
                    for (size_t i = 0; i < modes().size(); ++i) w.prediction[i] *= observation_response_[i];
            }, "Periodic PM prediction");
            if (observation_response_configured_) {
                bool finite = true; for (auto z : w.prediction) finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
                if (maximum(finite ? 0 : 1) != 0) throw UnrepresentableState("[Periodic PM] Observation response overflow");
            }
            w.valid = true; return w.prediction;
        }
        void pullback(const std::vector<double> & p, const Modes & g, Workspace & w,
                      Field & white_gradient, std::vector<double> & scalar_gradient) const {
            require(w.owner == this && w.valid && w.parameters == p && g.size() == modes().size(),
                    "[Periodic PM] Pullback needs matching forward and local modes");
            bool finite = true; for (auto z : g) finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
            if (maximum(finite ? 0 : 1) != 0) throw UnrepresentableState("[Periodic PM] Nonfinite Fourier cotangent");
            // Full-Hermitian Fourier pairing -> Euclidean real-cell pairing.
            // The normalized forward FFT contributes 1/M^3; inverse covers the weights.
            w.grid.fill_fourier_grid(0.); w.grid.set_grid_status_real(false);
            const double mesh_cells = observation_layout().global_cells;
            for (size_t i = 0; i < g.size(); ++i)
                w.grid.set_fourier_from_index(modes()[i].fourier_index, FML::GRID::ComplexType((observation_response_configured_ ? g[i] * observation_response_[i] : g[i]) / mesh_cells));
            w.grid.fftw_c2r();
            PMCotangent cotangent;
            FML::COMMUNICATION::collective_try([&] {
                cotangent.matter_density.resize(observation_layout().owned_cells);
                size_t q = 0;
                for (auto i : w.grid.get_real_range()) cotangent.matter_density[q++] = w.grid.get_real_from_index(i);
            }, "Periodic PM real cotangent");
            ParticleMeshAdjointT<Grid> adjoint(dynamics_);
            auto result = adjoint.pullback(cotangent, w.pm, reverse_);
            white_gradient = std::move(result.white); w.report = result.report;
            FML::COMMUNICATION::collective_try([&] {
                scalar_gradient.assign(p.size(), 0.);
                // Raw PM spectrum derivatives are global. BasicPosterior reduces
                // parameter contributions once, so supply these on rank zero only.
                if (FML::ThisTask == 0)
                    std::copy(result.spectrum.begin(), result.spectrum.end(), scalar_gradient.begin());
            }, "Periodic PM scalar contributions");
        }
        Field apply_field_metric(const Field & field, const std::vector<double> & p,
                                 Workspace & w, double exponent) const {
            require(w.owner == this, "[Periodic PM] Foreign metric workspace"); validate_state(field, p);
            require(in_support(p), "[Periodic PM] Metric parameters outside support"); controls({exponent});
            // Identity in whitened coordinates. No calibrated PM Fisher approximation.
            return field;
        }
        Modes mock(const Field & white, const std::vector<double> & p, std::uint64_t seed, Workspace & w) const {
            auto data = predict(white, p, w);
            auto noise_work = observation_.workspace(); auto noise_p = observation_.initial_parameters(); noise_p[0] = 1.;
            const auto noise = observation_.predict(observation_.white_noise(seed, 0x504d4e53u), noise_p, noise_work);
            const double amplitude = std::sqrt(std::exp(p.back()));
            if (!std::isfinite(amplitude)) throw UnrepresentableState("[Periodic PM] Unrepresentable mock noise");
            for (size_t i = 0; i < data.size(); ++i) data[i] += amplitude * noise[i];
            return data;
        }
        const Reconstruction & reconstruction(const Workspace & w) const {
            require(w.owner == this && w.valid, "[Periodic PM] Reconstruction needs successful forward");
            return w.pm.result();
        }
        // Checkpoint support is explicit in ParticleMeshCheckpoint.h. A generic
        // response() would omit PM physics and is deliberately not exposed.
    };
    template <class Grid>
    struct Execution<PeriodicPMModelT<Grid>> {
        using Forward = PeriodicPMModelT<Grid>;
        static void validate(const Forward & m, const Field & s, const std::vector<double> & p) { m.validate_state(s, p); }
        static void map_controls(const Forward & m, bool vary, int steps, double tolerance) {
            m.controls({double(vary), double(steps), tolerance});
        }
        static void hmc_controls(const Forward & m, bool vary, const FML::MCMC::HMCOptions & o) {
            m.controls({double(vary), double(o.warmup), double(o.samples), double(o.n_leapfrog), double(o.jitter),
                o.step_size, double(o.adapt_step_size), o.target_acceptance, double(o.adapt_mass),
                o.divergence_threshold, double(o.seed >> 32), double(o.seed & 0xffffffffu), double(o.verbose_every)});
        }
        static double sum(const Forward & m, double x) { return m.sum(x); }
        static double maximum(const Forward & m, double x) { return m.maximum(x); }
        template <class Likelihood>
        static Likelihood likelihood(const Forward & m, Modes data) {
            static_assert(std::is_same_v<Likelihood, GaussianFieldLikelihood>, "Periodic PM adapter requires Fourier Gaussian likelihood");
            return {m.modes(), m.configuration().likelihood_cutoff, m.volume(), std::move(data),
                    [&](const Modes & values) { m.validate_modes(values); }};
        }
        static FML::MCMC::HMCExecution hmc(const Forward & m) {
            return {[&m](double x) { return m.sum(x); },
                    [&m](Field & x, std::uint64_t seed, std::uint64_t iteration) {
                        for (size_t q = 0; q < x.size(); ++q) {
                            FML::RANDOM::COUNTER_BASED::Stream rng(seed, m.layout().global_cell_id(q), 0x484d434du);
                            FML::RANDOM::COUNTER_BASED::Stream step(rng.next64(), iteration, 0x53544550u);
                            x[q] = step.normal();
                        }
                    }, [&m](bool valid) { return m.maximum(valid ? 0 : 1) == 0; }};
        }
    };
    // Cartesian survey adapters assume one mesh for both input cells and observed
    // modes. Keep that factory disabled; PM checkpoints describe the two layouts separately.
    namespace DETAIL {
        template <class GridType>
        struct ForwardGrid<PeriodicPMModelT<GridType>> {
            using Grid = GridType;
            static Grid make(const PeriodicPMModelT<Grid> &) = delete;
        };
    }
    using PeriodicPMModel = PeriodicPMModelT<>;
    template <class Grid>
    using PeriodicPMPosteriorT = BasicPosterior<PeriodicPMModelT<Grid>>;
    using PeriodicPMPosterior = PeriodicPMPosteriorT<FML::GRID::FFTWGrid<3>>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif

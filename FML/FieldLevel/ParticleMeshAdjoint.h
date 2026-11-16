#ifndef FML_FIELD_LEVEL_PARTICLE_MESH_ADJOINT_H
#define FML_FIELD_LEVEL_PARTICLE_MESH_ADJOINT_H
#include <FML/FieldLevel/ParticleMesh.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Euclidean cotangents of compact owned output arrays. Empty fields mean zero.
    // Wrapped position cotangents can be supplied as displacement cotangents away
    // from wrap discontinuities. Scalar diagnostics are not differentiable outputs.
    struct PMCotangent {
        Field initial_density, matter_density;
        std::array<Field, 3> displacement, momentum;
    };
    struct PMAdjointOptions {
        size_t max_checkpoints{4}; // Additional retained midpoint states beyond the initial state.
    };
    struct PMAdjointReport {
        size_t replayed_steps{}, force_evaluations{}, peak_checkpoint_states{}, checkpoint_payload_bytes{};
    };
    struct PMGradient {
        Field white;
        std::vector<double> spectrum; // Replicated GLOBAL derivatives; do not MPI-sum these again.
        PMAdjointReport report;
    };

    template <class GridType = FML::GRID::FFTWGrid<3>>
    class ParticleMeshAdjointT {
        using Model = ParticleMeshForwardT<GridType>;
        using Grid = GridType;
        using Work = typename Model::Workspace;
        using Fields = std::array<Field, 3>;
        struct State { Fields displacement, momentum; };
        const Model & model;
        void require(bool ok, const char * text) const { model.require(ok, text); }
        static void copy_state(const State & source, Work & w) {
            FML::COMMUNICATION::collective_try([&] {
                w.output.displacement = source.displacement;
                w.output.momentum = source.momentum;
            }, "PM adjoint replay state");
        }
        static State checkpoint(const Work & w, PMAdjointReport & report, size_t & live) {
            State result;
            FML::COMMUNICATION::collective_try([&] {
                result.displacement = w.output.displacement;
                result.momentum = w.output.momentum;
            }, "PM adjoint checkpoint");
            report.peak_checkpoint_states = std::max(report.peak_checkpoint_states, ++live);
            return result;
        }
        template <class Visit>
        void derivative_stencil(const Work & w, size_t q, Visit visit) const {
            const int n = model.configuration().density_mesh;
            auto x = w.output.position(q);
            if (model.settings_.assignment == PMAssignment::TSC) {
                DETAIL::tsc_stencil(n, x, [&](auto node, double, const auto & dw) { visit(node, dw); });
                return;
            }
            int base[3]; double f[3];
            for (int d = 0; d < 3; ++d) {
                const double u = x[d] * n;
                base[d] = int(std::floor(u)); f[d] = u - base[d];
            }
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    for (int c = 0; c < 2; ++c) {
                        const int bit[3] = {a, b, c};
                        double weight[3];
                        for (int d = 0; d < 3; ++d) weight[d] = bit[d] ? f[d] : 1 - f[d];
                        std::array<double, 3> dw;
                        for (int d = 0; d < 3; ++d)
                            dw[d] = n * (bit[d] ? 1. : -1.) * weight[(d + 1) % 3] * weight[(d + 2) % 3];
                        const auto node = (std::uint64_t((base[0] + a) % n) * n + (base[1] + b) % n) * n +
                                          (base[2] + c) % n;
                        visit(node, dw);
                    }
        }
        void forward_step(size_t j, Work & w, PMAdjointReport & report) const {
            model.density(w); model.force(w); ++report.force_evaluations;
            model.kick(w, model.steps_[j].first_kick);
            for (int d = 0; d < 3; ++d)
                for (size_t q = 0; q < model.cells(); ++q)
                    w.output.displacement[d][q] += model.steps_[j].drift * w.output.momentum[d][q];
            model.finite(w.output);
            model.density(w); model.force(w); ++report.force_evaluations;
            model.kick(w, model.steps_[j].last_kick);
            model.finite(w.output);
            ++report.replayed_steps;
        }
        void replay(size_t begin, size_t end, const State & start, Work & w, PMAdjointReport & report) const {
            copy_state(start, w);
            for (size_t j = begin; j < end; ++j) forward_step(j, w, report);
        }
        // Transpose of the selected mass deposition, with the current particle geometry.
        // Routing is rebuilt at every geometry (particles need not remain on their owners).
        void deposition_pullback(Work & w, const Field & density_bar, Fields & position_bar) const {
            model.density(w);
            size_t q = 0;
            for (auto i : w.density.get_real_range())
                w.density.set_real_from_index(i, density_bar.empty() ? 0. : density_bar[q++]);
            const auto nodes = w.assignment.reverse(w.density);
            const double ratio = double(model.configuration().density_mesh) / model.configuration().mesh;
            for (size_t p = 0; p < model.cells(); ++p)
                derivative_stencil(w, p, [&](auto node, const auto & dw) {
                    const double value = ratio * ratio * ratio * w.assignment.at(nodes, node);
                    for (int d = 0; d < 3; ++d) position_bar[d][p] += value * dw[d];
                });
        }
        // position_bar += [d F(x)/dx]^T * (kick_integral * momentum_bar).
        // Includes BOTH the gather's position dependence and the source deposition.
        void force_pullback(Work & w, double integral, const Fields & momentum_bar,
                            Fields & position_bar, PMAdjointReport & report) const {
            model.density(w); model.force(w); ++report.force_evaluations;
            for (int c = 0; c < 3; ++c) {
                const auto nodes = w.assignment.reverse(w.force[c]);
                for (size_t q = 0; q < model.cells(); ++q)
                    derivative_stencil(w, q, [&](auto node, const auto & dw) {
                        const double value = integral * momentum_bar[c][q] * w.assignment.at(nodes, node);
                        for (int d = 0; d < 3; ++d) position_bar[d][q] += value * dw[d];
                    });
                // Same keys/geometry as mass deposition, even for zero cotangents:
                // all assignment nodes, including zero-weight nodes, are visited, so the routing stays compatible.
                w.assignment.deposit(w.force[c], [&](auto add) {
                    for (size_t q = 0; q < model.cells(); ++q)
                        model.stencil(w.output, q, [&](auto node, double weight) {
                            add(node, weight * integral * momentum_bar[c][q]);
                        });
                });
                w.force[c].set_grid_status_real(true);
            }
            // Each G_c is skew-adjoint on the real mesh: G_c^T = -G_c.
            // Normalized forward FFT + unnormalized inverse introduce no N factor.
            w.density.fill_fourier_grid(0.); w.density.set_grid_status_real(false);
            const int n = model.configuration().density_mesh;
            for (int c = 0; c < 3; ++c) {
                w.force[c].fftw_r2c();
                for (auto i : w.density.get_fourier_range()) {
                    auto k = w.density.get_fourier_integer_wavevector_from_index(i);
                    double square = 0; for (int v : k) square += double(v) * v;
                    if (square == 0 || std::abs(k[c]) == n / 2) continue;
                    const auto value = w.force[c].get_fourier_from_index(i) *
                                       FML::GRID::ComplexType(0, model.force_coefficient(k, square, c, true));
                    w.density.set_fourier_from_index(i, w.density.get_fourier_from_index(i) + value);
                }
            }
            w.density.fftw_c2r();
            const auto nodes = w.assignment.reverse(w.density);
            const double ratio = double(n) / model.configuration().mesh;
            for (size_t q = 0; q < model.cells(); ++q)
                derivative_stencil(w, q, [&](auto node, const auto & dw) {
                    const double value = ratio * ratio * ratio * w.assignment.at(nodes, node);
                    for (int d = 0; d < 3; ++d) position_bar[d][q] += value * dw[d];
                });
        }
        void reverse_step(size_t j, Work & w, Fields & gx, Fields & gp, PMAdjointReport & report) const {
            Fields old_position;
            FML::COMMUNICATION::collective_try([&] { old_position = w.output.displacement; },
                                              "PM adjoint step positions");
            forward_step(j, w, report);
            force_pullback(w, model.steps_[j].last_kick, gp, gx, report);
            for (int d = 0; d < 3; ++d)
                for (size_t q = 0; q < model.cells(); ++q) gp[d][q] += model.steps_[j].drift * gx[d][q];
            w.output.displacement = std::move(old_position);
            force_pullback(w, model.steps_[j].first_kick, gp, gx, report);
        }
        // Bounded binary subdivision. With no free midpoint slots, replay prefixes
        // from the nearest retained start. No full state or mesh tape is stored.
        void reverse_interval(size_t begin, size_t end, const State & start, size_t slots,
                              Work & w, Fields & gx, Fields & gp, PMAdjointReport & report, size_t & live) const {
            if (end - begin == 1) {
                copy_state(start, w); reverse_step(begin, w, gx, gp, report); return;
            }
            if (slots == 0) {
                for (size_t j = end; j-- > begin;) {
                    replay(begin, j, start, w, report);
                    reverse_step(j, w, gx, gp, report);
                }
                return;
            }
            const size_t mid = begin + (end - begin) / 2;
            replay(begin, mid, start, w, report);
            {
                const auto saved = checkpoint(w, report, live);
                reverse_interval(mid, end, saved, slots - 1, w, gx, gp, report, live);
            }
            --live;
            reverse_interval(begin, mid, start, slots, w, gx, gp, report, live);
        }
      public:
        explicit ParticleMeshAdjointT(const Model & forward) : model(forward) {}
        PMGradient pullback(const PMCotangent & cotangent, typename Model::Workspace & forward,
                            PMAdjointOptions options = {}) const {
            require(forward.owner == &model && forward.valid, "[PM adjoint] Requires successful matching forward");
            require(options.max_checkpoints <= 32, "[PM adjoint] max_checkpoints must be <= 32");
            model.linear_.controls({double(options.max_checkpoints)});
            require(model.cells() <= std::numeric_limits<size_t>::max() /
                        ((options.max_checkpoints + 1) * 6 * sizeof(double)), "[PM adjoint] Checkpoint byte overflow");
            auto valid_field = [](const Field & field, size_t n) {
                if (!field.empty() && field.size() != n) return false;
                for (double x : field) if (!std::isfinite(x)) return false;
                return true;
            };
            bool valid = valid_field(cotangent.initial_density, model.cells()) &&
                         valid_field(cotangent.matter_density, forward.output.density_layout.owned_cells);
            for (int d = 0; d < 3; ++d)
                valid &= valid_field(cotangent.displacement[d], model.cells()) && valid_field(cotangent.momentum[d], model.cells());
            require(valid, "[PM adjoint] Invalid local cotangent dimensions/values");
            auto w = model.workspace();
            w.output.particle_layout = forward.output.particle_layout;
            w.output.density_layout = forward.output.density_layout;
            Fields gx, gp;
            FML::COMMUNICATION::collective_try([&] {
                for (int d = 0; d < 3; ++d) {
                    gx[d] = cotangent.displacement[d].empty() ? Field(model.cells(), 0.) : cotangent.displacement[d];
                    gp[d] = cotangent.momentum[d].empty() ? Field(model.cells(), 0.) : cotangent.momentum[d];
                }
            }, "PM adjoint cotangents");
            auto parameters = model.linear_.initial_parameters();
            std::copy(forward.input_parameters.begin(), forward.input_parameters.end(), parameters.begin());
            const double a = model.settings_.scale_factors.front(), D = model.settings_.background.growth(a);
            const double velocity = a * a * model.settings_.background.E(a) * model.settings_.background.growth_rate(a);
            {
                auto initial = model.linear_.reconstruct(forward.input_white, parameters, w.initial, 0.);
                w.output.displacement = std::move(initial.displacement);
                FML::COMMUNICATION::collective_try([&] {
                    for (int d = 0; d < 3; ++d) {
                        w.output.momentum[d].resize(model.cells());
                        for (size_t q = 0; q < model.cells(); ++q) {
                            w.output.displacement[d][q] *= D;
                            w.output.momentum[d][q] = velocity * w.output.displacement[d][q];
                        }
                    }
                }, "PM adjoint initial particles");
            }
            PMGradient result;
            size_t live = 0;
            const auto start = checkpoint(w, result.report, live);
            // The final matter field is the selected deposition (the final FFT roundtrip is identity).
            FML::COMMUNICATION::collective_try([&] {
                w.output.displacement = forward.output.displacement;
            }, "PM adjoint final geometry");
            deposition_pullback(w, cotangent.matter_density, gx);
            reverse_interval(0, model.steps_.size(), start, options.max_checkpoints,
                             w, gx, gp, result.report, live);
            result.report.checkpoint_payload_bytes = result.report.peak_checkpoint_states * 6 * model.cells() * sizeof(double);

            // Reduce the IC displacement cotangent to an IC density cotangent in
            // Fourier space, then use the existing spectrum/whitening pullback.
            Grid grid = FML::GRID::make_grid<Grid>(model.configuration().mesh, 0, 0, model.domain());
            Modes modes;
            FML::COMMUNICATION::collective_try([&] { modes.assign(model.linear_.modes().size(), Complex(0)); },
                                              "PM adjoint initial Fourier cotangent");
            auto transform = [&](const Field & field) {
                size_t q = 0;
                for (auto i : grid.get_real_range()) grid.set_real_from_index(i, field.empty() ? 0. : field[q++]);
                grid.set_grid_status_real(true); grid.fftw_r2c();
            };
            transform(cotangent.initial_density);
            for (size_t i = 0; i < modes.size(); ++i)
                modes[i] = grid.get_fourier_from_index(model.linear_.modes()[i].fourier_index);
            for (int d = 0; d < 3; ++d) {
                for (size_t q = 0; q < model.cells(); ++q) gx[d][q] = D * (gx[d][q] + velocity * gp[d][q]);
                transform(gx[d]);
                for (size_t i = 0; i < modes.size(); ++i) {
                    const auto & k = model.linear_.modes()[i].n;
                    double square = 0; for (int v : k) square += double(v) * v;
                    modes[i] += Complex(0, -k[d] / (2 * M_PI * square)) *
                                Complex(grid.get_fourier_from_index(model.linear_.modes()[i].fourier_index));
                }
            }
            const double cells = model.layout().global_cells;
            for (auto & z : modes) z *= cells;
            parameters[model.parameters_.size()] = 1.; // Unit density transfer; bias is not a PM input.
            model.linear_.predict(forward.input_white, parameters, forward.initial);
            std::vector<double> local;
            model.linear_.pullback(parameters, modes, forward.initial, result.white, local);
            FML::COMMUNICATION::collective_try([&] { result.spectrum.resize(model.parameters_.size()); },
                                              "PM adjoint spectrum derivative");
            for (size_t j = 0; j < result.spectrum.size(); ++j) result.spectrum[j] = model.sum(local[j]);
            valid = true;
            for (double x : result.white) valid &= std::isfinite(x);
            for (double x : result.spectrum) valid &= std::isfinite(x);
            if (model.maximum(valid ? 0 : 1) != 0)
                throw UnrepresentableState("[PM adjoint] Unrepresentable gradient");
            return result;
        }
    };
    using ParticleMeshAdjoint = ParticleMeshAdjointT<>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif

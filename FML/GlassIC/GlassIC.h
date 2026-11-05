#ifndef FML_GLASS_IC_H
#define FML_GLASS_IC_H
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <FML/RandomFields/CounterBasedRandomField.h>
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/ShortRangeGravity/LongRangePM.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace FML::GLASSIC {
    using Grid = FML::GRID::FFTWGrid<3>;
    using Vector = std::array<double, 3>;
    struct Particle {
        Vector position{}, velocity{}, q{};
        std::uint64_t id{};
        double * get_pos() { return position.data(); }
        double * get_vel() { return velocity.data(); }
        double * get_q() { return q.data(); }
        std::uint64_t get_id() const { return id; }
        constexpr int get_ndim() const { return 3; }
    };
    using Particles = FML::PARTICLE::MPIParticles<Particle>;
    struct Load {
        Particles particles;
        std::uint64_t seed{};
        double boxsize{1}; // Physical comoving Mpc/h metadata; positions always in [0,1).
        bool perturbed{false};
    };
    namespace detail {
        inline void require(bool valid, const char * message) {
            int bad = !valid;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        template <class T>
        inline void same(T value, const char * message) {
            T lo = value, hi = value;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, message);
        }
        inline double wrap(double x) { return x - std::floor(x); }
        inline void sort(Particles & p) {
            auto & v = p.get_particles();
            std::sort(v.begin(), v.begin() + p.get_npart(), [](const auto & a, const auto & b) { return a.id < b.id; });
        }
        // MPIParticles requires reserved receive capacity. Count destinations first, then grow only local storage.
        inline void migrate(Particles & p) {
#ifdef USE_MPI
            const auto upper = FML::GatherFromTasks(&FML::xmax_domain);
            std::vector<unsigned long long> send(FML::NTasks, 0), recv(FML::NTasks, 0);
            for (size_t i = 0; i < p.get_npart(); ++i) {
                const auto x = p.get_particles_ptr()[i].position[0];
                const auto owner = std::upper_bound(upper.begin(), upper.end(), x) - upper.begin();
                ++send.at(size_t(owner));
            }
            MPI_Alltoall(send.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         recv.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
            size_t incoming = 0;
            bool valid = true;
            for (auto n : recv) {
                if (n > p.get_particles().max_size() - incoming) {
                    valid = false;
                    break;
                }
                incoming += size_t(n);
            }
            require(valid, "[GlassIC] Migration exceeds local storage range");
            if (p.get_particles().size() < incoming)
                p.get_particles().resize(incoming);
#endif
            p.communicate_particles();
            sort(p);
        }
        inline void check_grid(Grid & grid) {
            require(grid.get_nmesh() > 0 && grid.get_local_nx() > 0 && grid.get_n_extra_slices_left() >= 1 &&
                        grid.get_n_extra_slices_right() >= 1 &&
                        double(grid.get_local_x_start()) / grid.get_nmesh() == FML::xmin_domain &&
                        double(grid.get_local_x_start() + grid.get_local_nx()) / grid.get_nmesh() == FML::xmax_domain,
                    "[GlassIC] Need nonempty FFT slabs matching particle domains and one halo on each side");
        }
        inline void check_load(Load & load) {
            same(load.seed, "[GlassIC] Seed differs between ranks");
            same(load.boxsize, "[GlassIC] Box metadata differs between ranks");
            same(int(load.perturbed), "[GlassIC] Load state differs between ranks");
            bool good = load.particles.get_npart_total() > 0 && std::isfinite(load.boxsize) && load.boxsize > 0;
            for (size_t i = 0; i < load.particles.get_npart(); ++i) {
                const auto & p = load.particles.get_particles_ptr()[i];
                good &= p.id < load.particles.get_npart_total();
                for (int d = 0; d < 3; ++d)
                    good &= std::isfinite(p.position[d]) && p.position[d] >= 0 && p.position[d] < 1 &&
                            std::isfinite(p.velocity[d]) && std::isfinite(p.q[d]) && p.q[d] >= 0 && p.q[d] < 1;
                good &= p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain;
            }
            require(good, "[GlassIC] Invalid load, IDs, coordinates or ownership");
        }
    } // namespace detail
    // ID-based Poisson realization: no global particle array, independent of task/thread counts.
    inline Load poisson_load(std::uint64_t count, std::uint64_t seed, double boxsize = 1) {
        detail::same(count, "[GlassIC] Counts differ between ranks");
        detail::same(seed, "[GlassIC] Seeds differ between ranks");
        detail::same(boxsize, "[GlassIC] Box sizes differ between ranks");
        detail::require(count > 0 && count <= std::numeric_limits<size_t>::max() && boxsize > 0 &&
                            std::isfinite(boxsize),
                        "[GlassIC] Invalid count or boxsize");
        const auto tasks = std::uint64_t(FML::NTasks), rank = std::uint64_t(FML::ThisTask);
        const auto start = (count / tasks) * rank + std::min(count % tasks, rank);
        const auto n = count / tasks + (rank < count % tasks);
        detail::require(n <= std::vector<Particle>().max_size(), "[GlassIC] Local seed count exceeds storage range");
        std::vector<Particle> local(static_cast<size_t>(n));
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < local.size(); ++i) {
            auto & p = local[i];
            p.id = start + i;
            FML::RANDOM::COUNTER_BASED::Stream rng(seed, p.id, 0x474c4153u);
            for (int d = 0; d < 3; ++d)
                p.q[d] = p.position[d] = rng.uniform();
        }
        Load load;
        load.seed = seed;
        load.boxsize = boxsize;
        load.particles.move_from(std::move(local));
        detail::migrate(load.particles);
        return load;
    }
    struct Options {
        // mesh = 0: automatic, the smallest multiple of the task count (and of 2) >= 3 N^(1/3). With the split scale
        // tied to the mesh (split_cells/mesh) a coarse mesh makes the short-range pair work dominate: for N = 32^3
        // one force evaluation took 1.2 s with mesh 2 N^(1/3), 0.43 s with 3 N^(1/3) and 0.26 s with 4 N^(1/3)
        // (1 thread), at 4 x 8 bytes x mesh^3 of grid memory
        int mesh{0}, max_steps{1000}, min_steps{20}, stable_steps{5};
        double timestep{.1}, damping{2}, max_step_spacing{.05};
        // cutoff_tolerance 1e-4: the forces differ from those with 1e-6 by ~2e-6 of the rms force, ~1.6x faster
        double split_cells{1.25}, softening_spacing{.1}, cutoff_tolerance{1e-4};
        double force_tolerance{1e-3}, velocity_tolerance{1e-3}; // RMS vector norms / mean spacing.
    };
    struct Residuals {
        double force_rms{}, force_max{}, velocity_rms{}, velocity_max{}; // Box-coordinate units.
    };
    enum class Status { Converged, MaximumSteps, NonFinite };
    struct Result {
        Status status{Status::MaximumSteps};
        int steps{};
        Residuals residuals;
        std::vector<Residuals> history; // Initial state and every completed drift.
    };
    inline Residuals residuals(Load & load, const std::vector<Vector> & force) {
        detail::require(force.size() == load.particles.get_npart(), "[GlassIC] Force size mismatch");
        double f2 = 0, v2 = 0, fmax = 0, vmax = 0;
        int bad = 0;
        for (size_t i = 0; i < force.size(); ++i) {
            double a = 0, b = 0;
            for (int d = 0; d < 3; ++d) {
                a += force[i][d] * force[i][d];
                b += std::pow(load.particles.get_vel(i, d), 2);
            }
            if (!std::isfinite(a) || !std::isfinite(b))
                bad = 1;
            f2 += a;
            v2 += b;
            fmax = std::max(fmax, a);
            vmax = std::max(vmax, b);
        }
        FML::MaxOverTasks(&bad);
        FML::SumOverTasks(&f2);
        FML::SumOverTasks(&v2);
        FML::MaxOverTasks(&fmax);
        FML::MaxOverTasks(&vmax);
        if (bad || !std::isfinite(f2) || !std::isfinite(v2))
            return {NAN, NAN, NAN, NAN};
        const double n = double(load.particles.get_npart_total());
        return {std::sqrt(f2 / n), std::sqrt(fmax), std::sqrt(v2 / n), std::sqrt(vmax)};
    }
    // Persistent four-grid PM workspace. Short-range cells/ghosts rebuild after each drift.
    class Force {
        Grid density;
        std::array<Grid, 3> psi;
        FML::SHORT_RANGE_GRAVITY::Kernel kernel;
        std::uint64_t count;

      public:
        Force(int mesh,
              std::uint64_t n,
              double split_cells = 1.25,
              double softening_spacing = .1,
              double tolerance = 1e-6)
            : count(n) {
            detail::same(mesh, "[GlassIC] Mesh differs between ranks");
            detail::same(n, "[GlassIC] Particle count differs between ranks");
            detail::same(split_cells, "[GlassIC] Split differs between ranks");
            detail::same(softening_spacing, "[GlassIC] Softening differs between ranks");
            detail::same(tolerance, "[GlassIC] Cutoff differs between ranks");
            detail::require(mesh >= 4 && mesh % FML::NTasks == 0 && n > 0 && std::isfinite(split_cells) &&
                                split_cells > 0 && std::isfinite(softening_spacing) && softening_spacing > 0 &&
                                std::isfinite(tolerance) && tolerance > 0 && tolerance < 1,
                            "[GlassIC] Invalid force settings");
            FML::SHORT_RANGE_GRAVITY::Options o;
            o.split_scale = split_cells / mesh;
            o.softening_support = softening_spacing / std::cbrt(double(n));
            o.cutoff_tolerance = tolerance;
            o.coupling = 1 / (4 * std::acos(-1.) * double(n));
            // Kernel construction is identical on all ranks after the collective settings checks.
            kernel = FML::SHORT_RANGE_GRAVITY::Kernel(o);
            density = Grid(mesh, 1, 1);
            detail::check_grid(density);
            for (auto & g : psi)
                g = Grid(mesh, 1, 1);
        }
        std::vector<Vector> operator()(Load & load) {
            detail::check_load(load);
            detail::require(!load.perturbed && load.particles.get_npart_total() == count,
                            "[GlassIC] Force needs an unperturbed load with the configured count");
            FML::INTERPOLATION::particles_to_grid<3, Particle>(load.particles.get_particles_ptr(),
                                                               load.particles.get_npart(),
                                                               size_t(count),
                                                               density,
                                                               "CIC");
            density.fftw_r2c();
            FML::SHORT_RANGE_GRAVITY::apply_long_range_filter(density, kernel);
            // phi=delta/k^2, so grad(phi) is attractive. Undo the two CIC windows before interpolation.
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int slab = 0; slab < density.get_local_nx(); ++slab)
                for (auto i : density.get_fourier_range(slab, slab + 1)) {
                    const auto k = density.get_fourier_wavevector_from_index(i);
                    double k2 = 0, window = 1;
                    for (int d = 0; d < 3; ++d) {
                        k2 += k[d] * k[d];
                        const double x = k[d] / (2 * density.get_nmesh());
                        const double sinc = x == 0 ? 1 : std::sin(x) / x;
                        window *= sinc * sinc;
                    }
                    density.set_fourier_from_index(i,
                                                   k2 == 0 ? FML::GRID::ComplexType(0) :
                                                             density.get_fourier_from_index(i) *
                                                                 FML::GRID::FloatType(1 / (k2 * window * window)));
                }
            FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<3>(density, psi);
            for (auto & g : psi)
                g.communicate_boundaries();
            std::array<std::vector<FML::GRID::FloatType>, 3> values;
            FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, Particle>(
                psi,
                load.particles.get_particles_ptr(),
                load.particles.get_npart(),
                values,
                "CIC");
            using namespace FML::SHORT_RANGE_GRAVITY;
            std::vector<FML::SHORT_RANGE_GRAVITY::Particle> owned(load.particles.get_npart());
            for (size_t i = 0; i < owned.size(); ++i) {
                const auto & p = load.particles.get_particles_ptr()[i];
                owned[i] = {p.position, 1, p.id};
            }
            auto sources = exchange_slab_ghosts(owned, FML::xmin_domain, FML::xmax_domain, kernel);
            std::sort(sources.particles.begin(), sources.particles.end(), [](const auto & a, const auto & b) {
                return a.id < b.id;
            });
            auto short_force = CellList(sources.particles, kernel).compute(owned);
            std::vector<Vector> result(owned.size());
            for (size_t i = 0; i < owned.size(); ++i)
                for (int d = 0; d < 3; ++d)
                    result[i][d] =
                        -double(values[d][i]) - short_force.acceleration[i][d]; // Reverse BOTH components once.
            return result;
        }
    };
    /// The default relaxation mesh: the smallest multiple of the task count (and of 2) that is >= 3 N^(1/3), and >= 4
    inline int automatic_mesh(std::uint64_t count) {
        const int step = FML::NTasks % 2 == 0 ? FML::NTasks : 2 * FML::NTasks;
        const int target = std::max(4, int(std::ceil(3.0 * std::cbrt(double(count)) - 1e-9)));
        return ((target + step - 1) / step) * step;
    }
    inline Result relax(Load & load, Options o = {}) {
        for (int x : {o.max_steps, o.min_steps, o.stable_steps})
            detail::same(x, "[GlassIC] Relaxation settings differ between ranks");
        for (double x : {o.timestep, o.damping, o.max_step_spacing, o.force_tolerance, o.velocity_tolerance})
            detail::same(x, "[GlassIC] Relaxation settings differ between ranks");
        detail::require(o.max_steps >= 0 && o.min_steps >= 0 && o.stable_steps > 0 && o.timestep > 0 &&
                            std::isfinite(o.timestep) && o.damping > 0 && std::isfinite(o.damping) &&
                            o.max_step_spacing > 0 && std::isfinite(o.max_step_spacing) && o.force_tolerance >= 0 &&
                            std::isfinite(o.force_tolerance) && o.velocity_tolerance >= 0 &&
                            std::isfinite(o.velocity_tolerance),
                        "[GlassIC] Invalid relaxation settings");
        detail::check_load(load);
        if (o.mesh <= 0)
            o.mesh = automatic_mesh(load.particles.get_npart_total());
        Force force(o.mesh, load.particles.get_npart_total(), o.split_cells, o.softening_spacing, o.cutoff_tolerance);
        const double spacing = 1 / std::cbrt(double(load.particles.get_npart_total()));
        auto acceleration = force(load);
        Result result;
        int stable = 0;
        // Exact damped kick for frozen acceleration followed by drift; artificial time has no cosmological meaning.
        const double decay = std::exp(-o.damping * o.timestep), kick = -std::expm1(-o.damping * o.timestep) / o.damping;
        for (int step = 0;; ++step) {
            result.steps = step;
            result.residuals = residuals(load, acceleration);
            result.history.push_back(result.residuals);
            const auto & r = result.residuals;
            if (!std::isfinite(r.force_rms) || !std::isfinite(r.velocity_rms)) {
                result.status = Status::NonFinite;
                return result;
            }
            stable = (step >= o.min_steps && r.force_rms <= o.force_tolerance * spacing &&
                      r.velocity_rms <= o.velocity_tolerance * spacing) ?
                         stable + 1 :
                         0;
            if (stable >= o.stable_steps) {
                result.status = Status::Converged;
                return result;
            }
            if (step == o.max_steps)
                return result;
            // Global limiter scales the entire proposed drift, preserving relative velocities and mean momentum.
            double vmax = 0;
            int bad = 0;
            for (size_t i = 0; i < load.particles.get_npart(); ++i) {
                double v2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double v =
                        decay * load.particles.get_particles_ptr()[i].velocity[d] + kick * acceleration[i][d];
                    v2 += v * v;
                }
                bad |= !std::isfinite(v2);
                vmax = std::max(vmax, std::sqrt(v2));
            }
            FML::MaxOverTasks(&bad);
            FML::MaxOverTasks(&vmax);
            if (bad) {
                result.status = Status::NonFinite;
                return result;
            }
            const double limit = o.max_step_spacing * spacing;
            const double scale = vmax > limit / o.timestep ? (limit / vmax) / o.timestep : 1.;
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < load.particles.get_npart(); ++i) {
                auto & p = load.particles.get_particles_ptr()[i];
                for (int d = 0; d < 3; ++d) {
                    p.velocity[d] = scale * (decay * p.velocity[d] + kick * acceleration[i][d]);
                    p.q[d] = p.position[d] = detail::wrap(p.position[d] + o.timestep * p.velocity[d]);
                }
            }
            detail::migrate(load.particles);
            acceleration = force(load);
        }
    }
    // Apply precombined LPT displacement and velocity fields at the SAME original glass q, then migrate once.
    // displacement is in box lengths; velocity is in explicitly caller-chosen units. Relaxation velocities are
    // replaced.
    inline void apply_fields(Load & load, std::array<Grid, 3> & displacement, std::array<Grid, 3> & velocity) {
        detail::check_load(load);
        detail::require(!load.perturbed, "[GlassIC] ICs already applied");
        for (auto * fields : {&displacement, &velocity})
            for (auto & grid : *fields) {
                detail::check_grid(grid);
                detail::same(grid.get_nmesh(), "[GlassIC] IC meshes differ between ranks");
                detail::require(grid.get_grid_status_real(), "[GlassIC] IC fields must be in real space");
                bool valid = true;
                for (auto i : grid.get_real_range())
                    valid &= std::isfinite(grid.get_real_from_index(i));
                detail::require(valid, "[GlassIC] Nonfinite IC field");
                grid.communicate_boundaries();
            }
        detail::require(displacement[0].get_nmesh() == velocity[0].get_nmesh() &&
                            displacement[1].get_nmesh() == displacement[0].get_nmesh() &&
                            displacement[2].get_nmesh() == displacement[0].get_nmesh() &&
                            velocity[1].get_nmesh() == velocity[0].get_nmesh() &&
                            velocity[2].get_nmesh() == velocity[0].get_nmesh(),
                        "[GlassIC] IC field layouts differ");
        std::array<std::vector<FML::GRID::FloatType>, 3> dx, dv;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, Particle>(
            displacement,
            load.particles.get_particles_ptr(),
            load.particles.get_npart(),
            dx,
            "CIC");
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, Particle>(
            velocity,
            load.particles.get_particles_ptr(),
            load.particles.get_npart(),
            dv,
            "CIC");
        bool valid = true;
        for (size_t i = 0; i < load.particles.get_npart(); ++i)
            for (int d = 0; d < 3; ++d)
                valid &= std::isfinite(double(dx[d][i])) && std::isfinite(double(dv[d][i]));
        detail::require(valid, "[GlassIC] Nonfinite interpolated ICs");
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < load.particles.get_npart(); ++i) {
            auto & p = load.particles.get_particles_ptr()[i];
            p.q = p.position;
            for (int d = 0; d < 3; ++d) {
                p.position[d] = detail::wrap(p.q[d] + double(dx[d][i]));
                p.velocity[d] = dv[d][i];
            }
        }
        load.perturbed = true;
        detail::migrate(load.particles);
    }
    // Shared transfer-provider bridge. Query/validate the spectrum outside OpenMP before generating modes.
    // k_grid is in inverse box lengths; provider k is h/Mpc and provider P is (Mpc/h)^3.
    inline void density_from_provider(Grid & delta,
                                      const FML::INITIALCONDITIONS::TransferProvider & provider,
                                      FML::INITIALCONDITIONS::Species species,
                                      double a,
                                      double boxsize,
                                      std::uint64_t seed,
                                      bool fixed = false) {
        detail::same(delta.get_nmesh(), "[GlassIC] Density mesh differs between ranks");
        detail::same(a, "[GlassIC] Epoch differs between ranks");
        detail::same(boxsize, "[GlassIC] Physical box differs between ranks");
        detail::same(seed, "[GlassIC] Field seed differs between ranks");
        detail::same(int(fixed), "[GlassIC] Amplitude convention differs between ranks");
        detail::same(int(species), "[GlassIC] Species differs between ranks");
        detail::check_grid(delta);
        detail::require(a > 0 && std::isfinite(a) && boxsize > 0 && std::isfinite(boxsize),
                        "[GlassIC] Invalid provider epoch or box size");
        const size_t half = size_t((delta.get_nmesh() - 1) / 2);
        detail::require(half <= size_t(std::sqrt(double(std::numeric_limits<size_t>::max() / 3 - 1))),
                        "[GlassIC] Power lookup size overflow");
        std::vector<double> power(3 * half * half + 1, 0);
        bool valid = true;
        try {
            provider.require_ic_density_gauge();
            for (size_t n = 1; n < power.size(); ++n) {
                const double k = 2 * std::acos(-1.) * std::sqrt(double(n)) / boxsize;
                provider.check_query(species, k, a);
                power[n] = ((provider.power(species, k, a) / boxsize) / boxsize) / boxsize;
                valid &= std::isfinite(power[n]) && power[n] >= 0;
            }
        } catch (const std::exception &) {
            valid = false;
        }
        detail::require(valid, "[GlassIC] Provider gauge/coverage/power invalid for requested IC mesh");
        FML::RANDOM::COUNTER_BASED::generate_fourier<3>(
            delta,
            seed,
            0,
            [&](double k) {
                const auto n = size_t(std::llround(k * k / (4 * std::acos(-1.) * std::acos(-1.))));
                return power[n];
            },
            fixed);
    }
    // delta is supplied at the IC epoch: reuse the existing full 3LPT EdS construction, INCLUDING transverse term.
    inline void apply_3lpt(Load & load,
                           Grid & delta_fourier,
                           double dlogDdt,
                           FML::COSMOLOGY::LPT::Dealiasing dealiasing = FML::COSMOLOGY::LPT::Dealiasing::None) {
        detail::same(dlogDdt, "[GlassIC] Growth rate differs between ranks");
        detail::same(int(dealiasing), "[GlassIC] Dealiasing differs between ranks");
        detail::check_load(load);
        detail::require(!load.perturbed, "[GlassIC] ICs already applied");
        detail::check_grid(delta_fourier);
        detail::require(dealiasing == FML::COSMOLOGY::LPT::Dealiasing::None ||
                            dealiasing == FML::COSMOLOGY::LPT::Dealiasing::Padding,
                        "[GlassIC] Unsupported dealiasing convention");
        detail::require(!delta_fourier.get_grid_status_real() && std::isfinite(dlogDdt),
                        "[GlassIC] Need Fourier delta and finite growth rate");
        std::array<Grid, 3> dx, dv;
        FML::COSMOLOGY::LPT::compute_3LPT_displacement_field<3>(delta_fourier, dx, dv, dlogDdt, false, dealiasing);
        apply_fields(load, dx, dv);
    }
} // namespace FML::GLASSIC
#endif

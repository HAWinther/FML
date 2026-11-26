#ifndef TREEPM_HEADER
#define TREEPM_HEADER

//==========================================================================================
//
// A simple TreePM N-body code for collisionless particles (CDM, or CDM and baryons), GADGET-style:
//
//  * The particles are on a Peano-Hilbert domain (Global/Domain.h), rebuilt load-balanced every step.
//  * The force is split with a Gaussian filter (ShortRangeGravity.h, the split of GADGET-2): the long-range part is
//    solved on a PM grid (slabs) with the density from particles on any domain and the force interpolated back to them
//    (Interpolation/DistributedDeposit.h: particles_to_grid_any_domain, interpolate_grids_to_particles_any_domain),
//    the short-range part is summed with an octree (or a cell list) over the local particles and the ghosts within the
//    cutoff from the other tasks (ShortRangeGravity/SlabGhosts.h: exchange_domain_ghosts on the Peano-Hilbert domain),
//    with cubic-spline softening.
//  * Time stepping: kick-drift-kick with a common step (timestep_method = fixed: steps uniform in log a), or as
//    GADGET-2 (timestep_method = gadget): the PM force on a global step limited by the rms displacement
//    (MaxRMSDisplacementFac) and MaxSizeTimestep, and the short-range force on individual steps
//    dt_i = sqrt(2 eta eps / |a_i|) in power-of-two bins of the PM step, computed for the active particles only.
//
// Units: positions in [0,1) (box units), time variable tau = log(a), velocities v = a^2 dx/dt / (H0 B):
//   dx/dtau = v / (a^2 E(a)),   dv/dtau = -Gamma v - (3/2) OmegaM P(a) / (a E(a)) * D phi,   D^2 phi = delta
// with the models of Cosmology.h (LCDM, coupled quintessence / the Fading Dark Sector, covariant dark scattering).
//
// CDM and baryons as two sets of particles (TreePMParameters::baryons; TreePMParticle::species 0 = CDM, 1 = baryons):
// the source of the force is the CDM+baryon density, each species weighted by f_c(a), f_b(a) (the particle mass). With
// coupled quintessence the CDM also feels 2 beta^2 f_c(a) times the force of the CDM alone (a second force evaluation
// with only the CDM as source) and the friction -beta dphi/dlna, the baryons neither.
//
//==========================================================================================

#include "Cosmology.h"

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/DistributedDeposit.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/PeanoHilbertDomain/HilbertCurve.h>
#include <FML/ShortRangeGravity/LongRangePM.h>
#include <FML/ShortRangeGravity/Octree.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace FML {
    namespace NBODY {
        namespace TREEPM {

            namespace SR = FML::SHORT_RANGE_GRAVITY;
            constexpr int NDIM = 3;

            //======================================================================================
            /// The particle. Acc is the acceleration -D phi at the last force evaluation (box units, D^2 phi =
            /// delta). mass is the weight of the particle in the density (1, or f_c(a), f_b(a) for CDM and baryons;
            /// set by the code before every force evaluation)
            //======================================================================================
            struct TreePMParticle {
                double Pos[NDIM];
                double Vel[NDIM];
                double Acc[NDIM]{};
                double AccPM[NDIM]{}; // The long-range part of Acc (timestep_method = gadget)
                double AccSR[NDIM]{}; // The short-range part of Acc (timestep_method = gadget)
                long long int id;
                int bin{0};         // The short-range time bin (step dt_PM / 2^bin; timestep_method = gadget)
                int species{0}; // 0: CDM+baryons (one fluid) or CDM (two fluids), 1: baryons
                double mass{1.0};

                constexpr int get_ndim() const { return NDIM; }
                double * get_pos() { return Pos; }
                double * get_vel() { return Vel; }
                long long int get_id() const { return id; }
                void set_id(long long int _id) { id = _id; }
                double get_mass() const { return mass; }
            };

            //======================================================================================
            /// Parameters of a simulation (the model: ModelParameters)
            //======================================================================================
            struct TreePMParameters : ModelParameters {
                int Nmesh{128};                 // The PM grid (Nmesh^3; the number of tasks must divide it)
                std::string assignment{"CIC"};  // Density assignment and force interpolation of the PM part
                bool deconvolve{true};          // Divide out the window of the assignment and interpolation
                double asmth{1.25};             // The split scale r_s in PM cells (GADGET-2's ASMTH)
                double softening{1.0 / 30.0};   // Plummer-equivalent softening (comoving) / mean particle spacing
                double cutoff_tolerance{1e-4};  // The short-range pair force at the cutoff / Newton (GADGET-2: ~4.5 r_s)
                std::string backend{"tree"};    // The short-range sum: tree (monopole octree) or cell (cell list)
                double opening_angle{0.5};      // The tree's opening angle (0: exact)
                int leaf_size{8};               // Particles per tree leaf
                int hilbert_bits{21};           // Bits per coordinate of the Peano-Hilbert keys of the domain
                int Npart_1D{64};               // Particles per dimension (per species) for the softening
                double boxsize{100.0};          // Mpc/h (for output only)
                // Time steps: fixed (a common step, the list of the caller) or gadget (GADGET-2: the PM step from
                // max_rms_displacement and max_dlna, individual short-range steps sqrt(2 eta eps / |a|) in bins)
                std::string timestep_method{"fixed"};
                double eta{0.025};                 // GADGET's ErrTolIntAccuracy
                double max_rms_displacement{0.25}; // GADGET's MaxRMSDisplacementFac
                double max_dlna{0.025};            // GADGET's MaxSizeTimestep (in log a)
                int max_bins{20};                  // At most 2^max_bins short-range steps per PM step
                bool gadget() const { return timestep_method == "gadget"; }
            };

            //======================================================================================
            /// Statistics of one step
            //======================================================================================
            struct StepStats {
                double time_domain{0}, time_pm{0}, time_ghosts{0}, time_short{0}, time_move{0};
                long long ghosts{0};               // Ghosts received (summed over tasks)
                long long pairs{0}, nodes{0};      // Short-range pair and node interactions (summed over tasks)
                double max_imbalance{1};           // Largest number of particles on a task / mean
                int force_evaluations{0};
                // timestep_method = gadget: short-range sub-steps, active particle force evaluations, finest bin
                int nsub{0};
                long long nactive{0};
                int finest_bin{0};
            };

            //======================================================================================
            ///
            /// The simulation: the particles on a Peano-Hilbert domain, the PM grid and the short-range sum
            ///
            //======================================================================================
            class TreePMSimulation {
              public:
                using Particles = FML::PARTICLE::MPIParticles<TreePMParticle>;
                using Vector = SR::Vector;

                TreePMParameters p;
                Cosmology cosmo;
                Particles & part;
                double a{0.0};
                long long npart_total{0}, npart_species{0};
                StepStats last;

                TreePMSimulation(const TreePMParameters & params, Particles & particles, double aini)
                    : p(params), cosmo(make_cosmology(params)), part(particles), a(aini) {
                    assert_mpi(p.background_model(), "TreePM: gravity_model must be LCDM, CoupledQuintessence, "
                                                     "FadingDarkSector or CovariantDarkScattering");
                    assert_mpi(p.Nmesh % FML::NTasks == 0, "TreePM: the number of tasks must divide Nmesh");
                    assert_mpi(p.backend == "tree" or p.backend == "cell", "TreePM: backend must be tree or cell");
                    npart_total = (long long)part.get_npart_total();
                    npart_species = p.baryons ? npart_total / 2 : npart_total;
                    cdm_fifth = p.baryons and cosmo.has_cdm_fifth_force();
                    assert_mpi(p.timestep_method == "fixed" or p.gadget(), "TreePM: timestep_method must be fixed or gadget");
                    for (size_t i = 0; i < part.get_npart(); i++)
                        for (int d = 0; d < NDIM; d++)
                            part[i].Pos[d] = wrap(part[i].Pos[d]);
                    rebalance();
                    if (p.gadget()) {
                        compute_pm_forces(aini);
                        compute_short_forces(aini, nullptr);
                    } else {
                        compute_forces(aini);
                    }
                }

                /// Take one step from the current a to anew: a common kick-drift-kick step (timestep_method = fixed), or a
                /// PM step with individual short-range steps (timestep_method = gadget)
                void step(double anew);

                /// timestep_method = gadget: the PM (global) step in tau = log a as GADGET-2 (find_long_range_step):
                /// the rms displacement in a step is at most max_rms_displacement * min(mean particle spacing, r_s), and
                /// the step at most max_dlna. limit (if given) says which. Collective
                double gadget_pm_timestep(std::string * limit = nullptr) const;

                /// The force on every local particle (Acc) at a_now
                void compute_forces(double a_now);

                /// The split scale, cutoff and softening support in box units
                double split_scale() const { return p.asmth / double(p.Nmesh); }
                double softening_support() const { return 2.8 * p.softening / double(p.Npart_1D); }

                /// The acceleration -D phi (box units) of every local particle from the sources with weight w(particle)
                /// (D^2 phi = delta of those sources, normalized to their mean): the PM long-range part plus the
                /// short-range sum. The particles' masses are set to the weights
                template <class W>
                std::vector<Vector> gravity(W && weight) {
                    auto acc = pm_gravity(weight);
                    const auto sr = short_gravity(weight, nullptr);
                    for (size_t i = 0; i < acc.size(); i++)
                        for (int d = 0; d < NDIM; d++)
                            acc[i][d] += sr[i][d];
                    return acc;
                }
                /// The long-range (PM) part of gravity() for every local particle
                template <class W>
                std::vector<Vector> pm_gravity(W && weight);
                /// The short-range part of gravity() for the local particles with active[i] (all if nullptr); zero for the
                /// others. All the particles are sources
                template <class W>
                std::vector<Vector> short_gravity(W && weight, const std::vector<char> * active);

                /// timestep_method = gadget: the long-range force (AccPM) of every particle, and the short-range force
                /// (AccSR) of the active ones (all if nullptr); Acc = AccPM + AccSR
                void compute_pm_forces(double a_now);
                void compute_short_forces(double a_now, const std::vector<char> * active);

                void print_status(std::ostream & os) const;

              private:
                bool cdm_fifth{false};

                /// The source weights at a_now (all 1, or f_c, f_b)
                std::function<double(const TreePMParticle &)> source_weight(double a_now) const {
                    const double fc = cosmo.cdm_fraction(a_now);
                    if (p.baryons)
                        return [fc](const TreePMParticle & q) { return q.species == 1 ? 1.0 - fc : fc; };
                    return [](const TreePMParticle &) { return 1.0; };
                }
                static double cdm_weight(const TreePMParticle & q) { return q.species == 0 ? 1.0 : 0.0; }
                /// The kernel of the short-range force for a total source mass
                SR::Kernel make_kernel(double total_mass) const {
                    SR::Options o;
                    o.boxsize = 1.0;
                    o.split_scale = split_scale();
                    o.softening_support = softening_support();
                    o.cutoff_tolerance = p.cutoff_tolerance;
                    o.coupling = SR::fml_acceleration_coupling(1.0, total_mass, 1.0);
                    return SR::Kernel(o);
                }
                double set_masses(const std::function<double(const TreePMParticle &)> & weight) {
                    double total_mass = 0.0;
                    for (size_t i = 0; i < part.get_npart(); i++) {
                        part[i].mass = weight(part[i]);
                        total_mass += part[i].mass;
                    }
                    FML::SumOverTasks(&total_mass);
                    assert_mpi(total_mass > 0.0, "TreePM: no source mass");
                    return total_mass;
                }
                /// The short-range time bin of a particle in a PM step dtau (GADGET-2: dt = sqrt(2 eta eps / |a|), the
                /// acceleration d^2x/dtau^2 = (3/2) OmegaM P / (a^3 E^2) |Acc| in box units)
                int wanted_bin(const TreePMParticle & pt, double a_now, double dtau) const {
                    const double E = cosmo.E(a_now);
                    const double g2 = pt.Acc[0] * pt.Acc[0] + pt.Acc[1] * pt.Acc[1] + pt.Acc[2] * pt.Acc[2];
                    const double alpha = 1.5 * p.OmegaM * cosmo.poisson_factor(a_now) / (a_now * a_now * a_now * E * E) * std::sqrt(g2);
                    if (not(alpha > 0.0))
                        return 0;
                    const double eps = p.softening / double(p.Npart_1D);
                    const double dti = std::sqrt(2.0 * p.eta * eps / alpha);
                    int b = 0;
                    while (b < p.max_bins and dtau / std::ldexp(1.0, b) > dti)
                        b++;
                    return b;
                }
                void step_gadget(double anew);

                static double wrap(double x) {
                    x -= std::floor(x);
                    return x >= 1.0 ? 0.0 : x;
                }
                static double now() {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                }
                Fluid fluid_of(const TreePMParticle & pt) const {
                    return p.baryons ? (pt.species == 1 ? Fluid::Baryon : Fluid::CDM) : Fluid::CB;
                }

                /// A new load-balanced Peano-Hilbert domain from the particle keys (equal particles per task), and the
                /// particles moved to it
                void rebalance() {
                    namespace PH = FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;
                    const double t0 = now();
                    std::vector<PH::Key> keys(part.get_npart());
                    for (size_t i = 0; i < part.get_npart(); i++)
                        keys[i] = PH::key_from_position({part[i].Pos[0], part[i].Pos[1], part[i].Pos[2]}, p.hilbert_bits);
                    auto domain = FML::PeanoHilbertDomain::build(keys, unsigned(p.hilbert_bits), 16);
                    part.set_domain(FML::Domain::peano_hilbert(domain));
                    double n = double(part.get_npart()), nmax = n;
                    FML::MaxOverTasks(&nmax);
                    FML::SumOverTasks(&n);
                    last.max_imbalance = nmax / (n / FML::NTasks);
                    last.time_domain += now() - t0;
                }
            };

            template <class W>
            std::vector<TreePMSimulation::Vector> TreePMSimulation::pm_gravity(W && weight) {
                const size_t n = part.get_npart();
                const double total_mass = set_masses(weight);
                const SR::Kernel kernel = make_kernel(total_mass);

                // The long-range force: the filtered PM force on the grid, interpolated to the particles (on their
                // Peano-Hilbert domain). FML's PM gives +D phi
                double t0 = now();
                std::vector<Vector> acc(n);
                {
                    FML::GRID::FFTWGrid<NDIM> density(p.Nmesh);
                    FML::INTERPOLATION::particles_to_grid_any_domain<NDIM>(part.get_particles_ptr(), n, size_t(npart_total),
                                                                           density, p.assignment);
                    density.fftw_r2c();
                    SR::apply_long_range_filter(density, kernel);
                    std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> force;
                    FML::NBODY::ForceOptions fo;
                    fo.deconvolve = p.deconvolve;
                    FML::NBODY::compute_force_from_density_fourier<NDIM>(density, force, p.assignment, 1.0, fo);
                    std::vector<std::array<double, NDIM>> f;
                    const std::array<const FML::GRID::FFTWGrid<NDIM> *, NDIM> grids{&force[0], &force[1], &force[2]};
                    FML::INTERPOLATION::interpolate_grids_to_particles_any_domain<NDIM, NDIM, TreePMParticle, FML::GRID::FFTWGrid>(
                        grids, part.get_particles_ptr(), n, f, p.assignment);
                    for (size_t i = 0; i < n; i++)
                        for (int d = 0; d < NDIM; d++)
                            acc[i][d] = -f[i][d];
                }
                last.time_pm += now() - t0;
                return acc;
            }

            template <class W>
            std::vector<TreePMSimulation::Vector> TreePMSimulation::short_gravity(W && weight, const std::vector<char> * active) {
                const size_t n = part.get_npart();
                const double total_mass = set_masses(weight);
                const SR::Kernel kernel = make_kernel(total_mass);

                // The short-range force: the local particles and the ghosts within the cutoff are the sources, the
                // (active) local particles the targets
                double t0 = now();
                std::vector<SR::Particle> owned(n);
                for (size_t i = 0; i < n; i++) {
                    for (int d = 0; d < NDIM; d++)
                        owned[i].position[d] = part[i].Pos[d];
                    owned[i].mass = part[i].mass;
                    owned[i].id = std::uint64_t(part[i].id);
                }
                auto sources = SR::exchange_domain_ghosts(owned, part.get_domain(), kernel);
                long long nghost = (long long)sources.received_particles;
                FML::SumOverTasks(&nghost);
                last.ghosts += nghost;
                last.time_ghosts += now() - t0;

                t0 = now();
                std::vector<SR::Particle> subset;
                std::vector<size_t> index;
                if (active) {
                    for (size_t i = 0; i < n; i++)
                        if ((*active)[i]) {
                            subset.push_back(owned[i]);
                            index.push_back(i);
                        }
                }
                const auto & targets = active ? subset : owned;
                std::vector<Vector> acc(n, Vector{0.0, 0.0, 0.0});
                long long pairs = 0, nodes = 0;
                auto add = [&](const std::vector<Vector> & r) {
                    for (size_t k = 0; k < r.size(); k++) {
                        const size_t i = active ? index[k] : k;
                        acc[i] = r[k];
                    }
                };
                if (p.backend == "cell") {
                    const auto r = SR::CellList(sources.particles, kernel).compute(targets);
                    add(r.acceleration);
                    pairs = (long long)r.statistics.accepted_pairs;
                } else {
                    SR::TreeOptions to;
                    to.opening_angle = p.opening_angle;
                    to.leaf_capacity = size_t(p.leaf_size);
                    const auto r = SR::Octree(sources.particles, kernel, to).compute(targets);
                    add(r.acceleration);
                    pairs = (long long)r.statistics.accepted_pairs;
                    nodes = (long long)r.accepted_nodes;
                }
                FML::SumOverTasks(&pairs);
                FML::SumOverTasks(&nodes);
                last.pairs += pairs;
                last.nodes += nodes;
                last.time_short += now() - t0;
                last.force_evaluations++;
                return acc;
            }

            inline void TreePMSimulation::compute_pm_forces(double a_now) {
                auto acc = pm_gravity(source_weight(a_now));
                if (cdm_fifth) {
                    const auto acc_c = pm_gravity(cdm_weight);
                    const double fifth = cosmo.cdm_fifth_force(a_now);
                    for (size_t i = 0; i < part.get_npart(); i++)
                        if (part[i].species == 0)
                            for (int d = 0; d < NDIM; d++)
                                acc[i][d] += fifth * acc_c[i][d];
                }
                for (size_t i = 0; i < part.get_npart(); i++)
                    for (int d = 0; d < NDIM; d++) {
                        part[i].AccPM[d] = acc[i][d];
                        part[i].Acc[d] = part[i].AccPM[d] + part[i].AccSR[d];
                    }
            }

            inline void TreePMSimulation::compute_short_forces(double a_now, const std::vector<char> * active) {
                auto acc = short_gravity(source_weight(a_now), active);
                if (cdm_fifth) {
                    const auto acc_c = short_gravity(cdm_weight, active);
                    const double fifth = cosmo.cdm_fifth_force(a_now);
                    for (size_t i = 0; i < part.get_npart(); i++)
                        if (part[i].species == 0)
                            for (int d = 0; d < NDIM; d++)
                                acc[i][d] += fifth * acc_c[i][d];
                }
                for (size_t i = 0; i < part.get_npart(); i++) {
                    if (active and not(*active)[i])
                        continue;
                    for (int d = 0; d < NDIM; d++) {
                        part[i].AccSR[d] = acc[i][d];
                        part[i].Acc[d] = part[i].AccPM[d] + part[i].AccSR[d];
                    }
                }
            }

            inline double TreePMSimulation::gadget_pm_timestep(std::string * limit) const {
                // The rms of the drift velocity dx/dtau = v / (a^2 E)
                double v2 = 0.0;
                for (size_t i = 0; i < part.get_npart(); i++)
                    for (int d = 0; d < NDIM; d++)
                        v2 += part[i].Vel[d] * part[i].Vel[d];
                FML::SumOverTasks(&v2);
                const double vrms = std::sqrt(v2 / double(npart_total)) / (a * a * cosmo.E(a));
                const double length = std::min(1.0 / double(p.Npart_1D), split_scale());
                double dtau = p.max_dlna;
                std::string why = "MaxSizeTimestep";
                if (vrms > 0.0 and p.max_rms_displacement * length / vrms < dtau) {
                    dtau = p.max_rms_displacement * length / vrms;
                    why = "rms displacement";
                }
                if (limit)
                    *limit = why;
                return dtau;
            }

            inline void TreePMSimulation::compute_forces(double a_now) {
                // The source: all the particles (one fluid), or CDM and baryons weighted by f_c(a), f_b(a)
                const double fc = cosmo.cdm_fraction(a_now);
                auto acc = p.baryons ? gravity([&](const TreePMParticle & q) { return q.species == 1 ? 1.0 - fc : fc; })
                                     : gravity([](const TreePMParticle &) { return 1.0; });
                // Coupled quintessence with CDM and baryon particles: the CDM also feels 2 beta^2 f_c times the force
                // of the CDM alone
                if (cdm_fifth) {
                    const auto acc_c = gravity([](const TreePMParticle & q) { return q.species == 0 ? 1.0 : 0.0; });
                    const double fifth = cosmo.cdm_fifth_force(a_now);
                    for (size_t i = 0; i < part.get_npart(); i++)
                        if (part[i].species == 0)
                            for (int d = 0; d < NDIM; d++)
                                acc[i][d] += fifth * acc_c[i][d];
                }
                for (size_t i = 0; i < part.get_npart(); i++)
                    for (int d = 0; d < NDIM; d++)
                        part[i].Acc[d] = acc[i][d];
            }

            inline void TreePMSimulation::step(double anew) {
                if (p.gadget()) {
                    step_gadget(anew);
                    return;
                }
                last = StepStats{};
                const double amid = std::sqrt(a * anew);
                // Kick factors per fluid (with friction a kick is v -> F v + K Acc)
                auto kick = [&](double a1, double a2) {
                    std::array<double, 3> K, F;
                    for (auto fl : {Fluid::CB, Fluid::CDM, Fluid::Baryon}) {
                        K[int(fl)] = cosmo.kick_factor(a1, a2, fl);
                        F[int(fl)] = cosmo.friction_factor(a1, a2, fl);
                    }
                    for (size_t i = 0; i < part.get_npart(); i++) {
                        auto & pt = part[i];
                        const int f = int(fluid_of(pt));
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] = F[f] * pt.Vel[d] + K[f] * pt.Acc[d];
                    }
                };
                double t0 = now();
                kick(a, amid);
                const double D = cosmo.drift_factor(a, anew);
                for (size_t i = 0; i < part.get_npart(); i++)
                    for (int d = 0; d < NDIM; d++)
                        part[i].Pos[d] = wrap(part[i].Pos[d] + part[i].Vel[d] * D);
                last.time_move += now() - t0;
                a = anew;
                rebalance();
                compute_forces(a);
                t0 = now();
                kick(amid, anew);
                last.time_move += now() - t0;
            }

            /// A PM step as GADGET-2: half a long-range kick, the short-range force on individual steps (bins of the PM
            /// step, kick-drift-kick with sub-steps, the force of the active particles only, every particle drifted at
            /// every sub-step), and at the end (all synchronized) the domain, the PM force and the other half kick.
            /// The friction is applied with the short-range kicks (they cover the step of every particle once)
            inline void TreePMSimulation::step_gadget(double anew) {
                last = StepStats{};
                const double tau0 = std::log(a), tau1 = std::log(anew), dtau = tau1 - tau0;
                const double amid = std::sqrt(a * anew);
                double t0 = now();

                // The bins from the accelerations at a (the finest bin of all particles sets the sub-steps)
                int B = 0;
                for (size_t i = 0; i < part.get_npart(); i++) {
                    part[i].bin = wanted_bin(part[i], a, dtau);
                    B = std::max(B, part[i].bin);
                }
                FML::MaxOverTasks(&B);
                last.finest_bin = B;
                const long long nticks = 1LL << B;
                auto tau_of_tick = [&](double t) { return tau0 + dtau * t / double(nticks); };
                auto ticks_of_bin = [&](int b) { return 1LL << (B - b); };

                // Kick factors per fluid and bin (from tick t - half a step of the bin to t, and from t to t + half)
                constexpr int NF = 3;
                const Fluid fluids[NF] = {Fluid::CB, Fluid::CDM, Fluid::Baryon};
                std::array<std::vector<double>, NF> Kclose, Kopen, Fclose, Fopen;
                for (int f = 0; f < NF; f++) {
                    Kclose[f].assign(B + 1, 0.0);
                    Kopen[f].assign(B + 1, 0.0);
                    Fclose[f].assign(B + 1, 1.0);
                    Fopen[f].assign(B + 1, 1.0);
                }
                auto set_kick_factors = [&](long long t) {
                    for (int b = 0; b <= B; b++) {
                        const double nt = double(ticks_of_bin(b));
                        const double am = std::exp(tau_of_tick(t - 0.5 * nt)), ac = std::exp(tau_of_tick(double(t))),
                                     ap = std::exp(tau_of_tick(t + 0.5 * nt));
                        for (int f = 0; f < NF; f++) {
                            Kclose[f][b] = cosmo.kick_factor(am, ac, fluids[f]);
                            Kopen[f][b] = cosmo.kick_factor(ac, ap, fluids[f]);
                            Fclose[f][b] = cosmo.friction_factor(am, ac, fluids[f]);
                            Fopen[f][b] = cosmo.friction_factor(ac, ap, fluids[f]);
                        }
                    }
                };
                // The long-range kicks. The friction acts in the short-range kicks, which cover the whole PM step after the
                // opening kick: so the opening kick (given at a1) is weighted with exp(+int_a1^tau Gamma) (kick_factor /
                // friction_factor: after the friction of [a1, a2] it is the kick of [a1, a2]); the closing one (given at
                // a2, after all the friction) is the plain kick_factor
                auto pm_kick = [&](double a1, double a2, bool opening) {
                    std::array<double, NF> K;
                    for (int f = 0; f < NF; f++)
                        K[f] = cosmo.kick_factor(a1, a2, fluids[f]) /
                               (opening ? cosmo.friction_factor(a1, a2, fluids[f]) : 1.0);
                    for (size_t i = 0; i < part.get_npart(); i++) {
                        auto & pt = part[i];
                        const double k = K[int(fluid_of(pt))];
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] += k * pt.AccPM[d];
                    }
                };

                // Opening half kicks: long range (half the PM step) and short range (half of each particle's step)
                pm_kick(a, amid, true);
                set_kick_factors(0);
                for (size_t i = 0; i < part.get_npart(); i++) {
                    auto & pt = part[i];
                    const int f = int(fluid_of(pt));
                    for (int d = 0; d < NDIM; d++)
                        pt.Vel[d] = Fopen[f][pt.bin] * pt.Vel[d] + Kopen[f][pt.bin] * pt.AccSR[d];
                }
                last.time_move += now() - t0;

                std::vector<char> active;
                for (long long t = 1; t <= nticks; t++) {
                    // Drift everybody to tick t and keep them on their domain
                    t0 = now();
                    const double a_prev = std::exp(tau_of_tick(double(t - 1))), a_now = std::exp(tau_of_tick(double(t)));
                    const double D = cosmo.drift_factor(a_prev, a_now);
                    for (size_t i = 0; i < part.get_npart(); i++)
                        for (int d = 0; d < NDIM; d++)
                            part[i].Pos[d] = wrap(part[i].Pos[d] + part[i].Vel[d] * D);
                    last.time_move += now() - t0;
                    last.nsub++;

                    // The bins ending their step at tick t: b >= min_bin
                    int min_bin = B;
                    while (min_bin > 0 and t % ticks_of_bin(min_bin - 1) == 0)
                        min_bin--;
                    const bool full = (t == nticks);
                    if (full) {
                        // All synchronized: a new domain
                        a = anew;
                        rebalance();
                    } else {
                        t0 = now();
                        part.communicate_particles();
                        last.time_domain += now() - t0;
                    }
                    active.assign(part.get_npart(), 0);
                    long long nactive = 0;
                    for (size_t i = 0; i < part.get_npart(); i++) {
                        active[i] = full or part[i].bin >= min_bin;
                        nactive += active[i];
                    }
                    FML::SumOverTasks(&nactive);
                    if (nactive == 0)
                        continue;
                    last.nactive += nactive;
                    compute_short_forces(a_now, &active);
                    if (full)
                        compute_pm_forces(a_now);

                    // Closing half kick of the active particles, a new bin, an opening half kick (not at the end)
                    t0 = now();
                    set_kick_factors(t);
                    for (size_t i = 0; i < part.get_npart(); i++) {
                        if (not active[i])
                            continue;
                        auto & pt = part[i];
                        const int f = int(fluid_of(pt));
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] = Fclose[f][pt.bin] * pt.Vel[d] + Kclose[f][pt.bin] * pt.AccSR[d];
                        if (full)
                            continue;
                        // The new bin must be synchronized at t (any finer one is; a coarser one only if t is on it)
                        int b = std::min(wanted_bin(pt, a_now, dtau), B);
                        while (t % ticks_of_bin(b) != 0)
                            b++;
                        pt.bin = b;
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] = Fopen[f][b] * pt.Vel[d] + Kopen[f][b] * pt.AccSR[d];
                    }
                    last.time_move += now() - t0;
                }
                // Closing half of the long-range kick with the new PM force
                t0 = now();
                pm_kick(amid, anew, false);
                last.time_move += now() - t0;
            }

            inline void TreePMSimulation::print_status(std::ostream & os) const {
                if (FML::ThisTask != 0)
                    return;
                if (p.gadget())
                    os << "  short-range sub-steps " << last.nsub << " (finest bin " << last.finest_bin << "), active "
                       << "particle forces " << last.nactive << " (" << std::fixed << std::setprecision(2)
                       << double(last.nactive) / double(npart_total) << " per particle)\n"
                       << std::defaultfloat;
                os << "  force evaluations " << last.force_evaluations << ", ghosts " << last.ghosts << " ("
                   << std::fixed << std::setprecision(2) << double(last.ghosts) / double(npart_total)
                   << " per particle), short-range pairs " << last.pairs << ", nodes " << last.nodes
                   << ", particles per task max/mean " << last.max_imbalance << "\n"
                   << "  time [s]: domain " << last.time_domain << "  PM " << last.time_pm << "  ghosts "
                   << last.time_ghosts << "  short-range " << last.time_short << "  kick/drift " << last.time_move
                   << "\n"
                   << std::defaultfloat;
            }

        } // namespace TREEPM
    } // namespace NBODY
} // namespace FML
#endif

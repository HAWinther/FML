#ifndef FML_ANALYSIS_BARYONIFICATION_H
#define FML_ANALYSIS_BARYONIFICATION_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Halos.h>
#include <FML/Analysis/Output.h>
#include <FML/Analysis/ParticleUtils.h>
#include <FML/Analysis/PowerSpectrum.h>
#include <FML/Baryonification/Baryonification.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

//=============================================================================
// Position-only baryonification of a particle set (Schneider et al. 2019, arXiv:1810.08629; enclosed-mass
// transport, Schneider & Teyssier 2015, arXiv:1510.06034) with FML/Baryonification's finite-support profiles. The
// same procedure as SimulationCode's (baryonification = true): FoF halos, M200c around them, hosts = the resolved,
// non-truncated, non-overlapping halos above a mass cut whose profile support is below half the box; the particles
// are displaced and moved to the task that owns their new position.
//
// The particles should represent the total CDM + baryon matter field (one particle species). The density parameter
// of the particles (for 200c) is taken from their total mass.
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// Keys baryonification_<name>, the same as SimulationCode's. Masses Msun/h, lengths comoving Mpc/h; radius
        /// ratios say _over_r200
        //=============================================================================
        struct BaryonificationSettings {
            static constexpr const char * prefix = "baryonification_";
            bool output_particles, relax_collisionless;
            std::string model;
            double min_halo_mass_msunh, gas_mass_scale_msunh, stellar_mass_scale_msunh;
            double gas_mass_exponent, gas_core_radius_over_r200, gas_ejection_radius_over_r200;
            double truncation_radius_over_r200, stellar_radius_over_r200, stellar_amplitude;
            double stellar_total_exponent, stellar_central_exponent, relaxation_amplitude, relaxation_exponent;
            double concentration_norm, concentration_pivot_mass_msunh, concentration_mass_slope,
                concentration_redshift_power;
            double fof_linking_length, fof_buffer_mpch, min_radius_over_r200, max_radius_over_r200;
            int fof_min_particles, fof_nmesh_max, profile_radial_bins;

            BaryonificationSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("model",
                  model,
                  std::string("schneider2019"),
                  "Finite-support Schneider2019 profile family; illustrative, uncalibrated defaults");
                f("output_particles",
                  output_particles,
                  false,
                  "Write a separate corrected snapshot; velocities are unchanged");
                f("min_halo_mass_msunh", min_halo_mass_msunh, 1e12, "Minimum host M200c (Msun/h)");
                f("gas_mass_scale_msunh", gas_mass_scale_msunh, 1e14, "Gas-slope mass scale M_c (Msun/h)");
                f("stellar_mass_scale_msunh", stellar_mass_scale_msunh, 2.5e11, "Stellar-fraction pivot M1 (Msun/h)");
                f("gas_mass_exponent", gas_mass_exponent, 0.3, "Exponent mu in beta=3-(M_c/M200c)^mu");
                f("gas_core_radius_over_r200", gas_core_radius_over_r200, 0.1, "Gas core radius / R200c");
                f("gas_ejection_radius_over_r200", gas_ejection_radius_over_r200, 4.0, "Gas ejection radius / R200c");
                f("truncation_radius_over_r200", truncation_radius_over_r200, 4.0, "NFW truncation radius / R200c");
                f("stellar_radius_over_r200", stellar_radius_over_r200, 0.015, "Central stellar scale R_h / R200c");
                f("stellar_amplitude", stellar_amplitude, 0.09, "Stellar mass fraction at M1");
                f("stellar_total_exponent", stellar_total_exponent, 0.32, "Total stellar fraction power-law exponent");
                f("stellar_central_exponent",
                  stellar_central_exponent,
                  0.6,
                  "Central stellar fraction power-law exponent");
                f("relax_collisionless", relax_collisionless, true, "Apply collisionless contraction/expansion");
                f("relaxation_amplitude", relaxation_amplitude, 0.3, "Dimensionless collisionless relaxation amplitude");
                f("relaxation_exponent", relaxation_exponent, 2.0, "Dimensionless collisionless relaxation exponent");
                f("concentration_norm", concentration_norm, 5.0, "Illustrative c200c normalization; not a fitted relation");
                f("concentration_pivot_mass_msunh",
                  concentration_pivot_mass_msunh,
                  1e14,
                  "Concentration pivot mass (Msun/h)");
                f("concentration_mass_slope",
                  concentration_mass_slope,
                  -0.1,
                  "c=c0*(M200c/Mpivot)^slope*(1+z)^redshift_power");
                f("concentration_redshift_power",
                  concentration_redshift_power,
                  -1.0,
                  "Power of (1+z) in the concentration relation");
                f("fof_linking_length", fof_linking_length, 0.2, "FoF seed linking length / mean particle separation");
                f("fof_min_particles", fof_min_particles, 20, "Minimum particles in a FoF seed halo");
                f("fof_nmesh_max", fof_nmesh_max, 0, "Maximum FoF search mesh (0: automatic)");
                f("fof_buffer_mpch",
                  fof_buffer_mpch,
                  3.0,
                  "Halo/SO search buffer (comoving Mpc/h); larger than halo radii");
                f("profile_radial_bins", profile_radial_bins, 2048, "Number of positive profile-table radii");
                f("min_radius_over_r200", min_radius_over_r200, 1e-6, "First positive profile radius / R200c");
                f("max_radius_over_r200",
                  max_radius_over_r200,
                  30.0,
                  "Finite profile support / R200c; must be smaller than half the box");
            }

            FML::BARYONIFICATION::ModelParameters profile_model(double baryon_fraction) const {
                FML::BARYONIFICATION::ModelParameters p;
                p.baryon_fraction = baryon_fraction;
                p.gas_mass_scale = gas_mass_scale_msunh;
                p.stellar_mass_scale = stellar_mass_scale_msunh;
                p.gas_mass_exponent = gas_mass_exponent;
                p.gas_core_radius = gas_core_radius_over_r200;
                p.gas_ejection_radius = gas_ejection_radius_over_r200;
                p.truncation_radius = truncation_radius_over_r200;
                p.stellar_radius = stellar_radius_over_r200;
                p.stellar_amplitude = stellar_amplitude;
                p.stellar_total_exponent = stellar_total_exponent;
                p.stellar_central_exponent = stellar_central_exponent;
                p.relax_collisionless = relax_collisionless;
                p.relaxation_amplitude = relaxation_amplitude;
                p.relaxation_exponent = relaxation_exponent;
                return p;
            }

            void validate() const {
                using namespace FML::BARYONIFICATION::detail;
                require(model == "schneider2019", "baryonification_model must be schneider2019");
                require(positive(min_halo_mass_msunh) && positive(concentration_pivot_mass_msunh) &&
                            positive(concentration_norm),
                        "Baryonification masses (_msunh) and concentration must be finite and positive");
                require(std::isfinite(concentration_mass_slope) && std::isfinite(concentration_redshift_power),
                        "Baryonification concentration exponents must be finite");
                require(positive(fof_linking_length) && fof_min_particles >= 2 && fof_nmesh_max >= 0 &&
                            positive(fof_buffer_mpch),
                        "Baryonification: invalid FoF linking length, particle count, mesh or buffer_mpch");
                require(profile_radial_bins >= 32 && positive(min_radius_over_r200) && min_radius_over_r200 < 1.0 &&
                            positive(max_radius_over_r200) && max_radius_over_r200 > 1.0,
                        "Baryonification: invalid profile table radii or radial-bin count");
                profile_model(0.157).validate(); // the baryon fraction is checked when it is known
            }

            void write(std::ostream & out) const {
                auto copy = *this;
                copy.parameters([&](const char * name, const auto & value, const auto &, const char *) {
                    out << "# baryonification_" << name << " = " << value << '\n';
                });
            }

            double concentration(double M200c_msunh, double redshift) const {
                return concentration_norm * std::pow(M200c_msunh / concentration_pivot_mass_msunh, concentration_mass_slope) *
                       std::pow(1.0 + redshift, concentration_redshift_power);
            }
        };

        /// A host halo: center (box units), M200c (Msun/h), R200c (box units) and concentration
        struct BaryonHost {
            std::array<double, 3> center;
            double mass_msunh, radius_box, concentration;
        };

        /// What baryonify did: the hosts (all of them, on every task) and why the other FoF halos were not hosts
        struct BaryonificationResult {
            std::vector<BaryonHost> hosts;
            long long truncated{0}, nested{0}, unresolved{0}, below_cut{0}, too_large{0};
            double baryon_fraction{0.0};
            double Delta_200c{0.0}; // 200c in units of the mean density of the particles
            double particle_mass_msunh{0.0};
            double mean_density_msunh_mpch3{0.0};
            long long npart_total{0};
        };

        namespace detail {
            /// All the hosts on every task, sorted (a consistent order for any number of tasks). Collective
            inline std::vector<BaryonHost> gather_hosts(const std::vector<BaryonHost> & local) {
                std::vector<double> packed;
                for (const auto & h : local) {
                    packed.insert(packed.end(), h.center.begin(), h.center.end());
                    packed.push_back(h.mass_msunh);
                    packed.push_back(h.radius_box);
                    packed.push_back(h.concentration);
                }
#ifdef USE_MPI
                packed = FML::COMMUNICATION::gather_records_checked(packed, true);
#endif
                std::vector<BaryonHost> hosts;
                for (size_t i = 0; i + 5 < packed.size(); i += 6)
                    hosts.push_back(
                        {{packed[i], packed[i + 1], packed[i + 2]}, packed[i + 3], packed[i + 4], packed[i + 5]});
                std::sort(hosts.begin(), hosts.end(), [](const BaryonHost & a, const BaryonHost & b) {
                    return std::tie(a.center, a.mass_msunh, a.radius_box) <
                           std::tie(b.center, b.mass_msunh, b.radius_box);
                });
                return hosts;
            }
        } // namespace detail

        /// Find the hosts and displace the particles (positions only), then move them to the task owning their new
        /// position. baryon_fraction: Omega_b / Omega_m of the profiles. Collective; an error on any task throws on
        /// all tasks
        template <class P>
        BaryonificationResult baryonify(FML::PARTICLE::MPIParticles<P> & particles,
                                        const BaryonificationSettings & s,
                                        const AnalysisContext & context,
                                        double baryon_fraction) {
            using namespace FML::BARYONIFICATION;
            s.validate();
            context.validate();
            const double box = context.boxsize;
            const double redshift = context.redshift();
            const auto model = s.profile_model(baryon_fraction);
            model.validate();
            const ProfileNumerics numerics{size_t(s.profile_radial_bins), s.min_radius_over_r200, s.max_radius_over_r200};

            BaryonificationResult result;
            result.baryon_fraction = baryon_fraction;
            double total_mass = 0.0;
            for (size_t i = 0; i < particles.get_npart(); ++i)
                total_mass += FML::PARTICLE::GetMass(particles[i]);
            FML::SumOverTasks(&total_mass);
            result.npart_total = (long long)particles.get_npart_total();
            if (result.npart_total == 0 or not(total_mass > 0.0))
                throw std::runtime_error("Cannot baryonify an empty (or zero-mass) particle set");

            HaloSettings finder;
            finder.linking_length = s.fof_linking_length;
            finder.nmin_per_halo = s.fof_min_particles;
            finder.nmesh_max = s.fof_nmesh_max;
            finder.buffer_length_mpch = s.fof_buffer_mpch;
            result.Delta_200c = so_overdensity("200c", context, context.Omega_of_particles(total_mass));
            auto records = find_fof_halos<3, P>(particles.get_particles_ptr(),
                                                particles.get_npart(),
                                                finder,
                                                box,
                                                finder.linking_length,
                                                finder.nmin_per_halo,
                                                {result.Delta_200c},
                                                total_mass);
            flag_overlapping_so_halos<3>(records);

            result.particle_mass_msunh = total_mass * context.mass_to_msunh / double(result.npart_total);
            const double background_mass_per_box_volume = total_mass * context.mass_to_msunh;
            result.mean_density_msunh_mpch3 = background_mass_per_box_volume / (box * box * box);
            std::vector<BaryonHost> local;
            for (const auto & h : records) {
                if (h.so_truncated) {
                    ++result.truncated;
                    continue;
                }
                if (h.so_overlap) {
                    ++result.nested;
                    continue;
                }
                // An M200c from fewer particles than a FoF seed is noise, not a host
                if (h.so_npart[0] < s.fof_min_particles) {
                    ++result.unresolved;
                    continue;
                }
                const double mass = h.so_mass[0] * context.mass_to_msunh;
                if (mass < s.min_halo_mass_msunh or h.so_radius[0] <= 0.0) {
                    ++result.below_cut;
                    continue;
                }
                // The periodic Baryonifier needs a support (box units) below half the box
                if (s.max_radius_over_r200 * h.so_radius[0] >= 0.5) {
                    ++result.too_large;
                    continue;
                }
                local.push_back({h.so_pos, mass, h.so_radius[0], s.concentration(mass, redshift)});
            }
            FML::SumOverTasks(&result.truncated);
            FML::SumOverTasks(&result.nested);
            FML::SumOverTasks(&result.unresolved);
            FML::SumOverTasks(&result.below_cut);
            FML::SumOverTasks(&result.too_large);
            result.hosts = detail::gather_hosts(local);
            const auto & hosts = result.hosts;

            // Only the hosts whose support reaches a local particle (in x) are needed on this task: build the
            // profiles of those (in parallel) instead of all the hosts on every task
            double xlo = std::numeric_limits<double>::max(), xhi = -std::numeric_limits<double>::max();
            for (size_t i = 0; i < particles.get_npart(); ++i) {
                const double x = double(FML::PARTICLE::GetPos(particles[i])[0]);
                xlo = std::min(xlo, x);
                xhi = std::max(xhi, x);
            }
            std::vector<size_t> needed;
            if (particles.get_npart() > 0) {
                for (size_t i = 0; i < hosts.size(); ++i) {
                    const double support = numerics.maximum_radius * hosts[i].radius_box;
                    const double x = hosts[i].center[0];
                    double distance = 1.0;
                    for (int image = -1; image <= 1; ++image)
                        distance = std::min(distance, std::max({xlo - (x + image), (x + image) - xhi, 0.0}));
                    if (distance <= support)
                        needed.push_back(i);
                }
            }
            std::vector<std::optional<RadialMapping>> mappings(needed.size());
            std::string error;
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
            for (size_t j = 0; j < needed.size(); ++j) {
                const auto & h = hosts[needed[j]];
                try {
                    auto profiles = make_halo_profiles({h.mass_msunh, h.radius_box, h.concentration},
                                                       model,
                                                       background_mass_per_box_volume,
                                                       numerics);
                    mappings[j].emplace(profiles.mapping());
                } catch (const std::exception & e) {
#ifdef USE_OMP
#pragma omp critical
#endif
                    error = e.what();
                }
            }
            throw_if_any_task_failed(error, "Baryonification: building a halo profile");

            if (not hosts.empty()) {
                std::vector<HaloDisplacement> corrections;
                corrections.reserve(needed.size());
                for (size_t j = 0; j < needed.size(); ++j)
                    corrections.push_back({hosts[needed[j]].center, std::move(*mappings[j])});
                Baryonifier field(std::move(corrections), 1.0);
                field.apply(particles.get_particles_ptr(), particles.get_npart());
                redistribute_particles(particles);
            }
            return result;
        }

        /// The provenance lines (settings, rejection counts) of the baryonification outputs
        inline std::string baryonification_provenance(const BaryonificationResult & r,
                                                      const BaryonificationSettings & s,
                                                      const AnalysisContext & context) {
            std::ostringstream out;
            out << std::setprecision(10);
            out << "# Position-only baryonification: finite support, uniform background, uncalibrated "
                   "profiles/concentrations\n"
                << "# Schneider et al. (2019), arXiv:1810.08629; Schneider & Teyssier (2015), arXiv:1510.06034\n"
                << "# redshift = " << context.redshift() << "\n# baryon_fraction = " << r.baryon_fraction
                << "\n# boxsize_mpch = " << context.boxsize << "\n# particle_mass_msunh = " << r.particle_mass_msunh
                << "\n# mean_density_msunh_per_mpch3 = " << r.mean_density_msunh_mpch3
                << "\n# Delta_200c_over_mean = " << r.Delta_200c << "\n# selected_host_count = " << r.hosts.size()
                << "\n# rejected_truncated = " << r.truncated << "\n# rejected_nested = " << r.nested
                << "\n# rejected_unresolved = " << r.unresolved << "\n# rejected_below_cut = " << r.below_cut
                << "\n# rejected_support_over_half_box = " << r.too_large
                << "\n# velocities = unchanged\n# halo_mass_definition = 200c\n";
            s.write(out);
            return out.str();
        }

        /// The host catalogue: x, y, z (Mpc/h), M200c (Msun/h), R200c (Mpc/h), concentration. Collective
        inline void write_baryonification_hosts(const std::string & filename,
                                                const BaryonificationResult & r,
                                                const BaryonificationSettings & s,
                                                const AnalysisContext & context) {
            const double box = context.boxsize;
            write_on_root(filename, [&](std::ostream & out) {
                out << baryonification_provenance(r, s, context);
                out << "# x_mpch y_mpch z_mpch M200c_msunh R200c_mpch concentration\n";
                for (const auto & h : r.hosts)
                    out << h.center[0] * box << ' ' << h.center[1] * box << ' ' << h.center[2] * box << ' '
                        << h.mass_msunh << ' ' << h.radius_box * box << ' ' << h.concentration << '\n';
            });
        }

        /// P(k) after and before baryonification (same binning) and their ratio. Collective
        template <int NDIM>
        void write_baryonified_power_spectrum(const std::string & filename,
                                              const FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> & original,
                                              const FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> & corrected,
                                              const std::string & header) {
            write_on_root(filename, [&](std::ostream & out) {
                out << header;
                out << "# k_hmpc P_corrected_mpch3 P_original_mpch3 ratio ratio_valid\n";
                for (int i = 0; i < corrected.n; ++i) {
                    const double before = original.pofk[i], after = corrected.pofk[i];
                    if (not std::isfinite(before) or not std::isfinite(after))
                        throw std::runtime_error("Non-finite baryonification power spectrum");
                    const bool valid = before > 0.0 and std::isfinite(after / before);
                    out << corrected.kbin[i] << ' ' << after << ' ' << before << ' ' << (valid ? after / before : 0.0)
                        << ' ' << int(valid) << '\n';
                }
            });
        }

    } // namespace ANALYSIS
} // namespace FML
#endif

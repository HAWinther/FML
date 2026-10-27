#include "Cosmology.h"
#include "BaryonificationOutput.h"
#include "OutputUtils.h"
#include "HaloFinding.h"
#include "SnapshotPowerSpectrum.h"
#include <FML/Baryonification/Baryonification.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <climits>
#include <fstream>
#include <iomanip>
#include <tuple>

namespace COLASOLVER {
    namespace {
        using Particle = BaryonificationParticle;
        using Container = FML::PARTICLE::MPIParticles<Particle>;

        struct HostHalo {
            std::array<double, 3> center; // box units
            double mass_msunh, radius_box;
        };

        std::vector<HostHalo> all_hosts(const std::vector<HostHalo> & local) {
            std::vector<double> packed;
            for (const auto & h : local) {
                packed.insert(packed.end(), h.center.begin(), h.center.end());
                packed.push_back(h.mass_msunh);
                packed.push_back(h.radius_box);
            }
#ifdef USE_MPI
            int too_large = packed.size() > size_t(INT_MAX);
            FML::SumOverTasks(&too_large);
            if (too_large) throw std::runtime_error("Baryonification halo catalogue exceeds MPI count limit");
            const int count = int(packed.size());
            std::vector<int> counts(FML::NTasks), offsets(FML::NTasks);
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
            long long total = 0;
            for (int i = 0; i < FML::NTasks; ++i) {
                if (total + counts[i] > INT_MAX)
                    throw std::runtime_error("Baryonification replicated catalogue exceeds MPI count limit");
                offsets[i] = int(total);
                total += counts[i];
            }
            std::vector<double> gathered(size_t(total), 0.0);
            MPI_Allgatherv(packed.data(), count, MPI_DOUBLE, gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                           FML::communicator);
            packed = std::move(gathered);
#endif
            std::vector<HostHalo> hosts;
            for (size_t i = 0; i < packed.size(); i += 5)
                hosts.push_back({{packed[i], packed[i + 1], packed[i + 2]}, packed[i + 3], packed[i + 4]});
            // A consistent sum order across MPI decompositions (independent of local FoF enumeration).
            std::sort(hosts.begin(), hosts.end(), [](const HostHalo & a, const HostHalo & b) {
                return std::tie(a.center, a.mass_msunh, a.radius_box) < std::tie(b.center, b.mass_msunh, b.radius_box);
            });
            return hosts;
        }

        template <class Writer> void write_root(const std::string & filename, Writer writer) {
            int failed = 0;
            if (FML::ThisTask == 0) {
                try {
                    std::ofstream out(filename);
                    out << std::setprecision(17) << std::boolalpha;
                    if (out) writer(out);
                    out.flush();
                    failed = !out;
                } catch (const std::exception & e) {
                    std::cerr << e.what() << '\n';
                    failed = 1;
                }
            }
            FML::SumOverTasks(&failed);
            if (failed) throw std::runtime_error("Cannot write baryonification output " + filename);
        }

        void redistribute(Container & particles) {
            // Allocate from the actual destination counts, including initially empty ranks. A fixed buffer
            // factor could fail precisely where feedback moves particles across a domain boundary.
            std::vector<size_t> counts(size_t(FML::NTasks), 0);
            for (size_t i = 0; i < particles.get_npart(); ++i) {
                const int rank = std::min(FML::NTasks - 1, int(particles[i].pos[0] * FML::NTasks));
                ++counts[size_t(rank)];
            }
            FML::SumArrayOverTasks(counts.data(), FML::NTasks);
            auto & storage = particles.get_particles();
            storage.resize(std::max(storage.size(), counts[size_t(FML::ThisTask)]));
            particles.communicate_particles();
        }
    }

    void output_baryonified_snapshot(FML::Vector<Particle> snapshot,
                                     const SimulationParameters & params,
                                     const std::shared_ptr<Cosmology> & cosmo,
                                     double redshift, const std::string & folder) {
        using namespace FML::BARYONIFICATION;
        const auto & s = params.baryonification;
        s.validate();
        const double box = params.simulation.boxsize, a = 1.0 / (1.0 + redshift);
        const double omega = cosmo->get_OmegaCDM() + cosmo->get_Omegab();
        const auto model = s.profile_model(cosmo->get_Omegab() / omega);
        model.validate();
        const ProfileNumerics numerics{size_t(s.profile_radial_bins), s.min_radius_over_r200, s.max_radius_over_r200};
        Container particles;
        particles.move_from(std::move(snapshot));
        const auto total = particles.get_npart_total();
        if (total == 0) throw std::runtime_error("Cannot baryonify an empty snapshot");

        FoFSettings finder{};
        finder.linking_length = s.fof_linking_length;
        finder.nmin_per_halo = s.fof_min_particles;
        finder.nmesh_max = s.fof_nmesh_max;
        finder.buffer_length_mpch = s.fof_buffer_mpch;
        auto records = find_fof_halos<3, Particle>(
            particles.get_particles_ptr(), particles.get_npart(), finder, box, finder.linking_length,
            finder.nmin_per_halo, {so_overdensity("200c", *cosmo, a)}, double(total));
        flag_overlapping_so_halos<3>(records);

        // Exactly the mass convention used by the existing halo catalogue writer (Msun/h).
        const double mass_per_particle = halo_particle_mass_msunh(omega, box, 1.0 / 2997.92458, total);
        const double background_mass_per_box_volume = mass_per_particle * double(total);
        std::vector<HostHalo> local;
        long long truncated = 0, nested = 0, unresolved = 0, below_cut = 0, too_large = 0;
        for (const auto & h : records) {
            if (h.so_truncated) { ++truncated; continue; }
            if (h.so_overlap) { ++nested; continue; }
            // An M200c from fewer particles than a FoF seed is noise, not a host
            if (h.so_npart[0] < s.fof_min_particles) { ++unresolved; continue; }
            const double mass = h.so_mass[0] * mass_per_particle;
            if (mass < s.min_halo_mass_msunh || h.so_radius[0] <= 0.0) { ++below_cut; continue; }
            // The periodic Baryonifier needs a support (box units) below half the box: skip such a host rather than
            // aborting the run
            if (s.max_radius_over_r200 * h.so_radius[0] >= 0.5) { ++too_large; continue; }
            local.push_back({h.so_pos, mass, h.so_radius[0]});
        }
        FML::SumOverTasks(&truncated);
        FML::SumOverTasks(&nested);
        FML::SumOverTasks(&unresolved);
        FML::SumOverTasks(&below_cut);
        FML::SumOverTasks(&too_large);
        const auto hosts = all_hosts(local);
        std::vector<HaloDisplacement> corrections;
        std::vector<double> concentrations;
        corrections.reserve(hosts.size());
        for (const auto & h : hosts) {
            const double c = s.concentration_norm * std::pow(h.mass_msunh / s.concentration_pivot_mass_msunh,
                                                           s.concentration_mass_slope) *
                             std::pow(1.0 + redshift, s.concentration_redshift_power);
            auto profiles = make_halo_profiles({h.mass_msunh, h.radius_box, c}, model,
                                               background_mass_per_box_volume, numerics);
            corrections.push_back({h.center, profiles.mapping()});
            concentrations.push_back(c);
        }
        Baryonifier field(std::move(corrections), 1.0);

        // Measure both spectra on the copy, after the same FoF reordering. Identical binning/normalization
        // makes the empty-catalogue control exactly unity in every nonempty, positive-power bin.
        const auto original = measure_snapshot_power_spectrum<3>(particles.get_particles_ptr(), particles.get_npart(),
                                                                  total, params.pofk, box);
        if (!hosts.empty()) {
            field.apply(particles.get_particles_ptr(), particles.get_npart());
            redistribute(particles);
        }
        const auto corrected = measure_snapshot_power_spectrum<3>(particles.get_particles_ptr(), particles.get_npart(),
                                                                   total, params.pofk, box);
        const std::string z = redshift_to_string(redshift);
        auto provenance = [&](std::ostream & out) {
            out << "# Position-only baryonification: finite support, uniform background, uncalibrated profiles/concentrations\n"
                << "# redshift = " << redshift << "\n# baryon_fraction = " << model.baryon_fraction
                << "\n# boxsize_mpch = " << box << "\n# particle_mass_msunh = " << mass_per_particle
                << "\n# mean_density_msunh_per_mpch3 = " << background_mass_per_box_volume / (box * box * box)
                << "\n# selected_host_count = " << hosts.size() << "\n# rejected_truncated = " << truncated
                << "\n# rejected_nested = " << nested << "\n# rejected_unresolved = " << unresolved
                << "\n# rejected_below_cut = " << below_cut << "\n# rejected_support_over_half_box = " << too_large
                << "\n# velocities = unchanged\n# halo_mass_definition = 200c\n"
                << "# pofk_nmesh = " << params.pofk.nmesh << "\n# pofk_density_assignment_method = "
                << params.pofk.density_assignment_method << "\n# pofk_interlacing = " << params.pofk.interlacing
                << "\n# pofk_subtract_shotnoise = " << params.pofk.subtract_shotnoise << '\n';
            s.write(out);
        };
        write_root(folder + "/baryonification_z" + z + ".txt", [&](std::ostream & out) {
            provenance(out);
            out << "# x_mpch y_mpch z_mpch M200c_msunh R200c_mpch concentration\n";
            for (size_t i = 0; i < hosts.size(); ++i) {
                const auto & h = hosts[i];
                out << h.center[0] * box << ' ' << h.center[1] * box << ' ' << h.center[2] * box << ' '
                    << h.mass_msunh << ' ' << h.radius_box * box << ' ' << concentrations[i] << '\n';
            }
        });
        write_root(folder + "/pofk_baryonified_z" + z + ".txt", [&](std::ostream & out) {
            provenance(out);
            out << "# k_hmpc P_corrected_mpch3 P_original_mpch3 ratio ratio_valid\n";
            for (int i = 0; i < corrected.n; ++i) {
                const double before = original.pofk[i], after = corrected.pofk[i];
                if (!std::isfinite(before) || !std::isfinite(after))
                    throw std::runtime_error("Non-finite baryonification power spectrum");
                const bool valid = before > 0.0 && std::isfinite(after / before);
                out << corrected.kbin[i] << ' ' << after << ' ' << before << ' '
                    << (valid ? after / before : 0.0) << ' ' << int(valid) << '\n';
            }
        });

        if (s.output_particles) {
            const std::string prefix = folder + "/gadget_baryonified_z" + z;
            const double vel_norm = 100.0 * box / std::pow(a, 1.5);
            if (params.output.fileformat == ParticleFileFormat::GADGET) {
                FML::FILEUTILS::GADGET::GadgetWriter writer(3);
                writer.write_gadget_single(prefix + "." + std::to_string(FML::ThisTask), particles.get_particles_ptr(),
                    particles.get_npart(), total, FML::NTasks, a, box, cosmo->get_OmegaM(), cosmo->get_OmegaLambda(),
                    cosmo->get_h(), box, vel_norm);
            } else {
#ifdef USE_HDF5
                FML::FILEUTILS::GADGET::GadgetHDF5Writer writer(3);
                const std::string filename = FML::NTasks == 1 ? prefix + ".hdf5" :
                    prefix + "." + std::to_string(FML::ThisTask) + ".hdf5";
                writer.write_gadget_single(filename, particles.get_particles_ptr(), particles.get_npart(), total,
                    FML::NTasks, a, box, cosmo->get_OmegaM(), cosmo->get_OmegaLambda(), cosmo->get_h(), box, vel_norm);
#else
                throw std::runtime_error("Baryonified HDF5 output needs USE_HDF5=true");
#endif
            }
        }
        if (FML::ThisTask == 0)
            std::cout << "Baryonification at z=" << redshift << ": " << hosts.size()
                      << " hosts; rejected " << truncated << " truncated, " << nested << " nested, " << unresolved
                      << " with fewer than " << s.fof_min_particles << " particles in R200c and " << too_large
                      << " whose support reaches half the box\n";
    }
}

#ifndef FML_SNAPSHOTANALYZER_SNAPSHOTANALYZER_H
#define FML_SNAPSHOTANALYZER_SNAPSHOTANALYZER_H

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/PowerSpectrumBinning.h>
#include <FML/Baryonification/Baryonification.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Global/MPICounts.h>
#include <FML/VoidFinders/VoxelVoidFinder.h>
#include <FML/VoidFinders/SphericalVoidFinder.h>
#ifdef USE_CGAL
#include <FML/VoidFinders/ZobovVoidFinder.h>
#include <FML/Triangulation/DTFE.h>
#endif

#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace FML {
    namespace SNAPSHOTANALYZER {

        /// Minimal particle representation owned by the standalone analyzer.
        struct SnapshotParticle {
            std::array<double, 3> pos{{0.0, 0.0, 0.0}};
            std::array<double, 3> vel{{0.0, 0.0, 0.0}};
            double mass{0.0};
            long long id{0};
            int family{0};

            double * get_pos() { return pos.data(); }
            const double * get_pos() const { return pos.data(); }
            double * get_vel() { return vel.data(); }
            const double * get_vel() const { return vel.data(); }
            double get_mass() const { return mass; }
            void set_mass(double value) { mass = value; }
            long long get_id() const { return id; }
            void set_id(long long value) { id = value; }
            int get_family() const { return family; }
            void set_family(int value) { family = value; }
            int get_ndim() const { return 3; }
        };

        struct AnalyzerOptions {
            std::string snapshot_input;
            std::string output_file{"snapshot_analysis.txt"};
            bool pofk{true};
            int pofk_nmesh{256};
            int pofk_nbins{40};
            bool pofk_interlacing{true};
            bool pofk_subtract_shotnoise{true};
            std::string pofk_density_assignment_method{"PCS"};
            bool pofk_multipole{false};
            int pofk_multipole_nmesh{256};
            int pofk_multipole_nbins{40};
            int pofk_multipole_ellmax{4};
            bool pofk_multipole_interlacing{true};
            bool pofk_multipole_subtract_shotnoise{true};
            std::string pofk_multipole_density_assignment_method{"PCS"};
            double pofk_multipole_velocity_to_displacement{0.0};
            bool fof{false};
            double fof_linking_length{0.2};
            int fof_nmin{20};
            double fof_buffer_length_mpch{5.0};
            int fof_nmesh_max{512};
            std::string fof_output_file{"fof_catalogue.txt"};
            bool so{false};
            std::vector<double> so_overdensity{200.0};
            double so_initial_overdensity{200.0};
            double so_search_radius_mpch{5.0};
            int so_nmin_center{20};
            bool voids{false};
            int void_nmesh{0};
            double void_smoothing_scale{-1.0};
            std::string void_smoothing_method{"revolver"};
            std::string void_density_assignment_method{"CIC"};
            double void_min_density{1.0};
            std::string void_merging{"none"};
            double void_link_density_threshold{0.2};
            std::string void_output_file{"void_catalogue.txt"};
            bool spherical_voids{false};
            double spherical_void_delta{-0.8};
            double spherical_void_candidate_density_cut{0.5};
            double spherical_void_max_radius_mean_separation{8.0};
            double spherical_void_min_radius_mean_separation{2.0};
            int spherical_void_recentering_steps{0};
            double spherical_void_max_overlap{0.0};
            std::string spherical_void_output_file{"spherical_void_catalogue.txt"};
            bool zobov_voids{false};
            double zobov_buffer_fraction{0.5};
            double zobov_random_fraction{0.3};
            double zobov_min_density_cut{1.0};
            long long int zobov_min_particles{5};
            std::string zobov_void_output_file{"zobov_void_catalogue.txt"};
            bool dtfe{false};
            int dtfe_nmesh{128};
            double dtfe_buffer_fraction{0.5};
            double dtfe_random_fraction{0.3};
            std::string dtfe_sampling{"cell_average"};
            std::string dtfe_weighting{"volume"};
            int dtfe_nsub{2};
            bool dtfe_write_fields{false};
            std::string dtfe_fields_prefix{"dtfe_fields"};
            std::string dtfe_spectra_file{"dtfe_spectra.txt"};
            bool baryonification{false};
            std::string baryonification_output_prefix{"snapshot_baryonified"};
            std::string baryonification_host_catalogue{"baryonification_hosts.txt"};
            double baryonification_min_halo_mass_msunh{1.0e12};
            double baryonification_overdensity_mean{0.0};
            double baryonification_baryon_fraction{0.157};
            double baryonification_gas_mass_scale_msunh{1.0e14};
            double baryonification_gas_mass_exponent{0.3};
            double baryonification_gas_core_radius_over_r200{0.1};
            double baryonification_gas_ejection_radius_over_r200{4.0};
            double baryonification_truncation_radius_over_r200{4.0};
            double baryonification_stellar_mass_scale_msunh{2.5e11};
            double baryonification_stellar_amplitude{0.09};
            double baryonification_stellar_total_exponent{0.32};
            double baryonification_stellar_central_exponent{0.6};
            double baryonification_stellar_radius_over_r200{0.015};
            bool baryonification_relax_collisionless{true};
            double baryonification_relaxation_amplitude{0.3};
            double baryonification_relaxation_exponent{2.0};
            double baryonification_concentration_norm{5.0};
            double baryonification_concentration_pivot_mass_msunh{1.0e14};
            double baryonification_concentration_mass_slope{-0.1};
            double baryonification_concentration_redshift_power{-1.0};
            double baryonification_fof_linking_length{0.2};
            int baryonification_fof_min_particles{20};
            int baryonification_fof_nmesh_max{512};
            double baryonification_search_buffer_mpch{5.0};
            int baryonification_profile_radial_bins{2048};
            double baryonification_min_radius_over_r200{1e-6};
            double baryonification_max_radius_over_r200{30.0};

            explicit AnalyzerOptions(FML::FILEUTILS::LuaFileParser & lua) {
                snapshot_input = lua.read_string("snapshot_input", "");
                output_file = lua.read_string("output_file", output_file);
                pofk = lua.read_bool("pofk", pofk);
                pofk_nmesh = lua.read_int("pofk_nmesh", pofk_nmesh);
                pofk_nbins = lua.read_int("pofk_nbins", pofk_nbins);
                pofk_interlacing = lua.read_bool("pofk_interlacing", pofk_interlacing);
                pofk_subtract_shotnoise = lua.read_bool("pofk_subtract_shotnoise", pofk_subtract_shotnoise);
                pofk_density_assignment_method =
                    lua.read_string("pofk_density_assignment_method", pofk_density_assignment_method);
                pofk_multipole = lua.read_bool("pofk_multipole", pofk_multipole);
                pofk_multipole_nmesh = lua.read_int("pofk_multipole_nmesh", pofk_multipole_nmesh);
                pofk_multipole_nbins = lua.read_int("pofk_multipole_nbins", pofk_multipole_nbins);
                pofk_multipole_ellmax = lua.read_int("pofk_multipole_ellmax", pofk_multipole_ellmax);
                pofk_multipole_interlacing = lua.read_bool("pofk_multipole_interlacing", pofk_multipole_interlacing);
                pofk_multipole_subtract_shotnoise =
                    lua.read_bool("pofk_multipole_subtract_shotnoise", pofk_multipole_subtract_shotnoise);
                pofk_multipole_density_assignment_method =
                    lua.read_string("pofk_multipole_density_assignment_method", pofk_multipole_density_assignment_method);
                pofk_multipole_velocity_to_displacement =
                    lua.read_double("pofk_multipole_velocity_to_displacement", 0.0);
                fof = lua.read_bool("fof", fof);
                fof_linking_length = lua.read_double("fof_linking_length", fof_linking_length);
                fof_nmin = lua.read_int("fof_nmin", fof_nmin);
                fof_buffer_length_mpch = lua.read_double("fof_buffer_length_mpch", fof_buffer_length_mpch);
                fof_nmesh_max = lua.read_int("fof_nmesh_max", fof_nmesh_max);
                fof_output_file = lua.read_string("fof_output_file", fof_output_file);
                so = lua.read_bool("so", so);
                so_overdensity = lua.read_number_array<double>("so_overdensity", so_overdensity, false);
                so_initial_overdensity = lua.read_double("so_initial_overdensity", so_initial_overdensity);
                so_search_radius_mpch = lua.read_double("so_search_radius_mpch", so_search_radius_mpch);
                so_nmin_center = lua.read_int("so_nmin_center", so_nmin_center);
                voids = lua.read_bool("voids", voids);
                void_nmesh = lua.read_int("void_nmesh", void_nmesh);
                void_smoothing_scale = lua.read_double("void_smoothing_scale", void_smoothing_scale);
                void_smoothing_method = lua.read_string("void_smoothing_method", void_smoothing_method);
                void_density_assignment_method =
                    lua.read_string("void_density_assignment_method", void_density_assignment_method);
                void_min_density = lua.read_double("void_min_density", void_min_density);
                void_merging = lua.read_string("void_merging", void_merging);
                void_link_density_threshold = lua.read_double("void_link_density_threshold", void_link_density_threshold);
                void_output_file = lua.read_string("void_output_file", void_output_file);
                spherical_voids = lua.read_bool("spherical_voids", spherical_voids);
                spherical_void_delta = lua.read_double("spherical_void_delta", spherical_void_delta);
                spherical_void_candidate_density_cut =
                    lua.read_double("spherical_void_candidate_density_cut", spherical_void_candidate_density_cut);
                spherical_void_max_radius_mean_separation =
                    lua.read_double("spherical_void_max_radius_mean_separation", spherical_void_max_radius_mean_separation);
                spherical_void_min_radius_mean_separation =
                    lua.read_double("spherical_void_min_radius_mean_separation", spherical_void_min_radius_mean_separation);
                spherical_void_recentering_steps = lua.read_int("spherical_void_recentering_steps", spherical_void_recentering_steps);
                spherical_void_max_overlap = lua.read_double("spherical_void_max_overlap", spherical_void_max_overlap);
                spherical_void_output_file = lua.read_string("spherical_void_output_file", spherical_void_output_file);
                zobov_voids = lua.read_bool("zobov_voids", zobov_voids);
                zobov_buffer_fraction = lua.read_double("zobov_buffer_fraction", zobov_buffer_fraction);
                zobov_random_fraction = lua.read_double("zobov_random_fraction", zobov_random_fraction);
                zobov_min_density_cut = lua.read_double("zobov_min_density_cut", zobov_min_density_cut);
                zobov_min_particles = lua.read_int("zobov_min_particles", int(zobov_min_particles));
                zobov_void_output_file = lua.read_string("zobov_void_output_file", zobov_void_output_file);
                dtfe = lua.read_bool("dtfe", dtfe);
                dtfe_nmesh = lua.read_int("dtfe_nmesh", dtfe_nmesh);
                dtfe_buffer_fraction = lua.read_double("dtfe_buffer_fraction", dtfe_buffer_fraction);
                dtfe_random_fraction = lua.read_double("dtfe_random_fraction", dtfe_random_fraction);
                dtfe_sampling = lua.read_string("dtfe_sampling", dtfe_sampling);
                dtfe_weighting = lua.read_string("dtfe_weighting", dtfe_weighting);
                dtfe_nsub = lua.read_int("dtfe_nsub", dtfe_nsub);
                dtfe_write_fields = lua.read_bool("dtfe_write_fields", dtfe_write_fields);
                dtfe_fields_prefix = lua.read_string("dtfe_fields_prefix", dtfe_fields_prefix);
                dtfe_spectra_file = lua.read_string("dtfe_spectra_file", dtfe_spectra_file);
                baryonification = lua.read_bool("baryonification", baryonification);
                baryonification_output_prefix = lua.read_string("baryonification_output_prefix", baryonification_output_prefix);
                baryonification_host_catalogue = lua.read_string("baryonification_host_catalogue", baryonification_host_catalogue);
                baryonification_min_halo_mass_msunh = lua.read_double("baryonification_min_halo_mass_msunh", baryonification_min_halo_mass_msunh);
                baryonification_overdensity_mean = lua.read_double("baryonification_overdensity_mean", baryonification_overdensity_mean);
                baryonification_baryon_fraction = lua.read_double("baryonification_baryon_fraction", baryonification_baryon_fraction);
                baryonification_gas_mass_scale_msunh = lua.read_double("baryonification_gas_mass_scale_msunh", baryonification_gas_mass_scale_msunh);
                baryonification_gas_mass_exponent = lua.read_double("baryonification_gas_mass_exponent", baryonification_gas_mass_exponent);
                baryonification_gas_core_radius_over_r200 = lua.read_double("baryonification_gas_core_radius_over_r200", baryonification_gas_core_radius_over_r200);
                baryonification_gas_ejection_radius_over_r200 = lua.read_double("baryonification_gas_ejection_radius_over_r200", baryonification_gas_ejection_radius_over_r200);
                baryonification_truncation_radius_over_r200 = lua.read_double("baryonification_truncation_radius_over_r200", baryonification_truncation_radius_over_r200);
                baryonification_stellar_mass_scale_msunh = lua.read_double("baryonification_stellar_mass_scale_msunh", baryonification_stellar_mass_scale_msunh);
                baryonification_stellar_amplitude = lua.read_double("baryonification_stellar_amplitude", baryonification_stellar_amplitude);
                baryonification_stellar_total_exponent = lua.read_double("baryonification_stellar_total_exponent", baryonification_stellar_total_exponent);
                baryonification_stellar_central_exponent = lua.read_double("baryonification_stellar_central_exponent", baryonification_stellar_central_exponent);
                baryonification_stellar_radius_over_r200 = lua.read_double("baryonification_stellar_radius_over_r200", baryonification_stellar_radius_over_r200);
                baryonification_relax_collisionless = lua.read_bool("baryonification_relax_collisionless", baryonification_relax_collisionless);
                baryonification_relaxation_amplitude = lua.read_double("baryonification_relaxation_amplitude", baryonification_relaxation_amplitude);
                baryonification_relaxation_exponent = lua.read_double("baryonification_relaxation_exponent", baryonification_relaxation_exponent);
                baryonification_concentration_norm = lua.read_double("baryonification_concentration_norm", baryonification_concentration_norm);
                baryonification_concentration_pivot_mass_msunh = lua.read_double("baryonification_concentration_pivot_mass_msunh", baryonification_concentration_pivot_mass_msunh);
                baryonification_concentration_mass_slope = lua.read_double("baryonification_concentration_mass_slope", baryonification_concentration_mass_slope);
                baryonification_concentration_redshift_power = lua.read_double("baryonification_concentration_redshift_power", baryonification_concentration_redshift_power);
                baryonification_fof_linking_length = lua.read_double("baryonification_fof_linking_length", baryonification_fof_linking_length);
                baryonification_fof_min_particles = lua.read_int("baryonification_fof_min_particles", baryonification_fof_min_particles);
                baryonification_fof_nmesh_max = lua.read_int("baryonification_fof_nmesh_max", baryonification_fof_nmesh_max);
                baryonification_search_buffer_mpch = lua.read_double("baryonification_search_buffer_mpch", baryonification_search_buffer_mpch);
                baryonification_profile_radial_bins = lua.read_int("baryonification_profile_radial_bins", baryonification_profile_radial_bins);
                baryonification_min_radius_over_r200 = lua.read_double("baryonification_min_radius_over_r200", baryonification_min_radius_over_r200);
                baryonification_max_radius_over_r200 = lua.read_double("baryonification_max_radius_over_r200", baryonification_max_radius_over_r200);

                if (snapshot_input.empty())
                    throw std::runtime_error("Required Lua option 'snapshot_input' (Gadget file prefix) is empty");
                if (not pofk and not pofk_multipole and not fof and not voids and not spherical_voids and not zobov_voids and
                    not dtfe and not baryonification)
                    throw std::runtime_error("Enable at least one analysis: pofk, multipoles, fof, or a void finder");
                if (pofk_nmesh < 3 or pofk_nbins < 1 or pofk_multipole_nmesh < 3 or pofk_multipole_nbins < 1)
                    throw std::runtime_error("Power-spectrum mesh sizes must be >= 3 and bin counts >= 1");
                if (pofk_multipole and pofk_multipole_velocity_to_displacement <= 0.0)
                    throw std::runtime_error("Set positive pofk_multipole_velocity_to_displacement for RSD analysis");
                if (fof and (fof_linking_length <= 0.0 or fof_nmin < 1 or fof_buffer_length_mpch <= 0.0 or
                             fof_nmesh_max < 1))
                    throw std::runtime_error("Invalid FoF settings: check linking length, nmin, buffer, and mesh size");
                if (so and (not fof or so_overdensity.empty() or so_initial_overdensity <= 0.0 or
                            so_search_radius_mpch <= 0.0 or so_nmin_center < 0))
                    throw std::runtime_error("SO requires fof=true and positive thresholds, radii, and valid nmin");
                for (double delta : so_overdensity)
                    if (not std::isfinite(delta) or delta <= 0.0)
                        throw std::runtime_error("so_overdensity values must be finite and positive");
                if ((voids or spherical_voids) and
                    (void_nmesh < 0 or not std::isfinite(void_min_density) or void_min_density <= 0.0 or
                     not std::isfinite(void_smoothing_scale) or not std::isfinite(void_link_density_threshold) or
                     void_link_density_threshold <= 0.0 or
                     (void_nmesh > 0 and (void_nmesh < FML::NTasks or void_nmesh % FML::NTasks != 0))))
                    throw std::runtime_error("Invalid voxel void settings: check void_nmesh and void_min_density");
                if ((voids or zobov_voids) and void_merging != "none" and void_merging != "threshold" and
                    void_merging != "full")
                    throw std::runtime_error("void_merging must be 'none', 'threshold', or 'full'");
                if ((voids or spherical_voids) and void_smoothing_method != "revolver" and
                    void_smoothing_method != "gaussian")
                    throw std::runtime_error("void_smoothing_method must be 'revolver' or 'gaussian'");
                if (spherical_voids and (not std::isfinite(spherical_void_delta) or spherical_void_delta <= -1.0 or
                                         spherical_void_delta >= 0.0 or
                                         not std::isfinite(spherical_void_candidate_density_cut) or
                                         not std::isfinite(spherical_void_max_radius_mean_separation) or
                                         spherical_void_max_radius_mean_separation <= 0.0 or
                                         not std::isfinite(spherical_void_min_radius_mean_separation) or
                                         spherical_void_min_radius_mean_separation <= 0.0 or
                                         spherical_void_recentering_steps < 0 or
                                         not std::isfinite(spherical_void_max_overlap) or spherical_void_max_overlap < 0.0 or
                                         spherical_void_max_overlap >= 1.0))
                    throw std::runtime_error("Invalid spherical void settings");
                if (zobov_voids and (not std::isfinite(zobov_buffer_fraction) or zobov_buffer_fraction < 0.0 or
                                     not std::isfinite(zobov_random_fraction) or zobov_random_fraction < 0.0 or
                                     not std::isfinite(zobov_min_density_cut) or zobov_min_density_cut <= 0.0 or
                                     zobov_min_particles < 1))
                    throw std::runtime_error("Invalid ZOBOV settings");
                if (dtfe and (dtfe_nmesh < 3 or dtfe_nmesh < FML::NTasks or dtfe_nmesh % FML::NTasks != 0 or
                              dtfe_nsub < 1 or
                              not std::isfinite(dtfe_buffer_fraction) or dtfe_buffer_fraction < 0.0 or
                              not std::isfinite(dtfe_random_fraction) or dtfe_random_fraction < 0.0))
                    throw std::runtime_error("Invalid DTFE settings: grid must divide across tasks and fractions/nsub be valid");
                if (dtfe and dtfe_sampling != "grid_point" and dtfe_sampling != "cell_average")
                    throw std::runtime_error("dtfe_sampling must be 'grid_point' or 'cell_average'");
                if (dtfe and dtfe_weighting != "volume" and dtfe_weighting != "mass")
                    throw std::runtime_error("dtfe_weighting must be 'volume' or 'mass'");
                if (baryonification and (baryonification_output_prefix.empty() or baryonification_host_catalogue.empty() or
                                         not std::isfinite(baryonification_min_halo_mass_msunh) or
                                         baryonification_min_halo_mass_msunh <= 0.0 or
                                         not std::isfinite(baryonification_overdensity_mean) or
                                         baryonification_overdensity_mean < 0.0 or
                                         not std::isfinite(baryonification_fof_linking_length) or
                                         baryonification_fof_linking_length <= 0.0 or
                                         baryonification_fof_min_particles < 1 or
                                         baryonification_fof_nmesh_max < 1 or
                                         not std::isfinite(baryonification_search_buffer_mpch) or
                                         baryonification_search_buffer_mpch <= 0.0 or
                                         baryonification_profile_radial_bins < 32 or
                                         not std::isfinite(baryonification_min_radius_over_r200) or
                                         baryonification_min_radius_over_r200 <= 0.0 or
                                         baryonification_min_radius_over_r200 >= 1.0 or
                                         not std::isfinite(baryonification_max_radius_over_r200) or
                                         baryonification_max_radius_over_r200 <= 1.0 or
                                         not std::isfinite(baryonification_concentration_pivot_mass_msunh) or
                                         baryonification_concentration_pivot_mass_msunh <= 0.0))
                    throw std::runtime_error("Invalid baryonification settings");
            }
        };

        class SnapshotAnalyzer {
          public:
            explicit SnapshotAnalyzer(AnalyzerOptions options) : options(std::move(options)) {}

            void run() {
                FML::FILEUTILS::GADGET::GadgetReader reader(3);
                reader.read_gadget(options.snapshot_input, particles, 1.0, true, FML::ThisTask == 0);
                header = reader.get_header();
                if (not std::isfinite(header.BoxSize) or header.BoxSize <= 0.0)
                    throw std::runtime_error("Gadget snapshot has invalid BoxSize");
                if (options.baryonification)
                    validate_baryonification_output_paths();

                // Constant-mass Gadget species are represented by the header table; variable-mass
                // species were populated from the MASS block by GadgetReader.
                for (auto & p : particles)
                    if (header.mass[p.get_family()] > 0.0)
                        p.set_mass(header.mass[p.get_family()]);

                mpiparticles.create(particles, particles.size(), [](SnapshotParticle &) { return true; });
                if (FML::ThisTask == 0) {
                    std::cout << "Snapshot scale factor: " << header.time << "\n"
                              << "Box size [Mpc/h]: " << header.BoxSize << "\n"
                              << "Particles: " << mpiparticles.get_npart_total() << "\n";
                }

                std::ofstream output;
                if (FML::ThisTask == 0) {
                    output.open(options.output_file);
                    if (not output)
                        throw std::runtime_error("Could not open output file: " + options.output_file);
                    output << "# SnapshotAnalyzer standalone output\n"
                           << "# scale_factor " << std::setprecision(12) << header.time << "\n"
                           << "# box_size_Mpch " << header.BoxSize << "\n";
                }

                if (options.pofk)
                    measure_pofk(output);
                if (options.pofk_multipole)
                    measure_multipoles(output);
                if (options.fof)
                    find_halos();
                if (options.voids)
                    find_voids();
                if (options.spherical_voids)
                    find_spherical_voids();
                if (options.zobov_voids)
                    find_zobov_voids();
                if (options.dtfe)
                    analyze_dtfe();
                if (options.baryonification)
                    baryonify_snapshot();
            }

          private:
            AnalyzerOptions options;
            std::vector<SnapshotParticle> particles;
            FML::PARTICLE::MPIParticles<SnapshotParticle> mpiparticles;
            FML::FILEUTILS::GADGET::GadgetHeader header{};

            template <class Binning>
            void write_binning(std::ofstream & output, const std::string & label, Binning & binning) {
                binning.scale(header.BoxSize);
                if (FML::ThisTask != 0)
                    return;
                output << "\n# " << label << "\n# k_h_Mpc P_(Mpc_h)^3 modes\n";
                for (int i = 0; i < binning.n; i++)
                    output << std::setprecision(12) << binning.kbin[i] << " " << binning.pofk[i] << " "
                           << binning.count[i] << "\n";
            }

            void measure_pofk(std::ofstream & output) {
                using namespace FML::CORRELATIONFUNCTIONS;
                PowerSpectrumBinning<3> pofk(2.0 * 3.141592653589793,
                                              3.141592653589793 * options.pofk_nmesh,
                                              options.pofk_nbins, PowerSpectrumBinning<3>::LOG_SPACING);
                pofk.subtract_shotnoise = options.pofk_subtract_shotnoise;
                compute_power_spectrum<3>(options.pofk_nmesh,
                                          mpiparticles.get_particles_ptr(),
                                          mpiparticles.get_npart(),
                                          mpiparticles.get_npart_total(),
                                          pofk,
                                          options.pofk_density_assignment_method,
                                          options.pofk_interlacing);
                write_binning(output, "P(k)", pofk);
            }

            void measure_multipoles(std::ofstream & output) {
                using namespace FML::CORRELATIONFUNCTIONS;
                if (options.pofk_multipole_ellmax < 0 or options.pofk_multipole_ellmax % 2 != 0)
                    throw std::runtime_error("pofk_multipole_ellmax must be a non-negative even integer");
                std::vector<PowerSpectrumBinning<3>> pell(options.pofk_multipole_ellmax / 2 + 1,
                                                         PowerSpectrumBinning<3>(2.0 * 3.141592653589793,
                                                             3.141592653589793 * options.pofk_multipole_nmesh,
                                                             options.pofk_multipole_nbins,
                                                             PowerSpectrumBinning<3>::LOG_SPACING));
                for (auto & p : pell)
                    p.subtract_shotnoise = options.pofk_multipole_subtract_shotnoise;
                compute_power_spectrum_multipoles<3>(options.pofk_multipole_nmesh,
                                                     mpiparticles,
                                                     options.pofk_multipole_velocity_to_displacement,
                                                     pell,
                                                     options.pofk_multipole_density_assignment_method,
                                                     options.pofk_multipole_interlacing);
                for (size_t ell = 0; ell < pell.size(); ell++) {
                    auto binning = pell[ell];
                    write_binning(output, "P_ell=" + std::to_string(2 * ell) + "(k)", binning);
                }
            }

            // Friends-of-Friends grouping follows Davis et al. (1985), doi:10.1086/163168.
            void find_halos() {
                using Halo = FML::FOF::FoFHalo<SnapshotParticle, 3>;
                std::vector<Halo> halos;
                FML::FOF::FriendsOfFriends<SnapshotParticle, 3>(mpiparticles.get_particles_ptr(),
                                                                mpiparticles.get_npart(),
                                                                options.fof_linking_length,
                                                                options.fof_nmin,
                                                                true,
                                                                options.fof_buffer_length_mpch / header.BoxSize,
                                                                halos,
                                                                options.fof_nmesh_max);

                std::vector<FML::FOF::SOHalo<3>> so_halos;
                if (options.so) {
                    double total_mass = 0.0;
                    for (const auto & p : particles)
                        total_mass += p.get_mass();
                    FML::SumOverTasks(&total_mass);

                    std::vector<std::array<double, 3>> centers;
                    std::vector<double> initial_radius;
                    centers.reserve(halos.size());
                    initial_radius.reserve(halos.size());
                    for (const auto & h : halos) {
                        centers.push_back(h.pos);
                        initial_radius.push_back(1.5 * std::cbrt(3.0 * h.mass /
                            (4.0 * M_PI * options.so_initial_overdensity * total_mass)));
                    }
                    so_halos = FML::FOF::SphericalOverdensity<SnapshotParticle, 3>(
                        mpiparticles.get_particles_ptr(),
                        mpiparticles.get_npart(),
                        total_mass,
                        centers,
                        initial_radius,
                        options.so_overdensity,
                        options.so_search_radius_mpch / header.BoxSize,
                        options.so_nmin_center);
                }

                // FoF returns halo records on their owner tasks. Serialize appends for filesystems
                // that do not support concurrent writes to a shared file.
                for (int task = 0; task < FML::NTasks; task++) {
                    if (task == FML::ThisTask) {
                        std::ofstream catalog(options.fof_output_file,
                                              task == 0 ? std::ios::out : std::ios::app);
                        if (not catalog)
                            throw std::runtime_error("Could not open FoF catalogue: " + options.fof_output_file);
                        if (task == 0) {
                            catalog << "# FoF catalogue; Davis et al. (1985), doi:10.1086/163168\n"
                                    << "# Columns: id np mass_1e10_Msun_h x_Mpc_h y_Mpc_h z_Mpc_h "
                                       "vx_km_s vy_km_s vz_km_s sigma_x_km_s sigma_y_km_s sigma_z_km_s";
                            for (double delta : options.so_overdensity)
                                if (options.so)
                                    catalog << " SO_Delta" << delta << "_mass_1e10_Msun_h SO_Delta" << delta
                                            << "_radius_Mpc_h SO_Delta" << delta << "_npart";
                            if (options.so)
                                catalog << " SO_truncated";
                            catalog << "\n";
                        }
                        for (size_t ih = 0; ih < halos.size(); ih++) {
                            const auto & h = halos[ih];
                            catalog << h.id << " " << h.np << " " << std::setprecision(12) << h.mass << " "
                                    << h.pos[0] * header.BoxSize << " " << h.pos[1] * header.BoxSize << " "
                                    << h.pos[2] * header.BoxSize << " " << h.vel[0] << " " << h.vel[1] << " "
                                    << h.vel[2] << " " << h.vel_rms[0] << " " << h.vel_rms[1] << " "
                                    << h.vel_rms[2];
                            if (options.so)
                                for (size_t i = 0; i < options.so_overdensity.size(); i++)
                                    catalog << " " << so_halos[ih].mass[i] << " "
                                            << so_halos[ih].radius[i] * header.BoxSize << " "
                                            << so_halos[ih].npart[i];
                            if (options.so)
                                catalog << " " << (so_halos[ih].truncated ? 1 : 0);
                            catalog << "\n";
                        }
                    }
                    FML::Barrier();
                }
                if (FML::ThisTask == 0)
                    std::cout << "FoF catalogue written: " << options.fof_output_file << "\n";
            }

            // Voxel watershed follows the approach used by Revolver:
            // https://github.com/seshnadathur/Revolver
            void find_voids() {
                FML::VOIDFINDER::VoxelOptions voxel;
                voxel.Nmesh = options.void_nmesh;
                voxel.smoothing_scale = options.void_smoothing_scale;
                voxel.smoothing_method = options.void_smoothing_method;
                voxel.density_assignment_method = options.void_density_assignment_method;
                voxel.min_dens_cut = options.void_min_density;
                voxel.link_density_threshold = options.void_link_density_threshold;
                if (options.void_merging == "threshold")
                    voxel.merging = FML::VOIDFINDER::ZoneMerging::Threshold;
                else if (options.void_merging == "full")
                    voxel.merging = FML::VOIDFINDER::ZoneMerging::Full;
                else
                    voxel.merging = FML::VOIDFINDER::ZoneMerging::None;

                const auto void_catalogue = FML::VOIDFINDER::VoxelVoidFinder(
                    mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), voxel);
                FML::VOIDFINDER::write_catalogue(options.void_output_file, void_catalogue, header.BoxSize);
                if (FML::ThisTask == 0)
                    std::cout << "Voxel void catalogue written: " << options.void_output_file << " ("
                              << void_catalogue.size() << " voids)\n";
            }

            // Spherical underdensity catalog follows Padilla, Ceccarelli & Lambas (2005):
            // https://doi.org/10.1111/j.1365-2966.2005.09500.x
            void find_spherical_voids() {
                FML::VOIDFINDER::SphericalOptions spherical;
                spherical.delta_threshold = options.spherical_void_delta;
                spherical.candidate_density_cut = options.spherical_void_candidate_density_cut;
                spherical.grid.Nmesh = options.void_nmesh;
                spherical.grid.smoothing_scale = options.void_smoothing_scale;
                spherical.grid.smoothing_method = options.void_smoothing_method;
                spherical.grid.density_assignment_method = options.void_density_assignment_method;
                spherical.max_radius_in_mean_separation = options.spherical_void_max_radius_mean_separation;
                spherical.min_radius_in_mean_separation = options.spherical_void_min_radius_mean_separation;
                spherical.recentering_steps = options.spherical_void_recentering_steps;
                spherical.max_overlap = options.spherical_void_max_overlap;
                const auto catalogue = FML::VOIDFINDER::SphericalVoidFinder(
                    mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), spherical);
                FML::VOIDFINDER::write_catalogue(options.spherical_void_output_file, catalogue, header.BoxSize);
                if (FML::ThisTask == 0)
                    std::cout << "Spherical void catalogue written: " << options.spherical_void_output_file << " ("
                              << catalogue.size() << " voids)\n";
            }

            void find_zobov_voids() {
#ifndef USE_CGAL
                throw std::runtime_error("ZOBOV requires CGAL; rebuild SnapshotAnalyzer with USE_CGAL=true");
#else
                FML::VOIDFINDER::ZobovOptions zobov;
                zobov.buffer_fraction = options.zobov_buffer_fraction;
                zobov.random_fraction = options.zobov_random_fraction;
                zobov.min_dens_cut = options.zobov_min_density_cut;
                zobov.min_num_particles = options.zobov_min_particles;
                zobov.link_density_threshold = options.void_link_density_threshold;
                if (options.void_merging == "threshold")
                    zobov.merging = FML::VOIDFINDER::ZoneMerging::Threshold;
                else if (options.void_merging == "full")
                    zobov.merging = FML::VOIDFINDER::ZoneMerging::Full;
                const auto catalogue = FML::VOIDFINDER::ZobovVoidFinder(
                    mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), zobov);
                FML::VOIDFINDER::write_catalogue(options.zobov_void_output_file, catalogue, header.BoxSize);
                if (FML::ThisTask == 0)
                    std::cout << "ZOBOV void catalogue written: " << options.zobov_void_output_file << " ("
                              << catalogue.size() << " voids)\n";
#endif
            }

            struct BaryonHost {
                std::array<double, 3> center{};
                double mass_msunh{0.0};
                double radius_box{0.0};
            };

            void validate_baryonification_output_paths() {
                const auto norm = [](const std::string & s) {
                    return std::filesystem::absolute(s).lexically_normal();
                };
                std::vector<std::filesystem::path> inputs;
                for (int i = 0; i < header.num_files; i++)
                    inputs.push_back(norm(options.snapshot_input + "." + std::to_string(i)));
                std::vector<std::filesystem::path> analysis_outputs{norm(options.output_file)};
                if (options.fof)
                    analysis_outputs.push_back(norm(options.fof_output_file));
                if (options.voids)
                    analysis_outputs.push_back(norm(options.void_output_file));
                if (options.spherical_voids)
                    analysis_outputs.push_back(norm(options.spherical_void_output_file));
                if (options.zobov_voids)
                    analysis_outputs.push_back(norm(options.zobov_void_output_file));
                if (options.dtfe) {
                    analysis_outputs.push_back(norm(options.dtfe_spectra_file));
                    if (options.dtfe_write_fields)
                        for (int i = 0; i < FML::NTasks; i++)
                            analysis_outputs.push_back(norm(options.dtfe_fields_prefix + ".rank" +
                                                            std::to_string(i) + ".txt"));
                }
                const auto hosts = norm(options.baryonification_host_catalogue);
                std::vector<std::filesystem::path> snapshot_outputs;
                for (int i = 0; i < FML::NTasks; i++)
                    snapshot_outputs.push_back(norm(options.baryonification_output_prefix + "." + std::to_string(i)));
                for (const auto & input : inputs) {
                    if (hosts == input or std::find(analysis_outputs.begin(), analysis_outputs.end(), input) !=
                                              analysis_outputs.end() or
                        std::find(snapshot_outputs.begin(), snapshot_outputs.end(), input) != snapshot_outputs.end())
                        throw std::runtime_error("Baryonification output paths must differ from every input snapshot file");
                }
                std::vector<std::filesystem::path> all_outputs = analysis_outputs;
                all_outputs.push_back(hosts);
                all_outputs.insert(all_outputs.end(), snapshot_outputs.begin(), snapshot_outputs.end());
                auto sorted_outputs = all_outputs;
                std::sort(sorted_outputs.begin(), sorted_outputs.end());
                if (std::adjacent_find(sorted_outputs.begin(), sorted_outputs.end()) != sorted_outputs.end())
                    throw std::runtime_error("Baryonification output and analysis paths must all be distinct");
                int exists = 0;
                if (FML::ThisTask == 0) {
                    exists = std::filesystem::exists(hosts);
                    for (const auto & path : snapshot_outputs)
                        exists |= std::filesystem::exists(path);
                }
                FML::MaxOverTasks(&exists);
                if (exists)
                    throw std::runtime_error("Baryonification refuses to overwrite existing snapshot/catalogue files");
            }

            void baryonify_snapshot() {
                // The profile family and enclosed-mass mapping are implemented in FML/Baryonification.
                // This adapter supplies Gadget metadata, FoF/M200c hosts, and safe snapshot I/O.
                using namespace FML::BARYONIFICATION;
                using FoFHalo = FML::FOF::FoFHalo<SnapshotParticle, 3>;

                const auto input = std::filesystem::absolute(options.snapshot_input).lexically_normal();
                const auto output = std::filesystem::absolute(options.baryonification_output_prefix).lexically_normal();
                if (input == output)
                    throw std::runtime_error("Baryonification output prefix must differ from snapshot_input");
                int output_exists = 0;
                if (FML::ThisTask == 0)
                    for (int task = 0; task < FML::NTasks; task++)
                        output_exists |= std::filesystem::exists(options.baryonification_output_prefix + "." +
                                                                 std::to_string(task));
                if (FML::ThisTask == 0)
                    output_exists |= std::filesystem::exists(options.baryonification_host_catalogue);
                FML::MaxOverTasks(&output_exists);
                if (output_exists)
                    throw std::runtime_error("Baryonification refuses to overwrite existing snapshot/catalogue files");

                std::array<long long, 6> family_count{};
                for (const auto & p : mpiparticles)
                    family_count[size_t(p.get_family())]++;
                FML::SumArrayOverTasks(family_count.data(), family_count.size());
                int occupied_families = 0;
                for (auto n : family_count)
                    occupied_families += n > 0;
                if (occupied_families != 1)
                    throw std::runtime_error("Standalone baryonification currently requires one Gadget particle family "
                                             "representing the total CDM+baryon matter field");

                const double a = header.time;
                if (not std::isfinite(a) or a <= 0.0 or not std::isfinite(header.Omega0) or
                    header.Omega0 <= 0.0 or not std::isfinite(header.OmegaLambda) or
                    not std::isfinite(header.HubbleParam) or header.HubbleParam <= 0.0)
                    throw std::runtime_error("Baryonification needs valid scale factor and Gadget cosmology metadata");
                const double omega_k = 1.0 - header.Omega0 - header.OmegaLambda;
                const double e2 = header.Omega0 / (a * a * a) + omega_k / (a * a) + header.OmegaLambda;
                if (not std::isfinite(e2) or e2 <= 0.0)
                    throw std::runtime_error("Invalid Gadget expansion history for baryonification");
                const double omega_m_a = (header.Omega0 / (a * a * a)) / e2;
                if (not std::isfinite(omega_m_a) or omega_m_a <= 0.0)
                    throw std::runtime_error("Invalid Omega_m(a) for baryonification");
                const double delta_mean = options.baryonification_overdensity_mean > 0.0 ?
                                              options.baryonification_overdensity_mean : 200.0 / omega_m_a;
                if (not std::isfinite(delta_mean) or delta_mean <= 0.0)
                    throw std::runtime_error("Invalid baryonification SO overdensity");

                ModelParameters model;
                model.baryon_fraction = options.baryonification_baryon_fraction;
                model.gas_mass_scale = options.baryonification_gas_mass_scale_msunh;
                model.gas_mass_exponent = options.baryonification_gas_mass_exponent;
                model.gas_core_radius = options.baryonification_gas_core_radius_over_r200;
                model.gas_ejection_radius = options.baryonification_gas_ejection_radius_over_r200;
                model.truncation_radius = options.baryonification_truncation_radius_over_r200;
                model.stellar_mass_scale = options.baryonification_stellar_mass_scale_msunh;
                model.stellar_amplitude = options.baryonification_stellar_amplitude;
                model.stellar_total_exponent = options.baryonification_stellar_total_exponent;
                model.stellar_central_exponent = options.baryonification_stellar_central_exponent;
                model.stellar_radius = options.baryonification_stellar_radius_over_r200;
                model.relax_collisionless = options.baryonification_relax_collisionless;
                model.relaxation_amplitude = options.baryonification_relaxation_amplitude;
                model.relaxation_exponent = options.baryonification_relaxation_exponent;
                model.validate();

                double total_mass = 0.0;
                for (const auto & p : mpiparticles)
                    total_mass += p.get_mass();
                FML::SumOverTasks(&total_mass);
                if (total_mass <= 0.0)
                    throw std::runtime_error("Cannot baryonify an empty or zero-mass snapshot");

                std::vector<FoFHalo> fof_halos;
                FML::FOF::FriendsOfFriends<SnapshotParticle, 3>(mpiparticles.get_particles_ptr(),
                                                                mpiparticles.get_npart(),
                                                                options.baryonification_fof_linking_length,
                                                                options.baryonification_fof_min_particles,
                                                                true,
                                                                options.baryonification_search_buffer_mpch / header.BoxSize,
                                                                fof_halos,
                                                                options.baryonification_fof_nmesh_max);
                std::vector<std::array<double, 3>> centers;
                std::vector<double> initial_radius;
                for (const auto & h : fof_halos) {
                    centers.push_back(h.pos);
                    initial_radius.push_back(1.5 * std::cbrt(3.0 * h.mass /
                        (4.0 * M_PI * delta_mean * total_mass)));
                }
                const auto so = FML::FOF::SphericalOverdensity<SnapshotParticle, 3>(
                    mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), total_mass, centers,
                    initial_radius, {delta_mean}, options.baryonification_search_buffer_mpch / header.BoxSize,
                    options.baryonification_fof_min_particles);
                std::vector<BaryonHost> local_hosts;
                for (size_t i = 0; i < so.size(); i++) {
                    if (so[i].truncated or so[i].npart[0] < options.baryonification_fof_min_particles or
                        so[i].mass[0] * 1.0e10 < options.baryonification_min_halo_mass_msunh or
                        so[i].radius[0] <= 0.0)
                        continue;
                    if (options.baryonification_max_radius_over_r200 * so[i].radius[0] >= 0.5)
                        continue;
                    local_hosts.push_back({so[i].pos, so[i].mass[0] * 1.0e10, so[i].radius[0]});
                }

                std::vector<double> packed;
                for (const auto & h : local_hosts) {
                    packed.insert(packed.end(), h.center.begin(), h.center.end());
                    packed.push_back(h.mass_msunh);
                    packed.push_back(h.radius_box);
                }
                const auto all_packed = FML::COMMUNICATION::gather_records_checked(packed, true);
                std::vector<BaryonHost> hosts;
                for (size_t i = 0; i + 4 < all_packed.size(); i += 5)
                    hosts.push_back({{all_packed[i], all_packed[i + 1], all_packed[i + 2]},
                                     all_packed[i + 3], all_packed[i + 4]});

                FML::KDTREE::pointVec host_points;
                double largest_r200 = 0.0;
                for (const auto & h : hosts) {
                    host_points.emplace_back(h.center.begin(), h.center.end());
                    largest_r200 = std::max(largest_r200, h.radius_box);
                }
                std::vector<BaryonHost> unique_hosts;
                if (not hosts.empty()) {
                    const FML::KDTREE::KDTree host_tree(host_points, 1.0);
                    for (size_t i = 0; i < hosts.size(); i++) {
                        bool nested = false;
                        for (size_t j : host_tree.neighborhood_indices(host_points[i], largest_r200)) {
                            if (i == j)
                                continue;
                            const bool more_massive = hosts[j].mass_msunh > hosts[i].mass_msunh;
                            const bool tie_wins = hosts[j].mass_msunh == hosts[i].mass_msunh &&
                                (hosts[j].center < hosts[i].center ||
                                 (hosts[j].center == hosts[i].center && j < i));
                            if (not more_massive and not tie_wins)
                                continue;
                            double r2 = 0.0;
                            for (int d = 0; d < 3; d++) {
                                double dx = hosts[i].center[d] - hosts[j].center[d];
                                dx -= std::round(dx);
                                r2 += dx * dx;
                            }
                            if (r2 < hosts[j].radius_box * hosts[j].radius_box) {
                                nested = true;
                                break;
                            }
                        }
                        if (not nested)
                            unique_hosts.push_back(hosts[i]);
                    }
                }
                hosts = std::move(unique_hosts);

                const ProfileNumerics numerics{size_t(options.baryonification_profile_radial_bins),
                                               options.baryonification_min_radius_over_r200,
                                               options.baryonification_max_radius_over_r200};
                const double background_density = total_mass * 1.0e10; // box length is one in particle coordinates
                std::vector<HaloDisplacement> corrections;
                std::string profile_error;
                try {
                    for (const auto & h : hosts) {
                        const double support = numerics.maximum_radius * h.radius_box;
                        double distance = 1.0;
                        for (int image = -1; image <= 1; image++) {
                            const double x = h.center[0] + image;
                            distance = std::min(distance,
                                                std::max({FML::xmin_domain - x, x - FML::xmax_domain, 0.0}));
                        }
                        if (distance > support)
                            continue;
                        const double redshift = 1.0 / a - 1.0;
                        const double concentration = options.baryonification_concentration_norm *
                            std::pow(h.mass_msunh / options.baryonification_concentration_pivot_mass_msunh,
                                     options.baryonification_concentration_mass_slope) *
                            std::pow(1.0 + redshift, options.baryonification_concentration_redshift_power);
                        auto profiles = make_halo_profiles({h.mass_msunh, h.radius_box, concentration},
                                                           model, background_density, numerics);
                        corrections.push_back({h.center, profiles.mapping()});
                    }
                } catch (const std::exception & e) {
                    profile_error = e.what();
                }
                int profile_failed = profile_error.empty() ? 0 : 1;
                FML::MaxOverTasks(&profile_failed);
                if (profile_failed)
                    throw std::runtime_error(profile_error.empty() ?
                        "Baryonification profile construction failed on another MPI task" : profile_error);

                Baryonifier displacement(std::move(corrections), 1.0);
                displacement.apply(mpiparticles.get_particles_ptr(), mpiparticles.get_npart());
                redistribute_baryonified_particles();
                write_baryonification_catalogue(hosts, model, delta_mean, numerics);
                write_baryonified_gadget();
                if (FML::ThisTask == 0)
                    std::cout << "Baryonified snapshot written to " << options.baryonification_output_prefix
                              << " (" << hosts.size() << " unique hosts)\n";
            }

            void redistribute_baryonified_particles() {
                std::vector<size_t> counts(size_t(FML::NTasks), 0);
                for (size_t i = 0; i < mpiparticles.get_npart(); i++) {
                    const int destination = std::min(FML::NTasks - 1,
                                                     int(mpiparticles[i].get_pos()[0] * FML::NTasks));
                    counts[size_t(destination)]++;
                }
                FML::SumArrayOverTasks(counts.data(), counts.size());
                auto & storage = mpiparticles.get_particles();
                storage.resize(std::max(storage.size(), counts[size_t(FML::ThisTask)]));
                mpiparticles.communicate_particles();
            }

            void write_baryonification_catalogue(const std::vector<BaryonHost> & hosts,
                                                  const FML::BARYONIFICATION::ModelParameters & model,
                                                  double delta_mean,
                                                  const FML::BARYONIFICATION::ProfileNumerics & numerics) {
                int failed = 0;
                if (FML::ThisTask == 0) {
                    std::ofstream out(options.baryonification_host_catalogue);
                    failed = not bool(out);
                    if (out) {
                        out << std::setprecision(16)
                            << "# Position-only baryonification; Schneider et al. (2019), arXiv:1810.08629\n"
                            << "# Enclosed-mass transport: Schneider & Teyssier (2015), arXiv:1510.06034\n"
                            << "# Illustrative finite-support profiles; no hydrodynamical calibration\n"
                            << "# delta_mean_200c " << delta_mean << " scale_factor " << header.time
                            << " baryon_fraction " << model.baryon_fraction << " radial_bins " << numerics.radial_bins
                            << " max_radius_over_r200 " << numerics.maximum_radius << "\n"
                            << "# gas_mass_scale_msunh " << model.gas_mass_scale
                            << " gas_mass_exponent " << model.gas_mass_exponent
                            << " gas_core_over_r200 " << model.gas_core_radius
                            << " gas_ejection_over_r200 " << model.gas_ejection_radius
                            << " truncation_over_r200 " << model.truncation_radius << "\n"
                            << "# stellar_mass_scale_msunh " << model.stellar_mass_scale
                            << " stellar_amplitude " << model.stellar_amplitude
                            << " stellar_total_exponent " << model.stellar_total_exponent
                            << " stellar_central_exponent " << model.stellar_central_exponent
                            << " stellar_radius_over_r200 " << model.stellar_radius << "\n"
                            << "# relax_collisionless " << model.relax_collisionless
                            << " relaxation_amplitude " << model.relaxation_amplitude
                            << " relaxation_exponent " << model.relaxation_exponent << "\n"
                            << "# concentration_norm " << options.baryonification_concentration_norm
                            << " concentration_pivot_msunh " << options.baryonification_concentration_pivot_mass_msunh
                            << " concentration_mass_slope " << options.baryonification_concentration_mass_slope
                            << " concentration_redshift_power " << options.baryonification_concentration_redshift_power << "\n"
                            << "# x_box y_box z_box M200c_Msun_h R200c_box concentration\n";
                        for (const auto & h : hosts) {
                            const double z = 1.0 / header.time - 1.0;
                            const double c = options.baryonification_concentration_norm *
                                std::pow(h.mass_msunh / options.baryonification_concentration_pivot_mass_msunh,
                                         options.baryonification_concentration_mass_slope) *
                                std::pow(1.0 + z, options.baryonification_concentration_redshift_power);
                            out << h.center[0] << " " << h.center[1] << " " << h.center[2] << " "
                                << h.mass_msunh << " " << h.radius_box << " " << c << "\n";
                        }
                        out.flush();
                        failed |= not bool(out);
                    }
                }
                FML::SumOverTasks(&failed);
                if (failed)
                    throw std::runtime_error("Could not write baryonification host catalogue");
            }

            void write_baryonified_gadget() {
                using namespace FML::FILEUTILS::GADGET;
                const std::string filename = options.baryonification_output_prefix + "." +
                                             std::to_string(FML::ThisTask);
                std::array<long long, 6> count{};
                std::array<double, 6> mass_sum{};
                std::array<std::vector<size_t>, 6> local_index;
                for (size_t i = 0; i < mpiparticles.get_npart(); i++) {
                    const int family = mpiparticles[i].get_family();
                    local_index[size_t(family)].push_back(i);
                    count[size_t(family)]++;
                    mass_sum[size_t(family)] += mpiparticles[i].get_mass();
                }
                FML::SumArrayOverTasks(count.data(), count.size());
                FML::SumArrayOverTasks(mass_sum.data(), mass_sum.size());

                std::vector<GadgetSpecies> species;
                auto * part = mpiparticles.get_particles_ptr();
                for (int family = 0; family < 6; family++) {
                    if (count[size_t(family)] == 0)
                        continue;
                    const double mean_mass = mass_sum[size_t(family)] / double(count[size_t(family)]);
                    const double unit_mass = 3.0 * header.Omega0 * MplMpl_over_H0Msunh *
                        std::pow(header.BoxSize / HubbleLengthInMpch, 3) /
                        double(count[size_t(family)]) / 1.0e10;
                    if (mean_mass <= 0.0 or unit_mass <= 0.0)
                        throw std::runtime_error("Cannot preserve Gadget particle masses in output snapshot");
                    GadgetSpecies sp;
                    sp.type = family;
                    sp.npart = local_index[size_t(family)].size();
                    sp.npart_total = size_t(count[size_t(family)]);
                    sp.mass_fraction = mean_mass / unit_mass;
                    sp.get_pos = [part, &indices = local_index[size_t(family)]](size_t i, double * x) {
                        const auto * p = part[indices[i]].get_pos();
                        std::copy(p, p + 3, x);
                    };
                    sp.get_vel = [part, &indices = local_index[size_t(family)]](size_t i, double * v) {
                        const auto * p = part[indices[i]].get_vel();
                        std::copy(p, p + 3, v);
                    };
                    sp.get_id = [part, &indices = local_index[size_t(family)]](size_t i) {
                        return part[indices[i]].get_id();
                    };
                    sp.get_mass = [part, &indices = local_index[size_t(family)], mean_mass](size_t i) {
                        return part[indices[i]].get_mass() / mean_mass;
                    };
                    species.push_back(std::move(sp));
                }

                int failed = 0;
                try {
                    GadgetWriter writer(3);
                    writer.write_gadget_species(filename,
                                                std::move(species),
                                                FML::NTasks,
                                                header.time,
                                                header.BoxSize,
                                                header.Omega0,
                                                header.OmegaLambda,
                                                header.HubbleParam,
                                                header.BoxSize,
                                                1.0 / std::sqrt(header.time));
                } catch (...) {
                    failed = 1;
                }
                FML::MaxOverTasks(&failed);
                if (failed)
                    throw std::runtime_error("Could not write one or more baryonified Gadget snapshot files");
            }

            void analyze_dtfe() {
#ifndef USE_CGAL
                throw std::runtime_error("DTFE requires CGAL; rebuild SnapshotAnalyzer with USE_CGAL=true");
#else
                using namespace FML::TRIANGULATION;
                const auto sampling = options.dtfe_sampling == "grid_point" ? DTFESampling::GridPoint :
                                                                                DTFESampling::CellAverage;
                const auto weighting = options.dtfe_weighting == "mass" ? DTFEWeighting::Mass : DTFEWeighting::Volume;
                DTFE<SnapshotParticle> dtfe;
                dtfe.create(mpiparticles.get_particles_ptr(),
                            mpiparticles.get_npart(),
                            options.dtfe_buffer_fraction,
                            options.dtfe_random_fraction);

                FML::GRID::FFTWGrid<3> density;
                dtfe.density_to_grid(options.dtfe_nmesh, density, sampling, options.dtfe_nsub);
                DTFEVelocityGrids velocity;
                dtfe.velocity_to_grid(mpiparticles.get_particles_ptr(),
                                      options.dtfe_nmesh,
                                      velocity,
                                      sampling,
                                      weighting,
                                      options.dtfe_nsub,
                                      true,
                                      true,
                                      true,
                                      false);

                // DTFE gradients are with respect to box-normalized coordinates. Convert them to
                // physical inverse-length units before reporting their spectra.
                for (int ix = 0; ix < velocity.divergence.get_local_nx(); ix++)
                    for (int iy = 0; iy < options.dtfe_nmesh; iy++)
                        for (int iz = 0; iz < options.dtfe_nmesh; iz++) {
                            const std::array<int, 3> c{ix, iy, iz};
                            velocity.divergence.set_real(c, velocity.divergence.get_real(c) / header.BoxSize);
                            for (auto & component : velocity.vorticity)
                                component.set_real(c, component.get_real(c) / header.BoxSize);
                        }

                if (options.dtfe_write_fields)
                    write_dtfe_fields(density, velocity);

                std::ofstream spectra;
                if (FML::ThisTask == 0) {
                    spectra.open(options.dtfe_spectra_file);
                    if (not spectra)
                        throw std::runtime_error("Could not open DTFE spectra file: " + options.dtfe_spectra_file);
                    spectra << "# DTFE spectra; density in mean-density units; velocity in Gadget snapshot units; "
                               "derivative fields per Mpc/h\n"
                            << "# k_h_Mpc P units\n";
                }
                measure_dtfe_spectrum(density, "density_delta", true, spectra);
                for (int i = 0; i < 3; i++)
                    measure_dtfe_spectrum(velocity.velocity[i], "velocity_" + std::string(1, "xyz"[i]), false, spectra);
                measure_dtfe_spectrum(velocity.divergence, "velocity_divergence", false, spectra);
                for (int i = 0; i < 3; i++)
                    measure_dtfe_spectrum(velocity.vorticity[i], "vorticity_" + std::string(1, "xyz"[i]), false, spectra);
                if (FML::ThisTask == 0)
                    std::cout << "DTFE spectra written: " << options.dtfe_spectra_file << "\n";
#endif
            }

#ifdef USE_CGAL
            void measure_dtfe_spectrum(FML::GRID::FFTWGrid<3> & grid,
                                       const std::string & name,
                                       bool subtract_mean,
                                       std::ofstream & output) {
                const int nmesh = grid.get_nmesh();
                if (subtract_mean)
                    for (int ix = 0; ix < grid.get_local_nx(); ix++)
                        for (int iy = 0; iy < nmesh; iy++)
                            for (int iz = 0; iz < nmesh; iz++) {
                                const std::array<int, 3> c{ix, iy, iz};
                                grid.set_real(c, grid.get_real(c) - 1.0);
                            }
                grid.fftw_r2c();
                FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(
                    2.0 * M_PI, M_PI * nmesh, 40, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>::LOG_SPACING);
                pofk.subtract_shotnoise = false;
                FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<3>(grid, pofk);
                pofk.scale(header.BoxSize);
                if (FML::ThisTask == 0) {
                    output << "\n# " << name << "\n";
                    for (int i = 0; i < pofk.n; i++)
                        output << std::setprecision(12) << pofk.kbin[i] << " " << pofk.pofk[i] << " "
                               << pofk.count[i] << "\n";
                }
            }

            void write_dtfe_fields(FML::GRID::FFTWGrid<3> & density,
                                   FML::TRIANGULATION::DTFEVelocityGrids & velocity) {
                std::ostringstream rank;
                rank << std::setw(5) << std::setfill('0') << FML::ThisTask;
                const std::string filename = options.dtfe_fields_prefix + ".rank" + rank.str() + ".txt";
                std::ofstream field(filename);
                assert_mpi(field.good(), ("[SnapshotAnalyzer] Could not open DTFE field file " + filename).c_str());
                const int nmesh = density.get_nmesh();
                field << "# local DTFE field slab; x0 " << density.get_local_x_start() << " nx "
                      << density.get_local_nx() << " Nmesh " << nmesh << "\n"
                      << "# ix iy iz density vx vy vz divergence wx wy wz\n" << std::setprecision(12);
                for (int ix = 0; ix < density.get_local_nx(); ix++)
                    for (int iy = 0; iy < nmesh; iy++)
                        for (int iz = 0; iz < nmesh; iz++) {
                            const std::array<int, 3> c{ix, iy, iz};
                            field << density.get_local_x_start() + ix << " " << iy << " " << iz << " "
                                  << density.get_real(c);
                            for (const auto & component : velocity.velocity)
                                field << " " << component.get_real(c);
                            field << " " << velocity.divergence.get_real(c);
                            for (const auto & component : velocity.vorticity)
                                field << " " << component.get_real(c);
                            field << "\n";
                        }
            }
#endif
        };
    }
}

#endif

#ifndef FML_SNAPSHOTANALYZER_SNAPSHOTANALYZER_H
#define FML_SNAPSHOTANALYZER_SNAPSHOTANALYZER_H

// The standalone front end: read a snapshot (Gadget binary or HDF5), select the particles, build the
// AnalysisContext from the snapshot header (and options), then run the analyses. All the analyses are the shared
// code in FML/Analysis (no dependence on this executable or on the input format), so the same code can be called
// from a simulation.

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Baryonification.h>
#include <FML/Analysis/CosmicWeb.h>
#include <FML/Analysis/DTFE.h>
#include <FML/Analysis/Halos.h>
#include <FML/Analysis/Minkowski.h>
#include <FML/Analysis/Output.h>
#include <FML/Analysis/ParticleUtils.h>
#include <FML/Analysis/PowerSpectrum.h>
#include <FML/Analysis/Voids.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#ifdef USE_HDF5
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#endif
#include <FML/Global/MPICounts.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace FML {
    namespace SNAPSHOTANALYZER {

        namespace ANALYSIS = FML::ANALYSIS;

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

        //=============================================================================
        /// Reads Lua parameters and remembers the names it read, so that names in the file that are not parameters
        /// can be reported. Settings structs from FML/Analysis are read with their prefix (the same keys as in
        /// SimulationCode)
        //=============================================================================
        class LuaReader {
          public:
            explicit LuaReader(FML::FILEUTILS::LuaFileParser & lua) : lua(lua) {}

            template <class T>
            T get(const std::string & name, const T & default_value) {
                known.insert(name);
                if constexpr (std::is_same_v<T, bool>)
                    return lua.read_bool(name, default_value);
                else if constexpr (std::is_integral_v<T>)
                    return T(lua.read_int(name, int(default_value)));
                else if constexpr (std::is_floating_point_v<T>)
                    return T(lua.read_double(name, double(default_value)));
                else if constexpr (std::is_same_v<T, std::string>)
                    return lua.read_string(name, default_value);
                else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                    return lua.read_string_array(name, default_value, false);
                else if constexpr (std::is_same_v<T, std::vector<int>>)
                    return lua.read_number_array<int>(name, default_value, false);
                else if constexpr (std::is_same_v<T, std::vector<double>>)
                    return lua.read_number_array<double>(name, default_value, false);
                else
                    static_assert(sizeof(T) == 0, "LuaReader: unsupported parameter type");
            }

            template <class Settings>
            void read(Settings & settings) {
                settings.parameters([&](const char * name, auto & value, const auto & default_value, const char *) {
                    using T = std::decay_t<decltype(value)>;
                    value = get<T>(std::string(Settings::prefix) + name, T(default_value));
                });
            }

            /// Names that were parameters in earlier versions: an error with what to use instead (silently ignoring
            /// them would change the result). Other unknown names: a warning
            void check_names() const {
                static const std::map<std::string, std::string> renamed{
                    {"output_file", "output_folder (the files are named as in SimulationCode, e.g. pofk_z<z>.txt)"},
                    {"fof_nmin", "fof_nmin_per_halo"},
                    {"fof_output_file", "output_folder (halos_z<z>.txt)"},
                    {"so", "fof_so_definitions, e.g. {\"200m\", \"200c\"}"},
                    {"so_overdensity", "fof_so_definitions, e.g. {\"200m\", \"500c\", \"vir\"}"},
                    {"so_initial_overdensity", "nothing (the first sphere uses 200 times the mean density)"},
                    {"so_search_radius_mpch", "fof_buffer_length_mpch"},
                    {"so_nmin_center", "fof_so_nmin_center"},
                    {"pofk_multipole_velocity_to_displacement",
                     "nothing (computed from the snapshot cosmology, see cosmology_*)"},
                    {"baryonification_output_prefix",
                     "baryonification_output_particles = true (writes output_folder/gadget_baryonified_z<z>.*)"},
                    {"baryonification_host_catalogue", "output_folder (baryonification_z<z>.txt)"},
                    {"baryonification_overdensity_mean", "nothing (200c from the snapshot cosmology, see cosmology_*)"},
                    {"baryonification_search_buffer_mpch", "baryonification_fof_buffer_mpch"},
                    {"void_output_file", "output_folder (voids_z<z>.txt)"},
                    {"spherical_void_output_file", "output_folder (voids_spherical_z<z>.txt)"},
                    {"zobov_void_output_file", "output_folder (voids_zobov_z<z>.txt)"},
                    {"dtfe_spectra_file", "output_folder (dtfe_spectra_z<z>.txt)"},
                    {"dtfe_fields_prefix", "output_folder (dtfe_fields_z<z>.rank<NNNNN>.txt)"},
                };
                std::string errors;
                std::vector<std::string> unknown;
                for (auto & name : lua.get_global_variable_names()) {
                    if (known.count(name))
                        continue;
                    auto it = renamed.find(name);
                    if (it != renamed.end())
                        errors += "  " + name + " -> " + it->second + "\n";
                    else
                        unknown.push_back(name);
                }
                if (not errors.empty())
                    throw std::runtime_error("Parameters that are no longer used (use instead):\n" + errors);
                if (FML::ThisTask == 0)
                    for (auto & name : unknown)
                        std::cout << "Warning: [" << name << "] in the parameter file is not a parameter (typo?)\n";
            }

          private:
            FML::FILEUTILS::LuaFileParser & lua;
            std::set<std::string> known;
        };

        struct AnalyzerOptions {
            // Input
            std::string snapshot_input;
            std::string snapshot_format{"gadget"};
            std::string snapshot_read_mode{"files_per_task"};
            std::vector<int> particle_types;
            double snapshot_length_unit_mpch{1.0};
            double snapshot_mass_unit_msunh{1e10};
            double particle_allocation_factor{1.25};
            // The expansion history (for 200c/vir and redshift-space distortions). Default: LCDM from the snapshot
            // header (Omega0, OmegaLambda) without radiation
            double cosmology_Omega_r{0.0};
            std::string cosmology_hubble_table;
            // Output
            std::string output_folder{"."};
            // The analyses in FML/Analysis (keys as in SimulationCode where it has them)
            bool pofk{true};
            FML::ANALYSIS::PowerSpectrumSettings pofk_settings;
            bool pofk_multipole{false};
            FML::ANALYSIS::PowerSpectrumMultipoleSettings pofk_multipole_settings;
            bool bispectrum{false};
            FML::ANALYSIS::BispectrumSettings bispectrum_settings;
            bool cosmicweb{false};
            FML::ANALYSIS::CosmicWebSettings cosmicweb_settings;
            bool minkowski{false};
            FML::ANALYSIS::MinkowskiSettings minkowski_settings;
            bool fof{false};
            FML::ANALYSIS::HaloSettings fof_settings;
            bool baryonification{false};
            FML::ANALYSIS::BaryonificationSettings baryonification_settings;
            double baryonification_baryon_fraction{0.157};
            bool voids{false};
            bool spherical_voids{false};
            bool zobov_voids{false};
            FML::ANALYSIS::VoidSettings void_settings;
            FML::ANALYSIS::SphericalVoidSettings spherical_void_settings;
            FML::ANALYSIS::ZobovSettings zobov_settings;
            bool dtfe{false};
            FML::ANALYSIS::DTFESettings dtfe_settings;

            explicit AnalyzerOptions(FML::FILEUTILS::LuaFileParser & lua) {
                LuaReader r(lua);
                auto get = [&](const char * name, auto & value) { value = r.get(name, value); };
                get("snapshot_input", snapshot_input);
                get("snapshot_format", snapshot_format);
                get("snapshot_read_mode", snapshot_read_mode);
                get("particle_types", particle_types);
                get("snapshot_length_unit_mpch", snapshot_length_unit_mpch);
                get("snapshot_mass_unit_msunh", snapshot_mass_unit_msunh);
                get("particle_allocation_factor", particle_allocation_factor);
                get("cosmology_Omega_r", cosmology_Omega_r);
                get("cosmology_hubble_table", cosmology_hubble_table);
                get("output_folder", output_folder);
                get("pofk", pofk);
                r.read(pofk_settings);
                get("pofk_multipole", pofk_multipole);
                r.read(pofk_multipole_settings);
                get("bispectrum", bispectrum);
                r.read(bispectrum_settings);
                get("cosmicweb", cosmicweb);
                r.read(cosmicweb_settings);
                get("minkowski", minkowski);
                r.read(minkowski_settings);
                get("fof", fof);
                r.read(fof_settings);
                get("baryonification", baryonification);
                r.read(baryonification_settings);
                get("baryonification_baryon_fraction", baryonification_baryon_fraction);
                get("voids", voids);
                get("spherical_voids", spherical_voids);
                get("zobov_voids", zobov_voids);
                r.read(void_settings);
                r.read(spherical_void_settings);
                r.read(zobov_settings);
                get("dtfe", dtfe);
                r.read(dtfe_settings);
                r.check_names();
                validate();
            }

            void validate() const {
                if (snapshot_input.empty())
                    throw std::runtime_error("Required Lua option 'snapshot_input' (snapshot file prefix) is empty");
                if (snapshot_format != "gadget" and snapshot_format != "gadget_hdf5")
                    throw std::runtime_error("snapshot_format must be gadget or gadget_hdf5");
#ifndef USE_HDF5
                if (snapshot_format == "gadget_hdf5")
                    throw std::runtime_error("snapshot_format = gadget_hdf5 needs a USE_HDF5=true build");
#endif
                if (snapshot_read_mode != "files_per_task" and snapshot_read_mode != "all_tasks")
                    throw std::runtime_error("snapshot_read_mode must be files_per_task or all_tasks");
                if (output_folder.empty())
                    throw std::runtime_error("output_folder is empty");
                for (int type : particle_types)
                    if (type < 0 or type > 5)
                        throw std::runtime_error("particle_types must be Gadget types 0-5");
                if (not(snapshot_length_unit_mpch > 0.0) or not(snapshot_mass_unit_msunh > 0.0))
                    throw std::runtime_error("snapshot_length_unit_mpch and snapshot_mass_unit_msunh must be positive");
                if (not(particle_allocation_factor >= 1.0))
                    throw std::runtime_error("particle_allocation_factor must be >= 1");
                if (not(cosmology_Omega_r >= 0.0))
                    throw std::runtime_error("cosmology_Omega_r must be >= 0");
                if (not pofk and not pofk_multipole and not bispectrum and not cosmicweb and not minkowski and
                    not fof and not voids and not spherical_voids and not zobov_voids and not dtfe and
                    not baryonification)
                    throw std::runtime_error("Enable at least one analysis: pofk, pofk_multipole, bispectrum, "
                                             "cosmicweb, minkowski, fof, baryonification, voids, spherical_voids, "
                                             "zobov_voids or dtfe");
                if (pofk or baryonification)
                    pofk_settings.validate();
                if (pofk_multipole)
                    pofk_multipole_settings.validate();
                if (bispectrum)
                    bispectrum_settings.validate();
                if (cosmicweb)
                    cosmicweb_settings.validate();
                if (minkowski)
                    minkowski_settings.validate();
                if (fof)
                    fof_settings.validate();
                if (baryonification) {
                    baryonification_settings.validate();
                    if (not(baryonification_baryon_fraction > 0.0 and baryonification_baryon_fraction < 1.0))
                        throw std::runtime_error("baryonification_baryon_fraction must be in (0, 1)");
                }
                if (voids or spherical_voids or zobov_voids)
                    void_settings.validate();
                if (spherical_voids)
                    spherical_void_settings.validate();
                if (zobov_voids)
                    zobov_settings.validate();
                if (dtfe)
                    dtfe_settings.validate();
#ifndef USE_CGAL
                if (zobov_voids or dtfe)
                    throw std::runtime_error("zobov_voids and dtfe need a USE_CGAL=true build");
#endif
            }
        };

        class SnapshotAnalyzer {
          public:
            explicit SnapshotAnalyzer(AnalyzerOptions options) : options(std::move(options)) {}

            void run() {
                read_snapshot();
                make_context();
                prepare_output_folder();
                if (options.baryonification)
                    validate_baryonification_output_paths();

                if (options.pofk)
                    measure_pofk();
                if (options.pofk_multipole)
                    measure_multipoles();
                if (options.bispectrum)
                    measure_bispectrum();
                if (options.cosmicweb)
                    classify_cosmic_web();
                if (options.minkowski)
                    measure_minkowski_functionals();
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
                // Last: it moves the particles
                if (options.baryonification)
                    baryonify_snapshot();
            }

          private:
            AnalyzerOptions options;
            FML::PARTICLE::MPIParticles<SnapshotParticle> mpiparticles;
            FML::FILEUTILS::GADGET::GadgetHeader header{};
            ANALYSIS::AnalysisContext context;
            std::array<long long, 6> family_count{};
            std::vector<std::string> input_files;

            std::string output_path(const std::string & name) const { return options.output_folder + "/" + name; }
            std::string redshift_label() const { return ANALYSIS::redshift_label(context.redshift()); }

            /// The files of the snapshot. Every task reads the header of the first file
            std::vector<std::string> snapshot_files() {
                namespace G = FML::FILEUTILS::GADGET;
                const auto & prefix = options.snapshot_input;
                std::vector<std::string> files;
                if (options.snapshot_format == "gadget_hdf5") {
#ifdef USE_HDF5
                    G::GadgetHDF5Reader reader(3);
                    files = reader.get_filenames(prefix);
                    reader.read_header(files.front());
                    header = reader.get_header();
#endif
                } else {
                    // prefix.0, prefix.1, ... or a single file called prefix
                    const bool numbered = std::ifstream(prefix + ".0", std::ios::binary).good();
                    const std::string first = numbered ? prefix + ".0" : prefix;
                    std::ifstream fp(first, std::ios::binary);
                    if (not fp)
                        throw std::runtime_error("Cannot open the Gadget file " + prefix + ".0 (or " + prefix + ")");
                    G::GadgetReader reader(3);
                    reader.read_header(fp);
                    header = reader.get_header();
                    if (not numbered)
                        files = {prefix};
                    else
                        for (int i = 0; i < std::max(1, header.num_files); i++)
                            files.push_back(prefix + "." + std::to_string(i));
                }
                if (files.empty())
                    throw std::runtime_error("No snapshot files for " + prefix);
                return files;
            }

            /// Read one file and add its particles to the back of particles
            void read_file(const std::string & filename,
                           std::vector<SnapshotParticle> & particles,
                           bool only_keep_part_in_domain,
                           bool verbose) {
                namespace G = FML::FILEUTILS::GADGET;
                if (options.snapshot_format == "gadget_hdf5") {
#ifdef USE_HDF5
                    G::GadgetHDF5Reader reader(3);
                    reader.set_read_masses(true);
                    reader.read_gadget_single(filename, particles, only_keep_part_in_domain, verbose);
#endif
                } else {
                    G::GadgetReader reader(3);
                    reader.read_gadget_single(filename, particles, only_keep_part_in_domain, verbose);
                }
            }

            /// files_per_task: each task reads every NTasks-th file and the particles are then sent to the task
            /// owning their position (each file is read once). all_tasks: every task reads every file and keeps the
            /// particles in its domain
            void read_snapshot() {
                const auto files = snapshot_files();
                input_files = files;
                if (not std::isfinite(header.BoxSize) or header.BoxSize <= 0.0)
                    throw std::runtime_error("The snapshot has an invalid BoxSize");
                const bool per_task = options.snapshot_read_mode == "files_per_task";

                std::vector<SnapshotParticle> particles;
                std::string error;
                try {
                    for (size_t i = 0; i < files.size(); i++) {
                        if (per_task and int(i % size_t(FML::NTasks)) != FML::ThisTask)
                            continue;
                        if (FML::ThisTask == 0)
                            std::cout << "Reading " << files[i] << "\n";
                        read_file(files[i], particles, not per_task, i == 0);
                    }
                } catch (const std::exception & e) {
                    error = "Reading the snapshot: " + std::string(e.what());
                }
                ANALYSIS::throw_if_any_task_failed(error, "Reading the snapshot");

                // Constant-mass Gadget species are represented by the header table; variable-mass
                // species were populated from the MASS block (Masses dataset) by the reader.
                for (auto & p : particles)
                    if (header.mass[p.get_family()] > 0.0)
                        p.set_mass(header.mass[p.get_family()]);

                // The selected types, with room for particles moving between tasks; then free the read buffer
                const auto & types = options.particle_types;
                auto selected = [&](SnapshotParticle & p) {
                    return types.empty() or std::find(types.begin(), types.end(), p.get_family()) != types.end();
                };
                size_t nselected = 0;
                for (auto & p : particles)
                    nselected += selected(p);
                const size_t nallocate = size_t(double(nselected) * options.particle_allocation_factor) + 16;
                mpiparticles.create(particles, nallocate, selected);
                std::vector<SnapshotParticle>().swap(particles);
                if (per_task) {
                    ANALYSIS::redistribute_particles(mpiparticles);
                    auto & storage = mpiparticles.get_particles();
                    const size_t want =
                        size_t(double(mpiparticles.get_npart()) * options.particle_allocation_factor) + 16;
                    if (storage.size() < want)
                        storage.resize(want);
                }

                family_count.fill(0);
                for (const auto & p : mpiparticles)
                    family_count[size_t(p.get_family())]++;
                FML::SumArrayOverTasks(family_count.data(), family_count.size());
                if (mpiparticles.get_npart_total() == 0)
                    throw std::runtime_error("No particles selected (check particle_types)");
            }

            void make_context() {
                context.boxsize = header.BoxSize * options.snapshot_length_unit_mpch;
                context.scale_factor = header.time;
                context.mass_to_msunh = options.snapshot_mass_unit_msunh;
                context.velocity_to_kms = 1.0; // GadgetReader returns peculiar km/s (sqrt(a) times the stored value)
                if (not options.cosmology_hubble_table.empty())
                    context.background =
                        ANALYSIS::Background::from_table(options.cosmology_hubble_table, header.Omega0);
                else
                    context.background =
                        ANALYSIS::Background::LCDM(header.Omega0, header.OmegaLambda, options.cosmology_Omega_r);
                context.validate();

                double total_mass = 0.0;
                for (const auto & p : mpiparticles)
                    total_mass += p.get_mass();
                FML::SumOverTasks(&total_mass);
                if (FML::ThisTask == 0) {
                    std::cout << "Snapshot scale factor: " << header.time << " (z = " << context.redshift() << ")\n"
                              << "Box size [Mpc/h]: " << context.boxsize << "\n"
                              << "Particles selected: " << mpiparticles.get_npart_total() << " (Gadget types";
                    for (int i = 0; i < 6; i++)
                        if (family_count[i] > 0)
                            std::cout << " " << i << ": " << family_count[i];
                    std::cout << ")\n"
                              << "Header Omega0 = " << header.Omega0 << ", OmegaLambda = " << header.OmegaLambda
                              << "; Omega of the selected particles (from their mass) = "
                              << context.Omega_of_particles(total_mass) << "\n";
                }
            }

            void prepare_output_folder() {
                int failed = 0;
                if (FML::ThisTask == 0) {
                    std::error_code ec;
                    std::filesystem::create_directories(options.output_folder, ec);
                    failed = not std::filesystem::is_directory(options.output_folder);
                }
                FML::MaxOverTasks(&failed);
                if (failed)
                    throw std::runtime_error("Cannot create output_folder " + options.output_folder);
            }

            void measure_pofk() {
                const auto pofk = ANALYSIS::measure_power_spectrum<3>(mpiparticles.get_particles_ptr(),
                                                                      mpiparticles.get_npart(),
                                                                      mpiparticles.get_npart_total(),
                                                                      options.pofk_settings,
                                                                      context);
                const auto filename = output_path("pofk_z" + redshift_label() + ".txt");
                ANALYSIS::write_power_spectrum<3>(filename,
                                                  pofk,
                                                  "# P(k) of " + options.snapshot_input + "\n",
                                                  {{"modes", [&](int i) { return pofk.count[i]; }}});
                if (FML::ThisTask == 0)
                    std::cout << "P(k) written: " << filename << "\n";
            }

            void measure_bispectrum() {
                const auto bofk = ANALYSIS::measure_bispectrum<3>(mpiparticles.get_particles_ptr(),
                                                                  mpiparticles.get_npart(),
                                                                  mpiparticles.get_npart_total(),
                                                                  options.bispectrum_settings,
                                                                  context);
                const auto filename = output_path("bispectrum_z" + redshift_label() + ".txt");
                ANALYSIS::write_bispectrum<3>(filename, bofk, "# Bispectrum of " + options.snapshot_input + "\n");
                if (FML::ThisTask == 0)
                    std::cout << "Bispectrum written: " << filename << "\n";
            }

            void classify_cosmic_web() {
                const auto web = ANALYSIS::measure_cosmic_web(mpiparticles.get_particles_ptr(),
                                                              mpiparticles.get_npart(),
                                                              options.cosmicweb_settings,
                                                              context);
                const auto filename = output_path("cosmicweb_z" + redshift_label() + ".txt");
                ANALYSIS::write_cosmic_web(filename, web, "# Cosmic web (T-web) of " + options.snapshot_input + "\n");
                if (FML::ThisTask == 0)
                    std::cout << "Cosmic-web fractions written: " << filename << "\n";
            }

            void measure_minkowski_functionals() {
                const auto mf = ANALYSIS::measure_minkowski_functionals(mpiparticles.get_particles_ptr(),
                                                                        mpiparticles.get_npart(),
                                                                        options.minkowski_settings,
                                                                        context);
                const auto filename = output_path("minkowski_z" + redshift_label() + ".txt");
                ANALYSIS::write_minkowski_functionals(filename,
                                                      mf,
                                                      "# Minkowski functionals of " + options.snapshot_input + "\n");
                if (FML::ThisTask == 0)
                    std::cout << "Minkowski functionals written: " << filename << "\n";
            }

            void measure_multipoles() {
                const auto pell =
                    ANALYSIS::measure_power_spectrum_multipoles<3>(mpiparticles, options.pofk_multipole_settings, context);
                const auto filename = output_path("pofk_multipole_z" + redshift_label() + ".txt");
                std::ostringstream header_lines;
                header_lines << "# Redshift-space multipoles of " << options.snapshot_input
                             << " (mean over the three axes), velocity_to_displacement = "
                             << context.velocity_to_displacement() << " (box units per km/s)\n";
                ANALYSIS::write_power_spectrum_multipoles<3>(filename, pell, header_lines.str());
                if (FML::ThisTask == 0)
                    std::cout << "P_ell(k) written: " << filename << "\n";
            }

            // Friends-of-Friends grouping follows Davis et al. (1985), doi:10.1086/163168.
            void find_halos() {
                const auto catalogue = ANALYSIS::find_halos<3>(mpiparticles.get_particles_ptr(),
                                                               mpiparticles.get_npart(),
                                                               options.fof_settings,
                                                               context);
                ANALYSIS::write_halo_outputs<3>(options.output_folder, catalogue, options.fof_settings, context);
                long long nhalos = (long long)catalogue.halos.size();
                FML::SumOverTasks(&nhalos);
                if (FML::ThisTask == 0)
                    std::cout << "FoF: " << nhalos << " halos written to " << options.output_folder << "/halos_z"
                              << redshift_label() << ".*\n";
            }

            void find_voids() {
                const auto catalogue =
                    ANALYSIS::find_voxel_voids(mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), options.void_settings);
                const auto filename = output_path("voids_z" + redshift_label() + ".txt");
                ANALYSIS::write_void_catalogue(filename, catalogue, context, "# Voxel watershed voids (Revolver)\n");
                if (FML::ThisTask == 0)
                    std::cout << "Voxel void catalogue written: " << filename << " (" << catalogue.size() << " voids)\n";
            }

            void find_spherical_voids() {
                const auto catalogue = ANALYSIS::find_spherical_voids(mpiparticles.get_particles_ptr(),
                                                                      mpiparticles.get_npart(),
                                                                      options.spherical_void_settings,
                                                                      options.void_settings);
                const auto filename = output_path("voids_spherical_z" + redshift_label() + ".txt");
                ANALYSIS::write_void_catalogue(filename, catalogue, context, "# Spherical underdensity voids\n");
                if (FML::ThisTask == 0)
                    std::cout << "Spherical void catalogue written: " << filename << " (" << catalogue.size()
                              << " voids)\n";
            }

            void find_zobov_voids() {
#ifdef USE_CGAL
                const auto catalogue = ANALYSIS::find_zobov_voids(mpiparticles.get_particles_ptr(),
                                                                  mpiparticles.get_npart(),
                                                                  options.zobov_settings,
                                                                  options.void_settings);
                const auto filename = output_path("voids_zobov_z" + redshift_label() + ".txt");
                ANALYSIS::write_void_catalogue(filename, catalogue, context, "# ZOBOV voids\n");
                if (FML::ThisTask == 0)
                    std::cout << "ZOBOV void catalogue written: " << filename << " (" << catalogue.size()
                              << " voids)\n";
#endif
            }

            std::string baryonified_gadget_prefix() const {
                return output_path("gadget_baryonified_z" + redshift_label());
            }

            /// File i of the baryonified snapshot (the same format as the input)
            std::string baryonified_gadget_file(int i) const {
                return baryonified_gadget_prefix() + "." + std::to_string(i) +
                       (options.snapshot_format == "gadget_hdf5" ? ".hdf5" : "");
            }

            /// The baryonified Gadget files must not be (or overwrite) the input or an existing snapshot. Collective
            void validate_baryonification_output_paths() {
                if (not options.baryonification_settings.output_particles)
                    return;
                const auto norm = [](const std::string & s) { return std::filesystem::absolute(s).lexically_normal(); };
                int failed = 0;
                if (FML::ThisTask == 0) {
                    for (int i = 0; i < FML::NTasks; i++) {
                        const auto out = norm(baryonified_gadget_file(i));
                        failed |= std::filesystem::exists(out);
                        for (const auto & input : input_files)
                            failed |= out == norm(input);
                    }
                }
                FML::MaxOverTasks(&failed);
                if (failed)
                    throw std::runtime_error("Baryonification refuses to overwrite existing Gadget files " +
                                             baryonified_gadget_prefix() + ".* (or the input snapshot)");
            }

            void baryonify_snapshot() {
                int occupied_families = 0;
                for (auto n : family_count)
                    occupied_families += n > 0;
                if (occupied_families != 1)
                    throw std::runtime_error("Baryonification needs one Gadget particle type representing the total "
                                             "CDM+baryon matter field (select it with particle_types)");

                const auto original = ANALYSIS::measure_power_spectrum<3>(mpiparticles.get_particles_ptr(),
                                                                          mpiparticles.get_npart(),
                                                                          mpiparticles.get_npart_total(),
                                                                          options.pofk_settings,
                                                                          context);
                const auto result = ANALYSIS::baryonify(mpiparticles,
                                                        options.baryonification_settings,
                                                        context,
                                                        options.baryonification_baryon_fraction);
                const auto corrected = ANALYSIS::measure_power_spectrum<3>(mpiparticles.get_particles_ptr(),
                                                                           mpiparticles.get_npart(),
                                                                           mpiparticles.get_npart_total(),
                                                                           options.pofk_settings,
                                                                           context);
                const std::string z = redshift_label();
                ANALYSIS::write_baryonification_hosts(
                    output_path("baryonification_z" + z + ".txt"), result, options.baryonification_settings, context);
                std::ostringstream pofk_header;
                pofk_header << ANALYSIS::baryonification_provenance(result, options.baryonification_settings, context)
                            << "# pofk_nmesh = " << options.pofk_settings.nmesh
                            << "\n# pofk_density_assignment_method = " << options.pofk_settings.density_assignment_method
                            << "\n# pofk_interlacing = " << options.pofk_settings.interlacing
                            << "\n# pofk_subtract_shotnoise = " << options.pofk_settings.subtract_shotnoise << "\n";
                ANALYSIS::write_baryonified_power_spectrum<3>(
                    output_path("pofk_baryonified_z" + z + ".txt"), original, corrected, pofk_header.str());
                if (options.baryonification_settings.output_particles)
                    write_baryonified_gadget();
                if (FML::ThisTask == 0)
                    std::cout << "Baryonification: " << result.hosts.size() << " hosts; rejected " << result.truncated
                              << " truncated, " << result.nested << " nested, " << result.unresolved
                              << " unresolved, " << result.below_cut << " below the mass cut and " << result.too_large
                              << " whose support reaches half the box\n";
            }

            /// The baryonified particles as Gadget files (one per task), keeping types, IDs, velocities and masses
            /// (written in the Gadget unit 1e10 Msun/h)
            void write_baryonified_gadget() {
                using namespace FML::FILEUTILS::GADGET;
                const std::string filename = baryonified_gadget_file(FML::ThisTask);
                std::array<long long, 6> count{};
                std::array<double, 6> mass_sum{};
                std::array<double, 6> mass_max, mass_min_neg;
                mass_max.fill(-std::numeric_limits<double>::max());
                mass_min_neg.fill(-std::numeric_limits<double>::max());
                std::array<std::vector<size_t>, 6> local_index;
                for (size_t i = 0; i < mpiparticles.get_npart(); i++) {
                    const int family = mpiparticles[i].get_family();
                    local_index[size_t(family)].push_back(i);
                    count[size_t(family)]++;
                    mass_sum[size_t(family)] += mpiparticles[i].get_mass();
                    mass_max[size_t(family)] = std::max(mass_max[size_t(family)], mpiparticles[i].get_mass());
                    mass_min_neg[size_t(family)] = std::max(mass_min_neg[size_t(family)], -mpiparticles[i].get_mass());
                }
                FML::SumArrayOverTasks(count.data(), count.size());
                FML::SumArrayOverTasks(mass_sum.data(), mass_sum.size());
                FML::MaxArrayOverTasks(mass_max.data(), mass_max.size());
                FML::MaxArrayOverTasks(mass_min_neg.data(), mass_min_neg.size());

                std::vector<GadgetSpecies> species;
                auto * part = mpiparticles.get_particles_ptr();
                for (int family = 0; family < 6; family++) {
                    if (count[size_t(family)] == 0)
                        continue;
                    const double mean_mass = mass_sum[size_t(family)] / double(count[size_t(family)]);
                    // The writer's particle mass (Msun/h) for mass_fraction = 1 with the same header box size
                    const double unit_mass_msunh = 3.0 * header.Omega0 * MplMpl_over_H0Msunh *
                                                   std::pow(header.BoxSize / HubbleLengthInMpch, 3) /
                                                   double(count[size_t(family)]);
                    if (mean_mass <= 0.0 or unit_mass_msunh <= 0.0)
                        throw std::runtime_error("Cannot preserve Gadget particle masses in output snapshot");
                    GadgetSpecies sp;
                    sp.type = family;
                    sp.npart = local_index[size_t(family)].size();
                    sp.npart_total = size_t(count[size_t(family)]);
                    sp.mass_fraction = mean_mass * options.snapshot_mass_unit_msunh / unit_mass_msunh;
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
                    // Per-particle masses only if they differ (otherwise the mass table, as in the input)
                    const bool equal_masses = mass_max[size_t(family)] + mass_min_neg[size_t(family)] <=
                                              1e-12 * mean_mass;
                    if (not equal_masses)
                        sp.get_mass = [part, &indices = local_index[size_t(family)], mean_mass](size_t i) {
                            return part[indices[i]].get_mass() / mean_mass;
                        };
                    species.push_back(std::move(sp));
                }

                std::string error;
                try {
                    auto write = [&](auto && writer) {
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
                    };
                    if (options.snapshot_format == "gadget_hdf5") {
#ifdef USE_HDF5
                        write(GadgetHDF5Writer(3));
#endif
                    } else {
                        write(GadgetWriter(3));
                    }
                } catch (const std::exception & e) {
                    error = "Could not write the baryonified Gadget file " + filename + ": " + e.what();
                }
                ANALYSIS::throw_if_any_task_failed(error, "Writing the baryonified Gadget files");
                if (FML::ThisTask == 0)
                    std::cout << "Baryonified snapshot written to " << baryonified_gadget_prefix() << ".*\n";
            }

            void analyze_dtfe() {
#ifdef USE_CGAL
                const auto fields = ANALYSIS::compute_dtfe_fields(
                    mpiparticles.get_particles_ptr(), mpiparticles.get_npart(), options.dtfe_settings, context);
                const std::string z = redshift_label();
                if (options.dtfe_settings.write_fields)
                    ANALYSIS::write_dtfe_fields(output_path("dtfe_fields_z" + z), fields);
                const auto spectra = ANALYSIS::dtfe_spectra(fields, options.dtfe_settings, context);
                const auto filename = output_path("dtfe_spectra_z" + z + ".txt");
                ANALYSIS::write_dtfe_spectra(filename, spectra, context);
                if (FML::ThisTask == 0)
                    std::cout << "DTFE spectra written: " << filename << "\n";
#endif
            }
        };
    } // namespace SNAPSHOTANALYZER
} // namespace FML

#endif

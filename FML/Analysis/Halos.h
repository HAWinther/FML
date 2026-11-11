#ifndef FML_ANALYSIS_HALOS_H
#define FML_ANALYSIS_HALOS_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Output.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#ifdef USE_HDF5
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#endif

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
// Friends-of-friends halos and their spherical overdensity (SO) masses, the halo catalogue (text or HDF5) and the
// mass function. The halo finding is the same as SimulationCode's (fof = true): FoF (Davis et al. 1985), then
// for each fof_so_definitions entry (e.g. {"200m", "500c", "vir"}) a shrinking-sphere center and the radius where the
// mean enclosed density of the particles is Delta times the reference density. Halos whose SO center lies inside
// the radius (first definition) of a more massive halo are flagged as overlapping.
//
// Particles: positions in box units [0,1); masses (get_mass, 1 if absent) and velocities in the units given by the
// AnalysisContext. "m" (and "b") definitions are relative to the mean density of the particles analysed, "c" to the
// critical density: the density parameter of the particles is taken from their total mass.
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// FoF/SO settings. Keys fof_<name>, the same as SimulationCode's (so_nmin_center is extra)
        //=============================================================================
        struct HaloSettings {
            static constexpr const char * prefix = "fof_";
            int nmin_per_halo;
            double linking_length;
            int nmesh_max;
            double buffer_length_mpch;
            std::string output_fileformat;
            int massfunction_nbins;
            double massfunction_Mmin;
            double massfunction_Mmax;
            std::vector<std::string> so_definitions;
            int so_nmin_center;

            HaloSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmin_per_halo", nmin_per_halo, 20, "Minimum number of particles per halo");
                f("linking_length", linking_length, 0.2, "The linking length in units of the mean particle separation");
                f("nmesh_max", nmesh_max, 0, "Maximum grid size used to bin particles for the linking (0: automatic)");
                f("buffer_length_mpch",
                  buffer_length_mpch,
                  3.0,
                  "Buffer region from neighbor tasks and SO search radius (Mpc/h); larger than the largest halo");
                f("output_fileformat",
                  output_fileformat,
                  std::string("ascii"),
                  "The halo catalogue: halos_z<z>.txt (ascii) or halos_z<z>.hdf5 (HDF5, needs USE_HDF5)");
                f("massfunction_nbins", massfunction_nbins, 30, "Number of logarithmic bins of the mass function");
                f("massfunction_Mmin", massfunction_Mmin, 1e10, "Smallest mass of the mass function (Msun/h)");
                f("massfunction_Mmax", massfunction_Mmax, 1e16, "Largest mass of the mass function (Msun/h)");
                f("so_definitions",
                  so_definitions,
                  std::vector<std::string>{},
                  "Spherical overdensity masses around the FoF halos, e.g. {\"200m\", \"500c\", \"vir\"} (<Delta>m: "
                  "Delta times the mean density of the particles, <Delta>c: the critical density, vir: Bryan & Norman "
                  "1998). Empty: none");
                f("so_nmin_center",
                  so_nmin_center,
                  20,
                  "Stop the shrinking sphere when fewer particles than this are left");
            }

            void validate() const;
        };

        //=============================================================================
        /// A spherical overdensity definition: "vir" or <Delta><reference> with reference m or b (the mean density
        /// of the particles) or c (the critical density), e.g. 200m, 500c. Throws for anything else
        //=============================================================================
        struct SODefinition {
            bool virial{false};
            double Delta{0.0};
            char reference{'m'};
        };

        inline SODefinition parse_so_definition(const std::string & definition) {
            auto unknown = [&]() {
                return std::runtime_error("fof_so_definitions: unknown definition [" + definition +
                                          "] (<Delta>m or <Delta>b: mean density of the particles, <Delta>c: critical "
                                          "density, vir)");
            };
            if (definition == "vir")
                return {true, 0.0, 'm'};
            if (definition.size() < 2)
                throw unknown();
            SODefinition def;
            def.reference = definition.back();
            try {
                size_t used = 0;
                def.Delta = std::stod(definition.substr(0, definition.size() - 1), &used);
                if (used != definition.size() - 1)
                    throw unknown();
            } catch (...) {
                throw unknown();
            }
            if (not(def.Delta > 0.0) or not std::isfinite(def.Delta) or
                (def.reference != 'm' and def.reference != 'b' and def.reference != 'c'))
                throw unknown();
            return def;
        }

        inline void HaloSettings::validate() const {
            if (not(linking_length > 0.0) or nmin_per_halo < 2 or nmesh_max < 0 or not(buffer_length_mpch > 0.0))
                throw std::runtime_error("Invalid FoF settings: check fof_linking_length, fof_nmin_per_halo, "
                                         "fof_nmesh_max and fof_buffer_length_mpch");
            if (output_fileformat != "ascii" and output_fileformat != "HDF5")
                throw std::runtime_error("fof_output_fileformat must be ascii or HDF5");
#ifndef USE_HDF5
            if (output_fileformat == "HDF5")
                throw std::runtime_error("fof_output_fileformat = HDF5 needs USE_HDF5=true");
#endif
            if (massfunction_nbins < 1 or not(massfunction_Mmin > 0.0) or not(massfunction_Mmax > massfunction_Mmin))
                throw std::runtime_error("Invalid fof_massfunction_nbins, fof_massfunction_Mmin or fof_massfunction_Mmax");
            if (so_nmin_center < 0)
                throw std::runtime_error("fof_so_nmin_center must be >= 0");
            for (auto & def : so_definitions)
                parse_so_definition(def);
        }

        /// The overdensity of an SO definition in units of the mean density of the particles at the scale factor of
        /// the context. Omega_particles0: the density parameter of the particles today
        inline double so_overdensity(const std::string & definition,
                                     const AnalysisContext & context,
                                     double Omega_particles0) {
            const double a = context.scale_factor;
            const double E = context.background.E(a);
            const double Omega_particles_a = Omega_particles0 / (a * a * a * E * E);
            const auto def = parse_so_definition(definition);
            if (def.virial) {
                const double x = context.background.Omega_m(a) - 1.0;
                const double Delta_c = 18.0 * M_PI * M_PI + 82.0 * x - 39.0 * x * x;
                return Delta_c / Omega_particles_a;
            }
            return def.reference == 'c' ? def.Delta / Omega_particles_a : def.Delta;
        }

        /// A halo. Positions are in box units and masses and velocities in particle units until
        /// halos_to_physical_units (then Mpc/h, Msun/h, peculiar km/s)
        template <int NDIM>
        struct HaloRecord {
            long long int id;
            long long int np;
            double mass;
            std::array<double, NDIM> pos, vel, vel_rms;
            // Spherical overdensity (if so_definitions is not empty): center, velocity (inside the radius of the
            // first definition), and mass, radius, number of particles for each definition
            std::array<double, NDIM> so_pos{}, so_vel{}, so_vel_rms{};
            std::vector<double> so_mass, so_radius;
            std::vector<long long int> so_npart;
            bool so_truncated{false}; // The sphere reached the search radius (the buffer)
            bool so_overlap{false};   // The SO center is inside the radius (first definition) of a more massive halo
        };

        /// Run FoF (and SO around the FoF halos if so_thresholds is not empty) and return the halos found on this task
        /// (box/particle units). linking_length (mean separations) and nmin_per_halo are arguments (not taken from
        /// settings) so a caller can rescale them, e.g. for several particle species. total_mass: the total mass of
        /// all the particles (the mean density for SO, thresholds in units of it)
        template <int NDIM, class P>
        std::vector<HaloRecord<NDIM>> find_fof_halos(P * part,
                                                     size_t npart,
                                                     const HaloSettings & settings,
                                                     double boxsize,
                                                     double linking_length,
                                                     int nmin_per_halo,
                                                     const std::vector<double> & so_thresholds = {},
                                                     double total_mass = 0.0) {
            std::vector<FML::FOF::FoFHalo<P, NDIM>> groups;
            FML::FOF::FriendsOfFriends<P, NDIM>(part,
                                                npart,
                                                linking_length,
                                                nmin_per_halo,
                                                true,
                                                settings.buffer_length_mpch / boxsize,
                                                groups,
                                                settings.nmesh_max);
            std::vector<HaloRecord<NDIM>> halos;
            halos.reserve(groups.size());
            for (auto & g : groups) {
                HaloRecord<NDIM> h;
                h.id = (long long int)g.id;
                h.np = (long long int)g.np;
                h.mass = g.mass;
                for (int idim = 0; idim < NDIM; idim++) {
                    h.pos[idim] = g.pos[idim];
                    h.vel[idim] = g.vel[idim];
                    h.vel_rms[idim] = g.vel_rms[idim];
                }
                halos.push_back(h);
            }

            // Spherical overdensity around the FoF halos. The first shrinking sphere has 1.5 times the radius of a
            // sphere with the FoF mass and 200 times the mean density
            if (not so_thresholds.empty()) {
                std::vector<std::array<double, NDIM>> centers;
                std::vector<double> initial_radius;
                for (auto & h : halos) {
                    centers.push_back(h.pos);
                    initial_radius.push_back(1.5 * std::cbrt(3.0 * h.mass / (4.0 * M_PI * 200.0 * total_mass)));
                }
                auto so = FML::FOF::SphericalOverdensity<P, NDIM>(part,
                                                                  npart,
                                                                  total_mass,
                                                                  centers,
                                                                  initial_radius,
                                                                  so_thresholds,
                                                                  settings.buffer_length_mpch / boxsize,
                                                                  settings.so_nmin_center);
                for (size_t i = 0; i < halos.size(); i++) {
                    auto & h = halos[i];
                    h.so_pos = so[i].pos;
                    h.so_vel = so[i].vel;
                    h.so_vel_rms = so[i].vel_rms;
                    h.so_mass = so[i].mass;
                    h.so_radius = so[i].radius;
                    h.so_npart = so[i].npart;
                    h.so_truncated = so[i].truncated;
                }
            }
            return halos;
        }

        /// Flag the halos whose SO center lies inside the radius (of the first definition) of a more massive halo (on
        /// any task; ties: the lower ID is the main halo). Positions and radii in box units. Collective
        template <int NDIM>
        void flag_overlapping_so_halos(std::vector<HaloRecord<NDIM>> & halos) {
            std::vector<double> local;
            for (auto & h : halos) {
                for (int idim = 0; idim < NDIM; idim++)
                    local.push_back(h.so_pos[idim]);
                local.push_back(h.so_radius.empty() ? 0.0 : h.so_radius[0]);
                local.push_back(h.so_mass.empty() ? 0.0 : h.so_mass[0]);
                local.push_back(double(h.id));
            }
            const int stride = NDIM + 3;
            std::vector<double> all = local;
#ifdef USE_MPI
            all = FML::COMMUNICATION::gather_records_checked(local, true);
#endif
            const size_t nall = all.size() / stride;
            double rmax = 0.0;
            for (size_t j = 0; j < nall; j++)
                rmax = std::max(rmax, all[j * stride + NDIM]);
            if (nall == 0 or rmax <= 0.0)
                return;
            // Cells of size >= rmax
            const int ncell = std::max(1, std::min(256, int(1.0 / rmax)));
            auto cell_of = [&](const double * x, int idim) { return std::min(int(x[idim] * ncell), ncell - 1); };
            auto key_of = [&](const std::array<int, NDIM> & c) {
                long long int k = 0;
                for (int idim = 0; idim < NDIM; idim++)
                    k = k * ncell + ((c[idim] % ncell) + ncell) % ncell;
                return k;
            };
            std::map<long long int, std::vector<size_t>> cells;
            for (size_t j = 0; j < nall; j++) {
                std::array<int, NDIM> c;
                for (int idim = 0; idim < NDIM; idim++)
                    c[idim] = cell_of(&all[j * stride], idim);
                cells[key_of(c)].push_back(j);
            }
            for (auto & h : halos) {
                const double m = h.so_mass.empty() ? 0.0 : h.so_mass[0];
                std::array<int, NDIM> c0;
                for (int idim = 0; idim < NDIM; idim++)
                    c0[idim] = cell_of(h.so_pos.data(), idim);
                const int nr = std::min(1, ncell / 2);
                std::array<int, NDIM> off;
                off.fill(-nr);
                bool done = false;
                std::map<long long int, bool> visited;
                while (not done and not h.so_overlap) {
                    std::array<int, NDIM> c;
                    for (int idim = 0; idim < NDIM; idim++)
                        c[idim] = c0[idim] + off[idim];
                    const auto key = key_of(c);
                    if (not visited[key]) {
                        visited[key] = true;
                        auto it = cells.find(key);
                        if (it != cells.end())
                            for (auto j : it->second) {
                                const double * o = &all[j * stride];
                                const double mo = o[NDIM + 1];
                                if (not(mo > m or (mo == m and o[NDIM + 2] < double(h.id))))
                                    continue;
                                double r2 = 0.0;
                                for (int idim = 0; idim < NDIM; idim++) {
                                    double d = h.so_pos[idim] - o[idim];
                                    d -= std::round(d);
                                    r2 += d * d;
                                }
                                if (r2 < o[NDIM] * o[NDIM])
                                    h.so_overlap = true;
                            }
                    }
                    int idim = 0;
                    while (idim < NDIM) {
                        if (++off[idim] <= nr)
                            break;
                        off[idim] = -nr;
                        idim++;
                    }
                    done = idim == NDIM;
                }
            }
        }

        /// Box/particle units -> Mpc/h, Msun/h and peculiar km/s
        template <int NDIM>
        void halos_to_physical_units(std::vector<HaloRecord<NDIM>> & halos, const AnalysisContext & context) {
            const double mass_norm = context.mass_to_msunh;
            const double pos_norm = context.boxsize;
            const double vel_norm = context.velocity_to_kms;
            for (auto & g : halos) {
                g.mass *= mass_norm;
                for (int idim = 0; idim < NDIM; idim++) {
                    g.pos[idim] *= pos_norm;
                    g.vel[idim] *= vel_norm;
                    g.vel_rms[idim] *= vel_norm;
                    g.so_pos[idim] *= pos_norm;
                    g.so_vel[idim] *= vel_norm;
                    g.so_vel_rms[idim] *= vel_norm;
                }
                for (auto & m : g.so_mass)
                    m *= mass_norm;
                for (auto & r : g.so_radius)
                    r *= pos_norm;
            }
        }

        /// The halos of a particle set in physical units (on the task that found them) and what describes them
        template <int NDIM>
        struct HaloCatalogue {
            std::vector<HaloRecord<NDIM>> halos;
            std::vector<std::string> so_definitions;
            std::vector<double> so_thresholds; // Delta in units of the mean density of the particles
            double particle_mass_msunh{0.0};   // Mean particle mass
            double Omega_particles{0.0};       // Density parameter of the particles today (from their mass)
            long long int npart_total{0};
            double linking_length{0.0};
            int nmin_per_halo{0};
        };

        /// FoF + SO + overlap flags of a particle set, in physical units. Collective
        template <int NDIM, class P>
        HaloCatalogue<NDIM> find_halos(P * part,
                                       size_t npart,
                                       const HaloSettings & settings,
                                       const AnalysisContext & context) {
            settings.validate();
            context.validate();
            HaloCatalogue<NDIM> cat;
            double total_mass = 0.0;
            for (size_t i = 0; i < npart; i++)
                total_mass += FML::PARTICLE::GetMass(part[i]);
            FML::SumOverTasks(&total_mass);
            cat.npart_total = (long long int)npart;
            FML::SumOverTasks(&cat.npart_total);
            if (cat.npart_total == 0 or not(total_mass > 0.0))
                throw std::runtime_error("Halo finding: no particles (or zero total mass)");
            cat.particle_mass_msunh = total_mass * context.mass_to_msunh / double(cat.npart_total);
            cat.Omega_particles = context.Omega_of_particles(total_mass);
            cat.so_definitions = settings.so_definitions;
            for (auto & def : settings.so_definitions)
                cat.so_thresholds.push_back(so_overdensity(def, context, cat.Omega_particles));
            cat.linking_length = settings.linking_length;
            cat.nmin_per_halo = settings.nmin_per_halo;
            cat.halos = find_fof_halos<NDIM, P>(part,
                                                npart,
                                                settings,
                                                context.boxsize,
                                                settings.linking_length,
                                                settings.nmin_per_halo,
                                                cat.so_thresholds,
                                                total_mass);
            if (not cat.so_thresholds.empty())
                flag_overlapping_so_halos<NDIM>(cat.halos);
            halos_to_physical_units<NDIM>(cat.halos, context);
            return cat;
        }

        /// A mass function at the bin centers M = Mmin e^((i+1/2) dlogM): dn/dlnM counts the halos in
        /// [Mmin e^(i dlogM), Mmin e^((i+1) dlogM)) and n(>M) all halos (also those above Mmax) with mass >= the bin
        /// center. In (h/Mpc)^3. Collective
        struct MassFunction {
            std::vector<double> M, n_of_M, dndlnM;
        };

        template <int NDIM>
        MassFunction compute_mass_function(const std::vector<HaloRecord<NDIM>> & halos,
                                           const std::function<double(const HaloRecord<NDIM> &)> & mass_of,
                                           int nbins,
                                           double Mmin,
                                           double Mmax,
                                           double boxsize) {
            const double dlogM = std::log(Mmax / Mmin) / double(nbins);
            MassFunction mf;
            mf.M.resize(nbins);
            mf.n_of_M.assign(nbins, 0.0);
            mf.dndlnM.assign(nbins, 0.0);
            for (int i = 0; i < nbins; i++)
                mf.M[i] = Mmin * std::exp(dlogM * (i + 0.5));
            for (auto & g : halos) {
                const double x = std::log(mass_of(g) / Mmin) / dlogM;
                const double index = std::floor(x);
                if (index >= 0.0 and index < double(nbins))
                    mf.dndlnM[int(index)] += 1.0;
                const double icenter = std::floor(x - 0.5);
                if (icenter >= 0.0)
                    mf.n_of_M[int(std::min(icenter, double(nbins - 1)))] += 1.0;
            }
            FML::SumArrayOverTasks(mf.dndlnM.data(), nbins);
            FML::SumArrayOverTasks(mf.n_of_M.data(), nbins);
            for (int i = nbins - 2; i >= 0; i--)
                mf.n_of_M[i] += mf.n_of_M[i + 1];
            const double volume = std::pow(boxsize, NDIM);
            for (int i = 0; i < nbins; i++) {
                mf.n_of_M[i] /= volume;
                mf.dndlnM[i] /= volume * dlogM;
            }
            return mf;
        }

        /// Write a mass function: M (Msun/h), n(>M), dn/dlnM ((h/Mpc)^3) and extra columns, e.g. fits (collective)
        inline void write_mass_function(const std::string & filename,
                                        const MassFunction & mf,
                                        const std::string & header = "",
                                        const std::vector<std::pair<std::string, std::vector<double>>> & extra = {}) {
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "#  M (Msun/h)         n(M) (h/Mpc)^3         dndlogM(M) (h/Mpc)^3";
                for (auto & c : extra)
                    fp << "   " << c.first;
                fp << "\n";
                for (size_t i = 0; i < mf.M.size(); i++) {
                    fp << std::setw(15) << mf.M[i] << " " << std::setw(15) << mf.n_of_M[i] << " " << std::setw(15)
                       << mf.dndlnM[i];
                    for (auto & c : extra)
                        fp << " " << std::setw(15) << c.second.at(i);
                    fp << "\n";
                }
            });
        }

        namespace detail {
            /// All the halos as rows of doubles, gathered on task 0 (task order). Collective
            template <int NDIM>
            std::vector<double> gather_halo_rows(const std::vector<HaloRecord<NDIM>> & halos, int nso, int & stride) {
                const bool do_so = nso > 0;
                stride = 3 + 3 * NDIM + (do_so ? 3 * NDIM + 2 + 3 * nso : 0);
                std::vector<double> local;
                local.reserve(halos.size() * size_t(stride));
                for (auto & g : halos) {
                    local.push_back(double(g.id));
                    local.push_back(double(g.np));
                    local.push_back(g.mass);
                    for (int idim = 0; idim < NDIM; idim++)
                        local.push_back(g.pos[idim]);
                    for (int idim = 0; idim < NDIM; idim++)
                        local.push_back(g.vel[idim]);
                    for (int idim = 0; idim < NDIM; idim++)
                        local.push_back(g.vel_rms[idim]);
                    if (do_so) {
                        for (int idim = 0; idim < NDIM; idim++)
                            local.push_back(g.so_pos[idim]);
                        for (int idim = 0; idim < NDIM; idim++)
                            local.push_back(g.so_vel[idim]);
                        for (int idim = 0; idim < NDIM; idim++)
                            local.push_back(g.so_vel_rms[idim]);
                        local.push_back(g.so_truncated ? 1.0 : 0.0);
                        local.push_back(g.so_overlap ? 1.0 : 0.0);
                        for (int id = 0; id < nso; id++) {
                            local.push_back(g.so_mass[id]);
                            local.push_back(g.so_radius[id]);
                            local.push_back(double(g.so_npart[id]));
                        }
                    }
                }
#ifdef USE_MPI
                return FML::COMMUNICATION::gather_records_checked(local);
#else
                return local;
#endif
            }
        } // namespace detail

        /// The halo catalogue as one text file, written by task 0 (the same columns as SimulationCode's). Collective
        template <int NDIM>
        void write_halo_catalogue_ascii(const std::string & filename, const HaloCatalogue<NDIM> & cat) {
            const int nso = int(cat.so_definitions.size());
            int stride = 0;
            const auto all = detail::gather_halo_rows<NDIM>(cat.halos, nso, stride);
            write_on_root(filename, [&](std::ostream & fp) {
                fp << "#            np     ";
                fp << "mass (Msun/h)       ";
                fp << "pos[] (Mpc/h)                                   ";
                fp << "vel[] (peculiar km/s)                           ";
                fp << "vel_rms[]  (peculiar km/s)                      ";
                if (nso > 0) {
                    fp << "SO: pos[] (Mpc/h)  vel[] (km/s)  vel_rms[] (km/s)  truncated  overlap";
                    for (auto & def : cat.so_definitions)
                        fp << "  M_" << def << " (Msun/h)  R_" << def << " (Mpc/h)  N_" << def;
                }
                fp << "\n";
                const size_t nhalos = all.size() / size_t(stride);
                for (size_t i = 0; i < nhalos; i++) {
                    const double * row = &all[i * size_t(stride)];
                    fp << std::setw(15) << (long long int)row[1] << " ";
                    for (int c = 2; c < 3 + 3 * NDIM; c++)
                        fp << std::setw(15) << row[c] << " ";
                    if (nso > 0) {
                        const int c0 = 3 + 3 * NDIM;
                        for (int c = c0; c < c0 + 3 * NDIM; c++)
                            fp << std::setw(15) << row[c] << " ";
                        fp << std::setw(3) << int(row[c0 + 3 * NDIM]) << " " << std::setw(3)
                           << int(row[c0 + 3 * NDIM + 1]) << " ";
                        for (int id = 0; id < nso; id++) {
                            const double * so = row + c0 + 3 * NDIM + 2 + 3 * id;
                            fp << std::setw(15) << so[0] << " " << std::setw(15) << so[1] << " " << std::setw(10)
                               << (long long int)so[2] << " ";
                        }
                    }
                    fp << "\n";
                }
            });
        }

#ifdef USE_HDF5
        /// The halo catalogue as one HDF5 file, written by task 0 (the same layout as SimulationCode's). Collective
        template <int NDIM>
        void write_halo_catalogue_hdf5(const std::string & filename, const HaloCatalogue<NDIM> & cat, double redshift,
                                       double boxsize) {
            const int nso = int(cat.so_definitions.size());
            int stride = 0;
            const auto all = detail::gather_halo_rows<NDIM>(cat.halos, nso, stride);
            int failed = 0;
            std::string error;
            if (FML::ThisTask == 0) {
                try {
                    namespace H5 = FML::FILEUTILS::GADGET::HDF5;
                    const size_t nhalos = all.size() / size_t(stride);
                    auto column = [&](int c, int ncols) {
                        std::vector<double> v(nhalos * ncols);
                        for (size_t i = 0; i < nhalos; i++)
                            for (int j = 0; j < ncols; j++)
                                v[i * ncols + j] = all[i * stride + c + j];
                        return v;
                    };
                    auto to_ll = [](const std::vector<double> & v) {
                        std::vector<long long int> r(v.size());
                        for (size_t i = 0; i < v.size(); i++)
                            r[i] = (long long int)v[i];
                        return r;
                    };
                    H5::Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                                    H5Fclose,
                                    filename);
                    H5::Handle header(H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                      H5Gclose,
                                      "Header");
                    const double attrs_d[] = {redshift, boxsize, cat.linking_length, cat.particle_mass_msunh};
                    const char * names_d[] = {"Redshift", "BoxSize", "LinkingLength", "MeanParticleMass"};
                    for (int i = 0; i < 4; i++)
                        H5::write_attribute(header, names_d[i], H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, &attrs_d[i], 1);
                    const long long int nh = (long long int)nhalos, nmin = cat.nmin_per_halo;
                    H5::write_attribute(header, "NumHalos", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, &nh, 1);
                    H5::write_attribute(header, "MinParticlesPerHalo", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, &nmin, 1);
                    auto write_d = [&](hid_t loc, const std::string & name, int c, int ncols) {
                        auto v = column(c, ncols);
                        H5::write_dataset(loc, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, v.data(), nhalos, ncols);
                    };
                    auto write_ll = [&](hid_t loc, const std::string & name, int c) {
                        auto v = to_ll(column(c, 1));
                        H5::write_dataset(loc, name, H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, v.data(), nhalos, 1);
                    };
                    write_ll(file, "ID", 0);
                    write_ll(file, "NumParticles", 1);
                    write_d(file, "Mass", 2, 1);
                    write_d(file, "Position", 3, NDIM);
                    write_d(file, "Velocity", 3 + NDIM, NDIM);
                    write_d(file, "VelocityDispersion", 3 + 2 * NDIM, NDIM);
                    if (nso > 0) {
                        const int c0 = 3 + 3 * NDIM;
                        H5::Handle so(H5Gcreate2(file, "SO", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "SO");
                        write_d(so, "Position", c0, NDIM);
                        write_d(so, "Velocity", c0 + NDIM, NDIM);
                        write_d(so, "VelocityDispersion", c0 + 2 * NDIM, NDIM);
                        write_ll(so, "Truncated", c0 + 3 * NDIM);
                        write_ll(so, "Overlap", c0 + 3 * NDIM + 1);
                        for (int id = 0; id < nso; id++) {
                            const int c = c0 + 3 * NDIM + 2 + 3 * id;
                            H5::Handle g(H5Gcreate2(so,
                                                    cat.so_definitions[id].c_str(),
                                                    H5P_DEFAULT,
                                                    H5P_DEFAULT,
                                                    H5P_DEFAULT),
                                         H5Gclose,
                                         "SO/" + cat.so_definitions[id]);
                            H5::write_attribute(g,
                                                "Delta",
                                                H5T_NATIVE_DOUBLE,
                                                H5T_NATIVE_DOUBLE,
                                                &cat.so_thresholds[id],
                                                1);
                            write_d(g, "Mass", c, 1);
                            write_d(g, "Radius", c + 1, 1);
                            write_ll(g, "NumParticles", c + 2);
                        }
                    }
                } catch (const std::exception & e) {
                    error = e.what();
                    failed = 1;
                }
            }
            FML::MaxOverTasks(&failed);
            if (failed)
                throw std::runtime_error("Cannot write " + filename + (error.empty() ? "" : ": " + error));
        }
#endif

        /// The halo catalogue (halos_z<z>.txt or .hdf5) and the mass functions (massfunc_z<z>.txt for FoF, and
        /// massfunc_so<def>_z<z>.txt) in folder: the SimulationCode file names. Collective
        template <int NDIM>
        void write_halo_outputs(const std::string & folder,
                                const HaloCatalogue<NDIM> & cat,
                                const HaloSettings & settings,
                                const AnalysisContext & context) {
            using Halo = HaloRecord<NDIM>;
            const std::string z = redshift_label(context.redshift());
            auto mf = compute_mass_function<NDIM>(cat.halos,
                                                  [](const Halo & g) { return g.mass; },
                                                  settings.massfunction_nbins,
                                                  settings.massfunction_Mmin,
                                                  settings.massfunction_Mmax,
                                                  context.boxsize);
            write_mass_function(folder + "/massfunc_z" + z + ".txt", mf, "# FoF mass function\n");
            for (size_t id = 0; id < cat.so_definitions.size(); id++) {
                auto mfso = compute_mass_function<NDIM>(cat.halos,
                                                        [id](const Halo & g) { return g.so_mass[id]; },
                                                        settings.massfunction_nbins,
                                                        settings.massfunction_Mmin,
                                                        settings.massfunction_Mmax,
                                                        context.boxsize);
                std::ostringstream header;
                header << "# Spherical overdensity " << cat.so_definitions[id] << " (Delta = " << cat.so_thresholds[id]
                       << " times the mean density of the particles) around the FoF halos\n";
                write_mass_function(folder + "/massfunc_so" + cat.so_definitions[id] + "_z" + z + ".txt",
                                    mfso,
                                    header.str());
            }
#ifdef USE_HDF5
            if (settings.output_fileformat == "HDF5") {
                write_halo_catalogue_hdf5<NDIM>(folder + "/halos_z" + z + ".hdf5",
                                                cat,
                                                context.redshift(),
                                                context.boxsize);
                return;
            }
#endif
            write_halo_catalogue_ascii<NDIM>(folder + "/halos_z" + z + ".txt", cat);
        }

    } // namespace ANALYSIS
} // namespace FML
#endif

#ifndef READWRITERAMSES_HEADER
#define READWRITERAMSES_HEADER

#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {
    namespace FILEUTILS {

        /// Reading RAMSES files (DM only).
        namespace RAMSES {

            // The types of the fields in the ramses output
            // If you have nonstandard files then change it here
            using RamsesPosType = double;
            using RamsesVelType = double;
            using RamsesMassType = double;
            using RamsesLevelType = int;
            using RamsesTagType = char;
            using RamsesFamilyType = char;

            // For ID we determine the size when reading
            // one of these two
            using RamsesIDType = int;
            using RamsesLongIDType = size_t;

            // The header of a particle file
            struct ParticleFileHeader {
                int ncpu;
                int ndim;
                int npart;
                int localseed[4];
                int nstar_tot;
                int mstar_tot[2];
                int mstar_lost[2];
                int nsink;
            };

            //===========================================
            ///
            /// Read (and write) files related to RAMSES
            /// snapshots
            ///
            /// Only implemented read of particle files
            /// Only implemented write of ic_deltab IC file
            ///
            //===========================================

            class RamsesReader {
              private:
                // What is in the file and if we want to store it or not
                std::vector<std::string> entries_in_file{"POS", "VEL", "MASS", "ID", "LEVEL", "FAMILY", "TAG"};
                std::vector<bool> entries_to_store{true, true, true, true, true, true, true};

                // What we store when we read the files
                bool POS_STORE = true;
                bool VEL_STORE = true;
                bool MASS_STORE = false;
                bool ID_STORE = false;
                bool LEVEL_STORE = false;
                bool FAMILY_STORE = false;
                bool TAG_STORE = false;

                // File description
                std::string filepath{};
                int outputnr{0};

                // Total number of particles and particles in local domain
                size_t npart{0};
                size_t npart_in_domain{0};

                // Data obtained from info-file
                int ncpu{0};
                int levelmin{0};
                int levelmax{0};
                int ngridmax{0};
                int nstep_coarse{0};
                double aexp{0.0};
                double time{0.0};
                double boxlen{0.0};
                double h0{0.0};
                double omega_m{0.0};
                double omega_l{0.0};
                double omega_k{0.0};
                double omega_b{0.0};
                double unit_l{0.0};
                double unit_d{0.0};
                double unit_t{0.0};
                double boxlen_ini{0.0};

                // Book-keeping variables
                bool infofileread{false};
                size_t npart_read{0};

                double buffer_factor{1.0};
                bool keep_only_particles_in_domain{true};
                bool verbose{false};

                std::vector<int> npart_in_file{};
                std::vector<int> npart_in_domain_in_file{};

                void throw_error(std::string errormessage) const {
#ifdef USE_MPI
                    std::cout << errormessage << std::flush;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    abort();
#else
                    throw std::runtime_error(errormessage);
#endif
                }

              public:
                RamsesReader() = default;

                RamsesReader(std::string _filepath,
                             int _outputnr,
                             double _buffer_factor,
                             bool _keep_only_particles_in_domain,
                             bool _verbose = false)
                    : filepath(_filepath), outputnr(_outputnr), buffer_factor(_buffer_factor),
                      keep_only_particles_in_domain(_keep_only_particles_in_domain),
                      verbose(_verbose and FML::ThisTask == 0) {
                    read_info();
                }

                void set_file_format(std::vector<std::string> what_is_in_file) {
                    for (auto e : what_is_in_file) {
                        if (!(e == "POS" or e == "VEL" or e == "MASS" or e == "ID" or e == "LEVEL" or e == "FAMILY" or
                              e == "TAG"))
                            throw_error("[RamsesReader::set_file_format] Unknown field in file-entry (not "
                                        "POS,VEL,MASS,ID,LEVEL,FAMILY,TAG): " +
                                        e);
                    }
                    entries_in_file = what_is_in_file;
                }

                template <class T>
                void read_ramses_single(int ifile, std::vector<T> & p) {
                    if (!infofileread)
                        read_info();

                    std::string numberfolder = int_to_ramses_string(outputnr);
                    std::string numberfile = int_to_ramses_string(ifile + 1);
                    std::string partfile = filepath == "" ? "" : filepath + "/";
                    partfile = partfile + "output_" + numberfolder + "/part_" + numberfolder + ".out" + numberfile;
                    FILE * fp;
                    if ((fp = fopen(partfile.c_str(), "r")) == nullptr) {
                        throw_error("[RamsesReader::read_ramses_single] Error opening particle file " + partfile);
                    }

                    // When reading 1 by 1 we must put this to 0
                    npart_read = 0;

                    // Read header
                    auto header = read_particle_header(fp);
                    fclose(fp);

                    // Allocate particles
                    if (p.size() < size_t(header.npart))
                        p.resize(header.npart);

                    // Check that dimensions match
                    int ndim = p[0].get_ndim();
                    assert(ndim == header.ndim);

                    // Read the data
                    read_particle_file(ifile, p);
                }

                template <class T>
                void read_ramses(std::vector<T> & p) {
                    if (!infofileread)
                        read_info();

                    if (keep_only_particles_in_domain) {
                        size_t nallocate = buffer_factor > 1.0 ? size_t(double(npart_in_domain) * buffer_factor) : npart_in_domain;
                        p.reserve(nallocate);
                        p.resize(npart_in_domain);
                    } else {
                        p.reserve(npart * buffer_factor);
                    }

                    std::vector<long long int> npart_task(FML::NTasks);
                    npart_task[FML::ThisTask] = npart_in_domain;
#ifdef USE_MPI
                    long long int n = npart_in_domain;
                    MPI_Allgather(&n, 1, MPI_LONG_LONG, npart_task.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
#endif
                    if (verbose) {
                        std::cout << "\n";
                        std::cout << "=================================="
                                  << "\n";
                        std::cout << "Read Ramses Particle files:       "
                                  << "\n";
                        std::cout << "=================================="
                                  << "\n";
                        std::cout << "Folder containing output: " << filepath << "\n";
                        std::cout << "Outputnumber: " << int_to_ramses_string(outputnr) << "\n";
                        std::cout << "Npart total " << npart << " particles\n";
                        if (keep_only_particles_in_domain)
                            for (int i = 0; i < FML::NTasks; i++)
                                std::cout << "On task " << i << " we will store " << npart_task[i] << " particles\n";
                    }

                    // Loop and read all particle files
                    npart_read = 0;
                    for (int i = 0; i < ncpu; i++) {
                        read_particle_file(i, p);
                    }

                    if (verbose) {
                        std::cout << "Done reading particles\n";
                        std::cout << "==================================\n\n";
                    }
                }

                //====================================================
                // Integer to ramses string
                //====================================================

                std::string int_to_ramses_string(int i) {
                    std::stringstream rnum;
                    rnum << std::setfill('0') << std::setw(5) << i;
                    return rnum.str();
                }

                //====================================================
                // Read binary methods. The skips's are due to
                // files to be read are written by fortran code
                //====================================================

                int read_int(FILE * fp) {
                    int res;
                    read_section(fp, (char *)&res, 1);
                    return res;
                }

                int read_section(FILE * fp, char * buffer, int n) {
                    int skip1, skip2;
                    fread(&skip1, sizeof(int), 1, fp);
                    fread(buffer, sizeof(char), skip1, fp);
                    fread(&skip2, sizeof(int), 1, fp);
                    int bytes_per_element = skip1 / n;
                    assert(bytes_per_element * n == skip1);
                    assert(skip1 == skip2);
                    return bytes_per_element;
                }

                //====================================================
                // Read a ramses info file
                //====================================================

                void read_info() {
                    int ndim_loc;
                    std::string numbers = int_to_ramses_string(outputnr);
                    std::string infofile = filepath == "" ? "" : filepath + "/";
                    infofile = infofile + "output_" + numbers + "/info_" + numbers + ".txt";
                    FILE * fp;

                    // Open file
                    if ((fp = fopen(infofile.c_str(), "r")) == nullptr) {
                        throw_error("[RamsesReader::read_info] Error opening info file " + infofile);
                    }

                    // Read the info-file
                    fscanf(fp, "ncpu        =  %d\n", &ncpu);
                    fscanf(fp, "ndim        =  %d\n", &ndim_loc);
                    fscanf(fp, "levelmin    =  %d\n", &levelmin);
                    fscanf(fp, "levelmax    =  %d\n", &levelmax);
                    fscanf(fp, "ngridmax    =  %d\n", &ngridmax);
                    fscanf(fp, "nstep_coarse=  %d\n", &nstep_coarse);
                    fscanf(fp, "\n");
                    fscanf(fp, "boxlen      =  %lf\n", &boxlen);
                    fscanf(fp, "time        =  %lf\n", &time);
                    fscanf(fp, "aexp        =  %lf\n", &aexp);
                    fscanf(fp, "H0          =  %lf\n", &h0);
                    fscanf(fp, "omega_m     =  %lf\n", &omega_m);
                    fscanf(fp, "omega_l     =  %lf\n", &omega_l);
                    fscanf(fp, "omega_k     =  %lf\n", &omega_k);
                    fscanf(fp, "omega_b     =  %lf\n", &omega_b);
                    fscanf(fp, "unit_l      =  %lf\n", &unit_l);
                    fscanf(fp, "unit_d      =  %lf\n", &unit_d);
                    fscanf(fp, "unit_t      =  %lf\n", &unit_t);
                    fclose(fp);

                    // Calculate boxsize in Mpc/h
                    boxlen_ini = unit_l * h0 / 100.0 / aexp / 3.08567758e24;

                    // Read how many particles there is in the files
                    count_particles_in_files();
                    npart = 0;
                    npart_in_domain = 0;
                    for (int i = 0; i < ncpu; i++) {
                        npart += size_t(npart_in_file[i]);
                        npart_in_domain += size_t(npart_in_domain_in_file[i]);
                    }

                    if (verbose) {
                        std::cout << "\n";
                        std::cout << "=================================="
                                  << "\n";
                        std::cout << "Infofile data:                    "
                                  << "\n";
                        std::cout << "=================================="
                                  << "\n";
                        std::cout << "Filename     = " << infofile << "\n";
                        std::cout << "Box (Mpc/h)  = " << boxlen_ini << "\n";
                        std::cout << "ncpu         = " << ncpu << "\n";
                        std::cout << "npart        = " << npart << "\n";
                        std::cout << "aexp         = " << aexp << "\n";
                        std::cout << "H0           = " << h0 << "\n";
                        std::cout << "omega_m      = " << omega_m << "\n";
                        std::cout << "=================================="
                                  << "\n\n";
                    }

                    infofileread = true;
                }

                // Count how many particles are in each file and how many fall into the local domain
                void count_particles_in_files() {
                    npart_in_file.resize(ncpu);
                    npart_in_domain_in_file.resize(ncpu);

                    for (int i = 0; i < ncpu; i++) {
                        FILE * fp;
                        std::string numberfolder = int_to_ramses_string(outputnr);
                        std::string numberfile = int_to_ramses_string(i + 1);
                        std::string partfile = "";
                        if (filepath.compare("") != 0)
                            partfile = filepath + "/";
                        partfile = partfile + "output_" + numberfolder + "/part_" + numberfolder + ".out" + numberfile;

                        // Open file
                        if ((fp = fopen(partfile.c_str(), "r")) == nullptr) {
                            std::string error = "Error opening particle file " + partfile;
                            exit(0);
                        }

                        // Read header
                        auto header = read_particle_header(fp);

                        // Read x position
                        std::vector<char> buffer(header.npart * 8);
                        read_section(fp, buffer.data(), header.npart);

                        // Count how many positions fall into the local domain
                        RamsesPosType * pos = (RamsesPosType *)buffer.data();
                        int nindomain = 0;
                        for (int j = 0; j < header.npart; j++) {
                            if (pos[j] >= FML::xmin_domain and pos[j] < FML::xmax_domain)
                                nindomain++;
                        }
                        npart_in_domain_in_file[i] = nindomain;
                        npart_in_file[i] = header.npart;

                        fclose(fp);
                    }
                }

                ParticleFileHeader read_particle_header(FILE * fp) {
                    ParticleFileHeader head;
                    head.ncpu = read_int(fp);
                    head.ndim = read_int(fp);
                    head.npart = read_int(fp);
                    read_section(fp, (char *)head.localseed, 4);
                    head.nstar_tot = read_int(fp);
                    read_section(fp, (char *)head.mstar_tot, 2);
                    read_section(fp, (char *)head.mstar_lost, 2);
                    head.nsink = read_int(fp);
                    return head;
                }

                //====================================================
                // Store the positions if particle has positions
                //====================================================

                template <class T>
                void store_positions(RamsesPosType * pos, char * is_in_domain, T * p, const int dim, const int np) {
                    if constexpr (not FML::PARTICLE::has_get_pos<T>())
                        return;

                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            auto * x = FML::PARTICLE::GetPos(p[count++]);
                            x[dim] = pos[i];
                        }
                    }
                }

                //====================================================
                // Store the velocities if particle has velocities
                //====================================================
                template <class T>
                void store_velocity(RamsesVelType * vel, char * is_in_domain, T * p, const int dim, const int np) {
                    if constexpr (not FML::PARTICLE::has_get_vel<T>())
                        return;
                    RamsesVelType velfac = 100.0 * boxlen_ini / aexp;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            auto * v = FML::PARTICLE::GetVel(p[count++]);
                            v[dim] = vel[i] * velfac;
                        }
                    }
                }

                //====================================================
                // Store the mass if particle has mass
                //====================================================
                template <class T>
                void store_mass(RamsesMassType * mass, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_mass<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetMass(p[count++], mass[i]);
                        }
                    }
                }

                //====================================================
                // Store the id if particle has id
                //====================================================
                template <class T>
                void store_id(RamsesIDType * id, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_id<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetID(p[count++], id[i]);
                        }
                    }
                }

                //====================================================
                // Store the id if particle has id
                //====================================================
                template <class T>
                void store_longid(RamsesLongIDType * id, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_id<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetID(p[count++], id[i]);
                        }
                    }
                }

                //====================================================
                // Store the family if particle has family
                //====================================================
                template <class T>
                void store_family(RamsesFamilyType * family, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_family<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetFamily(p[count++], family[i]);
                        }
                    }
                }

                //====================================================
                // Store the tag if particle has tag
                //====================================================
                template <class T>
                void store_tag(RamsesTagType * tag, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_tag<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetTag(p[count++], tag[i]);
                        }
                    }
                }

                //====================================================
                // Store the level if particle has level
                //====================================================
                template <class T>
                void store_level(RamsesLevelType * level, char * is_in_domain, T * p, const int np) {
                    if constexpr (not FML::PARTICLE::has_set_level<T>())
                        return;
                    int count = 0;
                    for (int i = 0; i < np; i++) {
                        if (is_in_domain[i] == 1) {
                            FML::PARTICLE::SetLevel(p[count++], level[i]);
                        }
                    }
                }

                //====================================================
                // Read a single particle file
                //====================================================
                template <class T>
                void read_particle_file(const int i, std::vector<T> & p) {
                    std::string numberfolder = int_to_ramses_string(outputnr);
                    std::string numberfile = int_to_ramses_string(i + 1);
                    std::string partfile = "";
                    if (filepath.compare("") != 0)
                        partfile = filepath + "/";
                    partfile = partfile + "output_" + numberfolder + "/part_" + numberfolder + ".out" + numberfile;
                    FILE * fp;

                    // Local variables used to read into
                    // Open file
                    if ((fp = fopen(partfile.c_str(), "r")) == nullptr) {
                        std::string error = "Error opening particle file " + partfile;
                        exit(0);
                    }

                    // Read header
                    auto header = read_particle_header(fp);

                    // Store npcu globally
                    ncpu = header.ncpu;

                    // Allocate memory for buffer
                    std::vector<char> buffer(header.npart * 8);
                    std::vector<char> is_in_domain(header.npart);

                    // Verbose
                    assert(header.npart == npart_in_file[i]);
                    if (verbose)
                        std::cout << "Reading " << partfile << " npart = " << header.npart
                                  << " InDomain = " << npart_in_domain_in_file[i] << std::endl;

                    // Methods for reading each of the types
                    auto read_pos = [&](bool store) {
                        if (verbose)
                            std::cout << "Read POS ";

                        for (int j = 0; j < header.ndim; j++) {

                            int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                            if (bytes_per_element != sizeof(RamsesVelType))
                                throw_error("[RamsesReader::read_particle_file] Field POS has size " +
                                            std::to_string(bytes_per_element) + " but size set to " +
                                            std::to_string(sizeof(RamsesVelType)));

                            // Set the book-keeping array that tells us if a particle is in the domain or not
                            if (j == 0) {
                                if (!keep_only_particles_in_domain) {
                                    for (int i = 0; i < header.npart; i++) {
                                        is_in_domain[i] = 1;
                                    }
                                } else {
                                    RamsesPosType * pos = (RamsesPosType *)buffer.data();
                                    for (int i = 0; i < header.npart; i++) {
                                        is_in_domain[i] =
                                            (pos[i] >= FML::xmin_domain and pos[i] < FML::xmax_domain) ? 1 : 0;
                                    }
                                }
                            }

                            if (store)
                                store_positions((RamsesPosType *)buffer.data(),
                                                is_in_domain.data(),
                                                &p[npart_read],
                                                j,
                                                header.npart);
                        }
                    };

                    auto read_vel = [&](bool store) {
                        if (verbose)
                            std::cout << " VEL ";
                        for (int j = 0; j < header.ndim; j++) {
                            int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                            if (bytes_per_element != sizeof(RamsesVelType))
                                throw_error("[RamsesReader::read_particle_file] Field VEL has size " +
                                            std::to_string(bytes_per_element) + " but size set to " +
                                            std::to_string(sizeof(RamsesVelType)));
                            if (store)
                                store_velocity((RamsesVelType *)buffer.data(),
                                               is_in_domain.data(),
                                               &p[npart_read],
                                               j,
                                               header.npart);
                        }
                    };

                    auto read_mass = [&](bool store) {
                        if (verbose)
                            std::cout << " MASS ";
                        int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                        if (bytes_per_element != sizeof(RamsesMassType))
                            throw_error("[RamsesReader::read_particle_file] Field MASS has size " +
                                        std::to_string(bytes_per_element) + " but size set to " +
                                        std::to_string(sizeof(RamsesMassType)));
                        if (store)
                            store_mass(
                                (RamsesMassType *)buffer.data(), is_in_domain.data(), &p[npart_read], header.npart);
                    };

                    auto read_id = [&](bool store) {
                        int bytes_per_element = read_section(fp, buffer.data(), header.npart);

                        if (verbose)
                            std::cout << " ID (" << bytes_per_element << " bytes) ";

                        if (store) {
                            if (bytes_per_element == 4)
                                store_id(
                                    (RamsesIDType *)buffer.data(), is_in_domain.data(), &p[npart_read], header.npart);
                            else if (bytes_per_element == 8)
                                store_longid((RamsesLongIDType *)buffer.data(),
                                             is_in_domain.data(),
                                             &p[npart_read],
                                             header.npart);
                            else {
                                throw_error("[RamsesReader::read_particle_file] Field ID has size " +
                                            std::to_string(bytes_per_element) + " but we expected 4 or 8\n");
                            }
                        }
                    };

                    auto read_level = [&](bool store) {
                        if (verbose)
                            std::cout << " LEVEL ";
                        int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                        if (bytes_per_element != sizeof(RamsesLevelType))
                            throw_error("[RamsesReader::read_particle_file] Field LEVEL has size " +
                                        std::to_string(bytes_per_element) + " but size set to " +
                                        std::to_string(sizeof(RamsesLevelType)));
                        if (store)
                            store_level(
                                (RamsesLevelType *)buffer.data(), is_in_domain.data(), &p[npart_read], header.npart);
                    };

                    auto read_family = [&](bool store) {
                        if (verbose)
                            std::cout << " FAMILY ";
                        int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                        if (bytes_per_element != sizeof(RamsesFamilyType))
                            throw_error("[RamsesReader::read_particle_file] Field FAMILY has size " +
                                        std::to_string(bytes_per_element) + " but size set to " +
                                        std::to_string(sizeof(RamsesFamilyType)));
                        if (store)
                            store_family(
                                (RamsesFamilyType *)buffer.data(), is_in_domain.data(), &p[npart_read], header.npart);
                    };

                    auto read_tag = [&](bool store) {
                        if (verbose)
                            std::cout << " TAG ";
                        int bytes_per_element = read_section(fp, buffer.data(), header.npart);
                        if (bytes_per_element != sizeof(RamsesTagType))
                            throw_error("[RamsesReader::read_particle_file] Field TAG has size " +
                                        std::to_string(bytes_per_element) + " but size set to " +
                                        std::to_string(sizeof(RamsesTagType)));
                        if (store)
                            store_tag(
                                (RamsesTagType *)buffer.data(), is_in_domain.data(), &p[npart_read], header.npart);
                    };

                    // Do the actual reading
                    for (size_t i = 0; i < entries_in_file.size(); i++) {
                        auto entry = entries_in_file[i];
                        bool store_entry = entries_to_store[i];
                        if (i == 0)
                            assert(entry == "POS");
                        if (entry == "POS")
                            read_pos(store_entry);
                        else if (entry == "VEL")
                            read_vel(store_entry);
                        else if (entry == "MASS")
                            read_mass(store_entry);
                        else if (entry == "ID")
                            read_id(store_entry);
                        else if (entry == "LEVEL")
                            read_level(store_entry);
                        else if (entry == "FAMILY")
                            read_family(store_entry);
                        else if (entry == "TAG")
                            read_tag(store_entry);
                        else
                            throw_error("[RamsesReader::read_particle_file] Unknown file entry (not one of "
                                        "POS,VEL,MASS,ID,LEVEL,FAMILY,TAG): " +
                                        entry);
                    }
                    if (verbose)
                        std::cout << "\n";

                    // Update global variables
                    npart_read +=
                        keep_only_particles_in_domain ? size_t(npart_in_domain_in_file[i]) : size_t(npart_in_file[i]);
                    fclose(fp);
                }
            };

            //====================================================
            // Write a ramses ic_deltab file for a given cosmology
            // and levelmin
            //====================================================

            void write_icdeltab(const std::string filename,
                                const float _astart,
                                const float _omega_m,
                                const float _omega_l,
                                const float _boxlen_ini,
                                const float _h0,
                                const int _levelmin,
                                const int NDIM) {
                FILE * fp;
                int tmp, n1, n2, n3;
                float xoff1, xoff2, xoff3, dx;

                // Assume zero offset
                xoff1 = xoff2 = xoff3 = 0.0;

                // Assumes same in all directions
                n1 = n2 = n3 = 1 << _levelmin;

                // dx in Mpc (CHECK THIS)
                dx = _boxlen_ini / float(n1) * 100.0 / _h0;

                // Number of floats and ints we write
                tmp = (5 + NDIM) * sizeof(float) + NDIM * sizeof(int);

                // Open file
                if ((fp = fopen(filename.c_str(), "w")) == nullptr) {
                    throw std::runtime_error("[write_icdeltab] Error opening file " + filename);
                }

                // Write file
                fwrite(&tmp, sizeof(int), 1, fp);
                fwrite(&n1, sizeof(int), 1, fp);
                if (NDIM > 1)
                    fwrite(&n2, sizeof(int), 1, fp);
                if (NDIM > 2)
                    fwrite(&n3, sizeof(int), 1, fp);
                fwrite(&dx, sizeof(float), 1, fp);
                fwrite(&xoff1, sizeof(float), 1, fp);
                if (NDIM > 1)
                    fwrite(&xoff2, sizeof(float), 1, fp);
                if (NDIM > 2)
                    fwrite(&xoff3, sizeof(float), 1, fp);
                fwrite(&_astart, sizeof(float), 1, fp);
                fwrite(&_omega_m, sizeof(float), 1, fp);
                fwrite(&_omega_l, sizeof(float), 1, fp);
                fwrite(&_h0, sizeof(float), 1, fp);
                fwrite(&tmp, sizeof(int), 1, fp);
                fclose(fp);
            }
        } // namespace RAMSES
    }     // namespace FILEUTILS
} // namespace FML
#endif

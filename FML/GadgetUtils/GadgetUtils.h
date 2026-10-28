#ifndef GADGETUTILS_HEADER
#define GADGETUTILS_HEADER

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

//====================================================================================
//
// Read and write general Gadget files. Checks and swaps endian if needed.
// If the particle class has FAMILY methods then the type of the particle (0,1,2,3,4,5)
// is set in the particle. If not then we assume all is DM (family = 1)
//
// In an MPI setting then we have the option of only storing particles that fall inside the
// local domain
//
// If the files have position units other than Mpc/h then this can be set by gadget_pos_factor
// which is 1.0 for Mpc/h, 1000.0 for kpc/h and (Mpc/h / POSUNIT) in general.
//
// The particle class can be anything, but needs to have the methods
// auto *get_pos()
// auto *get_vel()
// idtype get_id()
// void set_pos(floattype *)
// void set_vel(floattype *)
// void set_id(idtype)
// together with get_ndim()
// if you want to store the corresponding quantities. If you don't have say set_id
// then we will ignore the IDs in the file
//
// We assume positions in the Particle class are in [0,1]. These are scaled to
// [0,Boxsize] when writing to file.
//
// When we read we return positions in [0,1).
//
// Reading: format 1 and format 2 (blocks preceded by a 4-character label block) files, either
// endianness, POS/VEL in float or double, 4 or 8 byte IDs (from the block size), and the MASS
// block (particles of the types with MassTable = 0). The masses (in 1e10 Msun/h) are stored with
// set_mass if the particle has it (only for the particles in the MASS block). Files with no
// particles are skipped. In format 2 files blocks other than POS, VEL, ID and MASS are skipped,
// in format 1 files the blocks after the MASS block are ignored.
//
// Writing: IDs are written with 4 bytes unless some ID does not fit in an int (or the total number
// of particles is larger than 2^31) in which case they are written with 8 bytes. Blocks larger
// than 2^31 bytes (the limit of the format) give an error: use more files.
//
// Errors are handled via throw_error in the class below (with MPI it aborts and otherwise
// throws a runtime error)
//
// Compile time defines:
// GADGET_LONG_INT_IDS  : Always write 8 byte IDs
// GADGET_ONLY_DM       : Only read type 1 particles and write all particles as type 1
//                        (GADGET_ONLY_READ_DM is the same)
//
//====================================================================================

#if defined(GADGET_ONLY_READ_DM) and not defined(GADGET_ONLY_DM)
#define GADGET_ONLY_DM
#endif

namespace FML {
    namespace FILEUTILS {

        /// Reading and writing GADGET files (DM only).
        namespace GADGET {

            // This is Mpl^2/(H0 Msun/h) used to set the mass of particles
            extern const double MplMpl_over_H0Msunh;
            // This is 1/H0 in units of Mpc/h used to set the mass of particles
            extern const double HubbleLengthInMpch;

            // The (smallest) ID type we write
#ifdef GADGET_LONG_INT_IDS
            using gadget_particle_id_type = long long int;
#else
            using gadget_particle_id_type = int;
#endif

            /// The GADGET header format
            // Do not change the order of the fields below as this is read as one piece of memory from file
            typedef struct GadgetHeader {
                // npart[1] gives the number of DM particles in the file, other particle types are ignored
                unsigned int npart[6]{0, 0, 0, 0, 0, 0};
                double mass[6]{0., 0., 0., 0., 0., 0.}; // mass[1] gives the particle mass
                double time{0.0};                       // Cosmological scale factor of snapshot
                double redshift{0.0};                   // Redshift of snapshot
                int flag_sfr{0};                        // Flags whether star formation is used
                int flag_feedback{0};                   // Flags whether feedback from star formation is included
                // If npartTotal[1] > 2^32, then total is this plus 2^32 * npartTotalHighWord[1]
                unsigned int npartTotal[6]{0, 0, 0, 0, 0, 0};
                int flag_cooling{0};     // Flags whether radiative cooling is included
                int num_files{0};        // The number of files that are used for a snapshot
                double BoxSize{0.0};     // Simulation box size (in code units)
                double Omega0{0.0};      // Matter density parameter
                double OmegaLambda{0.0}; // Lambda density parameter
                double HubbleParam{0.0}; // Hubble Parameter
                int flag_stellarage{0};  // flags whether the age of newly formed stars is recorded and saved
                int flag_metals{0};      // flags whether metal enrichment is included
                // High word of the total number of particles of each type
                unsigned int npartTotalHighWord[6]{0, 0, 0, 0, 0, 0};
                int flag_entropy_instead_u{0}; // Flags that IC-file contains entropy instead of u
                // Fills to 256 Bytes
                char fill[256 - sizeof(unsigned int) * 18 - sizeof(int) * 7 - sizeof(double) * 12];
            } GadgetHeader;
            static_assert(sizeof(GadgetHeader) == 256);

            void print_header_info(const GadgetHeader & header);

            /// Class for reading Gadget files. Checks and corrects for different endian-ness.
            class GadgetReader {
              private:
                GadgetHeader header{};

                bool endian_swap{false};
                bool header_is_read{false};
                // Format 2: every block is preceded by a block with a 4 character label
                bool format2{false};

                // The dimensions of the positions and velocities in the files.
                int NDIM{3};

                // The fields we assume is in the file (format 1). MASS is added if needed
                std::vector<std::string> fields_in_file = {"POS", "VEL", "ID"};

                void throw_error(std::string errormessage) const;

                /// Read a format 2 label block. Returns false at the end of the file
                bool read_label(std::ifstream & fp, std::string & label);

              public:
                GadgetReader() = default;
                GadgetReader(int ndim);

                /// Read a single gadget file and store the data in part. If only_keep_part_in_domain then
                /// we only keep the partiles that fall within the local domain
                template <class T, class Alloc = std::allocator<T>>
                void read_gadget_single(std::string filename,
                                        std::vector<T, Alloc> & part,
                                        bool only_keep_part_in_domain,
                                        bool verbose);

                /// Read all gadget files and store the data in part. If only_keep_part_in_domain then
                /// we only keep the partiles that fall within the local domain. If buffer_factor is > 1
                /// then we allocate corresponding extra storage in part
                template <class T, class Alloc = std::allocator<T>>
                void read_gadget(std::string filename,
                                 std::vector<T, Alloc> & part,
                                 double buffer_factor,
                                 bool only_keep_part_in_domain,
                                 bool verbose);

                /// Read a section (block) of a gadget file. The buffer is resized to the size of the block
                void read_section(std::ifstream & fp, std::vector<char> & buffer);

                /// Read the gadget header (detects the endianness and format 1 / format 2)
                void read_header(std::ifstream & fp);

                /// Get the header (assumes it has been read)
                GadgetHeader get_header();

                /// Get the number of gadget files
                int get_num_files(std::string filename = "");

                /// If non-standard file, set the fields that are in the file (POS,VEL,ID,MASS implemented)
                void set_fields_in_file(std::vector<std::string> fields);
            };

            /// One set of particles (one GADGET particle type) for write_gadget_species. The particles are accessed
            /// through functions so that species with different particle classes can go in the same file (see
            /// make_gadget_species)
            struct GadgetSpecies {
                int type{1};               // GADGET particle type (0-5)
                size_t npart{0};           // number of particles on this task
                size_t npart_total{0};     // total number of particles (over all tasks)
                double mass_fraction{1.0}; // Omega of this species / OmegaM (sets the particle mass)
                std::function<void(size_t, double *)> get_pos; // NDIM components of particle i (empty: none)
                std::function<void(size_t, double *)> get_vel; // NDIM components of particle i (empty: none)
                std::function<long long int(size_t)> get_id;   // ID of particle i (empty: none)
                /// Mass of particle i in units of the mean particle mass of the species (empty: all the same). If set
                /// the MassTable entry of the type is 0 and the masses are written (the MASS block, PartTypeN/Masses)
                std::function<double(size_t)> get_mass;
                /// Extra per-particle scalars (name, value of particle i), e.g. delta-f weights. Only written to
                /// HDF5 files (as PartTypeN/<name>)
                std::vector<std::pair<std::string, std::function<double(size_t)>>> extra_scalars;
            };

            /// A GadgetSpecies for an array of particles of class T (with get_pos, and optionally get_vel and get_id)
            template <class T>
            GadgetSpecies
            make_gadget_species(int type, T * part, size_t npart, size_t npart_total, double mass_fraction) {
                GadgetSpecies sp;
                sp.type = type;
                sp.npart = npart;
                sp.npart_total = npart_total;
                sp.mass_fraction = mass_fraction;
                const int ndim = FML::PARTICLE::GetNDIM(T());
                if constexpr (FML::PARTICLE::has_get_pos<T>())
                    sp.get_pos = [part, ndim](size_t i, double * x) {
                        const auto * p = FML::PARTICLE::GetPos(part[i]);
                        for (int idim = 0; idim < ndim; idim++)
                            x[idim] = double(p[idim]);
                    };
                if constexpr (FML::PARTICLE::has_get_vel<T>())
                    sp.get_vel = [part, ndim](size_t i, double * x) {
                        const auto * v = FML::PARTICLE::GetVel(part[i]);
                        for (int idim = 0; idim < ndim; idim++)
                            x[idim] = double(v[idim]);
                    };
                if constexpr (FML::PARTICLE::has_get_id<T>())
                    sp.get_id = [part](size_t i) { return (long long int)FML::PARTICLE::GetID(part[i]); };
                return sp;
            }

            /// Write files in GADGET format
            class GadgetWriter {
              private:
                GadgetHeader header{};

                int NDIM{3};

                void throw_error(std::string errormessage) const;

                /// Write n IDs (id(i)) with 4 bytes, or 8 bytes if long_ids
                void
                write_ids(std::ofstream & fp, size_t n, bool long_ids, const std::function<long long int(size_t)> & id);

              public:
                GadgetWriter() = default;
                GadgetWriter(int ndim);

                /// Write a single gadget file with multiple species. pos_norm is to convert from user units to
                /// positions in [0, box) vel_norm is to convert from user units to sqrt(a) dxdt in units of km/s
                /// OmegaFamilyOverOmegaM is used to set the mass (the mass is m ~ OmegaM  * Box^3/Npart *
                /// OmegaFamilyOverOmegaM). If you only have CDM particles in the usual spot then
                /// this array is just (0,1,0,0,0,0). If you have baryons in 0 with OmegaB=0.05 and cdm in 1 with
                /// OmegaCDM = 0.25 then this is (1/6,5/6,0,0,0,0)
                /// No MPI communication if the particles have no family and OmegaFamilyOverOmegaM[1] = 1 (otherwise
                /// the number of particles of each type is summed over the tasks: all tasks must call this)
                template <class T>
                void write_gadget_single(std::string filename,
                                         T * part,
                                         size_t NumPart,
                                         size_t NumPartTot,
                                         int NumberOfFilesToWrite,
                                         double aexp,
                                         double Boxsize,
                                         double OmegaM,
                                         double OmegaLambda,
                                         double HubbleParam,
                                         double pos_norm,
                                         double vel_norm,
                                         std::vector<double> OmegaFamilyOverOmegaM = {0., 1., 0., 0., 0., 0.});

                /// Write a single gadget file with several sets of particles (e.g. CDM, baryons and neutrinos kept in
                /// separate containers), each its own particle type (types must differ). The mass of a particle of a
                /// species is mass_fraction * OmegaM * rho_crit * Box^3 / npart_total. IDs are written as they are
                /// (make them unique). No MPI communication (the totals are given). Extra scalars are not written.
                /// Species with get_mass get MassTable = 0 and their masses in a MASS block after the IDs
                void write_gadget_species(std::string filename,
                                          std::vector<GadgetSpecies> species,
                                          int NumberOfFilesToWrite,
                                          double aexp,
                                          double Boxsize,
                                          double OmegaM,
                                          double OmegaLambda,
                                          double HubbleParam,
                                          double pos_norm,
                                          double vel_norm);

                /// Write a gadget section (error if larger than 2^31 - 1 bytes, the limit of the format)
                void write_section(std::ofstream & fp, std::vector<char> & buffer, size_t bytes);

                /// Write the gadget header (DM only)
                void write_header(std::ofstream & fp,
                                  unsigned int NumPart,
                                  size_t NumPartTot,
                                  int NumberOfFilesToWrite,
                                  double aexp,
                                  double Boxsize,
                                  double OmegaM,
                                  double OmegaLambda,
                                  double HubbleParam);

                /// Write the gadget header when multiple species
                void write_header_general(std::ofstream & fp,
                                          std::vector<size_t> npart_family,
                                          std::vector<size_t> npart_family_tot,
                                          std::vector<double> mass_in_1e10_msunh,
                                          int NumberOfFilesToWrite,
                                          double aexp,
                                          double Boxsize,
                                          double OmegaM,
                                          double OmegaLambda,
                                          double HubbleParam);
            };

            template <typename T>
            T swap_endian(T u) {
                static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
                union {
                    T u;
                    unsigned char u8[sizeof(T)];
                } source, dest;

                source.u = u;

                for (size_t k = 0; k < sizeof(T); k++)
                    dest.u8[k] = source.u8[sizeof(T) - k - 1];

                return dest.u;
            }

            template <typename T>
            void swap_endian_vector(T * vec, size_t n) {
                for (size_t i = 0; i < n; i++) {
                    vec[i] = swap_endian(vec[i]);
                }
            }

            template <class T, class Alloc>
            void GadgetReader::read_gadget(std::string fileprefix,
                                           std::vector<T, Alloc> & part,
                                           double buffer_factor,
                                           bool only_keep_part_in_domain,
                                           bool verbose) {

                verbose = verbose and FML::ThisTask == 0;

                // Read the number of particles and the number of files
                std::string filename = fileprefix + ".0";
                std::ifstream fp(filename.c_str(), std::ios::binary);
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_gadget_all] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }
                read_header(fp);
                fp.close();

                // Number of files
                const int nfiles = header.num_files;

                // Allocate memory for particle vector and reset it
                // If we only keep in domain we reserve for buffer_factor more particles
                // (but will resize if that is too little)
#ifdef GADGET_ONLY_DM
                size_t npartTotal = (size_t(header.npartTotalHighWord[1]) << 32) + size_t(header.npartTotal[1]);
#else
                size_t npartTotal = 0;
                for (int i = 0; i < 6; i++)
                    npartTotal += (size_t(header.npartTotalHighWord[i]) << 32) + size_t(header.npartTotal[i]);
#endif

                size_t nalloc = npartTotal;
                if (only_keep_part_in_domain)
                    nalloc = size_t(double(npartTotal) * buffer_factor) / FML::NTasks;
                part.clear();
                part.reserve(nalloc);

                // Read all the files
                for (int i = 0; i < nfiles; i++) {
                    filename = fileprefix + "." + std::to_string(i);
                    if (verbose)
                        std::cout << "Reading file " << filename << "\n";
                    read_gadget_single(filename, part, only_keep_part_in_domain, verbose);
                }
            }

            template <class T, class Alloc>
            void GadgetReader::read_gadget_single(std::string filename,
                                                  std::vector<T, Alloc> & part,
                                                  bool only_keep_part_in_domain,
                                                  bool verbose) {

                verbose = verbose and FML::ThisTask == 0;

                // Open file and read the header
                std::ifstream fp(filename.c_str(), std::ios::binary);
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_gadget_single] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }
                read_header(fp);
                if (verbose)
                    print_header_info(header);

                // Positions normalized by the boxsize in the file
                const double pos_norm = 1.0 / header.BoxSize;

                // Velocities normalized to peculiar in km/s
                const double vel_norm = sqrt(header.time);

                // The particles of type t are [type_start[t], type_start[t+1]) in the file
                size_t type_start[7];
                type_start[0] = 0;
                for (int t = 0; t < 6; t++)
                    type_start[t + 1] = type_start[t] + size_t(header.npart[t]);
                const size_t NumPartFileTot = type_start[6];
                auto type_of = [&](size_t i) {
                    int t = 0;
                    while (i >= type_start[t + 1])
                        t++;
                    return t;
                };

                // Nothing to read
                if (NumPartFileTot == 0)
                    return;

#ifndef GADGET_ONLY_DM
                if constexpr (FML::PARTICLE::has_set_family<T>() == false) {
                    if (NumPartFileTot != size_t(header.npart[1])) {
                        std::string errormessage = "[GadgetReader::read_gadget_single] Particle type does not have "
                                                   "set_family, but Gadget file contains multiple species! Either use "
                                                   "define GADGET_ONLY_DM or add methods to particle\n";
                        throw_error(errormessage);
                    }
                }
#endif

                // The number of particles in the MASS block (types with MassTable = 0)
                size_t NumPartWithMass = 0;
                for (int t = 0; t < 6; t++)
                    if (header.mass[t] == 0.0)
                        NumPartWithMass += header.npart[t];

                // Check if the vector has enough capacity to store the elements if not
                // reallocate it. We add the particles to the back of the part array
                if (part.capacity() < part.size() + NumPartFileTot)
                    part.reserve(part.size() + NumPartFileTot);

                // Which particles in the file we keep (in the domain, and type 1 for GADGET_ONLY_DM)
                std::vector<char> keep(NumPartFileTot, 1);
                const size_t index_start = part.size();
                bool pos_is_read = false;

                // Size of the elements in a block (float or double, int or long long) with n * ncomp elements
                auto element_size = [&](const std::string & field, size_t bytes, size_t n, size_t ncomp) {
                    for (size_t size : {size_t(4), size_t(8)})
                        if (bytes == size * n * ncomp)
                            return size;
                    throw_error("[GadgetReader::read_gadget_single] The " + field + " block in " + filename + " has " +
                                std::to_string(bytes) + " bytes, expected 4 or 8 bytes times " +
                                std::to_string(n * ncomp) + "\n");
                    return size_t(0);
                };
                // Element i of a block of floats/doubles (as double)
                auto real_value = [&](const std::vector<char> & buffer, size_t size, size_t i) -> double {
                    if (size == 4)
                        return double(reinterpret_cast<const float *>(buffer.data())[i]);
                    return reinterpret_cast<const double *>(buffer.data())[i];
                };
                auto swap_block = [&](std::vector<char> & buffer, size_t size) {
                    if (not endian_swap)
                        return;
                    if (size == 4)
                        swap_endian_vector(reinterpret_cast<int32_t *>(buffer.data()), buffer.size() / 4);
                    else
                        swap_endian_vector(reinterpret_cast<int64_t *>(buffer.data()), buffer.size() / 8);
                };
                auto check_pos_is_read = [&](const std::string & field) {
                    if (not pos_is_read)
                        throw_error("[GadgetReader::read_gadget_single] The " + field + " block comes before POS in " +
                                    filename + " (POS has to be read first)\n");
                };

                // Process a block
                std::vector<char> buffer;
                auto process_block = [&](const std::string & field) {
                    if (field == "POS") {
                        const size_t size = element_size(field, buffer.size(), NumPartFileTot, NDIM);
                        swap_block(buffer, size);
                        if (verbose)
                            std::cout << "Reading POS bytes = " << buffer.size() << "\n";
                        std::vector<double> x(NDIM);
                        for (size_t i = 0; i < NumPartFileTot; i++) {
                            for (int idim = 0; idim < NDIM; idim++) {
                                x[idim] = real_value(buffer, size, NDIM * i + idim) * pos_norm;
                                if (x[idim] >= 1.0)
                                    x[idim] -= 1.0;
                                if (x[idim] < 0.0)
                                    x[idim] += 1.0;
                            }
                            assert_mpi(x[0] >= 0.0 and x[0] < 1.0,
                                       "[read_gadet_single] Particle has x position outside the boxsize even after "
                                       "periodic wrap");
                            const int type = type_of(i);
#ifdef GADGET_ONLY_DM
                            if (type != 1) {
                                keep[i] = 0;
                                continue;
                            }
#endif
                            if (only_keep_part_in_domain and
                                not(x[0] >= FML::xmin_domain and x[0] < FML::xmax_domain)) {
                                keep[i] = 0;
                                continue;
                            }
                            part.push_back(T{});
                            auto & p = part.back();
                            if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                                auto * pos = FML::PARTICLE::GetPos(p);
                                for (int idim = 0; idim < NDIM; idim++)
                                    pos[idim] = x[idim];
                            }
                            if constexpr (FML::PARTICLE::has_set_family<T>())
                                FML::PARTICLE::SetFamily(p, type);
                        }
                        pos_is_read = true;
                    } else if (field == "VEL") {
                        check_pos_is_read(field);
                        const size_t size = element_size(field, buffer.size(), NumPartFileTot, NDIM);
                        swap_block(buffer, size);
                        if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                            size_t index = index_start;
                            for (size_t i = 0; i < NumPartFileTot; i++) {
                                if (not keep[i])
                                    continue;
                                auto * vel = FML::PARTICLE::GetVel(part[index++]);
                                for (int idim = 0; idim < NDIM; idim++)
                                    vel[idim] = real_value(buffer, size, NDIM * i + idim) * vel_norm;
                            }
                        }
                    } else if (field == "ID") {
                        check_pos_is_read(field);
                        const size_t size = element_size(field, buffer.size(), NumPartFileTot, 1);
                        swap_block(buffer, size);
                        if constexpr (FML::PARTICLE::has_set_id<T>()) {
                            size_t index = index_start;
                            for (size_t i = 0; i < NumPartFileTot; i++) {
                                if (not keep[i])
                                    continue;
                                const long long int id =
                                    size == 4 ? (long long int)(reinterpret_cast<int32_t *>(buffer.data())[i]) :
                                                (long long int)(reinterpret_cast<int64_t *>(buffer.data())[i]);
                                FML::PARTICLE::SetID(part[index++], id);
                            }
                        }
                    } else if (field == "MASS") {
                        check_pos_is_read(field);
                        const size_t size = element_size(field, buffer.size(), NumPartWithMass, 1);
                        swap_block(buffer, size);
                        if constexpr (FML::PARTICLE::has_set_mass<T>()) {
                            size_t index = index_start;
                            size_t count = 0;
                            for (size_t i = 0; i < NumPartFileTot; i++) {
                                const bool has_mass = header.mass[type_of(i)] == 0.0;
                                const double mass = has_mass ? real_value(buffer, size, count++) : 0.0;
                                if (not keep[i])
                                    continue;
                                if (has_mass)
                                    FML::PARTICLE::SetMass(part[index], mass);
                                index++;
                            }
                        }
                    } else {
                        if (verbose)
                            std::cout << "Skipping block [" << field << "]\n";
                    }
                };

                if (format2) {
                    // The blocks are labeled: read all of them and use the ones we know
                    std::string label;
                    while (read_label(fp, label)) {
                        read_section(fp, buffer);
                        if (label == "MASS" and NumPartWithMass == 0)
                            continue;
                        process_block(label);
                    }
                } else {
                    auto fields = fields_in_file;
                    if (NumPartWithMass > 0 and std::find(fields.begin(), fields.end(), "MASS") == fields.end())
                        fields.push_back("MASS");
                    for (auto & field : fields) {
                        if (field == "MASS") {
                            if (NumPartWithMass == 0)
                                continue;
                            // Files without the MASS block (all masses in the MassTable equal to 0)
                            if (fp.peek() == std::ifstream::traits_type::eof())
                                break;
                        }
                        read_section(fp, buffer);
                        process_block(field);
                    }
                }
                if (not pos_is_read)
                    throw_error("[GadgetReader::read_gadget_single] No POS block in " + filename + "\n");
            }

            // For multiple species
            template <class T>
            void GadgetWriter::write_gadget_single(std::string filename,
                                                   T * part,
                                                   size_t NumPart,
                                                   [[maybe_unused]] size_t NumPartTot,
                                                   int NumberOfFilesToWrite,
                                                   double aexp,
                                                   double Boxsize,
                                                   double OmegaM,
                                                   double OmegaLambda,
                                                   double HubbleParam,
                                                   double pos_norm,
                                                   double vel_norm,
                                                   std::vector<double> OmegaFamilyOverOmegaM) {

                std::vector<char> buffer;
                float * float_buffer;

                // Make filename using
                std::ofstream fp(filename.c_str(), std::ios::binary | std::ios::out);
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetWrite::write_gadget_single] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }

                // The family (GADGET type) of a particle (-1 if it is not written)
                auto family_of = [&]([[maybe_unused]] size_t i) -> int {
#ifndef GADGET_ONLY_DM
                    if constexpr (FML::PARTICLE::has_get_family<T>()) {
                        const int family = int(FML::PARTICLE::GetFamily(part[i]));
                        return (family >= 0 and family < 6) ? family : -1;
                    }
#endif
                    return 1;
                };
                constexpr bool use_family =
#ifdef GADGET_ONLY_DM
                    false;
#else
                    FML::PARTICLE::has_get_family<T>();
#endif

                // Count how many of each type we have
                std::vector<size_t> npart_family(6, 0);
                std::vector<size_t> npart_family_tot(6, 0);
                if constexpr (use_family) {
                    // Multiple species
                    for (size_t i = 0; i < NumPart; i++) {
                        const int family = family_of(i);
                        if (family >= 0)
                            npart_family[family]++;
                    }
                } else {
                    // Only CDM
                    npart_family[1] = NumPart;
                    npart_family_tot[1] = NumPartTot;
                }
#ifdef GADGET_ONLY_DM
                OmegaFamilyOverOmegaM = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
#endif

                // If we only have CDM and OmegaCDM/OmegaM = 1 then we know the total number of particles
                // (this avoids MPI calls), otherwise sum over tasks
                if (use_family or OmegaFamilyOverOmegaM[1] != 1.0) {
                    npart_family_tot = npart_family;
                    FML::SumArrayOverTasks(npart_family_tot.data(), npart_family_tot.size());
                }

                // Take as input std::vector<double> OmegaFamilyOverOmegaM(6, 1.0);
                std::vector<double> mass_in_1e10_msunh(6, 0.0);
                for (int i = 0; i < 6; i++) {
                    mass_in_1e10_msunh[i] = npart_family_tot[i] == 0 ?
                                                0.0 :
                                                3.0 * OmegaM * OmegaFamilyOverOmegaM[i] * MplMpl_over_H0Msunh *
                                                    std::pow(Boxsize / HubbleLengthInMpch, 3) /
                                                    double(npart_family_tot[i]) / 1e10;
                }

                // Count how many particles to write
                size_t ntowrite = 0;
                for (int i = 0; i < 6; i++) {
                    ntowrite += npart_family[i];
                }
                if (ntowrite > size_t(UINT_MAX))
                    throw_error(
                        "[GadgetWriter::write_gadget_single] Too many particles for one file. Use more files\n");

                write_header_general(fp,
                                     npart_family,
                                     npart_family_tot,
                                     mass_in_1e10_msunh,
                                     NumberOfFilesToWrite,
                                     aexp,
                                     Boxsize,
                                     OmegaM,
                                     OmegaLambda,
                                     HubbleParam);

                // The indices of the particles we write sorted by family
                std::vector<size_t> order;
                order.reserve(ntowrite);
                if constexpr (use_family) {
                    for (int curfamily = 0; curfamily < 6; curfamily++)
                        if (npart_family[curfamily] > 0)
                            for (size_t j = 0; j < NumPart; j++)
                                if (family_of(j) == curfamily)
                                    order.push_back(j);
                } else {
                    for (size_t j = 0; j < NumPart; j++)
                        order.push_back(j);
                }

                // POS
                if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                    const size_t bytes = ntowrite * NDIM * sizeof(float);
                    buffer.resize(bytes);
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    for (size_t count = 0; count < ntowrite; count++) {
                        auto * pos = FML::PARTICLE::GetPos(part[order[count]]);
                        for (int idim = 0; idim < NDIM; idim++)
                            float_buffer[NDIM * count + idim] = float(pos[idim]) * pos_norm;
                    }
                    write_section(fp, buffer, bytes);
                }

                // VEL
                if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                    const size_t bytes = ntowrite * NDIM * sizeof(float);
                    buffer.resize(bytes);
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    for (size_t count = 0; count < ntowrite; count++) {
                        auto * vel = FML::PARTICLE::GetVel(part[order[count]]);
                        for (int idim = 0; idim < NDIM; idim++)
                            float_buffer[NDIM * count + idim] = float(vel[idim]) * vel_norm;
                    }
                    write_section(fp, buffer, bytes);
                }

                // ID (8 bytes if some ID does not fit in an int or if there are more than 2^31 particles)
                if constexpr (FML::PARTICLE::has_get_id<T>()) {
                    size_t ntot = 0;
                    for (auto n : npart_family_tot)
                        ntot += n;
                    bool long_ids = sizeof(gadget_particle_id_type) == 8 or ntot > size_t(INT_MAX) + 1;
                    for (size_t j = 0; j < NumPart and not long_ids; j++) {
                        const long long int id = (long long int)FML::PARTICLE::GetID(part[j]);
                        long_ids = id > INT_MAX or id < INT_MIN;
                    }
                    write_ids(fp, ntowrite, long_ids, [&](size_t count) {
                        return (long long int)FML::PARTICLE::GetID(part[order[count]]);
                    });
                }
            }

            inline void GadgetWriter::write_gadget_species(std::string filename,
                                                           std::vector<GadgetSpecies> species,
                                                           int NumberOfFilesToWrite,
                                                           double aexp,
                                                           double Boxsize,
                                                           double OmegaM,
                                                           double OmegaLambda,
                                                           double HubbleParam,
                                                           double pos_norm,
                                                           double vel_norm) {
                std::sort(species.begin(), species.end(), [](auto & a, auto & b) { return a.type < b.type; });
                std::vector<size_t> npart_type(6, 0), npart_type_tot(6, 0);
                std::vector<double> mass_in_1e10_msunh(6, 0.0);
                for (size_t i = 0; i < species.size(); i++) {
                    auto & sp = species[i];
                    if (sp.type < 0 or sp.type > 5 or (i > 0 and species[i - 1].type == sp.type))
                        throw_error(
                            "[GadgetWriter::write_gadget_species] The particle types must be different and in 0-5");
                    npart_type[sp.type] = sp.npart;
                    npart_type_tot[sp.type] = sp.npart_total;
                    mass_in_1e10_msunh[sp.type] = sp.npart_total == 0 ?
                                                      0.0 :
                                                      3.0 * OmegaM * sp.mass_fraction * MplMpl_over_H0Msunh *
                                                          std::pow(Boxsize / HubbleLengthInMpch, 3) /
                                                          double(sp.npart_total) / 1e10;
                }
                // Per-particle masses (MASS block) for the species with get_mass: their MassTable entry is 0
                std::vector<double> mean_mass_of_type = mass_in_1e10_msunh;
                for (auto & sp : species)
                    if (sp.get_mass)
                        mass_in_1e10_msunh[sp.type] = 0.0;

                size_t ntowrite = 0;
                for (auto & sp : species)
                    ntowrite += sp.npart;
                if (ntowrite > size_t(UINT_MAX))
                    throw_error(
                        "[GadgetWriter::write_gadget_species] Too many particles for one file. Use more files\n");

                std::ofstream fp(filename.c_str(), std::ios::binary | std::ios::out);
                if (not fp.is_open())
                    throw_error("[GadgetWriter::write_gadget_species] File " + filename + " is not open\n");
                write_header_general(fp,
                                     npart_type,
                                     npart_type_tot,
                                     mass_in_1e10_msunh,
                                     NumberOfFilesToWrite,
                                     aexp,
                                     Boxsize,
                                     OmegaM,
                                     OmegaLambda,
                                     HubbleParam);

                std::vector<char> buffer;
                auto all_have = [&](auto member) {
                    for (auto & sp : species)
                        if (sp.npart > 0 and not(sp.*member))
                            return false;
                    return true;
                };
                auto write_vector_block = [&](auto member, double norm) {
                    const size_t bytes = ntowrite * NDIM * sizeof(float);
                    buffer.resize(bytes);
                    float * float_buffer = reinterpret_cast<float *>(buffer.data());
                    std::vector<double> x(NDIM);
                    size_t count = 0;
                    for (auto & sp : species)
                        for (size_t j = 0; j < sp.npart; j++) {
                            (sp.*member)(j, x.data());
                            for (int idim = 0; idim < NDIM; idim++)
                                float_buffer[NDIM * count + idim] = float(x[idim] * norm);
                            count++;
                        }
                    write_section(fp, buffer, bytes);
                };
                if (all_have(&GadgetSpecies::get_pos))
                    write_vector_block(&GadgetSpecies::get_pos, pos_norm);
                if (all_have(&GadgetSpecies::get_vel))
                    write_vector_block(&GadgetSpecies::get_vel, vel_norm);
                if (all_have(&GadgetSpecies::get_id)) {
                    // 8 byte IDs if some ID does not fit in an int or if there are more than 2^31 particles
                    size_t ntot = 0;
                    for (auto & sp : species)
                        ntot += sp.npart_total;
                    bool long_ids = sizeof(gadget_particle_id_type) == 8 or ntot > size_t(INT_MAX) + 1;
                    std::vector<long long int> ids;
                    ids.reserve(ntowrite);
                    for (auto & sp : species)
                        for (size_t j = 0; j < sp.npart; j++) {
                            ids.push_back(sp.get_id(j));
                            long_ids = long_ids or ids.back() > INT_MAX or ids.back() < INT_MIN;
                        }
                    write_ids(fp, ntowrite, long_ids, [&](size_t count) { return ids[count]; });
                }
                size_t nmass = 0;
                for (auto & sp : species)
                    nmass += sp.get_mass ? sp.npart : 0;
                if (nmass > 0) {
                    const size_t bytes = sizeof(float) * nmass;
                    buffer.resize(bytes);
                    float * float_buffer = reinterpret_cast<float *>(buffer.data());
                    size_t count = 0;
                    for (auto & sp : species)
                        if (sp.get_mass)
                            for (size_t j = 0; j < sp.npart; j++)
                                float_buffer[count++] = float(mean_mass_of_type[sp.type] * sp.get_mass(j));
                    write_section(fp, buffer, bytes);
                }
            }

        } // namespace GADGET
    } // namespace FILEUTILS
} // namespace FML

#endif

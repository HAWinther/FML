#ifndef GADGETUTILS_HEADER
#define GADGETUTILS_HEADER

#include <cassert>
#include <climits>
#include <cmath>
#include <fstream>
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
// Errors are handled via throw_error in the class below (with MPI it aborts and otherwise
// throws a runtime error)
//
// Compile time defines:
// GADGET_LONG_INT_IDS  : Use long long int for IDs otherwise use int
//
//====================================================================================

namespace FML {
    namespace FILEUTILS {

        /// Reading and writing GADGET files (DM only).
        namespace GADGET {

            // This is Mpl^2/(H0 Msun/h) used to set the mass of particles
            extern const double MplMpl_over_H0Msunh;
            // This is 1/H0 in units of Mpc/h used to set the mass of particles
            extern const double HubbleLengthInMpch;

            // The ID type we use
#ifdef GADGET_LONG_INT_IDS
            using gadget_particle_id_type = long long int;
#else
            using gadget_particle_id_type = int;
#endif

            /// The GADGET header format
            // Do not change the order of the fields below as this is read as one piece of memory from file
            typedef struct {
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
                GadgetHeader header;

                bool endian_swap{false};
                bool header_is_read{false};

                // The dimensions of the positions and velocities in the files.
                int NDIM{3};

                // The fields we assume is in the file
                std::vector<std::string> fields_in_file = {"POS", "VEL", "ID"};

                void throw_error(std::string errormessage) const;
                void set_endian_swap();

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

                /// Read a section of a gadget file
                void read_section(std::ifstream & fp, std::vector<char> & buffer);

                /// Read the gadget header
                void read_header(std::ifstream & fp);

                /// Get the header (assumes it has been read)
                GadgetHeader get_header();

                /// Get the number of gadget files
                int get_num_files(std::string filename = "");

                /// If non-standard file, set the fields that are in the file (only POS,VEL,ID implmented)
                void set_fields_in_file(std::vector<std::string> fields);
            };

            /// Write files in GADGET format
            class GadgetWriter {
              private:
                GadgetHeader header;

                int NDIM{3};

                void throw_error(std::string errormessage) const;

              public:
                GadgetWriter() = default;
                GadgetWriter(int ndim);

                /// Write a single gadget file with multiple species. pos_norm is to convert from user units to
                /// positions in [0, box) vel_norm is to convert from user units to sqrt(a) dxdt in units of km/s
                /// OmegaFamilyOverOmegaM is used to set the mass (the mass is m ~ OmegaM  * Box^3/Npart *
                /// OmegaFamilyOverOmegaM). If you only have CDM particles in the usual spot then
                /// this array is just (0,1,0,0,0,0). If you have baryons in 0 with OmegaB=0.05 and cdm in 1 with
                /// OmegaCDM = 0.25 then this is (1/6,5/6,0,0,0,0)
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

                /// Write a gadget section
                void write_section(std::ofstream & fp, std::vector<char> & buffer, int bytes);

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
            void swap_endian_vector(T * vec, int n) {
                for (int i = 0; i < n; i++) {
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
                // (but will resize if that is too litte)
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

                std::vector<char> buffer;
                float * float_buffer;
                gadget_particle_id_type * id_buffer;

                // Open file and get the number of bytes
                std::ifstream fp(filename.c_str(), std::ios::binary);
                size_t bytes_in_file;
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_gadget_single] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }
                fp.seekg(0, fp.end);
                bytes_in_file = fp.tellg();
                fp.seekg(0, fp.beg);
                if (bytes_in_file <= 256) {
                    std::string errormessage =
                        "[GadgetReader::read_gadget_single] The file contains less than just a header!\n";
                    throw_error(errormessage);
                }

                // Warning if the file is too large
                if(size_t(int(bytes_in_file)) != bytes_in_file){
                  std::cout << "Warning: The file is huge! Sections might have too much data that cannot be stored in an int\n";
                }

                // Read header
                read_header(fp);
                if (verbose)
                    print_header_info(header);

                // Positions normalized by the boxsize in the file
                const double pos_norm = 1.0 / header.BoxSize;

                // Velocities normalized to peculiar in km/s
                const double vel_norm = sqrt(header.time);

                // Compute how many bytes per particle
                unsigned int NumPartDM = header.npart[1];
                unsigned int NumPartFileTot = 0;
                for (int i = 0; i < 6; i++) {
                    NumPartFileTot += header.npart[i];
                }

#ifndef GADGET_ONLY_DM
                if constexpr (FML::PARTICLE::has_set_family<T>() == false) {
                    if (NumPartFileTot != NumPartDM) {
                        std::string errormessage = "[GadgetReader::read_gadget_single] Particle type does not have "
                                                   "set_family, but Gadget file contains multiple species! Either use "
                                                   "define GADGET_ONLY_DM or add methods to particle\n";
                        throw_error(errormessage);
                    }
                }
#endif

                const int bytes_per_particle = (bytes_in_file / NumPartFileTot);

                // Expected bytes if file based on the fields
                int bytes_per_particle_expected = 0;
                for (auto & field : fields_in_file) {
                    if (field == "POS")
                        bytes_per_particle_expected += NDIM * sizeof(float);
                    else if (field == "VEL")
                        bytes_per_particle_expected += NDIM * sizeof(float);
                    else if (field == "ID")
                        bytes_per_particle_expected += sizeof(gadget_particle_id_type);
                    else {
                        std::cout << "Warning: unknown field in file [" << field
                                  << "]. Only POS, VEL and ID are read and implemented\n";
                        assert(false);
                    }
                }
                if (bytes_per_particle != bytes_per_particle_expected) {
                    std::string errormessage =
                        "[GadgetReader::read_gadget_single] BytesPerParticle = " + std::to_string(bytes_per_particle) +
                        " != ExpectedBytes =" + std::to_string(bytes_per_particle_expected) +
                        ". Change the ID size in GadgetUtils? Otherwise check that "
                        "fields_in_file in this file is correct!\n";
                    throw_error(errormessage);
                }

                // Check if the vector has enough capacity to store the elements if not
                // reallcate it. We add the particles to the back of the part array
#ifdef GADGET_ONLY_DM
                if (part.capacity() < part.size() + NumPartDM)
                    part.reserve(part.size() + NumPartDM);
#else
                if (part.capacity() < part.size() + NumPartFileTot)
                    part.reserve(part.size() + NumPartFileTot);
#endif

                // Allocate buffer
                size_t bytes = NDIM * sizeof(float) * NumPartFileTot;
                buffer = std::vector<char>(bytes);
                float_buffer = reinterpret_cast<float *>(buffer.data());
                id_buffer = reinterpret_cast<gadget_particle_id_type *>(buffer.data());

                // Allocate temp storage for knowing if a particles is in the domain or not
                std::vector<char> is_in_domain;
                if (only_keep_part_in_domain) {
                    is_in_domain = std::vector<char>(NumPartFileTot, 1);
                    // We need to read POS first to know if a particle is in the domain
                    FML::assert_mpi(fields_in_file[0] == "POS", "Error: Position has to be first in file if only_keep_part_in_domain is set");
                }

                // Read the file
                size_t index_start = part.size();
                for (auto & field : fields_in_file) {

                    if (field == "POS") {
                        // Read particle positions and assign to particles
                        bytes = sizeof(float) * NDIM * NumPartFileTot;
                        if (verbose)
                            std::cout << "Reading POS bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(float) * NDIM << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        size_t index = index_start;
                        for (unsigned int i = 0; i < NumPartFileTot; i++) {
                            auto x = float_buffer[NDIM * i] * pos_norm;
                            if (x >= 1.0)
                                x -= 1.0;
                            if (x < 0.0)
                                x += 1.0;
                            assert_mpi(x >= 0.0 and x < 1.0,
                                       "[read_gadet_single] Particle has x position outside the boxsize even after "
                                       "periodic wrap");
                            if (only_keep_part_in_domain) {
                                if (not(x >= FML::xmin_domain and x < FML::xmax_domain)) {
                                    is_in_domain[i] = 0;
                                    continue;
                                }
                            }

#ifdef GADGET_ONLY_DM
                            if (i < header.npart[0])
                                continue;
                            if (i >= header.npart[0] + header.npart[1])
                                continue;
#endif
                            part.push_back(T{});

                            // Check if positions exists in Particle and assign
                            if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                                auto * pos = FML::PARTICLE::GetPos(part[index++]);
                                for (int idim = 0; idim < NDIM; idim++) {
                                    pos[idim] = float_buffer[NDIM * i + idim] * pos_norm;
                                    if (endian_swap)
                                        pos[idim] = swap_endian(pos[idim]);
                                    if (pos[idim] >= 1.0)
                                        pos[idim] -= 1.0;
                                    if (pos[idim] < 0.0)
                                        pos[idim] += 1.0;
                                }
                            }
                        }

#ifndef GADGET_ONLY_DM
                        // Set the family. Not needed if we only read DM
                        if constexpr (FML::PARTICLE::has_set_family<T>()) {
                            index = index_start;
                            for (int type = 0; type < 6; type++) {
                                for (unsigned int i = 0; i < header.npart[type]; i++) {
                                    FML::PARTICLE::SetFamily(part[index + i], type);
                                }
                                index += header.npart[type];
                            }
                        }
#endif

                    } else if (field == "VEL") {
                        // Read particle velocities and assign to particles
                        bytes = sizeof(float) * NDIM * NumPartFileTot;
                        if (verbose)
                            std::cout << "Reading VEL bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(float) * NDIM << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        // Check if velocities exists in Particle
                        if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                            size_t index = index_start;
                            for (unsigned int i = 0; i < NumPartFileTot; i++) {
                                if (only_keep_part_in_domain) {
                                    if (is_in_domain[i] == 0)
                                        continue;
                                }
#ifdef GADGET_ONLY_DM
                                if (i < header.npart[0])
                                    continue;
                                if (i >= header.npart[0] + header.npart[1])
                                    continue;
#endif
                                auto * vel = FML::PARTICLE::GetVel(part[index++]);
                                for (int idim = 0; idim < NDIM; idim++) {
                                    vel[idim] = float_buffer[NDIM * i + idim] * vel_norm;
                                    if (endian_swap)
                                        vel[idim] = swap_endian(vel[idim]);
                                }
                            }
                        }
                    } else if (field == "ID") {
                        // Read particle IDs (if they exist) and assign to particles
                        bytes = sizeof(gadget_particle_id_type) * NumPartFileTot;
                        if (verbose)
                            std::cout << "Reading ID bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(gadget_particle_id_type) << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        // Check if particle has ID
                        if constexpr (FML::PARTICLE::has_set_id<T>()) {
                            size_t index = index_start;
                            for (unsigned int i = 0; i < NumPartFileTot; i++) {
                                if (only_keep_part_in_domain) {
                                    if (is_in_domain[i] == 0)
                                        continue;
                                }
#ifdef GADGET_ONLY_DM
                                if (i < header.npart[0])
                                    continue;
                                if (i >= header.npart[0] + header.npart[1])
                                    continue;
#endif
                                if (endian_swap) {
                                    FML::PARTICLE::SetID(part[index++], swap_endian(id_buffer[i]));
                                } else {
                                    FML::PARTICLE::SetID(part[index++], id_buffer[i]);
                                }
                            }
                        }
                    }
                }
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
                gadget_particle_id_type * id_buffer;

                // Make filename using
                std::ofstream fp(filename.c_str(), std::ios::binary | std::ios::out);
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetWrite::write_gadget_single] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }

                // Count how many of each type we have
                std::vector<size_t> npart_family(6, 0);
#ifdef GADGET_ONLY_READ_DM
                npart_family[1] = NumPart;
                OmegaFamilyOverOmegaM = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
#else
                if constexpr (FML::PARTICLE::has_get_family<T>()) {
                    for (size_t i = 0; i < NumPart; i++) {
                        auto family = FML::PARTICLE::GetFamily(part[i]);
                        if (family >= 0 and family < 6)
                            npart_family[family]++;
                    }
                } else {
                    npart_family[1] = NumPart;
                }
#endif

                std::vector<size_t> npart_family_tot = npart_family;
                FML::SumArrayOverTasks(npart_family_tot.data(), npart_family_tot.size());

                // Take as input std::vector<double> OmegaFamilyOverOmegaM(6, 1.0);
                std::vector<double> mass_in_1e10_msunh(6, 0.0);
                for (int i = 0; i < 6; i++) {
                    mass_in_1e10_msunh[i] = npart_family_tot[i] == 0 ? 0.0 : 3.0 * OmegaM * OmegaFamilyOverOmegaM[i] * MplMpl_over_H0Msunh *
                                            std::pow(Boxsize / HubbleLengthInMpch, 3) / double(npart_family_tot[1]) /
                                            1e10;
                }

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

                // Count how many particles to write
                unsigned int ntowrite = 0;
                for (int i = 0; i < 6; i++) {
                    ntowrite += npart_family[i];
                }
                unsigned int bytes = ntowrite * NDIM * sizeof(float);
                buffer = std::vector<char>(bytes);

                // If particles have family then sort POS of them by family in the buffer
                if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                    bytes = ntowrite * NDIM * sizeof(float);
                    buffer.resize(bytes);
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    size_t count = 0;
                    for (int curfamily = 0; curfamily < 6; curfamily++) {
                        if (npart_family[curfamily] > 0) {
                            for (size_t j = 0; j < ntowrite; j++) {
                                auto * pos = FML::PARTICLE::GetPos(part[j]);
                                if constexpr (FML::PARTICLE::has_get_family<T>()) {
                                    auto family = FML::PARTICLE::GetFamily(part[j]);
                                    if (family == curfamily) {
                                        for (int idim = 0; idim < NDIM; idim++)
                                            float_buffer[NDIM * count + idim] = float(pos[idim]) * pos_norm;
                                        count++;
                                    }
                                } else {
                                    for (int idim = 0; idim < NDIM; idim++)
                                        float_buffer[NDIM * count + idim] = float(pos[idim]) * pos_norm;
                                    count++;
                                }
                            }
                        }
                    }
                    assert(count == ntowrite);
                    write_section(fp, buffer, bytes);
                }

                // If particles have family then sort VEL of them by family in the buffer
                if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                    bytes = ntowrite * NDIM * sizeof(float);
                    buffer.resize(bytes);
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    size_t count = 0;
                    for (int curfamily = 0; curfamily < 6; curfamily++) {
                        if (npart_family[curfamily] > 0) {
                            for (size_t j = 0; j < ntowrite; j++) {
                                if constexpr (FML::PARTICLE::has_get_family<T>()) {
                                    auto family = FML::PARTICLE::GetFamily(part[j]);
                                    if (family == curfamily) {
                                        auto * vel = FML::PARTICLE::GetVel(part[j]);
                                        for (int idim = 0; idim < NDIM; idim++)
                                            float_buffer[NDIM * count + idim] = float(vel[idim]) * vel_norm;
                                        count++;
                                    }
                                } else {
                                    auto * vel = FML::PARTICLE::GetVel(part[j]);
                                    for (int idim = 0; idim < NDIM; idim++)
                                        float_buffer[NDIM * count + idim] = float(vel[idim]) * vel_norm;
                                    count++;
                                }
                            }
                        }
                    }
                    assert(count == ntowrite);
                    write_section(fp, buffer, bytes);
                }

                // If particles have family then sort ID of them by family in the buffer
                if constexpr (FML::PARTICLE::has_get_id<T>()) {
                    bytes = sizeof(gadget_particle_id_type) * ntowrite;
                    buffer.resize(bytes);
                    id_buffer = reinterpret_cast<gadget_particle_id_type *>(buffer.data());
                    size_t count = 0;
                    for (int curfamily = 0; curfamily < 6; curfamily++) {
                        if (npart_family[curfamily] > 0) {
                            for (size_t j = 0; j < ntowrite; j++) {
                                if constexpr (FML::PARTICLE::has_get_family<T>()) {
                                    auto family = FML::PARTICLE::GetFamily(part[j]);
                                    if (family == curfamily) {
                                        auto id = FML::PARTICLE::GetID(part[j]);
                                        id_buffer[count] = id;
                                        count++;
                                    }
                                } else {
                                    auto id = FML::PARTICLE::GetID(part[j]);
                                    id_buffer[count] = id;
                                    count++;
                                }
                            }
                        }
                    }
                    assert(count == ntowrite);
                    write_section(fp, buffer, bytes);
                }
            }

        } // namespace GADGET
    }     // namespace FILEUTILS
} // namespace FML

#endif

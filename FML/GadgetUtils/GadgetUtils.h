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
// Read Gadget files for DM particles. Checks and swaps endian if needed.
//
// The particle class can be anything, but needs to have the methods
// auto *get_pos()
// auto *get_vel()
// idtype get_id()
// void set_pos(floattype *)
// void set_vel(floattype *)
// void set_id(idtype)
// together with get_ndim()
//
// We assume positions in the Particle class are in [0,1]. These are scaled to
// [0,Boxsize] when writing to file. When we read we return positions in [0,1).
// If GADGETUTILS_NO_BOXSIZE_SCALING then we write and read the positions are provided
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

            /// The GADGET1 header format
            // Do not change the order of the fields below as this is read as one piece of memory from file
            typedef struct {
                unsigned int
                    npart[6];   // npart[1] gives the number of particles in the file, other particle types are ignored
                double mass[6]; // mass[1] gives the particle mass
                double time{0.0};     // Cosmological scale factor of snapshot
                double redshift{0.0}; // Redshift of snapshot
                int flag_sfr{0};      // Flags whether star formation is used
                int flag_feedback{0}; // Flags whether feedback from star formation is included
                unsigned int
                    npartTotal[6];   // If npartTotal[1] > 2^32, then total is this plus 2^32 * npartTotalHighWord[1]
                int flag_cooling{0}; // Flags whether radiative cooling is included
                int num_files{0};    // The number of files that are used for a snapshot
                double BoxSize{0.0}; // Simulation box size (in code units)
                double Omega0{0.0};  // Matter density parameter
                double OmegaLambda{0.0};            // Lambda density parameter
                double HubbleParam{0.0};            // Hubble Parameter
                int flag_stellarage{0};             // flags whether the age of newly formed stars is recorded and saved
                int flag_metals{0};                 // flags whether metal enrichment is included
                unsigned int npartTotalHighWord[6]; // High word of the total number of particles of each type
                int flag_entropy_instead_u{0};      // Flags that IC-file contains entropy instead of u
                char
                    fill[256 - sizeof(unsigned int) * 18 - sizeof(int) * 7 - sizeof(double) * 12]; // Fills to 256 Bytes
            } GadgetHeader;
            static_assert(sizeof(GadgetHeader) == 256);

            void print_header_info(GadgetHeader & header);

            /// Class for reading Gadget files. Checks and corrects for different endian-ness.
            class GadgetReader {
              private:
                GadgetHeader header;

                bool endian_swap{false};
                bool header_is_read{false};

                /// Mpc/h / Units_in_file, i.e. if the positions are in Mpc/h its 1 if kpc/h then 1000
                double gadget_pos_factor{1.0};

                /// The dimensions of the positions and velocities in the files.
                int NDIM{3};

                // The fields we assume is in the file
                std::vector<std::string> fields_in_file = {"POS", "VEL", "ID"};

                void throw_error(std::string errormessage) const;

              public:
                GadgetReader() = default;
                GadgetReader(double pos_factor = 1.0, int ndim = 3);

                template <class T>
                void read_gadget_single(std::string filename, std::vector<T> & part, bool verbose);
                template <class T>
                void read_gadget(std::string filename, std::vector<T> & part, bool verbose);
                void read_section(std::ifstream & fp, std::vector<char> & buffer);
                void read_header(std::ifstream & fp);

                GadgetHeader get_header();

                void set_endian_swap();
                int get_num_files(std::string filename = "");

                void set_fields_in_file(std::vector<std::string> fields);
            };

            /// Write files in GADGET format
            class GadgetWriter {
              private:
                GadgetHeader header;

                int NDIM{3};

                void throw_error(std::string errormessage) const;

              public:
                GadgetWriter(int ndim = 3);

                // pos_norm transforms from pos-units of particles to Mpc/h
                template <class T>
                void write_gadget_single(std::string filename,
                                         std::vector<T> & part,
                                         size_t TotNumPart,
                                         int NumberOfFilesToWrite,
                                         double aexp,
                                         double Boxsize,
                                         double OmegaM,
                                         double OmegaLambda,
                                         double HubbleParam,
                                         double pos_norm);
                void write_section(std::ofstream & fp, std::vector<char> & buffer, int bytes);
                void write_header(std::ofstream & fp,
                                  unsigned int NumPart,
                                  size_t TotNumPart,
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

            template <class T>
            void GadgetReader::read_gadget(std::string fileprefix, std::vector<T> & part, bool verbose) {

                // Read the number of particles and the number of files
                std::string filename = fileprefix + ".0";
                std::ifstream fp(filename.c_str(), std::ios::binary);
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_gadget_all] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }
                read_header(fp);
                fp.close();

                // Number of files
                const int nfiles = header.num_files;

                // Allocate memory for particle vector and reset it
                const size_t npartTotal = (size_t(header.npartTotalHighWord[1]) << 32) + size_t(header.npartTotal[1]);
                part = std::vector<T>();
                part.reserve(npartTotal);

                // Read all the files
                for (int i = 0; i < nfiles; i++) {
                    filename = fileprefix + "." + std::to_string(i);
                    if (verbose)
                        std::cout << "Reading file " << filename << "\n";
                    read_gadget_single(filename, part, verbose);
                }
            }

            template <class T>
            void GadgetReader::read_gadget_single(std::string filename, std::vector<T> & part, bool verbose) {
                std::vector<char> buffer;
                float * float_buffer;
                gadget_particle_id_type * id_buffer;

                // Open file and get the number of bytes
                std::ifstream fp(filename.c_str(), std::ios::binary);
                int bytes_in_file;
                if (!fp.is_open()) {
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

                // Read header
                read_header(fp);
                if (verbose)
                    print_header_info(header);

                // Positions normalized by the boxsize in the file
                const double pos_norm = 1.0 / header.BoxSize;

                // Velocities normalized to peculiar in km/s
                const double vel_norm = sqrt(header.time);

                // Boxsize might need to be scaled to Mpc/h
                header.BoxSize /= gadget_pos_factor;

                // Compute how many bytes per particle
                const int bytes_per_particle = (bytes_in_file / header.npart[1]);

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

                // Number of particles in the current file
                const int NumPart = header.npart[1];

                // Allocate particles. Add to the back of the part array
                int istart = part.size();
                part.resize(part.size() + header.npart[1]);

                // Allocate buffer
                size_t bytes = NDIM * sizeof(float) * NumPart;
                buffer = std::vector<char>(bytes);
                float_buffer = reinterpret_cast<float *>(buffer.data());
                id_buffer = reinterpret_cast<gadget_particle_id_type *>(buffer.data());

                // Read the file
                for (auto & field : fields_in_file) {

                    if (field == "POS") {
                        // Read particle positions and assign to particles
                        bytes = sizeof(float) * NDIM * NumPart;
                        if (verbose)
                            std::cout << "Reading POS bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(float) * NDIM << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        // Check if positions exists in Particle
                        if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                            for (int i = 0; i < NumPart; i++) {
                                auto * pos = FML::PARTICLE::GetPos(part[istart + i]);
                                for (int idim = 0; idim < NDIM; idim++) {
                                    pos[idim] = float_buffer[NDIM * i + idim] * pos_norm;
                                    if (endian_swap)
                                        pos[idim] = swap_endian(pos[idim]);
                                }
                            }
                        }
                    } else if (field == "VEL") {
                        // Read particle velocities and assign to particles
                        bytes = sizeof(float) * NDIM * NumPart;
                        if (verbose)
                            std::cout << "Reading VEL bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(float) * NDIM << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        // Check if velocities exists in Particle
                        if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                            for (int i = 0; i < NumPart; i++) {
                                auto * vel = FML::PARTICLE::GetVel(part[istart + i]);
                                for (int idim = 0; idim < NDIM; idim++) {
                                    vel[idim] = float_buffer[NDIM * i + idim] * vel_norm;
                                    if (endian_swap)
                                        vel[idim] = swap_endian(vel[idim]);
                                }
                            }
                        }
                    } else if (field == "ID") {
                        // Read particle IDs (if they exist) and assign to particles
                        bytes = sizeof(gadget_particle_id_type) * NumPart;
                        if (verbose)
                            std::cout << "Reading ID bytes = " << bytes
                                      << " BytesPerParticle: " << sizeof(gadget_particle_id_type) << "\n";
                        buffer.resize(bytes);
                        read_section(fp, buffer);

                        // Check if particle has ID
                        if constexpr (FML::PARTICLE::has_set_id<T>()) {
                            for (int i = 0; i < NumPart; i++) {
                                if (endian_swap) {
                                    FML::PARTICLE::SetID(part[istart + i], swap_endian(id_buffer[i]));
                                } else {
                                    FML::PARTICLE::SetID(part[istart + i], id_buffer[i]);
                                }
                            }
                        }
                    }
                }
            }

            template <class T>
            void GadgetWriter::write_gadget_single(std::string filename,
                                                   std::vector<T> & part,
                                                   size_t TotNumPart,
                                                   int NumberOfFilesToWrite,
                                                   double aexp,
                                                   double Boxsize,
                                                   double OmegaM,
                                                   double OmegaLambda,
                                                   double HubbleParam,
                                                   double pos_norm) {

                std::vector<char> buffer;
                float * float_buffer;
                gadget_particle_id_type * id_buffer;
                unsigned int NumPart = part.size();

                // If we don't have particles to write return
                if (NumPart == 0)
                    return;

                // Make filename using
                std::ofstream fp(filename.c_str(), std::ios::binary | std::ios::out);
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetWrite::write_gadget_single] File " + filename + " is not open\n";
                    throw_error(errormessage);
                }

                // Write header
                write_header(
                    fp, NumPart, TotNumPart, NumberOfFilesToWrite, aexp, Boxsize, OmegaM, OmegaLambda, HubbleParam);

                // Gather particle positions and write
                unsigned int bytes = NDIM * sizeof(float) * NumPart;
                buffer = std::vector<char>(bytes);
                if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    for (unsigned int i = 0; i < NumPart; i++) {
                        auto * pos = FML::PARTICLE::GetPos(part[i]);
                        for (int idim = 0; idim < NDIM; idim++)
                            float_buffer[NDIM * i + idim] = float(pos[idim]) * Boxsize * pos_norm;
                    }
                    write_section(fp, buffer, bytes);
                }

                // Gather particle velocities and write
                if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                    float_buffer = reinterpret_cast<float *>(buffer.data());
                    for (unsigned int i = 0; i < NumPart; i++) {
                        auto * vel = FML::PARTICLE::GetVel(part[i]);
                        for (int idim = 0; idim < NDIM; idim++)
                            float_buffer[NDIM * i + idim] = float(vel[idim]);
                    }
                    write_section(fp, buffer, bytes);
                }

                // Gather particle IDs and write
                if constexpr (FML::PARTICLE::has_get_id<T>()) {
                    bytes = sizeof(gadget_particle_id_type) * NumPart;
                    buffer.resize(bytes);
                    id_buffer = reinterpret_cast<gadget_particle_id_type *>(buffer.data());
                    for (unsigned int i = 0; i < NumPart; i++) {
                        id_buffer[i] = (gadget_particle_id_type)*FML::PARTICLE::GetID(part[i]);
                    }
                    write_section(fp, buffer, bytes);
                }
            }

        } // namespace GADGET
    }     // namespace FILEUTILS
} // namespace FML

#endif

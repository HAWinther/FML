#include "GadgetUtils.h"

namespace FML {
    namespace FILEUTILS {
        namespace GADGET {

            const double MplMpl_over_H0Msunh = 2.49264e21;
            const double HubbleLengthInMpch = 2997.92458;

            // Handle errors
            void GadgetReader::throw_error(std::string errormessage) const {
#ifdef USE_MPI
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

            // Handle errors
            void GadgetWriter::throw_error(std::string errormessage) const {
#ifdef USE_MPI
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

            void print_header_info(GadgetHeader & header) {
                std::cout << "\n";
                std::cout << "GadgetHeader:\n";
                std::cout << "aexp        " << header.time << "\n";
                std::cout << "Redshift    " << header.redshift << "\n";
                std::cout << "Boxsize     " << header.BoxSize << "\n";
                std::cout << "Omega0      " << header.Omega0 << "\n";
                std::cout << "OmegaLambda " << header.OmegaLambda << "\n";
                std::cout << "HubbleParam " << header.HubbleParam << "\n";
                std::cout << "numFiles    " << header.num_files << "\n";
                std::cout << "npart       " << header.npart[1] << "\n";
                std::cout << "npartTotal  " << (size_t(header.npartTotalHighWord[1]) << 32) + header.npartTotal[1]
                          << "\n";
                std::cout << "\n";
            }

            //==============================================================================================
            //==============================================================================================

            void GadgetReader::set_fields_in_file(std::vector<std::string> fields) { fields_in_file = fields; }

            GadgetReader::GadgetReader(int ndim) : NDIM(ndim) {}

            GadgetHeader GadgetReader::get_header() { return header; }

            void GadgetReader::read_section(std::ifstream & fp, std::vector<char> & buffer) {
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_section] File is not open\n";
                    throw_error(errormessage);
                }
                int bytes_start, bytes_end;
                fp.read((char *)&bytes_start, sizeof(bytes_start));
                if (endian_swap)
                    bytes_start = swap_endian(bytes_start);
                if (buffer.size() > 0) {
                    if (buffer.size() < size_t(bytes_start)) {
                        std::string errormessage = "[GadgetReader::read_section] Buffersize is too small\n";
                        throw_error(errormessage);
                    }
                } else {
                    buffer = std::vector<char>(bytes_start);
                }
                fp.read(buffer.data(), bytes_start);
                fp.read((char *)&bytes_end, sizeof(bytes_end));
                if (endian_swap)
                    bytes_end = swap_endian(bytes_end);
                if (bytes_start != bytes_end) {
                    std::string errormessage = "[GadgetReader::read_section] Error in file BytesStart != ByteEnd!\n";
                    throw_error(errormessage);
                }
            }

            void GadgetReader::read_header(std::ifstream & fp) {
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_header] File is not open\n";
                    throw_error(errormessage);
                }
                int bytes_start, bytes_end;
                fp.read((char *)&bytes_start, sizeof(bytes_start));
                fp.read((char *)&header, bytes_start);
                fp.read((char *)&bytes_end, sizeof(bytes_end));
                if (bytes_start != bytes_end) {
                    std::string errormessage = "[GadgetReader::read_section] Error in file BytesStart != ByteEnd!\n";
                    throw_error(errormessage);
                }

                // Check if endian of file needs to be changed
                // Swap endian of header and set flag so that we
                // swap the endian of the other fields also
                if (bytes_start != sizeof(header)) {
                    bytes_start = swap_endian(bytes_start);
                    bytes_end = swap_endian(bytes_end);
                    if (bytes_start != sizeof(header)) {
                        std::string errormessage =
                            "[GadgetReader::read_section] Error in file. ByteStart != sizeof(header)\n";
                        throw_error(errormessage);
                    }
                    swap_endian_vector(header.npart, 6);
                    swap_endian_vector(header.mass, 6);
                    swap_endian_vector(header.npartTotal, 6);
                    swap_endian_vector(header.npartTotalHighWord, 6);
                    header.time = swap_endian(header.time);
                    header.redshift = swap_endian(header.redshift);
                    header.flag_sfr = swap_endian(header.flag_sfr);
                    header.flag_feedback = swap_endian(header.flag_feedback);
                    header.flag_cooling = swap_endian(header.flag_cooling);
                    header.num_files = swap_endian(header.num_files);
                    header.BoxSize = swap_endian(header.BoxSize);
                    header.Omega0 = swap_endian(header.Omega0);
                    header.OmegaLambda = swap_endian(header.OmegaLambda);
                    header.HubbleParam = swap_endian(header.HubbleParam);
                    header.flag_stellarage = swap_endian(header.flag_stellarage);
                    header.flag_metals = swap_endian(header.flag_metals);
                    header.flag_entropy_instead_u = swap_endian(header.flag_entropy_instead_u);
                    endian_swap = true;
                }

                header_is_read = true;
            }

            //==============================================================================================
            //==============================================================================================

            GadgetWriter::GadgetWriter(int ndim) : NDIM(ndim) {}

            void GadgetWriter::write_section(std::ofstream & fp, std::vector<char> & buffer, int bytes) {
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetWriter::write_section] File is not open\n";
                    throw_error(errormessage);
                }
                if (buffer.size() < size_t(bytes)) {
                    std::string errormessage = "[GadgetWriter::write_section] Buffersize too small\n";
                    throw_error(errormessage);
                }
                fp.write((char *)&bytes, sizeof(bytes));
                fp.write(buffer.data(), bytes);
                fp.write((char *)&bytes, sizeof(bytes));
            }

            void GadgetWriter::write_header(std::ofstream & fp,
                                            unsigned int NumPart,
                                            size_t NumPartTot,
                                            int NumberOfFilesToWrite,
                                            double aexp,
                                            double Boxsize,
                                            double OmegaM,
                                            double OmegaLambda,
                                            double HubbleParam) {
                if (!fp.is_open()) {
                    std::string errormessage = "[GadgetWriter::write_header] File is not open\n";
                    throw_error(errormessage);
                }

                // Mass in 10^10 Msun/h. Assumptions: Boxsize in Mpc/h
                for (int i = 0; i < 6; i++) {
                    header.npart[i] = 0;
                    header.npartTotal[i] = 0;
                    header.npartTotalHighWord[i] = 0;
                    header.mass[i] = 0.0;
                }
                header.npart[1] = NumPart;
                header.npartTotal[1] = (unsigned int)NumPartTot;
                header.npartTotalHighWord[1] = (unsigned int)(NumPartTot >> 32);
                header.mass[1] = 3.0 * OmegaM * MplMpl_over_H0Msunh * std::pow(Boxsize / HubbleLengthInMpch, 3) /
                                 double(NumPartTot) / 1e10;
                header.time = aexp;
                header.redshift = 1.0 / aexp - 1.0;
                header.flag_sfr = 0;
                header.flag_feedback = 0;
                header.flag_cooling = 0;
                header.flag_stellarage = 0;
                header.flag_metals = 0;
                header.flag_stellarage = 0;
                header.flag_metals = 0;
                header.flag_entropy_instead_u = 0;
                header.num_files = NumberOfFilesToWrite;
                header.BoxSize = Boxsize;
                header.Omega0 = OmegaM;
                header.OmegaLambda = OmegaLambda;
                header.HubbleParam = HubbleParam;

                int bytes = sizeof(header);
                fp.write((char *)&bytes, sizeof(bytes));
                fp.write((char *)&header, bytes);
                fp.write((char *)&bytes, sizeof(bytes));
            }

            void GadgetReader::set_endian_swap() { endian_swap = true; }

            int GadgetReader::get_num_files(std::string filename) {
                if (!header_is_read) {
                    std::ifstream fp(filename.c_str(), std::ios::binary);
                    if (!fp.is_open()) {
                        std::string errormessage = "[GadgetReader::get_num_files] File " + filename + " is not open\n";
                        throw_error(errormessage);
                    }
                    read_header(fp);
                }
                return header.num_files;
            }
        } // namespace GADGET
    }     // namespace FILEUTILS
} // namespace FML

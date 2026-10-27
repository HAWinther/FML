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

            void print_header_info(const GadgetHeader & header) {
                std::cout << "\n";
                std::cout << "GadgetHeader:\n";
                std::cout << "aexp          " << header.time << "\n";
                std::cout << "Redshift      " << header.redshift << "\n";
                std::cout << "Boxsize       " << header.BoxSize << " Mpc/h\n";
                std::cout << "Omega0        " << header.Omega0 << "\n";
                std::cout << "OmegaLambda   " << header.OmegaLambda << "\n";
                std::cout << "HubbleParam   " << header.HubbleParam << "\n";
                std::cout << "numFiles      " << header.num_files << "\n";
                for(int i = 0; i < 6; i++){
                  std::cout << "npart[" << i << "]      " << header.npart[i] << "\n";
                }
                for(int i = 0; i < 6; i++){
                  std::cout << "mass[" << i << "]       " << header.mass[i] << "\n";
                }
                for(int i = 0; i < 6; i++){
                  std::cout << "npartTotal[" << i << "] " << (size_t(header.npartTotalHighWord[i]) << 32) + header.npartTotal[i] << "\n";
                }
                std::cout << "\n";
            }

            //==============================================================================================
            //==============================================================================================

            void GadgetReader::set_fields_in_file(std::vector<std::string> fields) { fields_in_file = fields; }

            GadgetReader::GadgetReader(int ndim) : NDIM(ndim) {}

            GadgetHeader GadgetReader::get_header() { return header; }

            void GadgetReader::read_section(std::ifstream & fp, std::vector<char> & buffer) {
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_section] File is not open\n";
                    throw_error(errormessage);
                }

                int bytes_start, bytes_end;
                fp.read((char *)&bytes_start, sizeof(bytes_start));
                if (not fp)
                    throw_error("[GadgetReader::read_section] Could not read the block size (end of file?)\n");
                if (endian_swap)
                    bytes_start = swap_endian(bytes_start);
                if (bytes_start < 0)
                    throw_error("[GadgetReader::read_section] Negative block size in file\n");

                // NB: we do not swap endian-ness here, the caller will have to do that
                // depending on what type of data we are reading
                buffer.resize(size_t(bytes_start));
                fp.read(buffer.data(), bytes_start);

                fp.read((char *)&bytes_end, sizeof(bytes_end));
                if (not fp)
                    throw_error("[GadgetReader::read_section] The file ended inside a block\n");
                if (endian_swap)
                    bytes_end = swap_endian(bytes_end);

                if (bytes_start != bytes_end) {
                    std::string errormessage = "[GadgetReader::read_section] Error in file BytesStart != ByteEnd!\n";
                    throw_error(errormessage);
                }
            }

            bool GadgetReader::read_label(std::ifstream & fp, std::string & label) {
                if (fp.peek() == std::ifstream::traits_type::eof())
                    return false;
                std::vector<char> buffer;
                read_section(fp, buffer);
                if (buffer.size() != 8)
                    throw_error("[GadgetReader::read_label] Expected a label block of 8 bytes in a format 2 file\n");
                label = std::string(buffer.data(), 4);
                while (not label.empty() and (label.back() == ' ' or label.back() == '\0'))
                    label.pop_back();
                return true;
            }

            void GadgetReader::read_header(std::ifstream & fp) {
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetReader::read_header] File is not open\n";
                    throw_error(errormessage);
                }

                // Detect the format and the endianness from the first block size: 256 (the header, format 1)
                // or 8 (the label block, format 2)
                endian_swap = false;
                format2 = false;
                int bytes_start, bytes_end;
                fp.read((char *)&bytes_start, sizeof(bytes_start));
                if (not fp)
                    throw_error("[GadgetReader::read_header] Could not read the header\n");
                if (bytes_start == 8 or swap_endian(bytes_start) == 8) {
                    format2 = true;
                    endian_swap = bytes_start != 8;
                    fp.seekg(0, fp.beg);
                    std::string label;
                    read_label(fp, label);
                    if (label != "HEAD")
                        throw_error("[GadgetReader::read_header] Format 2 file without a HEAD block first\n");
                    fp.read((char *)&bytes_start, sizeof(bytes_start));
                    if (endian_swap)
                        bytes_start = swap_endian(bytes_start);
                } else if (bytes_start != int(sizeof(header))) {
                    // Ok lets try to swap endian and see if that works better
                    endian_swap = true;
                    bytes_start = swap_endian(bytes_start);
                }
                if (bytes_start != int(sizeof(header))) {
                    std::string errormessage = "[GadgetReader::read_header] Error: not a GADGET file (the first block "
                                               "is not a 256 byte header or a format 2 label in either endianness)\n";
                    throw_error(errormessage);
                }
                fp.read((char *)&header, bytes_start);
                fp.read((char *)&bytes_end, sizeof(bytes_end));
                if (not fp)
                    throw_error("[GadgetReader::read_header] The file ended inside the header\n");
                if(endian_swap)
                    bytes_end = swap_endian(bytes_end);

                if (bytes_start != bytes_end) {
                    std::string errormessage = "[GadgetReader::read_header] Error: we find that BytesStart != ByteEnd!\n";
                    throw_error(errormessage);
                }

                // Check if endian of file needs to be changed
                // Swap endian of header and set flag so that we
                // swap the endian of the other fields also
                if (endian_swap) {
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
                }

                header_is_read = true;
            }

            //==============================================================================================
            //==============================================================================================

            GadgetWriter::GadgetWriter(int ndim) : NDIM(ndim) {}

            void GadgetWriter::write_section(std::ofstream & fp, std::vector<char> & buffer, size_t bytes) {
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetWriter::write_section] File is not open\n";
                    throw_error(errormessage);
                }
                if (buffer.size() < bytes) {
                    std::string errormessage = "[GadgetWriter::write_section] Buffersize too small\n";
                    throw_error(errormessage);
                }
                if (bytes > size_t(INT_MAX)) {
                    std::string errormessage = "[GadgetWriter::write_section] A block of " + std::to_string(bytes) +
                                               " bytes is larger than the 2^31 - 1 bytes a GADGET block can have. "
                                               "Write the particles to more files\n";
                    throw_error(errormessage);
                }
                const int marker = int(bytes);
                fp.write((char *)&marker, sizeof(marker));
                fp.write(buffer.data(), bytes);
                fp.write((char *)&marker, sizeof(marker));
            }

            void GadgetWriter::write_ids(std::ofstream & fp,
                                         size_t n,
                                         bool long_ids,
                                         const std::function<long long int(size_t)> & id) {
                std::vector<char> buffer;
                if (long_ids) {
                    buffer.resize(n * sizeof(int64_t));
                    auto * id_buffer = reinterpret_cast<int64_t *>(buffer.data());
                    for (size_t i = 0; i < n; i++)
                        id_buffer[i] = int64_t(id(i));
                } else {
                    buffer.resize(n * sizeof(int32_t));
                    auto * id_buffer = reinterpret_cast<int32_t *>(buffer.data());
                    for (size_t i = 0; i < n; i++)
                        id_buffer[i] = int32_t(id(i));
                }
                write_section(fp, buffer, buffer.size());
            }

            // Simple thing only for dark matter particles
            void GadgetWriter::write_header(std::ofstream & fp,
                                            unsigned int NumPart,
                                            size_t NumPartTot,
                                            int NumberOfFilesToWrite,
                                            double aexp,
                                            double Boxsize,
                                            double OmegaM,
                                            double OmegaLambda,
                                            double HubbleParam) {
                if (not fp.is_open()) {
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
            
            void GadgetWriter::write_header_general(std::ofstream & fp,
                                            std::vector<size_t> npart_family,
                                            std::vector<size_t> npart_family_tot,
                                            std::vector<double> mass_in_1e10_msunh,
                                            int NumberOfFilesToWrite,
                                            double aexp,
                                            double Boxsize,
                                            double OmegaM,
                                            double OmegaLambda,
                                            double HubbleParam) {
                if (not fp.is_open()) {
                    std::string errormessage = "[GadgetWriter::write_header] File is not open\n";
                    throw_error(errormessage);
                }

                // Mass in 10^10 Msun/h. Assumptions: Boxsize in Mpc/h
                for (int i = 0; i < 6; i++) {
                    header.npart[i] = npart_family[i];
                    header.npartTotal[i] = (unsigned int) npart_family_tot[i];
                    header.npartTotalHighWord[i] = (unsigned int)(npart_family_tot[i] >> 32);
                    header.mass[i] = mass_in_1e10_msunh[i];
                }
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

            int GadgetReader::get_num_files(std::string filename) {
                // Read the header of filename if given (not one we read earlier, possibly of another snapshot)
                if (not header_is_read or not filename.empty()) {
                    std::ifstream fp(filename.c_str(), std::ios::binary);
                    if (not fp.is_open()) {
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

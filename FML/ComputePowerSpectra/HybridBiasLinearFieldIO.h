#ifndef FML_HYBRIDBIAS_LINEARFIELDIO_HEADER
#define FML_HYBRIDBIAS_LINEARFIELDIO_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================================================
// A file with the linear density field delta_L of the initial conditions on the particle lattice, the input of
// the hybrid bias spectra (SimulationCode writes it with hybridbias_output_linear_field = true; SnapshotAnalyzer
// reads it with hybridbias_linear_field_source = "file"). Little-endian binary:
//
//   char[8]  "FMLDLIN1"
//   int64    nmesh (the lattice: nmesh^ndim points)
//   int64    ndim
//   double   boxsize (Mpc/h)
//   double   redshift (of the field)
//   double   delta_L at the lattice points, nmesh^ndim values, index (ix * nmesh + iy) * nmesh + iz (x slowest)
//
// The lattice point (ix, iy, iz) is the starting point of the particle with id (ix * nmesh + iy) * nmesh + iz
// (FML's convention) or of the particle found there in an initial-conditions snapshot. From numpy:
//   with open(name, "wb") as f:
//       f.write(b"FMLDLIN1"); np.array([n, 3], "<i8").tofile(f); np.array([box, z], "<f8").tofile(f)
//       delta.astype("<f8").tofile(f)       # delta[ix, iy, iz], C order
// Each MPI task writes / reads its own x slices (no gathering of the field).
//=============================================================================================================

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            struct LinearFieldFileHeader {
                long long int nmesh{0};
                long long int ndim{0};
                double boxsize{0.0};
                double redshift{0.0};
            };

            namespace detail_io {
                constexpr char magic[8] = {'F', 'M', 'L', 'D', 'L', 'I', 'N', '1'};
                constexpr std::streamoff header_bytes = 8 + 2 * 8 + 2 * 8;

                inline bool little_endian() {
                    const uint16_t one = 1;
                    unsigned char c;
                    std::memcpy(&c, &one, 1);
                    return c == 1;
                }

                /// Throw on every task if any task failed (message from the lowest failing task's text if local)
                inline void collective_error(const std::string & local_error, const std::string & what) {
                    int failed = local_error.empty() ? 0 : 1;
                    FML::MaxOverTasks(&failed);
                    if (failed)
                        throw std::runtime_error(what + (local_error.empty() ? " (failed on another task)" : ": " + local_error));
                }
            } // namespace detail_io

            /// Read the header of a linear field file (any task; throws if it is not one)
            inline LinearFieldFileHeader read_linear_field_header(const std::string & filename) {
                std::ifstream in(filename, std::ios::binary);
                if (not in)
                    throw std::runtime_error("Cannot open the linear field file " + filename);
                char m[8];
                LinearFieldFileHeader h;
                in.read(m, 8);
                in.read(reinterpret_cast<char *>(&h.nmesh), 8);
                in.read(reinterpret_cast<char *>(&h.ndim), 8);
                in.read(reinterpret_cast<char *>(&h.boxsize), 8);
                in.read(reinterpret_cast<char *>(&h.redshift), 8);
                if (not in or std::memcmp(m, detail_io::magic, 8) != 0)
                    throw std::runtime_error(filename + " is not an FML linear field file (FMLDLIN1)");
                if (h.nmesh < 1 or h.ndim < 1 or h.ndim > 3 or not(h.boxsize > 0.0) or not std::isfinite(h.redshift))
                    throw std::runtime_error("Invalid header in the linear field file " + filename);
                in.seekg(0, std::ios::end);
                const double expected = double(detail_io::header_bytes) + 8.0 * std::pow(double(h.nmesh), double(h.ndim));
                if (double(in.tellg()) != expected)
                    throw std::runtime_error("The linear field file " + filename + " has the wrong size for nmesh = " +
                                             std::to_string(h.nmesh));
                return h;
            }

            /// Write a real-space grid (the lattice field) to filename. Collective: the tasks append their x slices
            /// in turn
            template <int N>
            void write_linear_field(const std::string & filename,
                                    const FML::GRID::FFTWGrid<N> & real_grid,
                                    double boxsize,
                                    double redshift) {
                if (not detail_io::little_endian())
                    throw std::runtime_error("write_linear_field: only little-endian machines are supported");
                const long long int nmesh = real_grid.get_nmesh();
                const int local_nx = int(real_grid.get_local_nx());
                std::string error;
                for (int task = 0; task < FML::NTasks; task++) {
                    if (task == FML::ThisTask and error.empty()) {
                        std::ofstream out(filename, task == 0 ? std::ios::binary | std::ios::trunc
                                                              : std::ios::binary | std::ios::app);
                        if (not out)
                            error = "cannot open " + filename;
                        if (error.empty() and task == 0) {
                            const long long int ndim = N;
                            out.write(detail_io::magic, 8);
                            out.write(reinterpret_cast<const char *>(&nmesh), 8);
                            out.write(reinterpret_cast<const char *>(&ndim), 8);
                            out.write(reinterpret_cast<const char *>(&boxsize), 8);
                            out.write(reinterpret_cast<const char *>(&redshift), 8);
                        }
                        if (error.empty()) {
                            std::vector<double> row(static_cast<size_t>(nmesh));
                            std::array<int, N> c{};
                            const long long int nrows_per_slice = (long long int)std::pow(double(nmesh), N - 2);
                            for (int ix = 0; ix < local_nx; ix++)
                                for (long long int r = 0; r < nrows_per_slice; r++) {
                                    c[0] = ix;
                                    long long int rr = r;
                                    for (int d = N - 2; d >= 1; d--) {
                                        c[d] = int(rr % nmesh);
                                        rr /= nmesh;
                                    }
                                    for (int iz = 0; iz < nmesh; iz++) {
                                        c[N - 1] = iz;
                                        row[size_t(iz)] = double(real_grid.get_real(c));
                                    }
                                    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(8 * nmesh));
                                }
                            if (not out)
                                error = "error writing " + filename;
                        }
                    }
#ifdef USE_MPI
                    MPI_Barrier(FML::communicator);
#endif
                }
                detail_io::collective_error(error, "write_linear_field");
            }

            /// Read a linear field file into a real-space grid (nmesh from the file; the x slices of this task).
            /// Collective
            template <int N>
            FML::GRID::FFTWGrid<N> read_linear_field(const std::string & filename, LinearFieldFileHeader & header) {
                std::string error;
                try {
                    if (not detail_io::little_endian())
                        throw std::runtime_error("only little-endian machines are supported");
                    header = read_linear_field_header(filename);
                    if (header.ndim != N)
                        throw std::runtime_error("the field has ndim = " + std::to_string(header.ndim));
                    if (header.nmesh < FML::NTasks)
                        throw std::runtime_error("nmesh is smaller than the number of MPI tasks");
                } catch (const std::exception & e) {
                    error = e.what();
                }
                detail_io::collective_error(error, "read_linear_field(" + filename + ")");

                const int nmesh = int(header.nmesh);
                FML::GRID::FFTWGrid<N> grid(nmesh);
                const long long int local_nx = grid.get_local_nx();
                const long long int x_start = grid.get_local_x_start();
                const long long int nrows_per_slice = (long long int)std::pow(double(nmesh), N - 2);
                try {
                    std::ifstream in(filename, std::ios::binary);
                    in.seekg(detail_io::header_bytes + std::streamoff(8) * std::streamoff(x_start) *
                                                           std::streamoff(nrows_per_slice) * std::streamoff(nmesh));
                    std::vector<double> row(static_cast<size_t>(nmesh));
                    std::array<int, N> c{};
                    for (int ix = 0; ix < local_nx; ix++)
                        for (long long int r = 0; r < nrows_per_slice; r++) {
                            in.read(reinterpret_cast<char *>(row.data()), std::streamsize(8 * nmesh));
                            c[0] = ix;
                            long long int rr = r;
                            for (int d = N - 2; d >= 1; d--) {
                                c[d] = int(rr % nmesh);
                                rr /= nmesh;
                            }
                            for (int iz = 0; iz < nmesh; iz++) {
                                if (not std::isfinite(row[size_t(iz)]))
                                    throw std::runtime_error("a value is not finite");
                                c[N - 1] = iz;
                                grid.set_real(c, FML::GRID::FloatType(row[size_t(iz)]));
                            }
                        }
                    if (not in)
                        throw std::runtime_error("error reading the values");
                } catch (const std::exception & e) {
                    error = e.what();
                }
                detail_io::collective_error(error, "read_linear_field(" + filename + ")");
                return grid;
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif

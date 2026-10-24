#ifndef GADGETUTILSHDF5_HEADER
#define GADGETUTILSHDF5_HEADER

#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

//====================================================================================
//
// Read and write GADGET snapshots in the HDF5 format (as written by GADGET-2/3/4, AREPO,
// GIZMO and read by yt, pynbody, nbodykit etc.)
//
// File layout:
//   /Header                 attributes NumPart_ThisFile, NumPart_Total, NumPart_Total_HighWord,
//                           MassTable, Time, Redshift, BoxSize, NumFilesPerSnapshot, Omega0,
//                           OmegaLambda, HubbleParam, Flag_* (as in the binary header)
//   /PartTypeN/Coordinates  (npart, 3) positions in [0, BoxSize)
//   /PartTypeN/Velocities   (npart, 3) sqrt(a) dx/dt in km/s (the standard GADGET convention)
//   /PartTypeN/ParticleIDs  (npart)
//
// A snapshot is either one file <fileprefix>.hdf5 or several files <fileprefix>.0.hdf5,
// <fileprefix>.1.hdf5, ... (NumFilesPerSnapshot in the header).
//
// The interface is the same as for the binary files (GadgetReader / GadgetWriter):
//  - Positions in the particles are in [0,1) (we divide by BoxSize when reading)
//  - Velocities are converted to peculiar velocities in km/s when reading (multiply by sqrt(a)).
//    If the file has UnitVelocity_in_cm_per_s (in the Parameters or Units group) we also
//    convert from these units to km/s
//  - When writing pos_norm and vel_norm convert from the units in the particles to positions in
//    [0,BoxSize) and sqrt(a) dx/dt in km/s
//  - If the particle has get_family/set_family the type (0,...,5) is the family. If not we only
//    read PartType1 (dark matter) and write everything as PartType1
//  - Any precision of the data in the file is fine for reading. We write positions and
//    velocities as float (or double if double_precision is set) and IDs as 64 bit unsigned
//
// Only compiled if USE_HDF5 is defined (link with -lhdf5)
//
//====================================================================================

namespace FML {
    namespace FILEUTILS {
        namespace GADGET {
            /// Does <fileprefix>.hdf5 or <fileprefix>.0.hdf5 exist? (available also without USE_HDF5)
            inline bool is_hdf5_snapshot(const std::string & fileprefix) {
                for (auto ending : {".hdf5", ".0.hdf5"}) {
                    std::ifstream fp(fileprefix + ending);
                    if (fp.good())
                        return true;
                }
                return false;
            }
        } // namespace GADGET
    }     // namespace FILEUTILS
} // namespace FML

#ifdef USE_HDF5
#include <hdf5.h>

namespace FML {
    namespace FILEUTILS {
        namespace GADGET {

            namespace HDF5 {

                [[noreturn]] inline void throw_error(const std::string & errormessage) {
#ifdef USE_MPI
                    std::cout << errormessage << "\n" << std::flush;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    abort();
#else
                    throw std::runtime_error(errormessage);
#endif
                }

                /// Owns a hid_t and closes it with the right function when going out of scope
                class Handle {
                    hid_t id{-1};
                    herr_t (*close)(hid_t){nullptr};

                  public:
                    Handle() = default;
                    Handle(hid_t id, herr_t (*close)(hid_t), const std::string & what) : id(id), close(close) {
                        if (id < 0)
                            throw_error("[GadgetHDF5] Failed to open/create " + what);
                    }
                    ~Handle() {
                        if (id >= 0 and close)
                            close(id);
                    }
                    Handle(const Handle &) = delete;
                    Handle & operator=(const Handle &) = delete;
                    Handle(Handle && other) noexcept : id(other.id), close(other.close) { other.id = -1; }
                    Handle & operator=(Handle && other) noexcept {
                        std::swap(id, other.id);
                        std::swap(close, other.close);
                        return *this;
                    }
                    operator hid_t() const { return id; }
                };

                inline bool exists(hid_t loc, const std::string & name) {
                    return H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0;
                }

                /// Read an attribute (any type in the file, converted to memtype). Returns false if it does
                /// not exist (or has a different number of elements than n)
                inline bool read_attribute(hid_t loc, const std::string & name, hid_t memtype, void * data, size_t n) {
                    if (H5Aexists(loc, name.c_str()) <= 0)
                        return false;
                    Handle attr(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Aclose, "attribute " + name);
                    Handle space(H5Aget_space(attr), H5Sclose, "dataspace of attribute " + name);
                    if (size_t(H5Sget_simple_extent_npoints(space)) != n)
                        return false;
                    return H5Aread(attr, memtype, data) >= 0;
                }

                inline void
                write_attribute(hid_t loc, const std::string & name, hid_t filetype, hid_t memtype, const void * data, size_t n) {
                    hsize_t dims[1] = {n};
                    Handle space(n == 1 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, dims, nullptr),
                                 H5Sclose,
                                 "dataspace for attribute " + name);
                    Handle attr(H5Acreate2(loc, name.c_str(), filetype, space, H5P_DEFAULT, H5P_DEFAULT),
                                H5Aclose,
                                "attribute " + name);
                    if (H5Awrite(attr, memtype, data) < 0)
                        throw_error("[GadgetHDF5] Failed to write attribute " + name);
                }

                /// Read rows [start, start + count) of a dataset with ncols columns (ncols = 1 for a 1D
                /// dataset) converting to memtype
                inline void read_rows(hid_t dataset,
                                      const std::string & name,
                                      hid_t memtype,
                                      void * data,
                                      hsize_t start,
                                      hsize_t count,
                                      hsize_t ncols) {
                    Handle filespace(H5Dget_space(dataset), H5Sclose, "dataspace of " + name);
                    hsize_t offset[2] = {start, 0};
                    hsize_t block[2] = {count, ncols};
                    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, block, nullptr) < 0)
                        throw_error("[GadgetHDF5] Failed to select rows in " + name);
                    hsize_t memdims[1] = {count * ncols};
                    Handle memspace(H5Screate_simple(1, memdims, nullptr), H5Sclose, "memory space for " + name);
                    if (H5Dread(dataset, memtype, memspace, filespace, H5P_DEFAULT, data) < 0)
                        throw_error("[GadgetHDF5] Failed to read " + name);
                }

                inline void write_dataset(hid_t loc,
                                          const std::string & name,
                                          hid_t filetype,
                                          hid_t memtype,
                                          const void * data,
                                          hsize_t nrows,
                                          hsize_t ncols) {
                    hsize_t dims[2] = {nrows, ncols};
                    Handle space(H5Screate_simple(ncols == 1 ? 1 : 2, dims, nullptr), H5Sclose, "dataspace for " + name);
                    // No modification times in the file so the same data gives identical files
                    Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "property list for " + name);
                    H5Pset_obj_track_times(dcpl, false);
                    Handle dataset(H5Dcreate2(loc, name.c_str(), filetype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
                                   H5Dclose,
                                   "dataset " + name);
                    if (nrows > 0 and H5Dwrite(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
                        throw_error("[GadgetHDF5] Failed to write dataset " + name);
                }
            } // namespace HDF5

            //====================================================================================
            /// Read GADGET HDF5 snapshots
            //====================================================================================
            class GadgetHDF5Reader {
              private:
                GadgetHeader header{};
                bool header_is_read{false};
                int NDIM{3};

                // Read the data in chunks of this many particles to limit the memory use
                size_t chunk_size{1 << 20};

                // Velocities in the file times this is km/s
                double velocity_unit_in_kms{1.0};

              public:
                GadgetHDF5Reader() = default;
                GadgetHDF5Reader(int ndim) : NDIM(ndim) {}

                /// The file(s) of a snapshot: <fileprefix>.hdf5 or <fileprefix>.0.hdf5, <fileprefix>.1.hdf5, ...
                std::vector<std::string> get_filenames(const std::string & fileprefix) {
                    {
                        std::ifstream fp(fileprefix + ".hdf5");
                        if (fp.good())
                            return {fileprefix + ".hdf5"};
                    }
                    const int nfiles = get_num_files(fileprefix + ".0.hdf5");
                    std::vector<std::string> filenames;
                    for (int i = 0; i < nfiles; i++)
                        filenames.push_back(fileprefix + "." + std::to_string(i) + ".hdf5");
                    return filenames;
                }

                /// Read the header of a file
                void read_header(const std::string & filename) {
                    HDF5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, filename);
                    read_header(file, filename);
                }

                void read_header(hid_t file, const std::string & filename) {
                    if (not HDF5::exists(file, "Header"))
                        HDF5::throw_error("[GadgetHDF5Reader] No Header group in " + filename);
                    HDF5::Handle h(H5Gopen2(file, "Header", H5P_DEFAULT), H5Gclose, "Header in " + filename);

                    header = GadgetHeader{};
                    std::array<unsigned long long, 6> npart{}, npart_total{}, highword{};
                    if (not HDF5::read_attribute(h, "NumPart_ThisFile", H5T_NATIVE_ULLONG, npart.data(), 6))
                        HDF5::throw_error("[GadgetHDF5Reader] No NumPart_ThisFile in the header of " + filename);
                    if (not HDF5::read_attribute(h, "NumPart_Total", H5T_NATIVE_ULLONG, npart_total.data(), 6))
                        npart_total = npart;
                    HDF5::read_attribute(h, "NumPart_Total_HighWord", H5T_NATIVE_ULLONG, highword.data(), 6);
                    for (int i = 0; i < 6; i++) {
                        // GADGET-4 writes NumPart_Total as 64 bit (and no high word)
                        const unsigned long long total = npart_total[i] + (highword[i] << 32);
                        header.npart[i] = (unsigned int)npart[i];
                        header.npartTotal[i] = (unsigned int)(total & 0xffffffffULL);
                        header.npartTotalHighWord[i] = (unsigned int)(total >> 32);
                    }
                    HDF5::read_attribute(h, "MassTable", H5T_NATIVE_DOUBLE, header.mass, 6);
                    if (not HDF5::read_attribute(h, "Time", H5T_NATIVE_DOUBLE, &header.time, 1))
                        HDF5::throw_error("[GadgetHDF5Reader] No Time in the header of " + filename);
                    if (not HDF5::read_attribute(h, "Redshift", H5T_NATIVE_DOUBLE, &header.redshift, 1))
                        header.redshift = 1.0 / header.time - 1.0;
                    if (not HDF5::read_attribute(h, "BoxSize", H5T_NATIVE_DOUBLE, &header.BoxSize, 1))
                        HDF5::throw_error("[GadgetHDF5Reader] No BoxSize in the header of " + filename);
                    if (not HDF5::read_attribute(h, "NumFilesPerSnapshot", H5T_NATIVE_INT, &header.num_files, 1))
                        header.num_files = 1;
                    HDF5::read_attribute(h, "Flag_Sfr", H5T_NATIVE_INT, &header.flag_sfr, 1);
                    HDF5::read_attribute(h, "Flag_Feedback", H5T_NATIVE_INT, &header.flag_feedback, 1);
                    HDF5::read_attribute(h, "Flag_Cooling", H5T_NATIVE_INT, &header.flag_cooling, 1);
                    HDF5::read_attribute(h, "Flag_StellarAge", H5T_NATIVE_INT, &header.flag_stellarage, 1);
                    HDF5::read_attribute(h, "Flag_Metals", H5T_NATIVE_INT, &header.flag_metals, 1);

                    // The cosmological parameters are in the Header (GADGET-2/3) or in Parameters (GADGET-4, AREPO)
                    // and the units (if not km/s) in Parameters or Units
                    auto read_from_any_group = [&](const std::string & name, double & value) {
                        if (HDF5::read_attribute(h, name, H5T_NATIVE_DOUBLE, &value, 1))
                            return true;
                        for (auto group : {"Parameters", "Units"}) {
                            if (not HDF5::exists(file, group))
                                continue;
                            HDF5::Handle g(H5Gopen2(file, group, H5P_DEFAULT), H5Gclose, group);
                            if (HDF5::read_attribute(g, name, H5T_NATIVE_DOUBLE, &value, 1))
                                return true;
                        }
                        return false;
                    };
                    read_from_any_group("Omega0", header.Omega0);
                    read_from_any_group("OmegaLambda", header.OmegaLambda);
                    read_from_any_group("HubbleParam", header.HubbleParam);
                    double unit_velocity_in_cm_per_s = 1e5;
                    read_from_any_group("UnitVelocity_in_cm_per_s", unit_velocity_in_cm_per_s);
                    velocity_unit_in_kms = unit_velocity_in_cm_per_s / 1e5;

                    header_is_read = true;
                }

                /// Get the header (assumes it has been read)
                GadgetHeader get_header() const { return header; }

                /// The number of files in the snapshot (from the header of filename)
                int get_num_files(const std::string & filename = "") {
                    if (not header_is_read)
                        read_header(filename);
                    return header.num_files;
                }

                /// Read the particles in chunks of this many (default 2^20)
                void set_chunk_size(size_t n) { chunk_size = std::max(size_t(1), n); }

                /// Read all the files in a snapshot and store the particles in part (same as GadgetReader::read_gadget)
                template <class T, class Alloc = std::allocator<T>>
                void read_gadget(std::string fileprefix,
                                 std::vector<T, Alloc> & part,
                                 double buffer_factor,
                                 bool only_keep_part_in_domain,
                                 bool verbose) {
                    verbose = verbose and FML::ThisTask == 0;
                    auto filenames = get_filenames(fileprefix);
                    read_header(filenames[0]);

                    size_t npart_total = 0;
                    for (int i = 0; i < 6; i++) {
                        if constexpr (not FML::PARTICLE::has_set_family<T>())
                            if (i != 1)
                                continue;
                        npart_total += (size_t(header.npartTotalHighWord[i]) << 32) + size_t(header.npartTotal[i]);
                    }
                    size_t nalloc = npart_total;
                    if (only_keep_part_in_domain)
                        nalloc = size_t(double(npart_total) * buffer_factor) / FML::NTasks;
                    part.clear();
                    part.reserve(nalloc);

                    for (auto & filename : filenames) {
                        if (verbose)
                            std::cout << "Reading file " << filename << "\n";
                        read_gadget_single(filename, part, only_keep_part_in_domain, verbose);
                    }
                }

                /// Read one file and add the particles to the back of part. If only_keep_part_in_domain then
                /// we only keep the particles that fall within the local domain
                template <class T, class Alloc = std::allocator<T>>
                void read_gadget_single(std::string filename,
                                        std::vector<T, Alloc> & part,
                                        bool only_keep_part_in_domain,
                                        bool verbose) {
                    verbose = verbose and FML::ThisTask == 0;

                    HDF5::Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, filename);
                    read_header(file, filename);
                    if (verbose)
                        print_header_info(header);

                    const double pos_norm = 1.0 / header.BoxSize;
                    const double vel_norm = std::sqrt(header.time) * velocity_unit_in_kms;

                    // Without family in the particle we only read dark matter
                    if constexpr (not FML::PARTICLE::has_set_family<T>()) {
                        size_t nother = 0;
                        for (int type = 0; type < 6; type++)
                            nother += (type == 1) ? 0 : header.npart[type];
                        if (nother > 0 and verbose)
                            std::cout << "[GadgetHDF5Reader] The file has " << nother
                                      << " particles that are not PartType1. The particle has no set_family so we "
                                         "only read PartType1\n";
                    }

                    std::vector<double> buffer;
                    std::vector<unsigned long long> id_buffer;
                    std::vector<char> is_in_domain;

                    for (int type = 0; type < 6; type++) {
                        if constexpr (not FML::PARTICLE::has_set_family<T>())
                            if (type != 1)
                                continue;
                        const size_t npart = header.npart[type];
                        if (npart == 0)
                            continue;

                        const std::string group = "PartType" + std::to_string(type);
                        if (not HDF5::exists(file, group))
                            HDF5::throw_error("[GadgetHDF5Reader] Header says there are particles of type " +
                                              std::to_string(type) + " but there is no " + group + " in " + filename);
                        HDF5::Handle g(H5Gopen2(file, group.c_str(), H5P_DEFAULT), H5Gclose, group);

                        // Open the datasets we need (and check their shape)
                        auto open_dataset = [&](const std::string & name, size_t ncols) {
                            if (not HDF5::exists(g, name))
                                HDF5::throw_error("[GadgetHDF5Reader] No " + group + "/" + name + " in " + filename);
                            HDF5::Handle d(H5Dopen2(g, name.c_str(), H5P_DEFAULT), H5Dclose, group + "/" + name);
                            HDF5::Handle space(H5Dget_space(d), H5Sclose, group + "/" + name);
                            hsize_t dims[2] = {0, 1};
                            const int rank = H5Sget_simple_extent_dims(space, dims, nullptr);
                            if (dims[0] != npart or (ncols > 1 and (rank != 2 or dims[1] != ncols)))
                                HDF5::throw_error("[GadgetHDF5Reader] " + group + "/" + name + " in " + filename +
                                                  " does not have the expected shape (" + std::to_string(npart) +
                                                  ", " + std::to_string(ncols) + ")");
                            return d;
                        };
                        HDF5::Handle coordinates = open_dataset("Coordinates", NDIM);
                        HDF5::Handle velocities, ids;
                        if constexpr (FML::PARTICLE::has_get_vel<T>())
                            velocities = open_dataset("Velocities", NDIM);
                        if constexpr (FML::PARTICLE::has_set_id<T>())
                            ids = open_dataset("ParticleIDs", 1);

                        if (part.capacity() < part.size() + npart and not only_keep_part_in_domain)
                            part.reserve(part.size() + npart);

                        for (size_t start = 0; start < npart; start += chunk_size) {
                            const size_t count = std::min(chunk_size, npart - start);

                            // Positions (we need these first to know which particles are in the domain)
                            buffer.resize(count * NDIM);
                            HDF5::read_rows(coordinates, group + "/Coordinates", H5T_NATIVE_DOUBLE, buffer.data(), start, count, NDIM);
                            is_in_domain.assign(count, 1);
                            const size_t index_start = part.size();
                            std::vector<double> x(NDIM);
                            for (size_t i = 0; i < count; i++) {
                                for (int idim = 0; idim < NDIM; idim++) {
                                    x[idim] = buffer[NDIM * i + idim] * pos_norm;
                                    if (x[idim] >= 1.0)
                                        x[idim] -= 1.0;
                                    if (x[idim] < 0.0)
                                        x[idim] += 1.0;
                                }
                                if (not(x[0] >= 0.0 and x[0] < 1.0))
                                    HDF5::throw_error("[GadgetHDF5Reader] Particle has x position outside the box "
                                                      "even after periodic wrap in " + filename);
                                if (only_keep_part_in_domain and not(x[0] >= FML::xmin_domain and x[0] < FML::xmax_domain)) {
                                    is_in_domain[i] = 0;
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

                            // Velocities
                            if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                                HDF5::read_rows(velocities, group + "/Velocities", H5T_NATIVE_DOUBLE, buffer.data(), start, count, NDIM);
                                size_t index = index_start;
                                for (size_t i = 0; i < count; i++) {
                                    if (not is_in_domain[i])
                                        continue;
                                    auto * vel = FML::PARTICLE::GetVel(part[index++]);
                                    for (int idim = 0; idim < NDIM; idim++)
                                        vel[idim] = buffer[NDIM * i + idim] * vel_norm;
                                }
                            }

                            // IDs
                            if constexpr (FML::PARTICLE::has_set_id<T>()) {
                                id_buffer.resize(count);
                                HDF5::read_rows(ids, group + "/ParticleIDs", H5T_NATIVE_ULLONG, id_buffer.data(), start, count, 1);
                                size_t index = index_start;
                                for (size_t i = 0; i < count; i++) {
                                    if (not is_in_domain[i])
                                        continue;
                                    FML::PARTICLE::SetID(part[index++], id_buffer[i]);
                                }
                            }
                        }
                    }
                }
            };

            //====================================================================================
            /// Write GADGET HDF5 snapshots
            //====================================================================================
            class GadgetHDF5Writer {
              private:
                GadgetHeader header{};
                int NDIM{3};
                bool double_precision{false};
                bool periodic_wrap{true};
                HDF5::Handle gcpl;

              public:
                GadgetHDF5Writer() = default;
                GadgetHDF5Writer(int ndim) : NDIM(ndim) {}

                /// Group creation properties: no modification times (so the same data gives identical files)
                HDF5::Handle & group_properties() {
                    if (gcpl < 0) {
                        gcpl = HDF5::Handle(H5Pcreate(H5P_GROUP_CREATE), H5Pclose, "group properties");
                        H5Pset_obj_track_times(gcpl, false);
                    }
                    return gcpl;
                }

                /// Write positions and velocities as double instead of float
                void set_double_precision(bool use_double) { double_precision = use_double; }

                /// Wrap the positions into [0, BoxSize) (the default, as GADGET expects). Turn off for data that is
                /// not a periodic box, e.g. a lightcone where the positions are relative to an observer
                void set_periodic_wrap(bool wrap) { periodic_wrap = wrap; }

                GadgetHeader get_header() const { return header; }

                /// Write a single file (same arguments as GadgetWriter::write_gadget_single). The filename should
                /// be <fileprefix>.hdf5 if there is one file and <fileprefix>.<i>.hdf5 otherwise
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
                                         std::vector<double> OmegaFamilyOverOmegaM = {0., 1., 0., 0., 0., 0.}) {

                    // Count how many of each type we have
                    auto family_of = [&](size_t i) -> int {
                        if constexpr (FML::PARTICLE::has_get_family<T>())
                            return int(FML::PARTICLE::GetFamily(part[i]));
                        (void)i;
                        return 1;
                    };
                    std::vector<size_t> npart_family(6, 0);
                    std::vector<size_t> npart_family_tot(6, 0);
                    if constexpr (FML::PARTICLE::has_get_family<T>()) {
                        for (size_t i = 0; i < NumPart; i++) {
                            const int family = family_of(i);
                            if (family >= 0 and family < 6)
                                npart_family[family]++;
                        }
                    } else {
                        npart_family[1] = NumPart;
                    }
                    npart_family_tot[1] = NumPartTot;

                    // If only CDM we know the total number (and avoid MPI calls), otherwise sum over tasks
                    if (OmegaFamilyOverOmegaM[1] != 1.0 or FML::PARTICLE::has_get_family<T>()) {
                        npart_family_tot = npart_family;
                        FML::SumArrayOverTasks(npart_family_tot.data(), npart_family_tot.size());
                    }

                    // Mass in 1e10 Msun/h (Boxsize in Mpc/h)
                    std::vector<double> mass(6, 0.0);
                    for (int i = 0; i < 6; i++)
                        mass[i] = npart_family_tot[i] == 0 ?
                                      0.0 :
                                      3.0 * OmegaM * OmegaFamilyOverOmegaM[i] * MplMpl_over_H0Msunh *
                                          std::pow(Boxsize / HubbleLengthInMpch, 3) / double(npart_family_tot[1]) / 1e10;

                    HDF5::Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, filename);

                    // The header
                    {
                        header = GadgetHeader{};
                        std::array<unsigned int, 6> npart{}, npart_total{}, highword{};
                        for (int i = 0; i < 6; i++) {
                            npart[i] = (unsigned int)npart_family[i];
                            npart_total[i] = (unsigned int)(npart_family_tot[i] & 0xffffffffULL);
                            highword[i] = (unsigned int)(npart_family_tot[i] >> 32);
                            header.npart[i] = npart[i];
                            header.npartTotal[i] = npart_total[i];
                            header.npartTotalHighWord[i] = highword[i];
                            header.mass[i] = mass[i];
                        }
                        header.time = aexp;
                        header.redshift = 1.0 / aexp - 1.0;
                        header.num_files = NumberOfFilesToWrite;
                        header.BoxSize = Boxsize;
                        header.Omega0 = OmegaM;
                        header.OmegaLambda = OmegaLambda;
                        header.HubbleParam = HubbleParam;

                        HDF5::Handle h(H5Gcreate2(file, "Header", H5P_DEFAULT, group_properties(), H5P_DEFAULT), H5Gclose, "Header");
                        const int zero = 0;
                        const int precision = double_precision ? 1 : 0;
                        auto uint6 = [&](const char * name, const std::array<unsigned int, 6> & v) {
                            HDF5::write_attribute(h, name, H5T_STD_U32LE, H5T_NATIVE_UINT, v.data(), 6);
                        };
                        auto dbl = [&](const char * name, double v) {
                            HDF5::write_attribute(h, name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &v, 1);
                        };
                        auto integer = [&](const char * name, int v) {
                            HDF5::write_attribute(h, name, H5T_STD_I32LE, H5T_NATIVE_INT, &v, 1);
                        };
                        uint6("NumPart_ThisFile", npart);
                        uint6("NumPart_Total", npart_total);
                        uint6("NumPart_Total_HighWord", highword);
                        HDF5::write_attribute(h, "MassTable", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, mass.data(), 6);
                        dbl("Time", header.time);
                        dbl("Redshift", header.redshift);
                        dbl("BoxSize", Boxsize);
                        integer("NumFilesPerSnapshot", NumberOfFilesToWrite);
                        dbl("Omega0", OmegaM);
                        dbl("OmegaLambda", OmegaLambda);
                        dbl("HubbleParam", HubbleParam);
                        integer("Flag_Sfr", zero);
                        integer("Flag_Cooling", zero);
                        integer("Flag_StellarAge", zero);
                        integer("Flag_Metals", zero);
                        integer("Flag_Feedback", zero);
                        integer("Flag_Entropy_ICs", zero);
                        integer("Flag_DoublePrecision", precision);
                    }

                    // The particles, one group per type
                    const hid_t filetype = double_precision ? H5T_IEEE_F64LE : H5T_IEEE_F32LE;
                    for (int type = 0; type < 6; type++) {
                        const size_t n = npart_family[type];
                        if (n == 0)
                            continue;
                        const std::string group = "PartType" + std::to_string(type);
                        HDF5::Handle g(H5Gcreate2(file, group.c_str(), H5P_DEFAULT, group_properties(), H5P_DEFAULT), H5Gclose, group);

                        std::vector<double> buffer;
                        auto gather = [&](auto && get, double norm) {
                            buffer.assign(n * NDIM, 0.0);
                            size_t count = 0;
                            for (size_t i = 0; i < NumPart; i++) {
                                if (family_of(i) != type)
                                    continue;
                                const auto * x = get(part[i]);
                                for (int idim = 0; idim < NDIM; idim++)
                                    buffer[NDIM * count + idim] = double(x[idim]) * norm;
                                count++;
                            }
                        };
                        if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                            gather([](auto & p) { return FML::PARTICLE::GetPos(p); }, pos_norm);
                            // Positions must be in [0, Boxsize) also after rounding to float
                            for (auto & x : buffer) {
                                if (not periodic_wrap)
                                    break;
                                if (x >= Boxsize)
                                    x -= Boxsize;
                                if (x < 0.0)
                                    x += Boxsize;
                                if (not double_precision and float(x) >= float(Boxsize))
                                    x = 0.0;
                            }
                            HDF5::write_dataset(g, "Coordinates", filetype, H5T_NATIVE_DOUBLE, buffer.data(), n, NDIM);
                        }
                        if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                            gather([](auto & p) { return FML::PARTICLE::GetVel(p); }, vel_norm);
                            HDF5::write_dataset(g, "Velocities", filetype, H5T_NATIVE_DOUBLE, buffer.data(), n, NDIM);
                        }
                        if constexpr (FML::PARTICLE::has_get_id<T>()) {
                            std::vector<unsigned long long> ids;
                            ids.reserve(n);
                            for (size_t i = 0; i < NumPart; i++)
                                if (family_of(i) == type)
                                    ids.push_back((unsigned long long)FML::PARTICLE::GetID(part[i]));
                            HDF5::write_dataset(g, "ParticleIDs", H5T_STD_U64LE, H5T_NATIVE_ULLONG, ids.data(), n, 1);
                        }
                    }
                }

                /// Write a single file with several sets of particles (e.g. CDM and baryons kept in separate
                /// containers), each its own particle type (see GadgetWriter::write_gadget_species). No MPI calls
                template <class T>
                void write_gadget_species(std::string filename,
                                          std::vector<GadgetSpecies<T>> species,
                                          int NumberOfFilesToWrite,
                                          double aexp,
                                          double Boxsize,
                                          double OmegaM,
                                          double OmegaLambda,
                                          double HubbleParam,
                                          double pos_norm,
                                          double vel_norm) {
                    std::sort(species.begin(), species.end(), [](auto & a, auto & b) { return a.type < b.type; });
                    std::array<unsigned int, 6> npart{}, npart_total{}, highword{};
                    std::vector<double> mass(6, 0.0);
                    for (size_t i = 0; i < species.size(); i++) {
                        auto & sp = species[i];
                        if (sp.type < 0 or sp.type > 5 or (i > 0 and species[i - 1].type == sp.type))
                            throw std::runtime_error("[GadgetHDF5Writer::write_gadget_species] The particle types must "
                                                     "be different and in 0-5");
                        npart[sp.type] = (unsigned int)sp.npart;
                        npart_total[sp.type] = (unsigned int)(sp.npart_total & 0xffffffffULL);
                        highword[sp.type] = (unsigned int)(sp.npart_total >> 32);
                        mass[sp.type] = sp.npart_total == 0 ?
                                            0.0 :
                                            3.0 * OmegaM * sp.mass_fraction * MplMpl_over_H0Msunh *
                                                std::pow(Boxsize / HubbleLengthInMpch, 3) / double(sp.npart_total) / 1e10;
                    }

                    HDF5::Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, filename);
                    {
                        header = GadgetHeader{};
                        for (int i = 0; i < 6; i++) {
                            header.npart[i] = npart[i];
                            header.npartTotal[i] = npart_total[i];
                            header.npartTotalHighWord[i] = highword[i];
                            header.mass[i] = mass[i];
                        }
                        header.time = aexp;
                        header.redshift = 1.0 / aexp - 1.0;
                        header.num_files = NumberOfFilesToWrite;
                        header.BoxSize = Boxsize;
                        header.Omega0 = OmegaM;
                        header.OmegaLambda = OmegaLambda;
                        header.HubbleParam = HubbleParam;

                        HDF5::Handle h(H5Gcreate2(file, "Header", H5P_DEFAULT, group_properties(), H5P_DEFAULT), H5Gclose, "Header");
                        const int zero = 0;
                        const int precision = double_precision ? 1 : 0;
                        auto uint6 = [&](const char * name, const std::array<unsigned int, 6> & v) {
                            HDF5::write_attribute(h, name, H5T_STD_U32LE, H5T_NATIVE_UINT, v.data(), 6);
                        };
                        auto dbl = [&](const char * name, double v) {
                            HDF5::write_attribute(h, name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &v, 1);
                        };
                        auto integer = [&](const char * name, int v) {
                            HDF5::write_attribute(h, name, H5T_STD_I32LE, H5T_NATIVE_INT, &v, 1);
                        };
                        uint6("NumPart_ThisFile", npart);
                        uint6("NumPart_Total", npart_total);
                        uint6("NumPart_Total_HighWord", highword);
                        HDF5::write_attribute(h, "MassTable", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, mass.data(), 6);
                        dbl("Time", header.time);
                        dbl("Redshift", header.redshift);
                        dbl("BoxSize", Boxsize);
                        integer("NumFilesPerSnapshot", NumberOfFilesToWrite);
                        dbl("Omega0", OmegaM);
                        dbl("OmegaLambda", OmegaLambda);
                        dbl("HubbleParam", HubbleParam);
                        integer("Flag_Sfr", zero);
                        integer("Flag_Cooling", zero);
                        integer("Flag_StellarAge", zero);
                        integer("Flag_Metals", zero);
                        integer("Flag_Feedback", zero);
                        integer("Flag_Entropy_ICs", zero);
                        integer("Flag_DoublePrecision", precision);
                    }

                    const hid_t filetype = double_precision ? H5T_IEEE_F64LE : H5T_IEEE_F32LE;
                    for (auto & sp : species) {
                        const size_t n = sp.npart;
                        if (n == 0)
                            continue;
                        const std::string group = "PartType" + std::to_string(sp.type);
                        HDF5::Handle g(H5Gcreate2(file, group.c_str(), H5P_DEFAULT, group_properties(), H5P_DEFAULT), H5Gclose, group);
                        std::vector<double> buffer;
                        auto gather = [&](auto && get, double norm) {
                            buffer.assign(n * NDIM, 0.0);
                            for (size_t i = 0; i < n; i++) {
                                const auto * x = get(sp.part[i]);
                                for (int idim = 0; idim < NDIM; idim++)
                                    buffer[NDIM * i + idim] = double(x[idim]) * norm;
                            }
                        };
                        if constexpr (FML::PARTICLE::has_get_pos<T>()) {
                            gather([](auto & p) { return FML::PARTICLE::GetPos(p); }, pos_norm);
                            for (auto & x : buffer) {
                                if (not periodic_wrap)
                                    break;
                                if (x >= Boxsize)
                                    x -= Boxsize;
                                if (x < 0.0)
                                    x += Boxsize;
                                if (not double_precision and float(x) >= float(Boxsize))
                                    x = 0.0;
                            }
                            HDF5::write_dataset(g, "Coordinates", filetype, H5T_NATIVE_DOUBLE, buffer.data(), n, NDIM);
                        }
                        if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                            gather([](auto & p) { return FML::PARTICLE::GetVel(p); }, vel_norm);
                            HDF5::write_dataset(g, "Velocities", filetype, H5T_NATIVE_DOUBLE, buffer.data(), n, NDIM);
                        }
                        if constexpr (FML::PARTICLE::has_get_id<T>()) {
                            std::vector<unsigned long long> ids(n);
                            for (size_t i = 0; i < n; i++)
                                ids[i] = (unsigned long long)FML::PARTICLE::GetID(sp.part[i]);
                            HDF5::write_dataset(g, "ParticleIDs", H5T_STD_U64LE, H5T_NATIVE_ULLONG, ids.data(), n, 1);
                        }
                    }
                }
            };

        } // namespace GADGET
    }     // namespace FILEUTILS
} // namespace FML

#endif
#endif

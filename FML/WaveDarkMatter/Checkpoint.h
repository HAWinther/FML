#ifndef FML_WAVE_CHECKPOINT_H
#define FML_WAVE_CHECKPOINT_H
#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
namespace FML::WAVEDM {
    namespace io_detail {
        inline void prefix_agrees(const std::string & prefix) {
            detail::require(!prefix.empty() && prefix.size() < 4096 && prefix.find('\0') == std::string::npos,
                            "[WaveDM] Invalid checkpoint prefix");
#ifdef USE_MPI
            std::array<char, 4096> root{};
            if (FML::ThisTask == 0)
                std::copy(prefix.begin(), prefix.end(), root.begin());
            MPI_Bcast(root.data(), int(root.size()), MPI_CHAR, 0, FML::communicator);
            detail::require(prefix == root.data(), "[WaveDM] Checkpoint prefixes differ between ranks");
#endif
        }
        inline std::string filename(const std::string & prefix, int rank) {
            return prefix + ".rank" + std::to_string(rank) + ".wave";
        }
        inline void hash_bytes(std::uint64_t & hash, const char * p, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                hash ^= static_cast<unsigned char>(p[i]);
                hash *= 1099511628211ull;
            }
        }
        inline std::array<char, 8> encode64(std::uint64_t bits) {
            std::array<char, 8> bytes{};
            for (int i = 0; i < 8; ++i)
                bytes[i] = char((bits >> (8 * i)) & 255);
            return bytes;
        }
        inline std::uint64_t decode64(const char * bytes) {
            std::uint64_t bits = 0;
            for (int i = 0; i < 8; ++i)
                bits |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
            return bits;
        }
        inline std::array<char, 8> encode(double value) {
            static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
            std::uint64_t bits;
            std::memcpy(&bits, &value, 8);
            return encode64(bits);
        }
        inline double decode(const char * bytes) {
            const auto bits = decode64(bytes);
            double x;
            std::memcpy(&x, &bits, 8);
            return x;
        }
        // The cells of a shard: x in [x0, x0 + nx), y in [y0, y0 + ny), all z, stored with z fastest, then y, then x.
        // Version 1 shards are x-slabs (y0 = 0, ny = n), version 2 shards any boxes (slabs or pencils)
        struct Box {
            ptrdiff_t x0{}, nx{}, y0{}, ny{};
        };
        inline size_t records(const Box & b, int n) { return size_t(b.nx) * size_t(b.ny) * n; }
        // The boxes must tile [0,n)^2
        inline bool tiles(const std::vector<Box> & boxes, int n) {
            size_t area = 0;
            for (auto & b : boxes) {
                if (b.nx < 0 || b.ny < 0 || b.x0 < 0 || b.y0 < 0 || b.x0 + b.nx > n || b.y0 + b.ny > n)
                    return false;
                area += size_t(b.nx) * size_t(b.ny);
            }
            if (area != size_t(n) * size_t(n))
                return false;
            for (size_t i = 0; i < boxes.size(); ++i)
                for (size_t j = i + 1; j < boxes.size(); ++j) {
                    const auto & a = boxes[i];
                    const auto & b = boxes[j];
                    if (a.nx * a.ny > 0 && b.nx * b.ny > 0 && a.x0 < b.x0 + b.nx && b.x0 < a.x0 + a.nx &&
                        a.y0 < b.y0 + b.ny && b.y0 < a.y0 + a.ny)
                        return false;
                }
            return true;
        }
        // Read the header line of a shard
        inline bool read_header(std::istream & in, int version, int n, const Box & box) {
            std::string magic;
            int v = 0, mesh = 0;
            Box b;
            if (!(in >> magic >> v >> mesh >> b.x0 >> b.nx))
                return false;
            if (v == 2 && !(in >> b.y0 >> b.ny))
                return false;
            if (v == 1)
                b.y0 = 0, b.ny = n;
            return in.get() == '\n' && magic == "FML_WAVE_SHARD" && v == version && mesh == n && b.x0 == box.x0 &&
                   b.nx == box.nx && b.y0 == box.y0 && b.ny == box.ny;
        }
        // Verify the complete shard in a bounded byte buffer on just one reader. No global field is gathered.
        inline bool verify(const std::string & name, int version, const Box & slab, int n) {
            std::ifstream in(name, std::ios::binary);
            if (!read_header(in, version, n, slab))
                return false;
            const auto offset = in.tellg();
            const auto size = records(slab, n) * 24;
            in.seekg(0, std::ios::end);
            if (in.tellg() != offset + std::streamoff(size) + std::streamoff(8))
                return false;
            in.seekg(offset);
            std::array<char, 65536> buffer{};
            std::uint64_t hash = 14695981039346656037ull;
            size_t left = size;
            while (left) {
                const auto count = std::min(left, buffer.size());
                if (!in.read(buffer.data(), std::streamsize(count)))
                    return false;
                hash_bytes(hash, buffer.data(), count);
                left -= count;
            }
            std::array<char, 8> stored{};
            return bool(in.read(stored.data(), 8)) && decode64(stored.data()) == hash;
        }
    } // namespace io_detail
    // New prefix only; shards are renamed first, manifest last. External density values are included. The shards are
    // the local boxes of the tasks (slabs or pencils); a checkpoint can be read with any number of tasks and grid type
    template <class Grid>
    void save_checkpoint(SolverT<Grid> & solver, const std::string & prefix) {
        io_detail::prefix_agrees(prefix);
        detail::require(sizeof(Real) <= 8, "[WaveDM] Checkpoint v1 supports float/double fields");
        const auto & wave = solver.wave();
        const size_t ncells = solver.local_size();
        const Complex * data = wave.get_real_grid();
        bool valid = wave.get_grid_status_real();
        for (size_t i = 0; i < ncells; ++i)
            valid &= std::isfinite(double(data[i].real())) && std::isfinite(double(data[i].imag()));
        detail::same(int(solver.has_external_density()), "[WaveDM] Checkpoint external-source flags differ");
        detail::require(valid, "[WaveDM] Checkpoint requires finite real-space wavefunction");
        const auto file = io_detail::filename(prefix, FML::ThisTask);
        std::ifstream exists(file), manifest(prefix + ".manifest");
        detail::require(!exists.good() && !manifest.good(), "[WaveDM] Checkpoint prefix exists");
        std::uint64_t hash = 14695981039346656037ull;
        std::ofstream out(file + ".tmp", std::ios::binary);
        const auto xr = wave.get_local_range(0), yr = wave.get_local_range(1);
        out << "FML_WAVE_SHARD 2 " << wave.get_nmesh() << ' ' << xr[0] << ' ' << xr[1] << ' ' << yr[0] << ' ' << yr[1]
            << '\n';
        for (size_t i = 0; i < ncells; ++i) {
            const auto z = data[i];
            const std::array<double, 3> fields{double(z.real()),
                                               double(z.imag()),
                                               solver.has_external_density() ? double(solver.external_density()[i]) :
                                                                               0};
            for (auto x : fields) {
                const auto bytes = io_detail::encode(x);
                out.write(bytes.data(), 8);
                io_detail::hash_bytes(hash, bytes.data(), 8);
            }
        }
        const auto checksum = io_detail::encode64(hash);
        out.write(checksum.data(), 8);
        out.close();
        detail::require(bool(out), "[WaveDM] Checkpoint shard write failed");
        detail::require(std::rename((file + ".tmp").c_str(), file.c_str()) == 0,
                        "[WaveDM] Checkpoint shard rename failed");
        long long box[4] = {xr[0], xr[1], yr[0], yr[1]};
        std::array<std::vector<long long>, 4> boxes;
        for (int k = 0; k < 4; ++k)
            boxes[k] = FML::GatherFromTasks(&box[k]);
        bool good = true;
        if (FML::ThisTask == 0) {
            std::ofstream meta(prefix + ".manifest.tmp");
            const auto p = solver.physics();
            meta << std::setprecision(17) << "FML_WAVE_CHECKPOINT 2 " << wave.get_nmesh() << ' ' << solver.boxsize() << ' '
                 << p.eta << ' ' << p.G << ' ' << solver.time() << ' ' << solver.scale_factor() << ' ' << solver.steps()
                 << ' ' << int(solver.has_external_density()) << ' ' << FML::NTasks << '\n';
            for (int r = 0; r < FML::NTasks; ++r)
                meta << boxes[0][r] << ' ' << boxes[1][r] << ' ' << boxes[2][r] << ' ' << boxes[3][r] << '\n';
            meta.close();
            good = bool(meta);
            if (good)
                good = std::rename((prefix + ".manifest.tmp").c_str(), (prefix + ".manifest").c_str()) == 0;
        }
        detail::require(good, "[WaveDM] Checkpoint manifest publication failed");
    }
    /// Load a checkpoint (written with any number of tasks, on slabs or pencils) onto a solver on grids of type Grid
    template <class Grid = FML::GRID::CFFTWGrid<3>>
    std::unique_ptr<SolverT<Grid>> load_checkpoint(const std::string & prefix) {
        io_detail::prefix_agrees(prefix);
        detail::require(sizeof(Real) <= 8, "[WaveDM] Checkpoint v1 supports float/double fields");
        std::ifstream meta(prefix + ".manifest");
        std::string magic, extra;
        int version = 0, n = 0, external = 0, ranks = 0;
        double box = 0, eta = 0, G = 0, time = 0, a = 0;
        std::uint64_t steps = 0;
        bool good = bool(meta >> magic >> version >> n >> box >> eta >> G >> time >> a >> steps >> external >> ranks);
        good &= magic == "FML_WAVE_CHECKPOINT" && (version == 1 || version == 2) && (external == 0 || external == 1) &&
                ranks > 0 &&
                ranks <= 1000000;
        detail::require(good, "[WaveDM] Missing or invalid checkpoint manifest");
        detail::same(n, "[WaveDM] Checkpoint meshes differ");
        detail::same(ranks, "[WaveDM] Checkpoint shard counts differ");
        detail::same(external, "[WaveDM] Checkpoint source flags differ");
        detail::same(steps, "[WaveDM] Checkpoint step counters differ");
        detail::same(version, "[WaveDM] Checkpoint versions differ");
        auto solver = std::make_unique<SolverT<Grid>>(n, box, Physics{eta, G}, time, a);
        solver->set_restart_steps(steps);
        std::vector<io_detail::Box> shards;
        for (int r = 0; good && r < ranks; ++r) {
            io_detail::Box b;
            if (!(meta >> b.x0 >> b.nx)) {
                good = false;
                break;
            }
            if (version == 2) {
                if (!(meta >> b.y0 >> b.ny))
                    good = false;
            } else {
                b.y0 = 0, b.ny = n;
            }
            shards.push_back(b);
        }
        good &= int(shards.size()) == ranks && io_detail::tiles(shards, n) && !(meta >> extra);
        detail::require(good, "[WaveDM] Invalid checkpoint partition");
        for (int r = 0; r < ranks; ++r) {
            long long b[4] = {shards[r].x0, shards[r].nx, shards[r].y0, shards[r].ny};
            for (auto x : b)
                detail::same(x, "[WaveDM] Checkpoint layouts differ");
            if (r % FML::NTasks == FML::ThisTask)
                good &= io_detail::verify(io_detail::filename(prefix, r), version, shards[r], n);
        }
        detail::require(good, "[WaveDM] Checkpoint checksum/header/length failure");
        auto & wave = solver->wave();
        Complex * data = wave.get_real_grid();
        const auto xr = wave.get_local_range(0), yr = wave.get_local_range(1);
        std::vector<Real> rho;
        if (external)
            rho.resize(solver->local_size());
        for (int r = 0; r < ranks; ++r) {
            const auto & s = shards[r];
            const auto xlo = std::max(xr[0], s.x0), xhi = std::min(xr[0] + xr[1], s.x0 + s.nx);
            const auto ylo = std::max(yr[0], s.y0), yhi = std::min(yr[0] + yr[1], s.y0 + s.ny);
            if (xlo >= xhi || ylo >= yhi)
                continue;
            std::ifstream in(io_detail::filename(prefix, r), std::ios::binary);
            std::getline(in, extra);
            const auto offset = in.tellg();
            std::array<char, 24> record{};
            // The rows (x, y) of the intersection: n records each
            for (auto x = xlo; x < xhi; ++x)
                for (auto y = ylo; y < yhi; ++y) {
                    in.seekg(offset + std::streamoff((size_t(x - s.x0) * s.ny + size_t(y - s.y0)) * n * 24));
                    const size_t first = (size_t(x - xr[0]) * size_t(yr[1]) + size_t(y - yr[0])) * n;
                    for (int z = 0; z < n; ++z) {
                        good &= bool(in.read(record.data(), 24));
                        const double re = io_detail::decode(record.data()), im = io_detail::decode(record.data() + 8),
                                     ext = io_detail::decode(record.data() + 16);
                        data[first + z] = Complex(Real(re), Real(im));
                        if (external)
                            rho[first + z] = Real(ext);
                        good &= std::isfinite(re) && std::isfinite(im) &&
                                std::isfinite(double(std::norm(data[first + z]))) && std::isfinite(ext) && ext >= 0;
                        if (!external)
                            good &= ext == 0;
                    }
                }
        }
        detail::require(good, "[WaveDM] Invalid or unrepresentable checkpoint samples");
        if (external)
            solver->set_external_density(rho);
        return solver;
    }
} // namespace FML::WAVEDM
#endif

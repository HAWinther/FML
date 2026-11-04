#ifndef FML_WAVE_CHECKPOINT_H
#define FML_WAVE_CHECKPOINT_H
#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
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
        struct Slab {
            ptrdiff_t start{}, nx{};
        };
        inline size_t records(const Slab & s, int n) { return size_t(s.nx) * size_t(n) * n; }
        // Verify the complete shard in a bounded byte buffer on just one reader. No global field is gathered.
        inline bool verify(const std::string & name, const Slab & slab, int n) {
            std::ifstream in(name, std::ios::binary);
            std::string magic;
            int version = 0, mesh = 0;
            ptrdiff_t start = 0, nx = 0;
            if (!(in >> magic >> version >> mesh >> start >> nx) || in.get() != '\n' || magic != "FML_WAVE_SHARD" ||
                version != 1 || mesh != n || start != slab.start || nx != slab.nx)
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
    // New prefix only; shards are renamed first, manifest last. External density values are included.
    inline void save_checkpoint(Solver & solver, const std::string & prefix) {
        io_detail::prefix_agrees(prefix);
        detail::require(sizeof(Real) <= 8, "[WaveDM] Checkpoint v1 supports float/double fields");
        const auto & wave = solver.wave();
        bool valid = !wave.spectral();
        for (size_t i = 0; i < wave.size(); ++i)
            valid &= std::isfinite(double(wave.data()[i].real())) && std::isfinite(double(wave.data()[i].imag()));
        detail::same(int(solver.has_external_density()), "[WaveDM] Checkpoint external-source flags differ");
        detail::require(valid, "[WaveDM] Checkpoint requires finite real-space wavefunction");
        const auto file = io_detail::filename(prefix, FML::ThisTask);
        std::ifstream exists(file), manifest(prefix + ".manifest");
        detail::require(!exists.good() && !manifest.good(), "[WaveDM] Checkpoint prefix exists");
        std::uint64_t hash = 14695981039346656037ull;
        std::ofstream out(file + ".tmp", std::ios::binary);
        out << "FML_WAVE_SHARD 1 " << wave.mesh() << ' ' << wave.local_start() << ' ' << wave.local_nx() << '\n';
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto z = wave.data()[i];
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
        auto start = static_cast<long long>(wave.local_start()), nx = static_cast<long long>(wave.local_nx());
        const auto starts = FML::GatherFromTasks(&start), widths = FML::GatherFromTasks(&nx);
        bool good = true;
        if (FML::ThisTask == 0) {
            std::ofstream meta(prefix + ".manifest.tmp");
            const auto p = solver.physics();
            meta << std::setprecision(17) << "FML_WAVE_CHECKPOINT 1 " << wave.mesh() << ' ' << solver.boxsize() << ' '
                 << p.eta << ' ' << p.G << ' ' << solver.time() << ' ' << solver.scale_factor() << ' ' << solver.steps()
                 << ' ' << int(solver.has_external_density()) << ' ' << FML::NTasks << '\n';
            for (int r = 0; r < FML::NTasks; ++r)
                meta << starts[r] << ' ' << widths[r] << '\n';
            meta.close();
            good = bool(meta);
            if (good)
                good = std::rename((prefix + ".manifest.tmp").c_str(), (prefix + ".manifest").c_str()) == 0;
        }
        detail::require(good, "[WaveDM] Checkpoint manifest publication failed");
    }
    inline std::unique_ptr<Solver> load_checkpoint(const std::string & prefix) {
        io_detail::prefix_agrees(prefix);
        detail::require(sizeof(Real) <= 8, "[WaveDM] Checkpoint v1 supports float/double fields");
        std::ifstream meta(prefix + ".manifest");
        std::string magic, extra;
        int version = 0, n = 0, external = 0, ranks = 0;
        double box = 0, eta = 0, G = 0, time = 0, a = 0;
        std::uint64_t steps = 0;
        bool good = bool(meta >> magic >> version >> n >> box >> eta >> G >> time >> a >> steps >> external >> ranks);
        good &= magic == "FML_WAVE_CHECKPOINT" && version == 1 && (external == 0 || external == 1) && ranks > 0 &&
                ranks <= 1000000;
        detail::require(good, "[WaveDM] Missing or invalid checkpoint manifest");
        detail::same(n, "[WaveDM] Checkpoint meshes differ");
        detail::same(ranks, "[WaveDM] Checkpoint shard counts differ");
        detail::same(external, "[WaveDM] Checkpoint source flags differ");
        detail::same(steps, "[WaveDM] Checkpoint step counters differ");
        auto solver = std::make_unique<Solver>(n, box, Physics{eta, G}, time, a);
        solver->set_restart_steps(steps);
        std::vector<io_detail::Slab> slabs;
        ptrdiff_t covered = 0;
        for (int r = 0; r < ranks; ++r) {
            io_detail::Slab s;
            if (!(meta >> s.start >> s.nx)) {
                good = false;
                break;
            }
            good &= s.nx >= 0 && s.nx <= n && (s.nx == 0 || (s.start == covered && s.start + s.nx <= n));
            if (s.nx > 0)
                covered += s.nx;
            slabs.push_back(s);
        }
        good &= covered == n && !(meta >> extra);
        detail::require(good, "[WaveDM] Invalid checkpoint slab partition");
        for (int r = 0; r < ranks; ++r) {
            long long start = slabs[r].start, nx = slabs[r].nx;
            detail::same(start, "[WaveDM] Checkpoint layouts differ");
            detail::same(nx, "[WaveDM] Checkpoint layouts differ");
            if (r % FML::NTasks == FML::ThisTask)
                good &= io_detail::verify(io_detail::filename(prefix, r), slabs[r], n);
        }
        detail::require(good, "[WaveDM] Checkpoint checksum/header/length failure");
        auto & wave = solver->wave();
        std::vector<Real> rho;
        if (external)
            rho.resize(wave.size());
        for (int r = 0; r < ranks; ++r) {
            const auto lo = std::max(wave.local_start(), slabs[r].start),
                       hi = std::min(wave.local_start() + wave.local_nx(), slabs[r].start + slabs[r].nx);
            if (lo >= hi)
                continue;
            std::ifstream in(io_detail::filename(prefix, r), std::ios::binary);
            std::getline(in, extra);
            const auto offset = in.tellg();
            in.seekg(offset + std::streamoff(size_t(lo - slabs[r].start) * n * n * 24));
            const size_t first = size_t(lo - wave.local_start()) * n * n, count = size_t(hi - lo) * n * n;
            std::array<char, 24> record{};
            for (size_t j = 0; j < count; ++j) {
                good &= bool(in.read(record.data(), 24));
                const double re = io_detail::decode(record.data()), im = io_detail::decode(record.data() + 8),
                             ext = io_detail::decode(record.data() + 16);
                wave.data()[first + j] = Complex(Real(re), Real(im));
                if (external)
                    rho[first + j] = Real(ext);
                good &= std::isfinite(re) && std::isfinite(im) &&
                        std::isfinite(double(std::norm(wave.data()[first + j]))) && std::isfinite(ext) && ext >= 0;
                if (!external)
                    good &= ext == 0;
            }
        }
        detail::require(good, "[WaveDM] Invalid or unrepresentable checkpoint samples");
        if (external)
            solver->set_external_density(rho);
        return solver;
    }
} // namespace FML::WAVEDM
#endif

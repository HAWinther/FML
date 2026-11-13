#ifndef FML_MCMC_HMC_CHECKPOINT_H
#define FML_MCMC_HMC_CHECKPOINT_H
#include <FML/MCMC/HMC.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <locale>
#include <iomanip>

namespace FML::MCMC::CHECKPOINT {
    using Bytes = std::vector<unsigned char>;
    inline std::uint64_t checksum(const Bytes & bytes) {
        std::uint64_t h = 14695981039346656037ULL;
        for (auto b : bytes) {
            h ^= b;
            h *= 1099511628211ULL;
        }
        return h; // Accidental-corruption detection, not authentication.
    }
    inline std::string build_tag() {
#ifdef __VERSION__
        std::string tag = std::string(__VERSION__);
#elif defined(_MSC_VER)
        std::string tag = "MSVC:" + std::to_string(_MSC_VER);
#else
        std::string tag = "unknown-compiler";
#endif
        tag += ":" + std::to_string(__cplusplus);
#ifdef _LIBCPP_VERSION
        tag += ":libc++:" + std::to_string(_LIBCPP_VERSION);
#endif
#ifdef __GLIBCXX__
        tag += ":libstdc++:" + std::to_string(__GLIBCXX__);
#endif
        return tag + ":size_t:" + std::to_string(sizeof(size_t));
    }
    struct Encoder {
        Bytes bytes;
        void integer(std::uint64_t x) {
            for (int i = 0; i < 8; ++i) {
                bytes.push_back(static_cast<unsigned char>(x & 255));
                x >>= 8;
            }
        }
        void real(double x) {
            static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
            std::uint64_t bits;
            std::memcpy(&bits, &x, 8);
            integer(bits);
        }
        void text(const std::string & x) {
            integer(x.size());
            bytes.insert(bytes.end(), x.begin(), x.end());
        }
        void vector(const std::vector<double> & x) {
            integer(x.size());
            for (double v : x)
                real(v);
        }
    };
    struct Decoder {
        const Bytes & bytes;
        size_t offset = 0;
        std::uint64_t integer() {
            if (bytes.size() - offset < 8)
                throw std::runtime_error("[HMC checkpoint] Truncated integer");
            std::uint64_t x = 0;
            for (int i = 0; i < 8; ++i)
                x |= std::uint64_t(bytes[offset++]) << (8 * i);
            return x;
        }
        size_t size(size_t limit) {
            const auto n = integer();
            if (n > limit)
                throw std::runtime_error("[HMC checkpoint] Invalid size/count");
            return size_t(n);
        }
        bool boolean() { return size(1) != 0; }
        double real() {
            auto bits = integer();
            double x;
            std::memcpy(&x, &bits, 8);
            return x;
        }
        std::string text(size_t limit = 16384) {
            const size_t n = size(limit);
            if (n > bytes.size() - offset)
                throw std::runtime_error("[HMC checkpoint] Truncated string");
            std::string x(bytes.begin() + offset, bytes.begin() + offset + n);
            offset += n;
            return x;
        }
        std::vector<double> vector(size_t expected) {
            if (size(expected) != expected || expected > (bytes.size() - offset) / 8)
                throw std::runtime_error("[HMC checkpoint] Wrong vector dimension");
            std::vector<double> x(expected);
            for (auto & v : x)
                v = real();
            return x;
        }
        void end() const {
            if (offset != bytes.size())
                throw std::runtime_error("[HMC checkpoint] Trailing payload");
        }
    };
    template <class T>
    std::string stream_text(const T & x) {
        std::ostringstream s;
        s.imbue(std::locale::classic());
        s << std::setprecision(std::numeric_limits<double>::max_digits10) << x;
        if (!s)
            throw std::runtime_error("[HMC checkpoint] RNG serialization failed");
        return s.str();
    }
    template <class T>
    void parse_stream(const std::string & text, T & x) {
        std::istringstream s(text);
        s.imbue(std::locale::classic());
        if (!(s >> x))
            throw std::runtime_error("[HMC checkpoint] Invalid RNG state");
        s >> std::ws;
        if (!s.eof())
            throw std::runtime_error("[HMC checkpoint] Trailing RNG state");
    }
    inline void options(Encoder & e, const HMCOptions & o) {
        e.integer(o.warmup);
        e.integer(o.samples);
        e.integer(o.n_leapfrog);
        e.integer(o.jitter);
        e.real(o.step_size);
        e.integer(o.adapt_step_size);
        e.real(o.target_acceptance);
        e.integer(o.adapt_mass);
        e.vector(o.lower);
        e.vector(o.upper);
        e.real(o.divergence_threshold);
        e.integer(o.seed);
        e.integer(o.verbose_every);
    }
    inline HMCOptions options(Decoder & d, size_t np) {
        HMCOptions o;
        o.warmup = int(d.size(INT_MAX));
        o.samples = int(d.size(INT_MAX));
        o.n_leapfrog = int(d.size(INT_MAX));
        o.jitter = d.boolean();
        o.step_size = d.real();
        o.adapt_step_size = d.boolean();
        o.target_acceptance = d.real();
        o.adapt_mass = d.boolean();
        o.lower = d.vector(np);
        o.upper = d.vector(np);
        o.divergence_threshold = d.real();
        o.seed = d.integer();
        o.verbose_every = int(d.size(INT_MAX));
        return o;
    }
    inline void state(Encoder & e, const HMCState & s, bool fields = true) {
        e.integer(s.initialized);
        e.integer(s.n_custom);
        options(e, s.options);
        if (fields) {
            e.vector(s.position);
            e.vector(s.gradient);
        } else {
            e.vector(std::vector<double>(s.position.begin() + s.n_custom, s.position.end()));
            e.vector(std::vector<double>(s.gradient.begin() + s.n_custom, s.gradient.end()));
        }
        e.vector(s.diagonal);
        e.vector(s.mean);
        e.vector(s.m2);
        e.real(s.potential);
        e.real(s.step_size);
        e.real(s.accept_sum);
        e.integer(s.next_iteration);
        e.integer(s.divergences);
        e.integer(s.count);
        e.integer(s.window);
        const auto & a = s.adaptation;
        for (double x : {a.mu, a.log_eps, a.log_eps_bar, a.h_bar, a.gamma, a.t0, a.kappa})
            e.real(x);
        e.integer(a.t);
        e.text(stream_text(s.rng));
        e.text(stream_text(s.gauss));
        e.text(stream_text(s.uniform));
    }
    inline HMCState state(Decoder & d, size_t n, size_t nc) {
        if (nc > n)
            throw std::runtime_error("[HMC checkpoint] Invalid custom dimension");
        HMCState s;
        s.initialized = d.boolean();
        s.n_custom = d.size(n);
        if (!s.initialized || s.n_custom != nc)
            throw std::runtime_error("[HMC checkpoint] Invalid sampler layout");
        const size_t np = n - nc;
        s.options = options(d, np);
        s.position = d.vector(n);
        s.gradient = d.vector(n);
        s.diagonal = d.vector(np);
        s.mean = d.vector(np);
        s.m2 = d.vector(np);
        s.potential = d.real();
        s.step_size = d.real();
        s.accept_sum = d.real();
        s.next_iteration = int(d.size(INT_MAX));
        s.divergences = long(d.size(LONG_MAX));
        s.count = long(d.size(LONG_MAX));
        s.window = d.size(INT_MAX);
        auto & a = s.adaptation;
        a.mu = d.real();
        a.log_eps = d.real();
        a.log_eps_bar = d.real();
        a.h_bar = d.real();
        a.gamma = d.real();
        a.t0 = d.real();
        a.kappa = d.real();
        a.t = long(d.size(LONG_MAX));
        parse_stream(d.text(), s.rng);
        parse_stream(d.text(), s.gauss);
        parse_stream(d.text(), s.uniform);
        if (s.gauss.mean() != 0 || s.gauss.stddev() != 1 || s.uniform.a() != 0 || s.uniform.b() != 1)
            throw std::runtime_error("[HMC checkpoint] Wrong RNG distribution parameters");
        return s;
    }
    inline size_t size_bound(size_t n, size_t np) {
        if (n > (std::numeric_limits<size_t>::max() - 131072) / 16 ||
            np > (std::numeric_limits<size_t>::max() - 131072 - 16 * n) / 128)
            throw std::overflow_error("[HMC checkpoint] State size overflow");
        return 131072 + 16 * n + 128 * np;
    }
    inline void write_file(const std::string & path, const Bytes & payload) {
        if (payload.size() > size_t(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("[HMC checkpoint] Output exceeds stream size");
        Encoder header;
        header.text("FML checkpoint 1");
        header.integer(payload.size());
        Encoder tail;
        tail.integer(checksum(payload));
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f)
            throw std::runtime_error("[HMC checkpoint] Cannot open output: " + path);
        f.write(reinterpret_cast<const char *>(header.bytes.data()), std::streamsize(header.bytes.size()));
        f.write(reinterpret_cast<const char *>(payload.data()), std::streamsize(payload.size()));
        f.write(reinterpret_cast<const char *>(tail.bytes.data()), std::streamsize(tail.bytes.size()));
        f.close();
        if (!f)
            throw std::runtime_error("[HMC checkpoint] Write/close failed: " + path);
    }
    inline Bytes read_file(const std::string & path, size_t limit) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f)
            throw std::runtime_error("[HMC checkpoint] Cannot open input: " + path);
        const auto end = f.tellg();
        if (end < 0 || std::uintmax_t(end) > std::uintmax_t(limit) + 64)
            throw std::runtime_error("[HMC checkpoint] Oversized file");
        Bytes file(static_cast<size_t>(end));
        f.seekg(0);
        f.read(reinterpret_cast<char *>(file.data()), std::streamsize(file.size()));
        if (!f)
            throw std::runtime_error("[HMC checkpoint] Read failed");
        Decoder d{file};
        if (d.text(32) != "FML checkpoint 1")
            throw std::runtime_error("[HMC checkpoint] Unknown format");
        const size_t n = d.size(limit);
        if (file.size() - d.offset < 8 || n != file.size() - d.offset - 8)
            throw std::runtime_error("[HMC checkpoint] Truncated/trailing file");
        Bytes payload(file.begin() + d.offset, file.begin() + d.offset + n);
        d.offset += n;
        if (checksum(payload) != d.integer())
            throw std::runtime_error("[HMC checkpoint] Checksum mismatch");
        d.end();
        return payload;
    }
    inline void save(const std::string & path, const HMCState & s) {
        Encoder e;
        e.text(build_tag());
        state(e, s);
        write_file(path, e.bytes);
    }
    inline HMCState load(const std::string & path, size_t n, size_t nc) {
        if (nc > n)
            throw std::runtime_error("[HMC checkpoint] Invalid custom dimension");
        const auto bytes = read_file(path, size_bound(n, n - nc));
        Decoder d{bytes};
        if (d.text() != build_tag())
            throw std::runtime_error("[HMC checkpoint] Build/library mismatch");
        auto s = state(d, n, nc);
        d.end();
        return s;
    }
} // namespace FML::MCMC::CHECKPOINT
#endif

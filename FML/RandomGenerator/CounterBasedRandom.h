#ifndef FML_COUNTER_BASED_RANDOM_HEADER
#define FML_COUNTER_BASED_RANDOM_HEADER
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace FML::RANDOM::COUNTER_BASED {
    /// Independent implementation of Philox4x32-10. Constants and known-answer
    /// tests follow DEShawResearch/random123 (tests/kat_vectors).
    inline std::array<uint32_t, 4> philox4x32(std::array<uint32_t, 4> counter, std::array<uint32_t, 2> key) {
        for (int round = 0; round < 10; round++) {
            const uint64_t p0 = uint64_t(0xD2511F53u) * counter[0];
            const uint64_t p1 = uint64_t(0xCD9E8D57u) * counter[2];
            counter = {uint32_t(p1 >> 32) ^ counter[1] ^ key[0],
                       uint32_t(p1),
                       uint32_t(p0 >> 32) ^ counter[3] ^ key[1],
                       uint32_t(p0)};
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
        return counter;
    }
    /// A midpoint on a 52-bit uniform grid, strictly inside (0,1).
    inline double uniform_open(uint32_t a, uint32_t b) {
        return (double((uint64_t(a) << 20) | (b >> 12)) + 0.5) * 0x1p-52;
    }

    /// Independent stream keyed by (seed, object ID, purpose, user stream).
    /// Integer Philox mapping is portable; transcendental rounding can vary across libm builds.
    class Stream {
        std::uint64_t object, block = 0;
        std::array<std::uint32_t, 2> key;
        std::array<std::uint32_t, 4> words{};
        int cursor = 4;

      public:
        Stream(std::uint64_t seed, std::uint64_t object_id, std::uint32_t purpose = 0, std::uint32_t stream = 0)
            : object(object_id), key{std::uint32_t(seed) ^ purpose, std::uint32_t(seed >> 32) ^ stream} {}
        std::uint32_t next32() {
            if (cursor == 4) {
                if (block == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Philox stream exhausted");
                words = philox4x32({std::uint32_t(object),
                                    std::uint32_t(object >> 32),
                                    std::uint32_t(block),
                                    std::uint32_t(block >> 32)},
                                   key);
                ++block;
                cursor = 0;
            }
            return words[cursor++];
        }
        std::uint64_t next64() {
            const auto high = next32();
            return (std::uint64_t(high) << 32) | next32();
        }
        double uniform() {
            const auto a = next32();
            const auto b = next32();
            return uniform_open(a, b);
        }
        std::uint64_t bounded(std::uint64_t n) {
            if (!n)
                throw std::invalid_argument("Empty integer sampling range");
            const std::uint64_t threshold = (std::uint64_t(0) - n) % n;
            while (true) {
                const auto value = next64();
                if (value >= threshold)
                    return value % n;
            }
        }
        double normal() {
            const double radius = std::sqrt(-2 * std::log(uniform()));
            return radius * std::cos(6.283185307179586476925286766559 * uniform());
        }
    };
} // namespace FML::RANDOM::COUNTER_BASED
#endif

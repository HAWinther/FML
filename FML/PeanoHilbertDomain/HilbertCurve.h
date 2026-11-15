#ifndef HILBERTCURVE_HEADER
#define HILBERTCURVE_HEADER

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace GRID {

        //==========================================================================================
        /// Peano-Hilbert keys in NDIM dimensions using Skilling's transpose algorithm: the one key implementation of
        /// FML (PeanoHilbertDomain, Global/Domain.h and AdaptiveMPIGrid use it)
        /// (J. Skilling, "Programming the Hilbert curve", AIP Conf. Proc. 707, 381 (2004)).
        ///
        /// Integer coordinates have LMAX bits per axis and the key has NDIM*LMAX bits, so it
        /// fits in 64 bits (2D: LMAX = 31, 3D: LMAX = 21).
        ///
        /// Hierarchical property: the top NDIM*l bits of the key only depend on the top l bits of
        /// each coordinate. A cell at level l therefore covers the contiguous, aligned key range
        /// [key, key + 2^(NDIM*(LMAX-l))) and its descendants are contiguous along the curve.
        //==========================================================================================
        template <int NDIM>
        struct HilbertCurve {
            static_assert(NDIM >= 2 and NDIM <= 6, "HilbertCurve: NDIM must be in [2,6]");

            using key_type = uint64_t;
            using coord_type = uint32_t;
            using Coord = std::array<coord_type, NDIM>;

            /// Maximum refinement depth (bits per axis)
            static constexpr int LMAX = (63 / NDIM) < 31 ? (63 / NDIM) : 31;

            /// Integer coordinates (bits bits per axis) -> Hilbert key
            static key_type key_from_coord(Coord X, int bits = LMAX) {
                if (bits == LMAX) {
                    // Table driven (same keys, about 4x faster in 3D)
                    const Tables & T = tables();
                    int s = 0;
                    key_type key = 0;
                    for (int l = LMAX - 1; l >= 0; l--) {
                        int o = 0;
                        for (int i = 0; i < NDIM; i++)
                            o |= int((X[i] >> l) & 1u) << i;
                        key = (key << NDIM) | T.digit[s * NCH + o];
                        s = T.next[s * NCH + o];
                    }
                    return key;
                }
                axes_to_transpose(X, bits);
                return interleave(X, bits);
            }

            /// The key of the cell at refinement level `level` (0..LMAX) containing X: the key of its first point,
            /// i.e. key_from_coord(X) with the bits below the level cleared. Only the top `level` levels are
            /// computed, so it is cheaper than the full key for coarse cells
            static key_type cell_key(const Coord & X, int level) {
                const Tables & T = tables();
                int s = 0;
                key_type key = 0;
                for (int l = LMAX - 1; l >= LMAX - level; l--) {
                    int o = 0;
                    for (int i = 0; i < NDIM; i++)
                        o |= int((X[i] >> l) & 1u) << i;
                    key = (key << NDIM) | T.digit[s * NCH + o];
                    s = T.next[s * NCH + o];
                }
                return level >= LMAX ? key : key << (NDIM * (LMAX - level));
            }

            /// Hilbert key -> integer coordinates (bits bits per axis)
            static Coord coord_from_key(key_type key, int bits = LMAX) {
                if (bits == LMAX) {
                    const Tables & T = tables();
                    int s = 0;
                    Coord X{};
                    for (int l = LMAX - 1; l >= 0; l--) {
                        const int dig = int((key >> (NDIM * l)) & key_type(NCH - 1));
                        const int o = T.octant[s * NCH + dig];
                        for (int i = 0; i < NDIM; i++)
                            X[i] |= coord_type((o >> i) & 1) << l;
                        s = T.next[s * NCH + o];
                    }
                    return X;
                }
                Coord X = deinterleave(key, bits);
                transpose_to_axes(X, bits);
                return X;
            }

          private:
            static constexpr int NCH = 1 << NDIM;

            // The curve (at LMAX bits) as a finite state machine: a sub-cube in state s maps the child in octant o
            // (bit i of o = the bit of axis i) to the key digit digit[s][o] and the child is in state next[s][o];
            // octant[s][digit] is the inverse. Derived once from the bitwise (Skilling) transform below, so the keys
            // are the same by construction: the state of a sub-cube is the order in which it visits its children,
            // which determines its orientation and hence the states of its children
            struct Tables {
                std::vector<uint16_t> digit, next, octant;
            };
            static const Tables & tables() {
                static const Tables T = make_tables();
                return T;
            }
            static Tables make_tables() {
                const int b = LMAX;
                auto order = [&](const Coord & anchor, int d) {
                    std::vector<uint16_t> m(NCH);
                    for (int o = 0; o < NCH; o++) {
                        Coord x = anchor;
                        for (int i = 0; i < NDIM; i++)
                            if ((o >> i) & 1)
                                x[i] |= coord_type(1) << (b - 1 - d);
                        Coord y = x;
                        axes_to_transpose(y, b);
                        m[o] = uint16_t((interleave(y, b) >> (NDIM * (b - 1 - d))) & key_type(NCH - 1));
                    }
                    return m;
                };
                std::vector<std::vector<uint16_t>> orders;
                std::vector<std::pair<Coord, int>> rep;
                auto state_of = [&](const Coord & anchor, int d) {
                    auto m = order(anchor, d);
                    for (size_t s = 0; s < orders.size(); s++)
                        if (orders[s] == m)
                            return int(s);
                    orders.push_back(m);
                    rep.push_back({anchor, d});
                    return int(orders.size()) - 1;
                };
                state_of(Coord{}, 0);
                std::vector<std::vector<uint16_t>> nxt;
                for (size_t s = 0; s < orders.size(); s++) {
                    const auto [a, d] = rep[s];
                    nxt.emplace_back(NCH);
                    for (int o = 0; o < NCH; o++) {
                        Coord c = a;
                        for (int i = 0; i < NDIM; i++)
                            if ((o >> i) & 1)
                                c[i] |= coord_type(1) << (b - 1 - d);
                        nxt[s][o] = uint16_t(state_of(c, d + 1));
                    }
                }
                Tables T;
                const size_t ns = orders.size();
                T.digit.resize(ns * NCH);
                T.next.resize(ns * NCH);
                T.octant.resize(ns * NCH);
                for (size_t s = 0; s < ns; s++)
                    for (int o = 0; o < NCH; o++) {
                        T.digit[s * NCH + o] = orders[s][o];
                        T.next[s * NCH + o] = nxt[s][o];
                        T.octant[s * NCH + orders[s][o]] = uint16_t(o);
                    }
                return T;
            }

            static void axes_to_transpose(Coord & X, int b) {
                const coord_type M = coord_type(1) << (b - 1);
                coord_type P, Q, t;
                // Inverse undo
                for (Q = M; Q > 1; Q >>= 1) {
                    P = Q - 1;
                    for (int i = 0; i < NDIM; i++) {
                        if (X[i] & Q) {
                            X[0] ^= P;
                        } else {
                            t = (X[0] ^ X[i]) & P;
                            X[0] ^= t;
                            X[i] ^= t;
                        }
                    }
                }
                // Gray encode
                for (int i = 1; i < NDIM; i++)
                    X[i] ^= X[i - 1];
                t = 0;
                for (Q = M; Q > 1; Q >>= 1)
                    if (X[NDIM - 1] & Q)
                        t ^= Q - 1;
                for (int i = 0; i < NDIM; i++)
                    X[i] ^= t;
            }

            static void transpose_to_axes(Coord & X, int b) {
                const uint64_t N = uint64_t(1) << b;
                coord_type P, t;
                // Gray decode by H ^ (H/2)
                t = X[NDIM - 1] >> 1;
                for (int i = NDIM - 1; i > 0; i--)
                    X[i] ^= X[i - 1];
                X[0] ^= t;
                // Undo excess work
                for (uint64_t Q = 2; Q != N; Q <<= 1) {
                    P = coord_type(Q - 1);
                    for (int i = NDIM - 1; i >= 0; i--) {
                        if (X[i] & coord_type(Q)) {
                            X[0] ^= P;
                        } else {
                            t = (X[0] ^ X[i]) & P;
                            X[0] ^= t;
                            X[i] ^= t;
                        }
                    }
                }
            }

            // Transposed form -> key. Most significant bit first: X[0]_b, X[1]_b, ..., X[0]_{b-1}, ...
            static key_type interleave(const Coord & X, int b) {
                key_type key = 0;
                for (int bit = b - 1; bit >= 0; bit--)
                    for (int i = 0; i < NDIM; i++)
                        key = (key << 1) | ((X[i] >> bit) & 1u);
                return key;
            }

            static Coord deinterleave(key_type key, int b) {
                Coord X{};
                int shift = NDIM * b - 1;
                for (int bit = b - 1; bit >= 0; bit--)
                    for (int i = 0; i < NDIM; i++, shift--)
                        X[i] |= coord_type((key >> shift) & 1u) << bit;
                return X;
            }
        };
    } // namespace GRID
} // namespace FML

// =====================================================================================================================
// 3D Peano-Hilbert keys of positions and grid cells (HilbertCurve<3>, checked against the resolution) for
// PeanoHilbertDomain (Global/Domain.h) and DistributedGrid3D
// =====================================================================================================================
namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT {

    using Key = uint64_t;
    using Position = std::array<double, 3>;

    /// Return the number of usable bits per coordinate for a 64-bit key.
    constexpr unsigned max_coordinate_bits() { return 21; }

    namespace detail {
        inline void validate_bits(unsigned bits) {
            if (bits == 0 || bits > max_coordinate_bits())
                throw std::invalid_argument("Peano-Hilbert bits must be in [1, 21]");
        }

        // The keys are those of HilbertCurve<3> (HilbertCurve.h: Skilling's transpose algorithm, the one key
        // implementation of FML), with the coordinates and key checked against the resolution
        inline Key key_from_integer_coordinates(const std::array<uint32_t, 3> & x, unsigned bits) {
            validate_bits(bits);
            const uint32_t limit = uint32_t(1) << bits;
            for (auto axis : x)
                if (axis >= limit)
                    throw std::out_of_range("Peano-Hilbert integer coordinate is outside its grid");
            return FML::GRID::HilbertCurve<3>::key_from_coord(x, int(bits));
        }

        inline std::array<uint32_t, 3> integer_coordinates_from_key(Key key, unsigned bits) {
            validate_bits(bits);
            if (key >= (Key(1) << (3 * bits)))
                throw std::out_of_range("Peano-Hilbert key is outside its configured resolution");
            return FML::GRID::HilbertCurve<3>::coord_from_key(key, int(bits));
        }
    } // namespace detail

    /// Map a point in the unit periodic cube [0,1)^3 to a Hilbert key.
    inline Key key_from_position(const Position & position, unsigned bits) {
        detail::validate_bits(bits);
        const uint32_t ngrid = uint32_t(1) << bits;
        std::array<uint32_t, 3> cell{};
        for (int axis = 0; axis < 3; ++axis) {
            const double x = position[axis];
            if (!std::isfinite(x) || x < 0.0 || x >= 1.0)
                throw std::out_of_range("Peano-Hilbert positions must be finite and in [0,1)");
            cell[axis] = std::min(uint32_t(x * ngrid), ngrid - 1);
        }
        return detail::key_from_integer_coordinates(cell, bits);
    }

    /// Map an integer grid coordinate to a Hilbert key.
    inline Key key_from_grid_coordinate(const std::array<uint32_t, 3> & coordinate, unsigned bits) {
        return detail::key_from_integer_coordinates(coordinate, bits);
    }

    /// Inverse of key_from_grid_coordinate for the same bit resolution.
    inline std::array<uint32_t, 3> grid_coordinate_from_key(Key key, unsigned bits) {
        return detail::integer_coordinates_from_key(key, bits);
    }

} // namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT

#endif

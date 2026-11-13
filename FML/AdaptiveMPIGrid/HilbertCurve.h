#ifndef HILBERTCURVE_HEADER
#define HILBERTCURVE_HEADER

#include <array>
#include <cstdint>

namespace FML {
    namespace GRID {

        //==========================================================================================
        /// Peano-Hilbert keys in NDIM dimensions using Skilling's transpose algorithm
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
                axes_to_transpose(X, bits);
                return interleave(X, bits);
            }

            /// Hilbert key -> integer coordinates (bits bits per axis)
            static Coord coord_from_key(key_type key, int bits = LMAX) {
                Coord X = deinterleave(key, bits);
                transpose_to_axes(X, bits);
                return X;
            }

          private:
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

#endif

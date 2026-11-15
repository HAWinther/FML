#ifndef FML_COMPUTEPOWERSPECTRA_SHELLOVERLAP_H
#define FML_COMPUTEPOWERSPECTRA_SHELLOVERLAP_H

// =====================================================================================================================
// The exact overlap between a grid cell and a spherical shell (or a ball), in 1, 2 and 3 dimensions.
//
// Used for the polyspectrum shells (PolyspectrumBinning::shell_overlap_weights): a Fourier cell gets the weight of the
// fraction of its volume inside the shell [k_low, k_high) instead of 0 or 1 from where its centre is. The shells are
// then truly spherical and continuous, and a measured value belongs to the exact k of its shell. The idea ("shell
// anti-aliasing") and the 3D primitive below follow CONCEPT (J. Dakin, S. Hannestad, T. Tram), src/analysis.py,
// get_bispec_overlap_cellshell and get_bispec_overlap_blockball_indefinite:
// https://github.com/jmd-dk/concept/blob/ab6d8d7d9a7c9e5010a206bbea3b22d0f15c4608/src/analysis.py#L2619 (GPL-3.0)
//
//   box_ball_overlap<N>(lo, hi, r)          volume of the box [lo, hi] inside the ball |x| < r (centre at 0)
//   cell_shell_overlap<N>(n, r_low, r_high) fraction of the unit cell centred at n inside r_low <= |x| < r_high
//
// The box is split along the coordinate planes into boxes in the positive orthant, and each of those is the
// inclusion-exclusion sum over its corners of the orthant volume W(a) = vol{x > a, |x| < r} (a >= 0): 1D r - a; 2D the
// integral of the circle; 3D the integral over x of the 2D area, with the closed-form primitive from CONCEPT.
// =====================================================================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace SHELLOVERLAP {
            namespace detail {

                /// 2D: area of {x > a, y > b, x^2 + y^2 < r^2} for a, b >= 0
                inline double orthant_2d(double a, double b, double r) {
                    const double r2 = r * r;
                    if (a * a + b * b >= r2)
                        return 0.0;
                    // The primitive of sqrt(r^2 - x^2)
                    auto S = [r2](double x) {
                        const double s = std::sqrt(std::max(0.0, r2 - x * x));
                        return 0.5 * (x * s + r2 * std::atan2(x, s));
                    };
                    const double xm = std::sqrt(r2 - b * b);
                    return S(xm) - S(a) - b * (xm - a);
                }

                /// 3D: the primitive in x of the area of {y > b, z > c, y^2 + z^2 < r^2 - x^2} (b, c >= 0), valid for
                /// 0 <= x <= sqrt(r^2 - b^2 - c^2). Ported from CONCEPT, get_bispec_overlap_blockball_indefinite (see
                /// the top of this file)
                inline double orthant_3d_primitive(double x, double b, double c, double r) {
                    const double r2 = r * r, b2 = b * b, c2 = c * c;
                    x = std::min(x, r);
                    const double x2 = x * x, x4 = x2 * x2;
                    double sqrt_y = r2 - x2 - b2;
                    sqrt_y = std::sqrt(sqrt_y > 0 ? sqrt_y : 0.0);
                    double sqrt_z = r2 - x2 - c2;
                    sqrt_z = std::sqrt(sqrt_z > 0 ? sqrt_z : 0.0);
                    const double atan_xy = std::atan2(x, sqrt_y);
                    const double atan_xz = std::atan2(x, sqrt_z);
                    const double atan_yz = std::atan2(sqrt_y * sqrt_z - b * c, b * sqrt_z + c * sqrt_y);
                    double xx = x4 * b2 * c2 +
                                r2 * (x4 * (b2 + c2) + 2 * x2 * b * (b * c2 - 2 * c * sqrt_y * sqrt_z) +
                                      r2 * (x4 + b2 * c2 + r2 * (-2 * x2 - (b2 + c2) + r2)));
                    double yy = 2 * x * r *
                                (-x2 * (b * c2 * sqrt_y + b2 * c * sqrt_z) +
                                 r2 * (b * sqrt_y * (r2 - x2 - c2) + c * sqrt_z * (r2 - x2 - b2)));
                    // atan2(0, 0) on the boundaries: the limits taken as in CONCEPT (relative to the scale r^8 here)
                    const double eps = 1e-4 * std::pow(std::max(r, 1e-300), 8) * 1e-12;
                    const bool xx_zero = std::fabs(xx) < eps, yy_zero = std::fabs(yy) < eps;
                    if (xx_zero)
                        xx = -1e-100;
                    if (yy_zero)
                        yy = 1e-200;
                    const double atan_xyz = std::atan2(yy, xx);
                    return x * b * c - x * (b * sqrt_y + c * sqrt_z) / 3.0 +
                           ((-b * (3 * r2 - b2) * atan_xy) - c * (3 * r2 - c2) * atan_xz +
                            x * (3 * r2 - x2) * atan_yz + r * r2 * atan_xyz) /
                               6.0;
                }

                /// 3D: volume of {x > a, y > b, z > c, |x| < r} for a, b, c >= 0
                inline double orthant_3d(double a, double b, double c, double r) {
                    const double s2 = r * r - b * b - c * c;
                    if (a * a >= s2)
                        return 0.0;
                    const double xm = std::sqrt(s2);
                    return std::max(0.0, orthant_3d_primitive(xm, b, c, r) - orthant_3d_primitive(a, b, c, r));
                }

                template <int N>
                double orthant(const std::array<double, N> & a, double r) {
                    if constexpr (N == 1)
                        return std::max(0.0, r - a[0]);
                    else if constexpr (N == 2)
                        return orthant_2d(a[0], a[1], r);
                    else
                        return orthant_3d(a[0], a[1], a[2], r);
                }

                /// A box in the positive orthant: inclusion-exclusion over the corners
                template <int N>
                double positive_box_ball(const std::array<double, N> & lo, const std::array<double, N> & hi, double r) {
                    double r2 = 0.0;
                    for (int d = 0; d < N; d++)
                        r2 += lo[d] * lo[d];
                    if (r2 >= r * r)
                        return 0.0;
                    double h2 = 0.0, volume = 1.0;
                    for (int d = 0; d < N; d++) {
                        h2 += hi[d] * hi[d];
                        volume *= hi[d] - lo[d];
                    }
                    if (h2 <= r * r)
                        return volume;
                    double sum = 0.0;
                    for (int mask = 0; mask < (1 << N); mask++) {
                        std::array<double, N> corner;
                        int sign = 1;
                        for (int d = 0; d < N; d++) {
                            corner[d] = (mask >> d) & 1 ? hi[d] : lo[d];
                            if ((mask >> d) & 1)
                                sign = -sign;
                        }
                        sum += sign * orthant<N>(corner, r);
                    }
                    return std::min(volume, std::max(0.0, sum));
                }

                /// Split along the coordinate planes and mirror into the positive orthant
                template <int N>
                double split_box_ball(std::array<double, N> lo, std::array<double, N> hi, double r, int axis) {
                    if (axis == N)
                        return positive_box_ball<N>(lo, hi, r);
                    if (hi[axis] <= 0.0) {
                        const double l = -hi[axis], h = -lo[axis];
                        lo[axis] = l;
                        hi[axis] = h;
                    }
                    if (lo[axis] < 0.0) {
                        auto lo2 = lo, hi2 = hi;
                        hi2[axis] = -lo[axis];
                        lo2[axis] = 0.0;
                        lo[axis] = 0.0;
                        return split_box_ball<N>(lo, hi, r, axis + 1) + split_box_ball<N>(lo2, hi2, r, axis + 1);
                    }
                    return split_box_ball<N>(lo, hi, r, axis + 1);
                }
            } // namespace detail

            /// The volume (length, area) of the box [lo, hi] inside the ball |x| < r centred at the origin. N = 1, 2, 3
            template <int N>
            double box_ball_overlap(const std::array<double, N> & lo, const std::array<double, N> & hi, double r) {
                static_assert(N >= 1 and N <= 3, "[box_ball_overlap] Only for N = 1, 2, 3");
                if (not(r > 0.0))
                    return 0.0;
                return detail::split_box_ball<N>(lo, hi, r, 0);
            }

            /// The fraction of the unit cell centred at n (in units of the cell size) inside the shell
            /// r_low <= |x| < r_high (same units). Cells far from the shell edges are 0 or 1 without any calculation
            template <int N>
            double cell_shell_overlap(const std::array<double, N> & n, double r_low, double r_high) {
                double d2 = 0.0;
                for (int d = 0; d < N; d++)
                    d2 += n[d] * n[d];
                const double d = std::sqrt(d2), half_diagonal = 0.5 * std::sqrt(double(N));
                if (d + half_diagonal <= r_low or d - half_diagonal >= r_high)
                    return 0.0;
                if (d - half_diagonal >= r_low and d + half_diagonal <= r_high)
                    return 1.0;
                std::array<double, N> lo, hi;
                for (int i = 0; i < N; i++) {
                    lo[i] = n[i] - 0.5;
                    hi[i] = n[i] + 0.5;
                }
                const double outer = d - half_diagonal >= r_high ? 0.0 : box_ball_overlap<N>(lo, hi, r_high);
                const double inner = d + half_diagonal <= r_low ? 0.0 : box_ball_overlap<N>(lo, hi, r_low);
                return std::clamp(outer - inner, 0.0, 1.0);
            }
        } // namespace SHELLOVERLAP
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif

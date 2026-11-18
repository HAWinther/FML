#ifndef FML_DELAUNAY_PREDICATES_HEADER
#define FML_DELAUNAY_PREDICATES_HEADER

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

namespace FML {
    namespace DELAUNAY {

        //==========================================================================================
        /// Exact geometric predicates for the Delaunay triangulation in 2D and 3D.
        ///
        /// The sign of a determinant is first computed in floating point together with an upper bound on its
        /// round-off error (the filters of Shewchuk 1997, "Adaptive precision floating-point arithmetic and fast
        /// robust geometric predicates", with the error bounds multiplied by a safety factor 4). If the sign is not
        /// certain the determinant is computed exactly with floating-point expansion arithmetic (sums of
        /// non-overlapping doubles, also from Shewchuk 1997) from the input coordinates. The results are therefore
        /// exact for any double input (barring overflow/underflow), which the triangulation relies on.
        ///
        /// Conventions (the same as CGAL's):
        ///   orient(p0, ..., pD)      = sign det[p1 - p0, ..., pD - p0]: +1 for a positively oriented simplex
        ///                              (2D: counterclockwise)
        ///   insphere(p0, ..., pD, q) = +1 if q is strictly inside the circumsphere of the positively oriented
        ///                              simplex (p0, ..., pD), -1 outside, 0 on the sphere
        //==========================================================================================

        namespace EXACT {

            /// x + y = a + b exactly
            inline void two_sum(double a, double b, double & x, double & y) {
                x = a + b;
                const double bv = x - a;
                const double av = x - bv;
                y = (a - av) + (b - bv);
            }
            /// x + y = a + b exactly, |a| >= |b|
            inline void fast_two_sum(double a, double b, double & x, double & y) {
                x = a + b;
                y = b - (x - a);
            }
            /// x + y = a * b exactly
            inline void two_product(double a, double b, double & x, double & y) {
                x = a * b;
                y = std::fma(a, b, -x);
            }

            /// A number as a sum of non-overlapping doubles in increasing magnitude (zero components removed). Only
            /// used when the floating-point filter cannot decide a sign. Small expansions (the usual case, e.g. for
            /// exactly representable differences) are stored inline, larger ones on the heap.
            class Expansion {
                static constexpr size_t NINLINE = 24;
                std::array<double, NINLINE> small{};
                std::vector<double> large{};
                size_t n{0};

              public:
                Expansion() = default;
                explicit Expansion(double a) {
                    if (a != 0.0)
                        push(a);
                }
                size_t size() const { return n; }
                double operator[](size_t i) const { return n <= NINLINE ? small[i] : large[i]; }
                void reserve(size_t m) {
                    if (m > NINLINE && large.capacity() < m)
                        large.reserve(m);
                }
                void push(double x) {
                    if (n < NINLINE) {
                        small[n++] = x;
                        return;
                    }
                    if (n == NINLINE)
                        large.assign(small.begin(), small.end());
                    large.push_back(x);
                    n++;
                }
                /// a - b exactly
                static Expansion difference(double a, double b) {
                    Expansion e;
                    double x, y;
                    two_sum(a, -b, x, y);
                    if (y != 0.0)
                        e.push(y);
                    if (x != 0.0)
                        e.push(x);
                    return e;
                }
                int sign() const {
                    if (n == 0)
                        return 0;
                    return (*this)[n - 1] > 0.0 ? 1 : -1;
                }

                /// a + b (Shewchuk's FAST-EXPANSION-SUM with zero elimination: a merge by magnitude, linear time; the
                /// inputs are nonadjacent, as all expansions made here are, so it is exact)
                friend Expansion operator+(const Expansion & e, const Expansion & f) {
                    if (e.n == 0)
                        return f;
                    if (f.n == 0)
                        return e;
                    Expansion h;
                    h.reserve(e.n + f.n);
                    size_t ie = 0, jf = 0;
                    double enow = e[0], fnow = f[0], Q, Qnew, hh;
                    auto take_e = [&]() { return (fnow > enow) == (fnow > -enow); };
                    if (take_e()) {
                        Q = enow;
                        ie++;
                        enow = ie < e.n ? e[ie] : 0.0;
                    } else {
                        Q = fnow;
                        jf++;
                        fnow = jf < f.n ? f[jf] : 0.0;
                    }
                    if (ie < e.n && jf < f.n) {
                        if (take_e()) {
                            fast_two_sum(enow, Q, Qnew, hh);
                            ie++;
                            enow = ie < e.n ? e[ie] : 0.0;
                        } else {
                            fast_two_sum(fnow, Q, Qnew, hh);
                            jf++;
                            fnow = jf < f.n ? f[jf] : 0.0;
                        }
                        Q = Qnew;
                        if (hh != 0.0)
                            h.push(hh);
                        while (ie < e.n && jf < f.n) {
                            if (take_e()) {
                                two_sum(Q, enow, Qnew, hh);
                                ie++;
                                enow = ie < e.n ? e[ie] : 0.0;
                            } else {
                                two_sum(Q, fnow, Qnew, hh);
                                jf++;
                                fnow = jf < f.n ? f[jf] : 0.0;
                            }
                            Q = Qnew;
                            if (hh != 0.0)
                                h.push(hh);
                        }
                    }
                    while (ie < e.n) {
                        two_sum(Q, enow, Qnew, hh);
                        ie++;
                        enow = ie < e.n ? e[ie] : 0.0;
                        Q = Qnew;
                        if (hh != 0.0)
                            h.push(hh);
                    }
                    while (jf < f.n) {
                        two_sum(Q, fnow, Qnew, hh);
                        jf++;
                        fnow = jf < f.n ? f[jf] : 0.0;
                        Q = Qnew;
                        if (hh != 0.0)
                            h.push(hh);
                    }
                    if (Q != 0.0)
                        h.push(Q);
                    return h;
                }
                /// e * b (Shewchuk's SCALE-EXPANSION with zero elimination)
                Expansion times(double b) const {
                    Expansion r;
                    if (n == 0 || b == 0.0)
                        return r;
                    r.reserve(2 * n);
                    double q, h;
                    two_product((*this)[0], b, q, h);
                    if (h != 0.0)
                        r.push(h);
                    for (size_t i = 1; i < n; i++) {
                        double p1, p0, sum;
                        two_product((*this)[i], b, p1, p0);
                        two_sum(q, p0, sum, h);
                        if (h != 0.0)
                            r.push(h);
                        fast_two_sum(p1, sum, q, h);
                        if (h != 0.0)
                            r.push(h);
                    }
                    if (q != 0.0)
                        r.push(q);
                    return r;
                }
                Expansion operator-() const {
                    Expansion r = *this;
                    for (size_t i = 0; i < r.n; i++) {
                        if (r.n <= NINLINE)
                            r.small[i] = -r.small[i];
                        else
                            r.large[i] = -r.large[i];
                    }
                    return r;
                }
                friend Expansion operator-(const Expansion & a, const Expansion & b) { return a + (-b); }
                friend Expansion operator*(const Expansion & a, const Expansion & b) {
                    Expansion r;
                    for (size_t j = 0; j < b.n; j++)
                        r = r + a.times(b[j]);
                    return r;
                }
            };

            //==========================================================================================
            // Fixed-size expansion arithmetic on stack arrays (Shewchuk's routines with zero elimination), for the
            // common exact case: all coordinate differences exactly representable (e.g. lattices, or coordinates on
            // a common grid as in PeriodicVoronoi)
            //==========================================================================================

            /// h = e * b; returns the length of h (at most 2 elen)
            inline int scale_expansion(int elen, const double * e, double b, double * h) {
                int hn = 0;
                if (elen == 0 || b == 0.0)
                    return 0;
                double q, hh;
                two_product(e[0], b, q, hh);
                if (hh != 0.0)
                    h[hn++] = hh;
                for (int i = 1; i < elen; i++) {
                    double p1, p0, sum;
                    two_product(e[i], b, p1, p0);
                    two_sum(q, p0, sum, hh);
                    if (hh != 0.0)
                        h[hn++] = hh;
                    fast_two_sum(p1, sum, q, hh);
                    if (hh != 0.0)
                        h[hn++] = hh;
                }
                if (q != 0.0)
                    h[hn++] = q;
                return hn;
            }

            /// h = e + f (FAST-EXPANSION-SUM with zero elimination); returns the length of h (at most elen + flen)
            inline int sum_expansion(int elen, const double * e, int flen, const double * f, double * h) {
                if (elen == 0) {
                    for (int i = 0; i < flen; i++)
                        h[i] = f[i];
                    return flen;
                }
                if (flen == 0) {
                    for (int i = 0; i < elen; i++)
                        h[i] = e[i];
                    return elen;
                }
                int ie = 0, jf = 0, hn = 0;
                double enow = e[0], fnow = f[0], Q, Qnew, hh;
                auto take_e = [&]() { return (fnow > enow) == (fnow > -enow); };
                auto next_e = [&]() {
                    ie++;
                    enow = ie < elen ? e[ie] : 0.0;
                };
                auto next_f = [&]() {
                    jf++;
                    fnow = jf < flen ? f[jf] : 0.0;
                };
                if (take_e()) {
                    Q = enow;
                    next_e();
                } else {
                    Q = fnow;
                    next_f();
                }
                if (ie < elen && jf < flen) {
                    if (take_e()) {
                        fast_two_sum(enow, Q, Qnew, hh);
                        next_e();
                    } else {
                        fast_two_sum(fnow, Q, Qnew, hh);
                        next_f();
                    }
                    Q = Qnew;
                    if (hh != 0.0)
                        h[hn++] = hh;
                    while (ie < elen && jf < flen) {
                        if (take_e()) {
                            two_sum(Q, enow, Qnew, hh);
                            next_e();
                        } else {
                            two_sum(Q, fnow, Qnew, hh);
                            next_f();
                        }
                        Q = Qnew;
                        if (hh != 0.0)
                            h[hn++] = hh;
                    }
                }
                while (ie < elen) {
                    two_sum(Q, enow, Qnew, hh);
                    next_e();
                    Q = Qnew;
                    if (hh != 0.0)
                        h[hn++] = hh;
                }
                while (jf < flen) {
                    two_sum(Q, fnow, Qnew, hh);
                    next_f();
                    Q = Qnew;
                    if (hh != 0.0)
                        h[hn++] = hh;
                }
                if (Q != 0.0)
                    h[hn++] = Q;
                return hn;
            }

            /// h = a * b - c * d for doubles; returns the length (at most 4)
            inline int two_two_cross(double a, double b, double c, double d, double * h) {
                double x[2], y[2];
                two_product(a, b, x[1], x[0]);
                two_product(-c, d, y[1], y[0]);
                const int nx = (x[0] != 0.0) ? 2 : 1;
                const double * px = (x[0] != 0.0) ? x : x + 1;
                const int ny = (y[0] != 0.0) ? 2 : 1;
                const double * py = (y[0] != 0.0) ? y : y + 1;
                return sum_expansion(x[1] != 0.0 ? nx : 0, px, y[1] != 0.0 ? ny : 0, py, h);
            }

            /// h = e * f with elen <= 8; returns the length (at most 2 elen flen). work must hold 2 elen flen doubles
            inline int
            mul_expansion(int elen, const double * e, int flen, const double * f, double * h, double * work) {
                int hn = 0;
                double s[16];
                for (int j = 0; j < flen; j++) {
                    const int sn = scale_expansion(elen, e, f[j], s);
                    const int un = sum_expansion(hn, h, sn, s, work);
                    for (int i = 0; i < un; i++)
                        h[i] = work[i];
                    hn = un;
                }
                return hn;
            }

            inline int sign_of(int n, const double * e) { return n == 0 ? 0 : (e[n - 1] > 0.0 ? 1 : -1); }

            /// Exact Shewchuk insphere when all differences x - e are exact doubles
            inline int insphere_exact_differences(const double (&ae)[3],
                                                  const double (&be)[3],
                                                  const double (&ce)[3],
                                                  const double (&de)[3]) {
                double ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
                const int nab = two_two_cross(ae[0], be[1], be[0], ae[1], ab);
                const int nbc = two_two_cross(be[0], ce[1], ce[0], be[1], bc);
                const int ncd = two_two_cross(ce[0], de[1], de[0], ce[1], cd);
                const int nda = two_two_cross(de[0], ae[1], ae[0], de[1], da);
                const int nac = two_two_cross(ae[0], ce[1], ce[0], ae[1], ac);
                const int nbd = two_two_cross(be[0], de[1], de[0], be[1], bd);
                // A three-term combination z1 * m1 + z2 * m2 + z3 * m3 (at most 24 components)
                auto combo = [](double z1,
                                int n1,
                                const double * m1,
                                double z2,
                                int n2,
                                const double * m2,
                                double z3,
                                int n3,
                                const double * m3,
                                double * out) {
                    double t1[8], t2[8], t3[8], t12[16];
                    const int k1 = scale_expansion(n1, m1, z1, t1);
                    const int k2 = scale_expansion(n2, m2, z2, t2);
                    const int k3 = scale_expansion(n3, m3, z3, t3);
                    const int k12 = sum_expansion(k1, t1, k2, t2, t12);
                    return sum_expansion(k12, t12, k3, t3, out);
                };
                double abc[24], bcd[24], cda[24], dab[24];
                const int nabc = combo(ae[2], nbc, bc, -be[2], nac, ac, ce[2], nab, ab, abc);
                const int nbcd = combo(be[2], ncd, cd, -ce[2], nbd, bd, de[2], nbc, bc, bcd);
                const int ncda = combo(ce[2], nda, da, de[2], nac, ac, ae[2], ncd, cd, cda);
                const int ndab = combo(de[2], nab, ab, ae[2], nbd, bd, be[2], nda, da, dab);
                auto lift = [](const double(&x)[3], double * out) {
                    double p[3][2], s01[4];
                    for (int d = 0; d < 3; d++)
                        two_product(x[d], x[d], p[d][1], p[d][0]);
                    auto len = [](const double * q, const double *& start) {
                        if (q[1] == 0.0)
                            return 0;
                        start = q[0] != 0.0 ? q : q + 1;
                        return q[0] != 0.0 ? 2 : 1;
                    };
                    const double *s0 = p[0], *s1 = p[1], *s2 = p[2];
                    const int n0 = len(p[0], s0), n1 = len(p[1], s1), n2 = len(p[2], s2);
                    const int n01 = sum_expansion(n0, s0, n1, s1, s01);
                    return sum_expansion(n01, s01, n2, s2, out);
                };
                double al[6], bl[6], cl[6], dl[6];
                const int nal = lift(ae, al), nbl = lift(be, bl), ncl = lift(ce, cl), ndl = lift(de, dl);
                // det = (dlift abc - clift dab) + (blift cda - alift bcd)
                double work[288];
                double t1[288], t2[288], t3[288], t4[288], s12[576], s34[576], det[1152];
                for (int i = 0; i < ndab; i++)
                    dab[i] = -dab[i];
                for (int i = 0; i < nbcd; i++)
                    bcd[i] = -bcd[i];
                const int n1 = mul_expansion(ndl, dl, nabc, abc, t1, work);
                const int n2 = mul_expansion(ncl, cl, ndab, dab, t2, work);
                const int n3 = mul_expansion(nbl, bl, ncda, cda, t3, work);
                const int n4 = mul_expansion(nal, al, nbcd, bcd, t4, work);
                const int n12 = sum_expansion(n1, t1, n2, t2, s12);
                const int n34 = sum_expansion(n3, t3, n4, t4, s34);
                const int n = sum_expansion(n12, s12, n34, s34, det);
                return sign_of(n, det);
            }

            // Exact versions of the determinants evaluated by the filters below (the same formulas, so the signs
            // agree whenever the filter is certain)

            inline int orient2d_shewchuk(const double * a, const double * b, const double * c) {
                const Expansion acx = Expansion::difference(a[0], c[0]), bcx = Expansion::difference(b[0], c[0]);
                const Expansion acy = Expansion::difference(a[1], c[1]), bcy = Expansion::difference(b[1], c[1]);
                return (acx * bcy - acy * bcx).sign();
            }

            inline int orient3d_shewchuk(const double * a, const double * b, const double * c, const double * d) {
                {
                    double x[3][3];
                    bool exact = true;
                    const double * p[3] = {a, b, c};
                    for (int k = 0; k < 3 && exact; k++)
                        for (int i = 0; i < 3; i++) {
                            double y;
                            two_sum(p[k][i], -d[i], x[k][i], y);
                            exact = exact && y == 0.0;
                        }
                    if (exact) {
                        // det = adz (bdx cdy - cdx bdy) + bdz (cdx ady - adx cdy) + cdz (adx bdy - bdx ady)
                        double m1[4], m2[4], m3[4], t1[8], t2[8], t3[8], t12[16], det[24];
                        const int n1 = two_two_cross(x[1][0], x[2][1], x[2][0], x[1][1], m1);
                        const int n2 = two_two_cross(x[2][0], x[0][1], x[0][0], x[2][1], m2);
                        const int n3 = two_two_cross(x[0][0], x[1][1], x[1][0], x[0][1], m3);
                        const int k1 = scale_expansion(n1, m1, x[0][2], t1);
                        const int k2 = scale_expansion(n2, m2, x[1][2], t2);
                        const int k3 = scale_expansion(n3, m3, x[2][2], t3);
                        const int k12 = sum_expansion(k1, t1, k2, t2, t12);
                        return sign_of(sum_expansion(k12, t12, k3, t3, det), det);
                    }
                }
                Expansion ad[3], bd[3], cd[3];
                for (int i = 0; i < 3; i++) {
                    ad[i] = Expansion::difference(a[i], d[i]);
                    bd[i] = Expansion::difference(b[i], d[i]);
                    cd[i] = Expansion::difference(c[i], d[i]);
                }
                const Expansion det = ad[0] * (bd[1] * cd[2] - bd[2] * cd[1]) +
                                      bd[0] * (cd[1] * ad[2] - cd[2] * ad[1]) + cd[0] * (ad[1] * bd[2] - ad[2] * bd[1]);
                return det.sign();
            }

            inline int incircle_shewchuk(const double * a, const double * b, const double * c, const double * d) {
                const Expansion adx = Expansion::difference(a[0], d[0]), ady = Expansion::difference(a[1], d[1]);
                const Expansion bdx = Expansion::difference(b[0], d[0]), bdy = Expansion::difference(b[1], d[1]);
                const Expansion cdx = Expansion::difference(c[0], d[0]), cdy = Expansion::difference(c[1], d[1]);
                const Expansion alift = adx * adx + ady * ady;
                const Expansion blift = bdx * bdx + bdy * bdy;
                const Expansion clift = cdx * cdx + cdy * cdy;
                const Expansion det =
                    alift * (bdx * cdy - cdx * bdy) + blift * (cdx * ady - adx * cdy) + clift * (adx * bdy - bdx * ady);
                return det.sign();
            }

            inline int insphere_shewchuk(const double * a,
                                         const double * b,
                                         const double * c,
                                         const double * d,
                                         const double * e) {
                {
                    double x[4][3];
                    bool exact = true;
                    const double * p[4] = {a, b, c, d};
                    for (int k = 0; k < 4 && exact; k++)
                        for (int i = 0; i < 3; i++) {
                            double y;
                            two_sum(p[k][i], -e[i], x[k][i], y);
                            exact = exact && y == 0.0;
                        }
                    if (exact)
                        return insphere_exact_differences(x[0], x[1], x[2], x[3]);
                }
                Expansion ae[3], be[3], ce[3], de[3];
                for (int i = 0; i < 3; i++) {
                    ae[i] = Expansion::difference(a[i], e[i]);
                    be[i] = Expansion::difference(b[i], e[i]);
                    ce[i] = Expansion::difference(c[i], e[i]);
                    de[i] = Expansion::difference(d[i], e[i]);
                }
                const Expansion ab = ae[0] * be[1] - be[0] * ae[1];
                const Expansion bc = be[0] * ce[1] - ce[0] * be[1];
                const Expansion cd = ce[0] * de[1] - de[0] * ce[1];
                const Expansion da = de[0] * ae[1] - ae[0] * de[1];
                const Expansion ac = ae[0] * ce[1] - ce[0] * ae[1];
                const Expansion bd = be[0] * de[1] - de[0] * be[1];
                const Expansion abc = ae[2] * bc - be[2] * ac + ce[2] * ab;
                const Expansion bcd = be[2] * cd - ce[2] * bd + de[2] * bc;
                const Expansion cda = ce[2] * da + de[2] * ac + ae[2] * cd;
                const Expansion dab = de[2] * ab + ae[2] * bd + be[2] * da;
                const Expansion alift = ae[0] * ae[0] + ae[1] * ae[1] + ae[2] * ae[2];
                const Expansion blift = be[0] * be[0] + be[1] * be[1] + be[2] * be[2];
                const Expansion clift = ce[0] * ce[0] + ce[1] * ce[1] + ce[2] * ce[2];
                const Expansion dlift = de[0] * de[0] + de[1] * de[1] + de[2] * de[2];
                const Expansion det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
                return det.sign();
            }
        } // namespace EXACT

        namespace FILTER {
            constexpr double eps = std::numeric_limits<double>::epsilon() / 2; // 2^-53
            // Shewchuk's first-stage error bounds, times a safety factor 4
            constexpr double ccwerrbound = 4 * (3.0 + 16.0 * eps) * eps;
            constexpr double o3derrbound = 4 * (7.0 + 56.0 * eps) * eps;
            constexpr double iccerrbound = 4 * (10.0 + 96.0 * eps) * eps;
            constexpr double isperrbound = 4 * (16.0 + 224.0 * eps) * eps;
        } // namespace FILTER

        /// Number of exact evaluations (for tests and diagnostics)
        struct PredicateStatistics {
            inline static std::atomic<size_t> n_exact{0};
            static void count() { n_exact.fetch_add(1, std::memory_order_relaxed); }
        };

        /// Shewchuk's orient2d(a, b, c): positive if a, b, c are counterclockwise
        inline int orient2d_shewchuk(const double * a, const double * b, const double * c) {
            const double detleft = (a[0] - c[0]) * (b[1] - c[1]);
            const double detright = (a[1] - c[1]) * (b[0] - c[0]);
            const double det = detleft - detright;
            const double errbound = FILTER::ccwerrbound * (std::fabs(detleft) + std::fabs(detright));
            if (det > errbound)
                return 1;
            if (-det > errbound)
                return -1;
            PredicateStatistics::count();
            return EXACT::orient2d_shewchuk(a, b, c);
        }

        /// Shewchuk's orient3d(a, b, c, d) = det[a - d, b - d, c - d]: positive if d is below the plane of a, b, c
        /// (counterclockwise seen from above), i.e. minus our orient
        inline int orient3d_shewchuk(const double * a, const double * b, const double * c, const double * d) {
            const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
            const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
            const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
            const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
            const double cdxady = cdx * ady, adxcdy = adx * cdy;
            const double adxbdy = adx * bdy, bdxady = bdx * ady;
            const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
            const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * std::fabs(adz) +
                                     (std::fabs(cdxady) + std::fabs(adxcdy)) * std::fabs(bdz) +
                                     (std::fabs(adxbdy) + std::fabs(bdxady)) * std::fabs(cdz);
            const double errbound = FILTER::o3derrbound * permanent;
            if (det > errbound)
                return 1;
            if (-det > errbound)
                return -1;
            PredicateStatistics::count();
            return EXACT::orient3d_shewchuk(a, b, c, d);
        }

        /// Shewchuk's incircle(a, b, c, d): positive if d is inside the circle through counterclockwise a, b, c
        inline int incircle_shewchuk(const double * a, const double * b, const double * c, const double * d) {
            const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
            const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
            const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
            const double alift = adx * adx + ady * ady;
            const double cdxady = cdx * ady, adxcdy = adx * cdy;
            const double blift = bdx * bdx + bdy * bdy;
            const double adxbdy = adx * bdy, bdxady = bdx * ady;
            const double clift = cdx * cdx + cdy * cdy;
            const double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
            const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
                                     (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
                                     (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
            const double errbound = FILTER::iccerrbound * permanent;
            if (det > errbound)
                return 1;
            if (-det > errbound)
                return -1;
            PredicateStatistics::count();
            return EXACT::incircle_shewchuk(a, b, c, d);
        }

        /// Shewchuk's insphere(a, b, c, d, e): positive if e is inside the sphere through a, b, c, d when
        /// orient3d_shewchuk(a, b, c, d) > 0
        inline int
        insphere_shewchuk(const double * a, const double * b, const double * c, const double * d, const double * e) {
            const double aex = a[0] - e[0], bex = b[0] - e[0], cex = c[0] - e[0], dex = d[0] - e[0];
            const double aey = a[1] - e[1], bey = b[1] - e[1], cey = c[1] - e[1], dey = d[1] - e[1];
            const double aez = a[2] - e[2], bez = b[2] - e[2], cez = c[2] - e[2], dez = d[2] - e[2];
            const double aexbey = aex * bey, bexaey = bex * aey, ab = aexbey - bexaey;
            const double bexcey = bex * cey, cexbey = cex * bey, bc = bexcey - cexbey;
            const double cexdey = cex * dey, dexcey = dex * cey, cd = cexdey - dexcey;
            const double dexaey = dex * aey, aexdey = aex * dey, da = dexaey - aexdey;
            const double aexcey = aex * cey, cexaey = cex * aey, ac = aexcey - cexaey;
            const double bexdey = bex * dey, dexbey = dex * bey, bd = bexdey - dexbey;
            const double abc = aez * bc - bez * ac + cez * ab;
            const double bcd = bez * cd - cez * bd + dez * bc;
            const double cda = cez * da + dez * ac + aez * cd;
            const double dab = dez * ab + aez * bd + bez * da;
            const double alift = aex * aex + aey * aey + aez * aez;
            const double blift = bex * bex + bey * bey + bez * bez;
            const double clift = cex * cex + cey * cey + cez * cez;
            const double dlift = dex * dex + dey * dey + dez * dez;
            const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
            const double aezp = std::fabs(aez), bezp = std::fabs(bez), cezp = std::fabs(cez), dezp = std::fabs(dez);
            const double aexbeyp = std::fabs(aexbey), bexaeyp = std::fabs(bexaey);
            const double bexceyp = std::fabs(bexcey), cexbeyp = std::fabs(cexbey);
            const double cexdeyp = std::fabs(cexdey), dexceyp = std::fabs(dexcey);
            const double dexaeyp = std::fabs(dexaey), aexdeyp = std::fabs(aexdey);
            const double aexceyp = std::fabs(aexcey), cexaeyp = std::fabs(cexaey);
            const double bexdeyp = std::fabs(bexdey), dexbeyp = std::fabs(dexbey);
            const double permanent =
                ((cexdeyp + dexceyp) * bezp + (dexbeyp + bexdeyp) * cezp + (bexceyp + cexbeyp) * dezp) * alift +
                ((dexaeyp + aexdeyp) * cezp + (aexceyp + cexaeyp) * dezp + (cexdeyp + dexceyp) * aezp) * blift +
                ((aexbeyp + bexaeyp) * dezp + (bexdeyp + dexbeyp) * aezp + (dexaeyp + aexdeyp) * bezp) * clift +
                ((bexceyp + cexbeyp) * aezp + (cexaeyp + aexceyp) * bezp + (aexbeyp + bexaeyp) * cezp) * dlift;
            const double errbound = FILTER::isperrbound * permanent;
            if (det > errbound)
                return 1;
            if (-det > errbound)
                return -1;
            PredicateStatistics::count();
            return EXACT::insphere_shewchuk(a, b, c, d, e);
        }

        /// orient(p0, ..., pD) = sign det[p1 - p0, ..., pD - p0] for D = 2, 3 (p is an array of D+1 pointers)
        template <int D>
        int orient(const std::array<const double *, D + 1> & p) {
            static_assert(D == 2 || D == 3, "orient: D must be 2 or 3");
            if constexpr (D == 2)
                return orient2d_shewchuk(p[0], p[1], p[2]);
            else
                return -orient3d_shewchuk(p[0], p[1], p[2], p[3]);
        }

        /// +1 if q is strictly inside the circumsphere of the positively oriented simplex p, -1 outside, 0 on it
        template <int D>
        int insphere(const std::array<const double *, D + 1> & p, const double * q) {
            static_assert(D == 2 || D == 3, "insphere: D must be 2 or 3");
            if constexpr (D == 2)
                return incircle_shewchuk(p[0], p[1], p[2], q);
            else
                return -insphere_shewchuk(p[0], p[1], p[2], p[3], q);
        }

    } // namespace DELAUNAY
} // namespace FML

#endif

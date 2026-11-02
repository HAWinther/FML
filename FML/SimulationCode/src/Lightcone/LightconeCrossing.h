#ifndef LIGHTCONE_CROSSING_HEADER
#define LIGHTCONE_CROSSING_HEADER

#include <array>
#include <cmath>

//=============================================================================
// When a particle crosses the past lightcone of the observer during a time step (no dependencies on the rest of
// SimulationCode, so it can be tested on its own: tests/test_lightcone_crossing.cpp)
//=============================================================================

/// One step apos -> apos_new of the lightcone. Distances r in units of the box. The lightcone is recorded from a_old
/// to a_new (= apos, apos_new unless plc_z_init / plc_z_finish is inside the step), frac_old, frac_new are these as
/// fractions of the step (the particles move linearly in a over the step)
struct LightconeStep {
    double apos{}, apos_new{};
    double a_old{}, a_new{};
    double r_lc_apos{}, r_lc_apos_new{};
    double r_lc_old{}, r_lc_new{};
    double r_lc_old_squared{}, r_lc_new_squared{};
    double frac_old{}, frac_new{};
};

/// The step apos -> apos_new of a lightcone that is recorded for a_init <= a <= a_finish. r_lc_of_a(a): the comoving
/// distance to the lightcone at a (units of the box)
template <class F>
LightconeStep make_lightcone_step(double apos, double apos_new, double a_init, double a_finish, F && r_lc_of_a) {
    LightconeStep s;
    s.apos = apos;
    s.apos_new = apos_new;
    s.r_lc_apos = r_lc_of_a(apos);
    s.r_lc_apos_new = r_lc_of_a(apos_new);
    s.a_old = std::max(apos, a_init);
    s.r_lc_old = r_lc_of_a(s.a_old);
    s.r_lc_old_squared = s.r_lc_old * s.r_lc_old;
    s.a_new = std::min(apos_new, a_finish);
    s.r_lc_new = r_lc_of_a(s.a_new);
    s.r_lc_new_squared = s.r_lc_new * s.r_lc_new;
    s.frac_old = s.a_old > apos ? (s.a_old - apos) / (apos_new - apos) : 0.0;
    s.frac_new = s.a_new < apos_new ? (s.a_new - apos) / (apos_new - apos) : 1.0;
    return s;
}

/// Does a particle at pos (relative to the observer, at apos) that moves delta_pos over the full step cross the
/// lightcone during the recorded part of the step? It does if it is inside r_lc_old at a_old and outside r_lc_new at
/// a_new. Then fraction_of_step (the fraction of the full step to the crossing) and a_cross are set.
/// The crossing is that of the L-PICOLA method (r_lc(a) approximated by the chord over the step). If the lightcone
/// starts or ends inside the step and the chord puts the crossing outside [frac_old, frac_new] (rejecting those
/// particles loses particles), |x + f dx| = r_lc(apos + f (apos_new - apos)) is solved by bisection in that interval
template <int NDIM, class F>
bool lightcone_crossing(const std::array<double, NDIM> & pos,
                        const std::array<double, NDIM> & delta_pos,
                        const LightconeStep & s,
                        F && r_lc_of_a,
                        double & fraction_of_step,
                        double & a_cross) {
    // The radial comoving distance to the particle before the step (at a_old)
    double r_old_squared = 0.0;
    for (int idim = 0; idim < NDIM; idim++) {
        double dx = s.frac_old == 0.0 ? pos[idim] : pos[idim] + delta_pos[idim] * s.frac_old;
        r_old_squared += dx * dx;
    }
    // Outside the lightcone at the start
    if (r_old_squared > s.r_lc_old_squared)
        return false;

    // The radial comoving distance to the particle after the step (at a_new)
    double r_new_squared = 0.0;
    for (int idim = 0; idim < NDIM; idim++) {
        double dx = s.frac_new == 1.0 ? pos[idim] + delta_pos[idim] : pos[idim] + delta_pos[idim] * s.frac_new;
        r_new_squared += dx * dx;
    }
    // Still inside the lightcone at the end
    if (r_new_squared <= s.r_lc_new_squared)
        return false;

    // The L-PICOLA crossing: approximative, but much less operations
    const double r_old = std::sqrt(r_old_squared);
    const double r_new = std::sqrt(r_new_squared);
    a_cross = s.apos + (s.apos_new - s.apos) * (s.r_lc_apos - r_old) / ((r_new - r_old) - (s.r_lc_apos_new - s.r_lc_apos));
    fraction_of_step = (a_cross - s.apos) / (s.apos_new - s.apos);

    // The lightcone starts or ends inside the step and the chord puts the crossing outside it: bisection
    const bool partial_step = s.frac_old > 0.0 or s.frac_new < 1.0;
    if (partial_step and (fraction_of_step < s.frac_old or fraction_of_step > s.frac_new)) {
        double flow = s.frac_old, fhigh = s.frac_new;
        for (int iter = 0; iter < 40; iter++) {
            const double f = 0.5 * (flow + fhigh);
            double r2 = 0.0;
            for (int idim = 0; idim < NDIM; idim++) {
                const double dx = pos[idim] + delta_pos[idim] * f;
                r2 += dx * dx;
            }
            const double r_lc = r_lc_of_a(s.apos + (s.apos_new - s.apos) * f);
            if (r2 > r_lc * r_lc)
                fhigh = f;
            else
                flow = f;
        }
        fraction_of_step = 0.5 * (flow + fhigh);
        a_cross = s.apos + (s.apos_new - s.apos) * fraction_of_step;
    }
    return true;
}

#endif

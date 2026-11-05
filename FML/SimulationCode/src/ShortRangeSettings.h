#ifndef SIMCODE_SHORT_RANGE_SETTINGS_H
#define SIMCODE_SHORT_RANGE_SETTINGS_H
#include "ParameterFields.h"
#include <FML/ShortRangeGravity/ShortRangeGravity.h>
namespace SIMCODE {
    struct ShortRangeSettings {
        bool enabled{false};
        std::string backend{"cell"};
        double split_scale{0}, softening_support{0}, cutoff_tolerance{1e-6}, opening_angle{.3};
        int table_points{16384};
        template <class F>
        void fields(F && f) {
            f.always_optional("force_short_range",
                              enabled,
                              false,
                              "Periodic Newtonian PM plus short-range forces; initial common-timestep N-body scope");
            f.if_true("force_short_range", [&] {
                f.required("force_short_range_split_scale",
                           split_scale,
                           0.,
                           "Comoving Gaussian split scale in Mpc/h",
                           check::positive());
                f.required("force_short_range_softening_support",
                           softening_support,
                           0.,
                           "Comoving spline support h in Mpc/h; h=2.8 Plummer-equivalent epsilon",
                           check::positive());
                f.always_optional("force_short_range_backend",
                                  backend,
                                  "cell",
                                  "Short-range backend: cell (P3M) or tree (monopole TreePM)",
                                  {"cell", "tree"});
                f.always_optional("force_short_range_cutoff_tolerance",
                                  cutoff_tolerance,
                                  1e-6,
                                  "Truncated pair-window tolerance, not a bound on relative net-force error",
                                  check::positive());
                f.always_optional("force_short_range_opening_angle",
                                  opening_angle,
                                  .3,
                                  "Tree opening angle; 0 gives exact leaf sums",
                                  check::in_range(0., 1.));
                f.always_optional("force_short_range_table_points",
                                  table_points,
                                  16384,
                                  "Kernel table resolution; 0 evaluates the analytic kernel",
                                  check::non_negative());
            });
        }
        FML::SHORT_RANGE_GRAVITY::Options options(double boxsize, double coupling = 1) const {
            if (!(boxsize > 0) || !std::isfinite(boxsize) || !std::isfinite(split_scale) || !(split_scale > 0) ||
                !std::isfinite(softening_support) || !(softening_support > 0) || table_points < 0 ||
                (backend != "cell" && backend != "tree") || !std::isfinite(opening_angle) || opening_angle < 0 ||
                opening_angle > 1)
                throw std::invalid_argument("Invalid short-range physical lengths, backend or accuracy settings");
            FML::SHORT_RANGE_GRAVITY::Options o;
            o.boxsize = 1;
            o.split_scale = split_scale / boxsize;
            o.softening_support = softening_support / boxsize;
            o.cutoff_tolerance = cutoff_tolerance;
            o.coupling = coupling;
            o.table_points = size_t(table_points);
            return o;
        }
        void validate(double boxsize) const {
            if (enabled) {
                FML::SHORT_RANGE_GRAVITY::Kernel kernel(options(boxsize));
            }
        }
    };
} // namespace SIMCODE
#endif

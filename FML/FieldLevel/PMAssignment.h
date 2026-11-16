#ifndef FML_FIELD_LEVEL_PM_ASSIGNMENT_H
#define FML_FIELD_LEVEL_PM_ASSIGNMENT_H
#include <array>
#include <cmath>
#include <cstdint>

namespace FML::FIELDLEVEL::FRAMEWORK {
    enum class PMAssignment { CIC = 2, TSC = 3 };
    namespace DETAIL {
        // Tensor-product quadratic B-spline. Position is in box fractions;
        // derivatives are with respect to those same fractions. At half-cell
        // support changes both the disappearing weight and derivative are zero.
        template <class Visit>
        void tsc_stencil(int mesh, const std::array<double, 3> & position, Visit visit) {
            int center[3];
            double weight[3][3], derivative[3][3];
            for (int d = 0; d < 3; ++d) {
                const double u = (position[d] - std::floor(position[d])) * mesh;
                center[d] = int(std::floor(u + .5));
                const double t = u - center[d], left = .5 - t, right = .5 + t;
                weight[d][0] = .5 * left * left;
                weight[d][1] = .75 - t * t;
                weight[d][2] = .5 * right * right;
                derivative[d][0] = -mesh * left;
                derivative[d][1] = -2 * mesh * t;
                derivative[d][2] = mesh * right;
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int c = 0; c < 3; ++c) {
                        const int slot[3] = {a, b, c};
                        const auto node = (std::uint64_t((center[0] + a - 1 + mesh) % mesh) * mesh +
                                           (center[1] + b - 1 + mesh) % mesh) * mesh +
                                           (center[2] + c - 1 + mesh) % mesh;
                        std::array<double, 3> dw;
                        for (int d = 0; d < 3; ++d)
                            dw[d] = derivative[d][slot[d]] * weight[(d + 1) % 3][slot[(d + 1) % 3]] *
                                    weight[(d + 2) % 3][slot[(d + 2) % 3]];
                        visit(node, weight[0][a] * weight[1][b] * weight[2][c], dw);
                    }
        }
    }
}
#endif

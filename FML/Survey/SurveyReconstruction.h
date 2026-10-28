#ifndef FML_SURVEY_RECONSTRUCTION_HEADER
#define FML_SURVEY_RECONSTRUCTION_HEADER
#include <FML/Survey/SurveyPower.h>
namespace FML::SURVEY {
    enum class ReconstructionConvention { RecIso, RecSym };
    struct ReconstructionOptions {
        double bias = 2, growth_rate = .8, smoothing_radius = 15;
        double minimum_random_fraction = .01;
        double relative_tolerance = sizeof(Real) == sizeof(float) ? 1e-5 : 1e-8;
        int max_iterations = 100;
        ReconstructionConvention convention = ReconstructionConvention::RecSym;
        std::function<bool(const Vector&)> mask; // Additional physical-space selection mask, optional.
    };
    struct SolverReport {
        bool converged = false;
        int iterations = 0;
        double initial_residual = 0, final_residual = 0, removed_mean = 0;
    };
    struct ReconstructionResult {
        std::vector<CatalogPoint> data, randoms; // Same local order and weights as the input catalogues.
        std::vector<Vector> data_displacement, random_displacement;
        SolverReport solver;
        double alpha = 0;
    };
    namespace detail {
        inline std::array<int, 3> neighbor(std::array<int, 3> c, int direction, int step, int n) {
            c[direction] += step;
            if (direction > 0)
                c[direction] = (c[direction] + n) % n;
            return c;
        }
        inline double mesh_dot(const Mesh& a, const Mesh& b) {
            double sum = 0;
            for (auto i : a.get_real_range())
                sum += double(a.get_real_from_index(i)) * b.get_real_from_index(i);
            FML::SumOverTasks(&sum);
            return sum;
        }
        inline double remove_mean(Mesh& a, const Geometry& g) {
            double sum = 0;
            for (auto i : a.get_real_range())
                sum += a.get_real_from_index(i);
            FML::SumOverTasks(&sum);
            sum /= double(g.nmesh) * g.nmesh * g.nmesh;
            for (auto i : a.get_real_range())
                a.set_real_from_index(i, Real(a.get_real_from_index(i) - sum));
            return sum;
        }
        inline void smooth(Mesh& a, const Geometry& g, double radius) {
            if (radius == 0)
                return;
            a.fftw_r2c();
            for (auto i : a.get_fourier_range()) {
                Vector k;
                double k2;
                a.get_fourier_wavevector_and_norm2_by_index(i, k, k2);
                const double filter = std::exp(-.5 * k2 * radius * radius / (g.boxsize * g.boxsize));
                a.set_fourier_from_index(i, a.get_fourier_from_index(i) * Real(filter));
            }
            a.fftw_c2r();
        }
        // SPD discrete -div[(I+beta*n*n)grad(phi)], using forward gradients and
        // their negative adjoint divergence. One-cell halos suffice, including cross terms.
        inline void reconstruction_operator(Mesh& phi, Mesh& output, std::array<Mesh, 3>& flux, const Geometry& g,
                                            double beta) {
            phi.communicate_boundaries();
            const double h = g.cellsize();
            for (auto i : phi.get_real_range()) {
                const auto c = phi.get_coord_from_index(i);
                Vector grad;
                for (int d = 0; d < 3; ++d)
                    grad[d] = (phi.get_real(neighbor(c, d, 1, g.nmesh)) - phi.get_real_from_index(i)) / h;
                const auto n = g.los(g.position(phi, i));
                const double radial = dot(grad, n);
                for (int d = 0; d < 3; ++d)
                    flux[d].set_real_from_index(i, Real(grad[d] + beta * n[d] * radial));
            }
            for (auto& f : flux)
                f.communicate_boundaries();
            for (auto i : phi.get_real_range()) {
                const auto c = phi.get_coord_from_index(i);
                double value = 0;
                for (int d = 0; d < 3; ++d)
                    value -= (flux[d].get_real_from_index(i) - flux[d].get_real(neighbor(c, d, -1, g.nmesh))) / h;
                output.set_real_from_index(i, Real(value));
            }
        }
        inline void inverse_laplacian(const Mesh& residual, Mesh& output, const Geometry& g) {
            for (auto i : residual.get_real_range())
                output.set_real_from_index(i, residual.get_real_from_index(i));
            output.fftw_r2c();
            for (auto i : output.get_fourier_range()) {
                const auto k = output.get_fourier_integer_wavevector_from_index(i);
                double eigenvalue = 0;
                for (int d = 0; d < 3; ++d)
                    eigenvalue += 4 * std::pow(std::sin(M_PI * k[d] / g.nmesh), 2) / (g.cellsize() * g.cellsize());
                output.set_fourier_from_index(i, eigenvalue > 0 ? output.get_fourier_from_index(i) / Real(eigenvalue)
                                                                : Complex(0));
            }
            output.fftw_c2r();
        }
    } // namespace detail
    struct DisplacementSolution {
        std::array<Mesh, 3> field;
        SolverReport report;
    };
    /// Solve for Psi=grad(phi), -div[(I+f/b nn)grad(phi)]=delta_tracer/b.
    /// The source mesh is real and its scale is delta_tracer/b (already smoothed).
    inline DisplacementSolution solve_displacement(const Mesh& source, const Geometry& g,
                                                   const ReconstructionOptions& options) {
        g.validate();
        require(source.get_nmesh() == g.nmesh && detail::is_real(source) && options.bias > 0 &&
                    std::isfinite(options.bias) && options.growth_rate >= 0 && std::isfinite(options.growth_rate) &&
                    options.relative_tolerance > 0 && options.relative_tolerance < 1 && options.max_iterations > 0,
                "Invalid reconstruction solver parameters/source grid");
        auto phi = detail::empty_mesh(g, 1), residual = detail::empty_mesh(g, 1), direction = detail::empty_mesh(g, 1);
        auto preconditioned = detail::empty_mesh(g, 1), ap = detail::empty_mesh(g, 1);
        std::array<Mesh, 3> flux{detail::empty_mesh(g, 1), detail::empty_mesh(g, 1), detail::empty_mesh(g, 1)};
        bool finite = true;
        for (auto i : source.get_real_range()) {
            const double x = source.get_real_from_index(i);
            finite &= std::isfinite(x);
            residual.set_real_from_index(i, Real(x));
        }
        require(finite, "Nonfinite reconstruction source");
        SolverReport report;
        report.removed_mean = detail::remove_mean(residual, g);
        const double initial2 = detail::mesh_dot(residual, residual);
        const double cells = double(g.nmesh) * g.nmesh * g.nmesh;
        report.initial_residual = report.final_residual = std::sqrt(initial2 / cells);
        report.converged = initial2 == 0;
        double rz = 0;
        if (!report.converged) {
            detail::inverse_laplacian(residual, preconditioned, g);
            for (auto i : direction.get_real_range())
                direction.set_real_from_index(i, preconditioned.get_real_from_index(i));
            rz = detail::mesh_dot(residual, preconditioned);
        }
        for (int iteration = 0; iteration < options.max_iterations && !report.converged; ++iteration) {
            detail::reconstruction_operator(direction, ap, flux, g, options.growth_rate / options.bias);
            const double pap = detail::mesh_dot(direction, ap);
            if (!(pap > 0) || !std::isfinite(pap) || !std::isfinite(rz))
                throw std::runtime_error("Reconstruction CG lost positive definiteness or finite arithmetic");
            const double step = rz / pap;
            for (auto i : phi.get_real_range()) {
                phi.set_real_from_index(i, Real(phi.get_real_from_index(i) + step * direction.get_real_from_index(i)));
                residual.set_real_from_index(i,
                                             Real(residual.get_real_from_index(i) - step * ap.get_real_from_index(i)));
            }
            report.iterations = iteration + 1;
            // Verify the actual equation residual, not just the recursive CG estimate.
            detail::reconstruction_operator(phi, ap, flux, g, options.growth_rate / options.bias);
            for (auto i : residual.get_real_range())
                residual.set_real_from_index(
                    i, Real(source.get_real_from_index(i) - report.removed_mean - ap.get_real_from_index(i)));
            const double residual2 = detail::mesh_dot(residual, residual);
            if (!std::isfinite(residual2))
                throw std::runtime_error("Nonfinite reconstruction residual");
            report.final_residual = std::sqrt(residual2 / cells);
            report.converged = residual2 <= initial2 * options.relative_tolerance * options.relative_tolerance;
            if (report.converged)
                break;
            detail::inverse_laplacian(residual, preconditioned, g);
            const double next = detail::mesh_dot(residual, preconditioned);
            const double beta = next / rz;
            rz = next;
            for (auto i : direction.get_real_range())
                direction.set_real_from_index(
                    i, Real(preconditioned.get_real_from_index(i) + beta * direction.get_real_from_index(i)));
        }
        // Return a cell-centered displacement, with no Nyquist derivative ambiguity.
        phi.communicate_boundaries();
        for (auto i : phi.get_real_range()) {
            const auto c = phi.get_coord_from_index(i);
            for (int d = 0; d < 3; ++d)
                flux[d].set_real_from_index(i, Real((phi.get_real(detail::neighbor(c, d, 1, g.nmesh)) -
                                                     phi.get_real(detail::neighbor(c, d, -1, g.nmesh))) /
                                                    (2 * g.cellsize())));
        }
        for (auto& field : flux)
            field.communicate_boundaries();
        return {std::move(flux), report};
    }
    inline std::vector<Vector> sample_displacement(const std::vector<CatalogPoint>& points,
                                                   const std::array<Mesh, 3>& fields, const Geometry& g) {
        g.validate();
        detail::validate_catalog(points, g);
        bool prepared = true;
        for (const auto& f : fields)
            prepared &= f.get_nmesh() == g.nmesh && detail::is_real(f) && f.get_n_extra_slices_left() >= 1 &&
                        f.get_n_extra_slices_right() >= 1;
        require(prepared, "Displacement fields require matching real meshes with communicated one-cell halos");
        struct Query {
            Vector position;
            unsigned long long index;
            int origin;
        };
        struct Answer {
            Vector value;
            unsigned long long index;
        };
        const int slab = g.nmesh / FML::NTasks;
        std::vector<std::vector<Query>> outgoing(FML::NTasks);
        for (size_t i = 0; i < points.size(); ++i) {
            const int ix = int((points[i].position[0] - g.minimum[0]) / g.cellsize());
            outgoing[ix / slab].push_back({points[i].position, i, FML::ThisTask});
        }
        const auto queries = detail::route(outgoing);
        std::vector<std::vector<Answer>> replies(FML::NTasks);
        for (const auto& query : queries) {
            Vector value{};
            detail::cic(query.position, g, [&](auto c, double w) {
                c[0] -= fields[0].get_local_x_start();
                for (int d = 0; d < 3; ++d)
                    value[d] += w * fields[d].get_real(c);
            });
            replies[query.origin].push_back({value, query.index});
        }
        const auto answers = detail::route(replies);
        std::vector<Vector> result(points.size());
        for (const auto& answer : answers)
            result[size_t(answer.index)] = answer.value;
        return result;
    }
    inline ReconstructionResult reconstruct(const std::vector<CatalogPoint>& data,
                                            const std::vector<CatalogPoint>& randoms, const Geometry& g,
                                            const ReconstructionOptions& options = {}) {
        g.validate();
        require(options.bias > 0 && std::isfinite(options.bias) && options.smoothing_radius >= 0 &&
                    std::isfinite(options.smoothing_radius) && options.minimum_random_fraction > 0 &&
                    std::isfinite(options.minimum_random_fraction),
                "Invalid reconstruction smoothing/random threshold");
        const double wd = weight_sum(data), wr = weight_sum(randoms);
        require(wd > 0 && wr > 0 && std::isfinite(wd) && std::isfinite(wr),
                "Reconstruction needs positive-weight data and random catalogues");
        ReconstructionResult result;
        result.alpha = wd / wr;
        auto density = paint_catalog(data, g), selection = paint_catalog(randoms, g);
        detail::smooth(density, g, options.smoothing_radius);
        detail::smooth(selection, g, options.smoothing_radius);
        const double minimum = options.minimum_random_fraction * wr / (double(g.nmesh) * g.nmesh * g.nmesh);
        for (auto i : density.get_real_range()) {
            const double r = selection.get_real_from_index(i);
            const bool active = r > minimum && (!options.mask || options.mask(g.position(density, i)));
            const double source = active ? (density.get_real_from_index(i) / (result.alpha * r) - 1) / options.bias : 0;
            density.set_real_from_index(i, Real(source));
        }
        selection.free();
        auto solution = solve_displacement(density, g, options);
        result.solver = solution.report;
        if (!result.solver.converged)
            throw std::runtime_error("Survey reconstruction did not converge; no shifted catalogue returned");
        result.data_displacement = sample_displacement(data, solution.field, g);
        result.random_displacement = sample_displacement(randoms, solution.field, g);
        auto shift = [&](const auto& input, const auto& displacement, bool rsd) {
            auto output = input;
            for (size_t i = 0; i < input.size(); ++i) {
                const auto n = g.los(input[i].position);
                const double radial = rsd ? options.growth_rate * dot(displacement[i], n) : 0;
                for (int d = 0; d < 3; ++d)
                    output[i].position[d] -= displacement[i][d] + radial * n[d];
            }
            return output;
        };
        result.data = shift(data, result.data_displacement, true);
        result.randoms =
            shift(randoms, result.random_displacement, options.convention == ReconstructionConvention::RecSym);
        // Coordinates are never wrapped or clipped; caller can enlarge the mesh if shifts exhaust its padding.
        return result;
    }
} // namespace FML::SURVEY
#endif

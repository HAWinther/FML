#ifndef FML_SURVEY_WINDOW_HEADER
#define FML_SURVEY_WINDOW_HEADER
#include <FML/Survey/SurveyPower.h>
namespace FML::SURVEY {
    struct WindowMultipoles {
        std::vector<int> ells;
        std::vector<double> edges, s, nvectors, shell_volume;
        std::vector<std::vector<double>> q;
        double alpha = 0, i2 = 0;
    };
    /// FFT correlation of the random selection field; mesh-node local LOS.
    /// At fixed order this uses O(Nmesh^3 log Nmesh) work and three mesh buffers.
    inline WindowMultipoles estimate_window(const std::vector<CatalogPoint> & randoms,
                                            const Geometry & g,
                                            const std::vector<double> & edges,
                                            double alpha,
                                            double i2,
                                            int maxell = 8,
                                            bool subtract_self = true) {
        g.validate();
        detail::validate_edges(edges);
        require(alpha > 0 && i2 > 0 && std::isfinite(alpha) && std::isfinite(i2) && maxell >= 0 && maxell <= 8 &&
                    maxell % 2 == 0 && edges.back() <= g.boxsize * .5,
                "Invalid window normalization/order or separation beyond half the padded box");
        auto selection = paint_catalog(randoms, g);
        Vector low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
        for (const auto & p : randoms)
            if (p.weight > 0)
                for (int d = 0; d < 3; ++d) {
                    low[d] = std::min(low[d], p.position[d]);
                    high[d] = std::max(high[d], p.position[d]);
                }
        bool padded = true;
        for (int d = 0; d < 3; ++d) {
            FML::MinOverTasks(&low[d]);
            FML::MaxOverTasks(&high[d]);
            // Include the CIC cloud extent, so periodic images cannot enter requested separations.
            padded &= high[d] - low[d] + 2 * g.cellsize() < g.boxsize * .5;
        }
        require(padded, "Window FFT needs survey extent plus CIC support below half the box in each direction");
        const double cellvolume = std::pow(g.cellsize(), 3);
        for (auto i : selection.get_real_range())
            selection.set_real_from_index(i, Real(selection.get_real_from_index(i) * alpha / cellvolume));
        auto fourier = selection;
        fourier.fftw_r2c();
        WindowMultipoles result;
        result.edges = edges;
        result.alpha = alpha;
        result.i2 = i2;
        const size_t nb = edges.size() - 1;
        for (int ell = 0; ell <= maxell; ell += 2)
            result.ells.push_back(ell);
        result.s.assign(nb, 0);
        result.nvectors.assign(nb, 0);
        result.shell_volume.resize(nb);
        result.q.assign(result.ells.size(), std::vector<double>(nb));
        std::vector<std::vector<double>> coefficients;
        for (auto ell : result.ells)
            coefficients.push_back(detail::legendre_coefficients(ell));
        auto separation = [&](const Mesh & mesh, FML::IndexIntType i) {
            auto c = mesh.get_coord_from_index(i);
            c[0] += mesh.get_local_x_start();
            Vector s;
            for (int d = 0; d < 3; ++d)
                s[d] = (c[d] > g.nmesh / 2 ? c[d] - g.nmesh : c[d]) * g.cellsize();
            return s;
        };
        // Lattice vectors per bin: all, and with s > 0 (the shell average of ell > 0 has no s = 0 term)
        std::vector<double> nonzero(nb, 0);
        for (auto i : selection.get_real_range()) {
            auto s = separation(selection, i);
            const double r = std::sqrt(dot(s, s));
            const int b = detail::bin(r, edges);
            if (b < 0)
                continue;
            ++result.nvectors[b];
            nonzero[b] += r > 0;
            result.s[b] += r;
        }
        detail::sum(result.s);
        detail::sum(result.nvectors);
        detail::sum(nonzero);
        // The shell average over the lattice vectors of a bin is not the angular average: the vectors of a
        // shell are not isotropic (cubic symmetry, few vectors at small s), so the average of L_ell(shat.n)
        // over them is not zero for ell >= 4, and the large isotropic part of the pair function leaks into
        // Q_4, Q_6, Q_8 (errors O(0.1-1) at s of a few cells, O(0.03) at 10 cells). With G(v) the pair
        // function of a tensor and F(v) its angular factor, each shell gets
        //   mean(G F) - mean(G) [mean(F) - <F>_sphere]
        // which is exact for G isotropic on the shell (the anisotropic part times the lattice anisotropy
        // remains, a second-order error).
        std::vector<std::vector<double>> correction(result.ells.size(), std::vector<double>(nb, 0));
        auto sphere_moment = [](const std::array<int, 3> & e) {
            // <x^a y^b z^c> over the unit sphere: (a-1)!!(b-1)!!(c-1)!!/(a+b+c+1)!! for a, b, c even, else 0
            auto odd_factorial = [](int n) {
                double f = 1;
                for (int j = n; j > 1; j -= 2)
                    f *= j;
                return f;
            };
            if (e[0] % 2 || e[1] % 2 || e[2] % 2)
                return 0.0;
            return odd_factorial(e[0] - 1) * odd_factorial(e[1] - 1) * odd_factorial(e[2] - 1) /
                   odd_factorial(e[0] + e[1] + e[2] + 1);
        };
        const auto tensors = detail::moments(g.local_los ? maxell : 0);
        // Exact mesh self-pair subtraction, including CIC clouds and node LOS: for every tensor the 27 offsets
        // b - a of the CIC cells a, b of the same random, in one pass over the randoms
        const size_t nt = tensors.size();
        std::vector<std::vector<double>> self_pairs(nt, std::vector<double>(27, 0));
        if (subtract_self) {
            std::vector<double> total(nt * 27, 0);
#ifdef USE_OMP
#pragma omp parallel
#endif
            {
                std::vector<double> local(nt * 27, 0), angular(nt);
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
                for (size_t i = 0; i < randoms.size(); ++i) {
                    const auto & p = randoms[i];
                    if (p.weight == 0)
                        continue;
                    const double factor = alpha * alpha * p.weight * p.weight / cellvolume;
                    detail::cic(p.position, g, [&](auto a, double wa) {
                        Vector x;
                        for (int d = 0; d < 3; ++d)
                            x[d] = g.minimum[d] + a[d] * g.cellsize();
                        const auto n = g.los(x);
                        for (size_t t = 0; t < nt; ++t)
                            angular[t] = factor * wa * detail::monomial(n, tensors[t]);
                        detail::cic(p.position, g, [&](auto b, double wb) {
                            const int index = ((b[0] - a[0] + 1) * 3 + b[1] - a[1] + 1) * 3 + b[2] - a[2] + 1;
                            for (size_t t = 0; t < nt; ++t)
                                local[t * 27 + index] += angular[t] * wb;
                        });
                    });
                }
#ifdef USE_OMP
#pragma omp critical
#endif
                for (size_t j = 0; j < total.size(); ++j)
                    total[j] += local[j];
            }
            detail::sum(total);
            for (size_t t = 0; t < nt; ++t)
                for (int j = 0; j < 27; ++j)
                    self_pairs[t][j] = total[t * 27 + j];
        }
        for (const auto & tensor : tensors) {
            std::vector<double> gsum(nb, 0);
            std::vector<std::vector<double>> fsum(result.ells.size(), std::vector<double>(nb, 0));
            const auto & self = self_pairs[&tensor - tensors.data()];
            auto correlation = detail::empty_mesh(g);
            for (auto i : selection.get_real_range())
                correlation.set_real_from_index(
                    i,
                    Real(selection.get_real_from_index(i) * detail::monomial(g.los(g.position(selection, i)), tensor)));
            correlation.fftw_r2c();
            for (auto i : correlation.get_fourier_range())
                correlation.set_fourier_from_index(i,
                                                   std::conj(correlation.get_fourier_from_index(i)) *
                                                       fourier.get_fourier_from_index(i));
            correlation.fftw_c2r();
            for (auto i : correlation.get_real_range()) {
                auto s = separation(correlation, i);
                const double r = std::sqrt(dot(s, s));
                const int b = detail::bin(r, edges);
                if (b < 0)
                    continue;
                double value = correlation.get_real_from_index(i) * g.volume();
                std::array<int, 3> delta;
                bool near = true;
                for (int d = 0; d < 3; ++d) {
                    delta[d] = int(std::lround(s[d] / g.cellsize()));
                    near &= std::abs(delta[d]) <= 1;
                }
                if (near)
                    value -= self[((delta[0] + 1) * 3 + delta[1] + 1) * 3 + delta[2] + 1];
                const auto shat = unit(s);
                if (r > 0)
                    gsum[b] += value;
                for (size_t l = 0; l < result.ells.size(); ++l) {
                    const int ell = result.ells[l];
                    if (r == 0 && ell > 0)
                        continue;
                    double factor;
                    if (!g.local_los)
                        factor = detail::legendre(ell, dot(shat, unit(g.fixed_los)));
                    else
                        factor = tensor.degree <= ell ? coefficients[l][tensor.degree] * tensor.multiplicity *
                                                            detail::monomial(shat, tensor) :
                                                        0;
                    result.q[l][b] += (2 * ell + 1) * factor * value / i2;
                    if (r > 0)
                        fsum[l][b] += factor;
                }
            }
            detail::sum(gsum);
            for (size_t l = 0; l < result.ells.size(); ++l) {
                const int ell = result.ells[l];
                if (ell == 0)
                    continue; // the angular factor of the monopole is 1 on any shell
                detail::sum(fsum[l]);
                double sphere;
                if (!g.local_los)
                    sphere = 0; // <L_ell(shat.n)> over the sphere, ell > 0
                else
                    sphere = tensor.degree <= ell ?
                                 coefficients[l][tensor.degree] * tensor.multiplicity * sphere_moment(tensor.exponent) :
                                 0;
                for (size_t b = 0; b < nb; ++b)
                    if (nonzero[b] > 0)
                        correction[l][b] -=
                            (2 * ell + 1) * (gsum[b] / nonzero[b]) * (fsum[l][b] / nonzero[b] - sphere) / i2;
            }
        }
        for (auto & q : result.q)
            detail::sum(q);
        for (size_t b = 0; b < nb; ++b) {
            const double count = result.nvectors[b];
            result.s[b] = count ? result.s[b] / count : .5 * (edges[b] + edges[b + 1]);
            result.shell_volume[b] = 4 * M_PI / 3 * (std::pow(edges[b + 1], 3) - std::pow(edges[b], 3));
            for (size_t l = 0; l < result.ells.size(); ++l) {
                // ell > 0: average over the s > 0 vectors (a bin with only s = 0 has no direction: 0)
                const double n = result.ells[l] == 0 ? count : nonzero[b];
                result.q[l][b] = n ? result.q[l][b] / n + correction[l][b] : 0;
            }
        }
        return result;
    }
    namespace detail {
        inline double bessel(int ell, double x) {
            if (x == 0)
                return ell ? 0 : 1;
            if (x < ell + 1.0) {
                double leading = 1;
                for (int j = 1; j <= ell; ++j)
                    leading *= x / (2 * j + 1);
                double term = 1, sum = 1;
                for (int j = 1; j < 100; ++j) {
                    term *= -x * x / (2 * j * (2 * ell + 2 * j + 1.0));
                    sum += term;
                    if (std::abs(term) < 1e-16 * std::max(1.0, std::abs(sum)))
                        break;
                }
                return leading * sum;
            }
            double j0 = std::sin(x) / x;
            if (!ell)
                return j0;
            double j1 = (j0 - std::cos(x)) / x;
            for (int j = 1; j < ell; ++j) {
                const double next = (2 * j + 1) * j1 / x - j0;
                j0 = j1;
                j1 = next;
            }
            return j1;
        }
        // Positive half of the 16-point Gauss-Legendre rule.
        inline constexpr std::array<double, 8> gx{.09501250983763744,
                                                  .2816035507792589,
                                                  .4580167776572274,
                                                  .6178762444026438,
                                                  .7554044083550030,
                                                  .8656312023878318,
                                                  .9445750230732326,
                                                  .9894009349916499};
        inline constexpr std::array<double, 8> gw{.1894506104550685,
                                                  .1826034150449236,
                                                  .1691565193950025,
                                                  .1495959888165767,
                                                  .1246289712555339,
                                                  .0951585116824928,
                                                  .0622535239386479,
                                                  .0271524594117541};
        template <class Function>
        double quadrature(double a, double b, Function f) {
            double sum = 0;
            for (size_t j = 0; j < gx.size(); ++j)
                sum += gw[j] * (f(.5 * (a + b) + .5 * (b - a) * gx[j]) + f(.5 * (a + b) - .5 * (b - a) * gx[j]));
            return .5 * (b - a) * sum;
        }
        inline double coupling(int out, int in, int window) {
            const double value = (2 * out + 1) * .5 * quadrature(-1, 1, [&](double mu) {
                                     return legendre(out, mu) * legendre(in, mu) * legendre(window, mu);
                                 });
            return std::abs(value) < 1e-12 ? 0 : value;
        }
    } // namespace detail
    struct WindowMatrix {
        std::vector<int> input_ells, output_ells;
        std::vector<double> input_edges, output_k, output_edges;
        std::vector<std::vector<double>> values; // [output ell-major row][input ell-major bandpower]
        bool global_integral_constraint = false;
        std::vector<double> apply(const std::vector<double> & bandpowers) const {
            if (values.empty() || bandpowers.size() != values[0].size())
                throw std::invalid_argument("Window matrix input has wrong number of bandpowers");
            for (const auto & row : values)
                if (row.size() != bandpowers.size())
                    throw std::invalid_argument("Window matrix has inconsistent row lengths");
            std::vector<double> result(values.size());
            for (size_t r = 0; r < result.size(); ++r)
                result[r] = std::inner_product(values[r].begin(), values[r].end(), bandpowers.begin(), 0.0);
            return result;
        }
    };
    /// Local-plane-parallel configuration-space window convolution. Input P is
    /// piecewise constant in each k bin. Output is evaluated at output_k nodes.
    /// Optional conventional global IC: xi_l -> xi_l - C Q_l, making convolved P0(0)=0.
    inline WindowMatrix make_window_matrix(const WindowMultipoles & w,
                                           const std::vector<double> & input_edges,
                                           const std::vector<double> & output_k,
                                           const std::vector<int> & input_ells = {0, 2, 4},
                                           const std::vector<int> & output_ells = {0, 2, 4},
                                           bool global_ic = true) {
        auto validate = [](bool ok, const char * message) {
            if (!ok)
                throw std::invalid_argument(message);
        };
        detail::validate_edges(input_edges, false);
        detail::validate_ells(input_ells, 4, false);
        detail::validate_ells(output_ells, 4, false);
        const size_t ns = w.s.size(), nin = input_edges.size() - 1;
        bool valid = !output_k.empty() && w.q.size() == w.ells.size() && w.shell_volume.size() == ns && ns > 0 &&
                     w.edges.size() == ns + 1 && (w.nvectors.empty() || w.nvectors.size() == ns);
        validate(valid, "Invalid window table array dimensions (s, edges, shell_volume, q, nvectors)");
        detail::validate_edges(w.edges, false);
        for (double k : output_k)
            valid &= std::isfinite(k) && k >= 0;
        for (const auto & q : w.q) {
            valid &= q.size() == ns;
            for (double value : q)
                valid &= std::isfinite(value);
        }
        detail::validate_ells(w.ells, 8, false);
        for (size_t i = 0; i < ns; ++i)
            valid &= std::isfinite(w.s[i]) && w.s[i] >= 0 && w.shell_volume[i] > 0;
        const int required = *std::max_element(input_ells.begin(), input_ells.end()) +
                             *std::max_element(output_ells.begin(), output_ells.end());
        for (int ell = 0; ell <= required; ell += 2)
            valid &= std::find(w.ells.begin(), w.ells.end(), ell) != w.ells.end();
        validate(valid, "Invalid/incomplete window table (need window multipoles through input ell + output ell)");
        // The s integrals use 16-point Gauss-Legendre nodes in every separation bin, with xi of the input
        // bandpower evaluated exactly at the nodes and Q linearly interpolated between the bin separations w.s
        // (constant before the first and after the last bin with lattice vectors, zero beyond the last edge).
        // A single point per bin (xi and the Bessel kernel at w.s only) is not converged unless k ds << 1: for
        // ds = 10 Mpc/h it gave errors of 10-25% at k = 0.1-0.2 h/Mpc.
        std::vector<size_t> used; // bins with lattice vectors (all bins for a table without nvectors)
        for (size_t s = 0; s < ns; ++s)
            if (w.nvectors.empty() || w.nvectors[s] > 0)
                used.push_back(s);
        validate(!used.empty(), "Window table has no separation bin with lattice vectors");
        std::vector<double> nodes, node_weights;
        for (size_t s = 0; s < ns; ++s) {
            const double lo = w.edges[s], hi = w.edges[s + 1];
            for (size_t j = 0; j < detail::gx.size(); ++j)
                for (double sign : {-1.0, 1.0}) {
                    const double x = .5 * (lo + hi) + sign * .5 * (hi - lo) * detail::gx[j];
                    nodes.push_back(x);
                    node_weights.push_back(4 * M_PI * x * x * .5 * (hi - lo) * detail::gw[j]);
                }
        }
        const size_t nn = nodes.size();
        // Q_j at the nodes
        std::vector<std::vector<double>> qnode(w.ells.size(), std::vector<double>(nn));
        for (size_t node = 0; node < nn; ++node) {
            const double x = nodes[node];
            size_t hi = 0;
            while (hi < used.size() && w.s[used[hi]] < x)
                ++hi;
            for (size_t j = 0; j < w.ells.size(); ++j) {
                const auto & qj = w.q[j];
                if (hi == 0)
                    qnode[j][node] = qj[used.front()];
                else if (hi == used.size())
                    qnode[j][node] = qj[used.back()];
                else {
                    const size_t a = used[hi - 1], b = used[hi];
                    const double t = (x - w.s[a]) / (w.s[b] - w.s[a]);
                    qnode[j][node] = (1 - t) * qj[a] + t * qj[b];
                }
            }
        }
        auto qindex = [&](int ell) { return size_t(std::find(w.ells.begin(), w.ells.end(), ell) - w.ells.begin()); };
        WindowMatrix matrix;
        matrix.input_ells = input_ells;
        matrix.output_ells = output_ells;
        matrix.input_edges = input_edges;
        matrix.output_k = output_k;
        matrix.global_integral_constraint = global_ic;
        matrix.values.assign(output_ells.size() * output_k.size(), std::vector<double>(input_ells.size() * nin));
        double volume = 0;
        for (size_t node = 0; node < nn; ++node)
            volume += node_weights[node] * qnode[qindex(0)][node];
        validate(!global_ic || (volume > 0 && std::isfinite(volume)),
                 "Global integral constraint needs positive integrated Q0");
        // Bessel kernels of the outputs at the nodes
        std::vector<std::vector<std::vector<double>>> kernel(
            output_ells.size(),
            std::vector<std::vector<double>>(output_k.size(), std::vector<double>(nn)));
        for (size_t out = 0; out < output_ells.size(); ++out)
            for (size_t k = 0; k < output_k.size(); ++k)
                for (size_t node = 0; node < nn; ++node)
                    kernel[out][k][node] =
                        node_weights[node] * detail::bessel(output_ells[out], output_k[k] * nodes[node]);
        for (size_t l = 0; l < input_ells.size(); ++l) {
            const int in = input_ells[l];
            // Window factor of each output multipole: sum_j coupling(out, in, ell_j) Q_j
            std::vector<std::vector<double>> factor(5, std::vector<double>(nn, 0));
            for (int ell = 0; ell <= 4; ell += 2)
                for (size_t j = 0; j < w.ells.size(); ++j) {
                    const double c = detail::coupling(ell, in, w.ells[j]);
                    if (c != 0)
                        for (size_t node = 0; node < nn; ++node)
                            factor[ell][node] += c * qnode[j][node];
                }
            for (size_t b = 0; b < nin; ++b) {
                std::vector<double> xi(nn);
                for (size_t node = 0; node < nn; ++node)
                    xi[node] = (in % 4 ? -1 : 1) / (2 * M_PI * M_PI) *
                               detail::quadrature(input_edges[b], input_edges[b + 1], [&](double k) {
                                   return k * k * detail::bessel(in, k * nodes[node]);
                               });
                double ic = 0;
                if (global_ic)
                    for (size_t node = 0; node < nn; ++node)
                        ic += node_weights[node] * factor[0][node] * xi[node] / volume;
                for (size_t out = 0; out < output_ells.size(); ++out) {
                    const int ell = output_ells[out];
                    const size_t jq = qindex(ell);
                    std::vector<double> xi_out(nn);
                    for (size_t node = 0; node < nn; ++node)
                        xi_out[node] = factor[ell][node] * xi[node] - ic * qnode[jq][node];
                    for (size_t k = 0; k < output_k.size(); ++k) {
                        double value = 0;
                        for (size_t node = 0; node < nn; ++node)
                            value += kernel[out][k][node] * xi_out[node];
                        matrix.values[out * output_k.size() + k][l * nin + b] = (ell % 4 ? -1 : 1) * value;
                    }
                }
            }
        }
        return matrix;
    }
    /// Continuous k^2-weighted output-shell averages, useful for measured bandpowers.
    /// Unlike make_window_matrix at output_k nodes, this integrates across each output bin.
    inline WindowMatrix make_binned_window_matrix(const WindowMultipoles & window,
                                                  const std::vector<double> & input_edges,
                                                  const std::vector<double> & output_edges,
                                                  const std::vector<int> & input_ells = {0, 2, 4},
                                                  const std::vector<int> & output_ells = {0, 2, 4},
                                                  bool global_ic = true) {
        detail::validate_edges(output_edges, false);
        const size_t nb = output_edges.size() - 1;
        std::vector<double> nodes, weights;
        for (size_t b = 0; b < nb; ++b) {
            const double lo = output_edges[b], hi = output_edges[b + 1];
            const double volume = (hi * hi * hi - lo * lo * lo) / 3;
            for (size_t j = 0; j < detail::gx.size(); ++j)
                for (double sign : {-1.0, 1.0}) {
                    const double k = .5 * (lo + hi) + sign * .5 * (hi - lo) * detail::gx[j];
                    nodes.push_back(k);
                    weights.push_back(.5 * (hi - lo) * detail::gw[j] * k * k / volume);
                }
        }
        const auto sampled = make_window_matrix(window, input_edges, nodes, input_ells, output_ells, global_ic);
        WindowMatrix result;
        result.input_edges = input_edges;
        result.output_edges = output_edges;
        result.input_ells = input_ells;
        result.output_ells = output_ells;
        result.global_integral_constraint = global_ic;
        result.output_k.resize(nb);
        result.values.assign(output_ells.size() * nb, std::vector<double>(sampled.values[0].size()));
        for (size_t b = 0; b < nb; ++b) {
            for (size_t q = 16 * b; q < 16 * (b + 1); ++q)
                result.output_k[b] += weights[q] * nodes[q];
            for (size_t ell = 0; ell < output_ells.size(); ++ell)
                for (size_t q = 16 * b; q < 16 * (b + 1); ++q)
                    for (size_t col = 0; col < result.values[0].size(); ++col)
                        result.values[ell * nb + b][col] += weights[q] * sampled.values[ell * nodes.size() + q][col];
        }
        return result;
    }
} // namespace FML::SURVEY
#endif

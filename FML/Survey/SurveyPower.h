#ifndef FML_SURVEY_POWER_HEADER
#define FML_SURVEY_POWER_HEADER
#include <FML/Survey/SurveyMesh.h>
namespace FML::SURVEY {
    struct PowerOptions {
        std::vector<double> edges;
        std::vector<int> ells{0, 2, 4};
        double i2 = 0; // <=0: derive from random nbar.
        bool compensate_cic = true;
        bool subtract_shotnoise = true;
        // Shot noise of the CIC-painted mesh including its aliased images (Jing 2005): the continuum value times
        // prod_i [1 - 2/3 sin^2(k_i h/2)], divided by the CIC window^2 when compensating. false: the continuum
        // value, which after compensation is too small by 0.7% at k h = 1.5 along an axis (growing as (k h)^4),
        // and without compensation too large by 1 - prod_i [1 - 2/3 sin^2(k_i h/2)]
        bool aliased_shotnoise = true;
    };
    struct PowerSpectrum {
        std::vector<int> ells;
        std::vector<double> edges, k, nmodes;
        std::vector<std::vector<double>> power, raw, shotnoise;
        double alpha = 0, i2 = 0;
    };
    namespace detail {
        inline void validate_ells(const std::vector<int>& ells, int maximum = 8, bool collective = true) {
            bool ok = !ells.empty();
            for (size_t i = 0; i < ells.size(); ++i)
                ok &= ells[i] >= 0 && ells[i] <= maximum && ells[i] % 2 == 0 &&
                      std::find(ells.begin(), ells.begin() + i, ells[i]) == ells.begin() + i;
            if (collective)
                require(ok, "Survey multipoles must be distinct even orders within the supported range");
            else if (!ok)
                throw std::invalid_argument(
                    "Survey multipoles must be distinct even orders within the supported range");
        }
        // A_l(k) = FFT[F(x) L_l(khat.los(x))], with F in weighted counts per cell.
        inline std::vector<Mesh> angular_transforms(const Mesh& field, const Geometry& g,
                                                    const std::vector<int>& ells) {
            validate_ells(ells, 4);
            const int maxell = *std::max_element(ells.begin(), ells.end());
            std::vector<Mesh> output;
            std::vector<std::vector<double>> coefficients;
            for (auto ell : ells) {
                output.emplace_back(g.nmesh, 0, 0);
                output.back().set_grid_status_real(false);
                output.back().fill_fourier_grid(Complex(0));
                coefficients.push_back(legendre_coefficients(ell));
            }
            const auto tensors = moments(g.local_los ? maxell : 0);
            for (const auto& tensor : tensors) {
                auto moment = empty_mesh(g);
                for (auto i : field.get_real_range()) {
                    const double value = monomial(g.los(g.position(field, i)), tensor);
                    moment.set_real_from_index(i, Real(field.get_real_from_index(i) * value));
                }
                moment.fftw_r2c();
                for (auto i : moment.get_fourier_range()) {
                    Vector k;
                    double kmag;
                    moment.get_fourier_wavevector_and_norm_by_index(i, k, kmag);
                    if (kmag > 0)
                        for (auto& x : k)
                            x /= kmag;
                    for (size_t l = 0; l < ells.size(); ++l) {
                        double factor;
                        if (!g.local_los)
                            factor = legendre(ells[l], dot(k, unit(g.fixed_los)));
                        else
                            factor = tensor.degree <= ells[l]
                                         ? coefficients[l][tensor.degree] * tensor.multiplicity * monomial(k, tensor)
                                         : 0;
                        output[l].set_fourier_from_index(i, output[l].get_fourier_from_index(i) +
                                                                Real(factor) * moment.get_fourier_from_index(i));
                    }
                }
            }
            return output;
        }
    } // namespace detail
    /// Collective. Catalogues may start on arbitrary ranks; returned binned arrays are global.
    /// shifted_randoms: post-reconstruction negative field; original randoms still define I2/alpha/selection.
    inline PowerSpectrum estimate_power(const std::vector<CatalogPoint>& data, const std::vector<CatalogPoint>& randoms,
                                        const Geometry& g, const PowerOptions& options,
                                        const std::vector<CatalogPoint>* shifted_randoms = nullptr) {
        g.validate();
        detail::validate_edges(options.edges);
        detail::validate_ells(options.ells, 4);
        const double wd = weight_sum(data), wr = weight_sum(randoms);
        require(wd > 0 && wr > 0 && std::isfinite(wd) && std::isfinite(wr),
                "Survey power needs nonempty positive-weight data and randoms");
        PowerSpectrum result;
        result.alpha = wd / wr;
        result.i2 = normalization(randoms, result.alpha, options.i2);
        result.ells = options.ells;
        result.edges = options.edges;
        auto density = paint_catalog(data, g);
        const auto& negative = shifted_randoms ? *shifted_randoms : randoms;
        const double wn = weight_sum(negative);
        require(wn > 0, "Shifted random catalogue has zero weight");
        const double alpha_negative = wd / wn;
        auto selection = paint_catalog(negative, g);
        for (auto i : density.get_real_range())
            density.set_real_from_index(i, density.get_real_from_index(i) -
                                               Real(alpha_negative) * selection.get_real_from_index(i));
        selection.free();
        auto ells = options.ells;
        if (std::find(ells.begin(), ells.end(), 0) == ells.end())
            ells.push_back(0);
        auto transforms = detail::angular_transforms(density, g, ells);
        const size_t monopole = std::find(ells.begin(), ells.end(), 0) - ells.begin();
        const size_t nb = options.edges.size() - 1, nl = options.ells.size();
        result.k.assign(nb, 0);
        result.nmodes.assign(nb, 0);
        result.raw.assign(nl, std::vector<double>(nb));
        result.shotnoise = result.raw;
        result.power = result.raw;
        const auto tensors = detail::moments(*std::max_element(ells.begin(), ells.end()));
        std::vector<double> shot(tensors.size(), 0);
        auto add_noise = [&](const auto& points, double factor) {
            for (const auto& p : points) {
                const auto n = g.los(p.position);
                for (size_t j = 0; j < tensors.size(); ++j)
                    shot[j] += factor * p.weight * p.weight * detail::monomial(n, tensors[j]);
            }
        };
        add_noise(data, 1);
        add_noise(negative, alpha_negative * alpha_negative);
        detail::sum(shot);
        std::vector<std::vector<double>> coefficients;
        for (auto ell : options.ells)
            coefficients.push_back(detail::legendre_coefficients(ell));
        const double fftnorm = double(g.nmesh) * g.nmesh * g.nmesh;
        for (auto i : transforms[0].get_fourier_range()) {
            Vector k;
            double kmag;
            transforms[0].get_fourier_wavevector_and_norm_by_index(i, k, kmag);
            if (kmag == 0)
                continue;
            for (auto& x : k)
                x /= g.boxsize;
            kmag /= g.boxsize;
            const int b = detail::bin(kmag, options.edges);
            if (b < 0)
                continue;
            const auto ik = transforms[0].get_fourier_integer_wavevector_from_index(i);
            const double count = FML::GRID::fourier_mode_multiplicity(ik[2], g.nmesh);
            const double window = options.compensate_cic ? detail::cic_window(k, g) : 1;
            double alias = 1;
            if (options.aliased_shotnoise)
                for (auto v : k)
                    alias *= 1 - 2. / 3. * std::pow(std::sin(.5 * v * g.cellsize()), 2);
            // the continuum shot noise is not affected by the compensation; the aliased one is in the raw power
            const double noise_factor = options.aliased_shotnoise ? alias / (window * window) : 1;
            Vector khat = k;
            for (auto& x : khat)
                x /= kmag;
            result.nmodes[b] += count;
            result.k[b] += count * kmag;
            for (size_t l = 0; l < nl; ++l) {
                const int ell = options.ells[l];
                double noise = 0;
                if (!g.local_los)
                    noise = shot[0] * detail::legendre(ell, dot(khat, unit(g.fixed_los)));
                else
                    for (size_t j = 0; j < tensors.size(); ++j) {
                        const auto& t = tensors[j];
                        if (t.degree <= ell)
                            noise += coefficients[l][t.degree] * t.multiplicity * detail::monomial(khat, t) * shot[j];
                    }
                const auto product =
                    transforms[monopole].get_fourier_from_index(i) * std::conj(transforms[l].get_fourier_from_index(i));
                result.raw[l][b] +=
                    count * (2 * ell + 1) * product.real() * fftnorm * fftnorm / (window * window * result.i2);
                result.shotnoise[l][b] += count * (2 * ell + 1) * noise_factor * noise / result.i2;
            }
        }
        detail::sum(result.k);
        detail::sum(result.nmodes);
        for (size_t l = 0; l < nl; ++l) {
            detail::sum(result.raw[l]);
            detail::sum(result.shotnoise[l]);
        }
        for (size_t b = 0; b < nb; ++b) {
            const double count = result.nmodes[b];
            result.k[b] = count ? result.k[b] / count : .5 * (options.edges[b] + options.edges[b + 1]);
            for (size_t l = 0; l < nl; ++l)
                if (count) {
                    result.raw[l][b] /= count;
                    result.shotnoise[l][b] /= count;
                    result.power[l][b] = result.raw[l][b] - (options.subtract_shotnoise ? result.shotnoise[l][b] : 0);
                }
        }
        return result;
    }
} // namespace FML::SURVEY
#endif

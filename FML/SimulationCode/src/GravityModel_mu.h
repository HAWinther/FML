#ifndef GRAVITYMODEL_MU_HEADER
#define GRAVITYMODEL_MU_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Global/Global.h>
#include <FML/NBody/NBody.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//=============================================================================
/// Phenomenological modified gravity: the Poisson equation for the potential that moves the particles is
/// nabla^2 Phi = 4 pi G mu(a,k) a^2 rho_m delta. No screening (the modification is applied to all scales and all
/// densities, as in linear theory). The growth factors (also 2LPT/3LPT) use G_eff/G = mu(a,k).
///
/// Parametrisations (gravity_model_mu_parametrization):
///   DE    : mu(a) = 1 + mu0 Omega_DE(a) / Omega_DE(a=1) (the "late-time" parametrisation of Planck 2015 XIV / DES;
///           the shared quasi-static theory FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h)
///   BZ    : mu(a,k) = (1 + beta1 lambda1^2 k^2 a^s) / (1 + lambda1^2 k^2 a^s) (Bertschinger & Zukin 2008, as in Planck
///           2015 XIV), k in h/Mpc and lambda1 in Mpc/h
///   table : mu(a) or mu(a,k) from a file (e.g. computed with hi_class): rows "a mu" or "a k mu" (k in h/Mpc) on a
///           regular grid (any order of the rows). Splined in (log a, log k); outside the table the closest value is
///           used
//=============================================================================
template <int NDIM>
class GravityModelMu final : public GravityModel<NDIM> {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using DVector2D = FML::INTERPOLATION::SPLINE::DVector2D;

  private:
    std::string parametrization{"DE"};
    double mu0{0.0};
    double BZ_beta1{1.0};
    double BZ_lambda1_mpch{0.0};
    double BZ_s{4.0};
    std::string table_filename;
    bool table_has_k{false};
    Spline mu_of_loga_spline;
    Spline2D mu_of_loga_logk_spline;

  public:
    explicit GravityModelMu(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "mu") {}

    bool scale_dependent() const { return parametrization == "BZ" or (parametrization == "table" and table_has_k); }

    /// mu(a, k) with k in h/Mpc
    double mu_of_a_k(double a, double k_hmpc) const {
        if (parametrization == "DE")
            return this->qs().mu(std::log(a), 0.0);
        if (parametrization == "BZ") {
            const double x = BZ_lambda1_mpch * BZ_lambda1_mpch * k_hmpc * k_hmpc * std::pow(a, BZ_s);
            return (1.0 + BZ_beta1 * x) / (1.0 + x);
        }
        if (table_has_k) {
            // A table needs k > 0: use the smallest k in the table for k = 0
            const double logk = k_hmpc > 0.0 ? std::log(k_hmpc) : -1e30;
            return mu_of_loga_logk_spline.eval(std::log(a), logk);
        }
        return mu_of_loga_spline(std::log(a));
    }

    /// The force (mu(a, k) delta(k) if it depends on k) and the growth are the defaults of GravityModel
    double GeffOverG(double a, double koverH0 = 0) const override { return mu_of_a_k(a, koverH0 * this->H0_hmpc); }
    bool geff_scale_dependent() const override { return scale_dependent(); }

    //========================================================================
    // Read the parameters and the table
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        parametrization = param.get<std::string>("gravity_model_mu_parametrization");
        if (parametrization == "DE") {
            mu0 = param.get<double>("gravity_model_mu_mu0");
        } else if (parametrization == "BZ") {
            auto p = param.get<std::vector<double>>("gravity_model_mu_BZ_params");
            FML::assert_mpi(p.size() == 3, "gravity_model_mu_BZ_params must be {beta1, lambda1 (Mpc/h), s}");
            BZ_beta1 = p[0];
            BZ_lambda1_mpch = p[1];
            BZ_s = p[2];
        } else if (parametrization == "table") {
            table_filename = param.get<std::string>("gravity_model_mu_filename");
            read_table();
        } else {
            throw std::runtime_error("Unknown gravity_model_mu_parametrization [" + parametrization +
                                     "]. Options: DE, BZ, table");
        }
    }

    void init() override {
        if (parametrization == "DE") {
            FML::assert_mpi(this->cosmo->get_OmegaLambda() > 0.0,
                            "gravity_model_mu_parametrization = DE needs Omega_DE > 0");
            this->quasi_static = std::make_shared<const FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_MuSigma>(
                this->cosmo->background(), mu0);
        }
        GravityModel<NDIM>::init();
    }

    /// Read a table with rows "a mu" or "a k mu" (regular grid in a and k, any order)
    void read_table() {
        const auto rows = FML::FILEUTILS::loadtxt(table_filename);
        const size_t ncol = rows.empty() ? 0 : rows[0].size();
        for (auto & row : rows)
            FML::assert_mpi(row.size() == ncol and (ncol == 2 or ncol == 3),
                            "gravity_model_mu_filename: every row must be 'a mu' or 'a k mu'");
        FML::assert_mpi(rows.size() >= 2, "gravity_model_mu_filename: too few rows");
        table_has_k = ncol == 3;
        auto unique_sorted = [&](int col) {
            std::vector<double> v;
            for (auto & r : rows)
                v.push_back(r[col]);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            return v;
        };
        auto avals = unique_sorted(0);
        for (auto a : avals)
            FML::assert_mpi(a > 0.0, "gravity_model_mu_filename: a must be > 0");
        DVector loga(avals.size());
        for (size_t i = 0; i < avals.size(); i++)
            loga[i] = std::log(avals[i]);
        auto index_of = [](const std::vector<double> & v, double x) {
            return size_t(std::lower_bound(v.begin(), v.end(), x) - v.begin());
        };
        if (not table_has_k) {
            FML::assert_mpi(avals.size() == rows.size(), "gravity_model_mu_filename: repeated values of a");
            DVector mu(avals.size());
            for (auto & r : rows)
                mu[index_of(avals, r[0])] = r[1];
            mu_of_loga_spline.create(loga, mu, "mu(log a)");
            return;
        }
        auto kvals = unique_sorted(1);
        for (auto k : kvals)
            FML::assert_mpi(k > 0.0, "gravity_model_mu_filename: k must be > 0");
        FML::assert_mpi(avals.size() >= 2 and kvals.size() >= 2 and avals.size() * kvals.size() == rows.size(),
                        "gravity_model_mu_filename: the rows 'a k mu' must form a regular grid in a and k");
        DVector logk(kvals.size());
        for (size_t i = 0; i < kvals.size(); i++)
            logk[i] = std::log(kvals[i]);
        DVector2D mu(avals.size(), DVector(kvals.size(), std::nan("")));
        for (auto & r : rows)
            mu[index_of(avals, r[0])][index_of(kvals, r[1])] = r[2];
        for (auto & v : mu)
            for (auto x : v)
                FML::assert_mpi(not std::isnan(x), "gravity_model_mu_filename: missing (a, k) points in the grid");
        mu_of_loga_logk_spline.create(loga, logk, mu, "mu(log a, log k)");
    }

    void info_model() const override {
        if (FML::ThisTask == 0) {
            std::cout << "# Parametrization  : " << parametrization << "\n";
            if (parametrization == "DE")
                std::cout << "# mu0              : " << mu0 << "\n";
            if (parametrization == "BZ")
                std::cout << "# beta1, lambda1 (Mpc/h), s : " << BZ_beta1 << " " << BZ_lambda1_mpch << " " << BZ_s
                          << "\n";
            if (parametrization == "table")
                std::cout << "# Table            : " << table_filename << (table_has_k ? " (a, k, mu)" : " (a, mu)")
                          << "\n";
            for (double a : {1.0, 0.5, 0.25}) {
                std::cout << "# mu(a = " << a << ") at k = 0.01 / 0.1 / 1 h/Mpc: ";
                for (double k : {0.01, 0.1, 1.0})
                    std::cout << mu_of_a_k(a, k) << " ";
                std::cout << "\n";
            }
        }
    }
};

#endif

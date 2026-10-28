#ifndef GRAVITYMODEL_MBETA_HEADER
#define GRAVITYMODEL_MBETA_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

/// (m(a),beta(a)) model
template <int NDIM>
class GravityModelmbeta : public GravityModelScreened<NDIM> {

    // Take in parameters used to define m(a), beta(a)
    std::vector<double> mbeta_params;

    // Spline for phi(a)/Mpl needed for screening method
    FML::INTERPOLATION::SPLINE::Spline phi_over_Mpl_of_a_spline;

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    GravityModelmbeta() : GravityModelScreened<NDIM>("mbeta") {}
    GravityModelmbeta(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "mbeta")

    {
        assert(this->get_name() == "mbeta");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModelScreened<NDIM>::info();
        if (FML::ThisTask == 0) {
            for (size_t i = 0; i < mbeta_params.size(); i++)
                std::cout << "# Params[" << i << "]        : " << mbeta_params[i] << "\n";
        }
        this->info_screening();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Internal method: m(a)/H0 and beta(a) and phi(a)/Mpl
    //========================================================================
    double phi_over_mpl_of_a(double a) const { return phi_over_Mpl_of_a_spline(a); }
    double m_over_H0_of_a(double a) const {
        return mbeta_params[2] * std::pow(a, mbeta_params[3]);

        // For example f(R) would be like this:
        // ...with mbeta_params[0] = 1.0/sqrt(6.0), mbeta_params[1] = 0 and mbeta_params[2] = fofr0
        // double OmegaM = this->cosmo->get_OmegaM();
        // double OmegaLambda = 1.0 - OmegaM;
        // double a3 = a * a * a;
        // double fac = OmegaM / a3 + 4.0 * OmegaLambda;
        // double fac0 = OmegaM + 4.0 * OmegaLambda;
        // double mass2 = fac * std::pow(fac / fac0, 2.0) / (2.0 * mbeta_params[2]);
        // return std::sqrt(mass2);
    }
    double beta_of_a(double a) const { return mbeta_params[0] * std::pow(a, mbeta_params[1]); }
    double mass_over_H0_squared(double a) const {
        double moverH0 = m_over_H0_of_a(a);
        return moverH0 * moverH0;
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double mass2a2 = a * a * mass_over_H0_squared(a);
        double koverH02 = koverH0 * koverH0;
        double beta = beta_of_a(a);
        return 1.0 + 2.0 * beta * beta * koverH02 / (koverH02 + mass2a2);
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta
    // and D^2 phi + m^2 phi = C delta where Phi = Phi_GR + phi
    // With screening C is a function of the newtonian potential
    //========================================================================
    void compute_screened_fifth_force(double a,
                                      double H0Box,
                                      double norm_poisson_equation,
                                      const typename GravityModelScreened<NDIM>::CouplingFunction & coupling,
                                      FFTWGrid<NDIM> & density_fourier,
                                      FFTWGrid<NDIM> & density_fifth_force) const override {
        // PhiCrit is phi(a) / (Mpl * 2 * beta) and screening-factor is PhiCrit/Phi_N
        double PhiCrit = phi_over_Mpl_of_a_spline(a) / (2.0 * beta_of_a(a));

        // Approximate screening method
        auto screening_function = [=, eff = this->screening_efficiency](double PhiNewton) {
            double screenfac = std::abs(PhiCrit / PhiNewton) * eff;
            return screenfac > 1.0 ? 1.0 : screenfac;
        };

        if (FML::ThisTask == 0)
            std::cout << "Adding fifth-force for (m(a),beta(a)) model (screening)\n";
        FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                  density_fifth_force,
                                                                  coupling,
                                                                  screening_function,
                                                                  norm_poisson_equation * std::pow(H0Box / a, 2));
    }

    std::string fifth_force_label() const override { return "(m(a),beta(a)) models"; }

    virtual void init() override {
        // phi(a)/Mpl is only needed for the screening method
        if (this->use_screening_method) {
            // For beta(a) = beta0 a^n and m(a) = m0 H0 a^m the integrand below goes as a^p with p = n - 3 - 2m:
            // p > 0 is needed for the integral from a = 0 to converge
            const double p = mbeta_params[1] - 3.0 - 2.0 * mbeta_params[3];
            FML::assert_mpi(
                p > 0.0,
                "Error in m(a) = m0H0a^m beta(a) = beta0 a^n models: n-4-2m <= -1.0 => phi(a) integral diverges");

            // phi_over_Mpl_of_a = 9 * OmegaM0 * Int_0^a beta(a) / [a^3(m/H0)^2] dloga
            double OmegaM0 = this->cosmo->get_OmegaM();
            auto integrand = [=](double a) {
                return 9.0 * OmegaM0 * beta_of_a(a) / (a * a * a * mass_over_H0_squared(a));
            };

            // The integral from 0 to amin is integrand(amin) / p (exact for the power laws; it was left out, which
            // is 3-25% of phi(a = 1) for p = 0.5-0.2), the rest numerically in log(a)
            const double amin = 0.001;
            const double amax = 1.0;
            const int npts = 10000;
            std::vector<double> a_array(npts, amin), loga_array(npts, std::log(amin)), phi_array(npts, 0.0);
            phi_array[0] = integrand(amin) / p;
            for (size_t i = 1; i < npts; i++) {
                loga_array[i] = std::log(amin) + std::log(amax / amin) * i / double(npts - 1);
                a_array[i] = std::exp(loga_array[i]);
                phi_array[i] = phi_array[i - 1] + integrand(a_array[i]) * (loga_array[i] - loga_array[i - 1]);
            }
            phi_over_Mpl_of_a_spline = Spline(a_array, phi_array, "phi(a)/Mpl spline for m(a),beta(a) models");
        }

        // Important: call base-call initializer in the end
        GravityModelScreened<NDIM>::init();
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        mbeta_params = param.get<std::vector<double>>("gravity_model_mbeta_params");
        FML::assert_mpi(mbeta_params.size() == 4,
                        "gravity_model_mbeta_params must have 4 numbers (beta0, beta power, m0, m power)");
        this->read_screening_parameters(param);
        this->scaledependent_growth = true;
    }
};

#endif

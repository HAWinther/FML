#ifndef GRAVITYMODEL_FOFR_HEADER
#define GRAVITYMODEL_FOFR_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

/// f(R) model
template <int NDIM>
class GravityModelFofR final : public GravityModel<NDIM> {
  protected:
    double fofr0{0.0};
    double nfofr{1.0};
    bool use_screening_method{true};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelFofR() : GravityModel<NDIM>("f(R)") {}
    GravityModelFofR(std::shared_ptr<BackgroundCosmology> cosmo)
        : GravityModel<NDIM>(cosmo, "f(R)")

    {
        assert(this->get_name() == "f(R)");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# fofr0    : " << fofr0 << "\n";
            std::cout << "# nfofr    : " << nfofr << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Internal method: m^2(a) / H0^2
    //========================================================================
    double mass_over_H0_squared(double a) const {
        double OmegaM = this->cosmo->get_OmegaM();
        double OmegaLambda = 1.0 - OmegaM;
        double a3 = a * a * a;
        double fac = OmegaM / a3 + 4.0 * OmegaLambda;
        double fac0 = OmegaM + 4.0 * OmegaLambda;
        double mass2 = fac * std::pow(fac / fac0, nfofr + 1.0) / ((nfofr + 1.0) * fofr0);
        return mass2;
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double mass2a2 = a * a * mass_over_H0_squared(a);
        double koverH02 = koverH0 * koverH0;
        return 1.0 + (1.0 / 3.0) * koverH02 / (koverH02 + mass2a2);
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta
    // and D^2 phi + m^2 phi = C delta where Phi = Phi_GR + phi
    // With screening C is a function of the newtonian potential
    //========================================================================
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Compute fifth-force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        auto coupling = [&](double kBox) { return GeffOverG(a, kBox / H0Box) - 1.0; };
        FFTWGrid<NDIM> density_fifth_force;

        if (use_screening_method) {

            // Approximate screening method
            const double OmegaM = this->cosmo->get_OmegaM();
            auto screening_function_fofr = [=](double PhiNewton) {
                double PhiCrit =
                    1.5 * fofr0 *
                    std::pow((OmegaM + 4.0 * (1.0 - OmegaM)) / (1.0 / (a * a * a) * OmegaM + 4.0 * (1.0 - OmegaM)),
                             nfofr + 1.0);
                double screenfac = std::abs(PhiCrit / PhiNewton);
                return screenfac > 1.0 ? 1.0 : screenfac;
            };

            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force f(R) (screening)\n";
            FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                      density_fifth_force,
                                                                      coupling,
                                                                      screening_function_fofr,
                                                                      norm_poisson_equation * std::pow(H0Box / a, 2));

        } else {

            // No screening - linear evolution
            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force f(R) (linear)\n";
            FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_fifth_force, coupling);
        }

        // Add on density corresponding to the gravitational force to get total force
        auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                auto delta = density_fourier.get_fourier_from_index(fourier_index);
                auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                density_fifth_force.set_fourier_from_index(fourier_index, delta + delta_fifth_force);
            }
        }

        // Compute total force
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        use_screening_method = param.get<bool>("gravity_model_screening");
        fofr0 = param.get<double>("gravity_model_fofr_fofr0");
        nfofr = param.get<double>("gravity_model_fofr_nfofr");
        this->scaledependent_growth = true;
    }
};

#endif

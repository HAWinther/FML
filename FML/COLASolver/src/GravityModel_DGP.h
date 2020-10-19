#ifndef GRAVITYMODEL_DGP_HEADER
#define GRAVITYMODEL_DGP_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

/// DGP model
template <int NDIM>
class GravityModelDGP final : public GravityModel<NDIM> {
  protected:
    double rcH0_DGP;
    bool use_screening_method{true};
    std::string smoothing_filter{"tophat"};
    double smoothing_scale_over_boxsize{0.0};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelDGP() : GravityModel<NDIM>("DGP") {}
    GravityModelDGP(std::shared_ptr<BackgroundCosmology> cosmo) : GravityModel<NDIM>(cosmo, "DGP") {}

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# rcH0    : " << rcH0_DGP << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Internal method: the beta function
    //========================================================================
    double get_beta_dgp(double a) const {
        double E = this->cosmo->HoverH0_of_a(a);
        double dlogHdloga = this->cosmo->dlogHdloga_of_a(a);
        return 1.0 + 2.0 * rcH0_DGP * E * (1.0 + dlogHdloga / 3.0);
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        return 1.0 + 1.0 / (3.0 * get_beta_dgp(a));
    }
    
    //========================================================================
    // For the LPT equations there is a modified >= 2LPT factor
    //========================================================================
    double source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double betadgp = get_beta_dgp(a);
        // The base-class contains how to deal with massive neutrinos so we multiply this in
        // to avoid having to also implement it here
        return GravityModel<NDIM>::source_factor_2LPT(a, koverH0) *
               (1.0 - (2.0 * rcH0_DGP * rcH0_DGP * this->cosmo->get_OmegaM()) /
                          (9.0 * a * a * a * (betadgp * betadgp * betadgp) * (1 + 1.0 / (3.0 * betadgp))));
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta (GR)
    // and D^2 phi_DGP = C delta where Phi = Phi_GR + phi_DGP
    // With screening we compute C as a function of delta
    //========================================================================
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Compute fifth-force
        auto coupling = [&]([[maybe_unused]] double kBox) { return GeffOverG(a, kBox / H0Box) - 1.0; };
        FFTWGrid<NDIM> density_fifth_force;

        if (use_screening_method) {

            // Approximate screening method
            const double OmegaM = this->cosmo->get_OmegaM();
            auto screening_function_dgp = [=](double density_contrast) {
                double fac = 8.0 * OmegaM * std::pow(rcH0_DGP * (GeffOverG(a) - 1.0), 2) * density_contrast;
                return fac < 1e-10 ? 0.0 : 2.0 * (std::sqrt(1.0 + fac) - 1) / fac;
            };

            FML::NBODY::compute_delta_fifth_force_density_screening(density_fourier,
                                                                    density_fifth_force,
                                                                    coupling,
                                                                    screening_function_dgp,
                                                                    smoothing_scale_over_boxsize,
                                                                    smoothing_filter);

        } else {

            // No screening - linear equation
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
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        use_screening_method = param.get<bool>("gravity_model_screening");
        rcH0_DGP = param.get<double>("gravity_model_dgp_rcH0overc");
        smoothing_scale_over_boxsize = param.get<double>("gravity_model_dgp_smoothing_scale_over_boxsize");
        smoothing_filter = param.get<std::string>("gravity_model_dgp_smoothing_filter");
        this->scaledependent_growth = false;
    }
};

#endif

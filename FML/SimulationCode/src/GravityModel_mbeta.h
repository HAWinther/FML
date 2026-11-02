#ifndef GRAVITYMODEL_MBETA_HEADER
#define GRAVITYMODEL_MBETA_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MBeta.h>

/// (m(a),beta(a)) model: beta(a), m(a), phi(a), mu(k, a) and the screening from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_MBeta.h)
template <int NDIM>
class GravityModelmbeta : public GravityModelScreened<NDIM> {

    // Take in parameters used to define m(a), beta(a)
    std::vector<double> mbeta_params;


  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    explicit GravityModelmbeta(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "mbeta") {}

    //========================================================================
    // Print some info
    //========================================================================
    void info_model() const override {
        if (FML::ThisTask == 0) {
            for (size_t i = 0; i < mbeta_params.size(); i++)
                std::cout << "# Params[" << i << "]        : " << mbeta_params[i] << "\n";
        }
        this->info_screening();
    }

    //========================================================================
    // The shared theory (built in init): beta(a), m(a), phi(a), mu(k, a) and the screening
    //========================================================================
    using QuasiStaticGravity_MBeta = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_MBeta;
    const QuasiStaticGravity_MBeta & mbeta() const { return static_cast<const QuasiStaticGravity_MBeta &>(this->qs()); }
    double phi_over_mpl_of_a(double a) const { return mbeta().phi_over_Mpl(a); }

    std::string fifth_force_label() const override { return "(m(a),beta(a)) models"; }

    void init() override {
        // phi(a)/Mpl (only needed for the screening method) needs beta(a) = beta0 a^n and m(a) = m0 H0 a^m with
        // n - 3 - 2m > 0 for its integral from a = 0 to converge
        if (this->use_screening_method)
            FML::assert_mpi(
                mbeta_params[1] - 3.0 - 2.0 * mbeta_params[3] > 0.0,
                "Error in m(a) = m0H0a^m beta(a) = beta0 a^n models: n-4-2m <= -1.0 => phi(a) integral diverges");
        this->quasi_static = std::make_shared<const QuasiStaticGravity_MBeta>(this->cosmo->background(),
                                                                             mbeta_params[0],
                                                                             mbeta_params[1],
                                                                             mbeta_params[2],
                                                                             mbeta_params[3],
                                                                             this->ahigh);
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
    }
};

#endif

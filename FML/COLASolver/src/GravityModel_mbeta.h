#ifndef GRAVITYMODEL_MBETA_HEADER
#define GRAVITYMODEL_MBETA_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

/// (m(a),beta(a)) model
template <int NDIM>
class GravityModelmbeta : public GravityModel<NDIM> {
  
  // Take in parameters used to define m(a), beta(a)
  std::vector<double> mbeta_params;

    // For using the screening approximation
    bool use_screening_method{true};
    bool screening_enforce_largescale_linear{false};
    double screening_linear_scale_hmpc{0.0};
    double screening_efficiency{1.0};

    // Spline for phi(a)/Mpl needed for screening method
    FML::INTERPOLATION::SPLINE::Spline phi_over_Mpl_of_a_spline;

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    GravityModelmbeta() : GravityModel<NDIM>("mbeta") {}
    GravityModelmbeta(std::shared_ptr<Cosmology> cosmo)
        : GravityModel<NDIM>(cosmo, "mbeta")

    {
        assert(this->get_name() == "mbeta");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            for(size_t i = 0; i < mbeta_params.size(); i++)
              std::cout << "# Params[" << i << "]        : " << mbeta_params[i] << "\n";
            std::cout << "# Screening method : " << use_screening_method << "\n";
            if (use_screening_method) {
                std::cout << "# Enforce correct linear evolution : " << screening_enforce_largescale_linear << "\n";
                std::cout << "# Scale for which we enforce this  : " << screening_linear_scale_hmpc << " h/Mpc\n";
                std::cout << "# Screening efficiency             : " << screening_efficiency << "\n";
            }
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Internal method: m(a)/H0 and beta(a) and phi(a)/Mpl
    //========================================================================
    double phi_over_mpl_of_a(double a) const {
        return phi_over_Mpl_of_a_spline(a);
    }
    double m_over_H0_of_a(double a) const {
        return mbeta_params[2] * std::pow(a, mbeta_params[3]); 
    
        // For example f(R) would be like this: 
        // ...with mbeta_params[0] = 1.0/sqrt(6.0), mbeta_params[1] = 0 and mbeta_params[2] = fofr0
        //double OmegaM = this->cosmo->get_OmegaM();
        //double OmegaLambda = 1.0 - OmegaM;
        //double a3 = a * a * a;
        //double fac = OmegaM / a3 + 4.0 * OmegaLambda;
        //double fac0 = OmegaM + 4.0 * OmegaLambda;
        //double mass2 = fac * std::pow(fac / fac0, 2.0) / (2.0 * mbeta_params[2]);
        //return std::sqrt(mass2);
    }
    double beta_of_a(double a) const {
        return mbeta_params[0] * std::pow(a, mbeta_params[1]); 
    }
    double mass_over_H0_squared(double a) const {
        double moverH0 = m_over_H0_of_a(a);  
        return moverH0 * moverH0;
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double mass2a2  = a * a * mass_over_H0_squared(a);
        double koverH02 = koverH0 * koverH0;
        double beta     = beta_of_a(a);
        return 1.0 + 2.0 * beta * beta * koverH02 / (koverH02 + mass2a2);
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

            // PhiCrit is phi(a) / (Mpl * 2 * beta) and screening-factor is PhiCrit/Phi_N
            double PhiCrit = phi_over_Mpl_of_a_spline(a) / (2.0 * beta_of_a(a));

            // Approximate screening method
            auto screening_function = [=](double PhiNewton) {
                double screenfac = std::abs(PhiCrit / PhiNewton) * screening_efficiency;
                return screenfac > 1.0 ? 1.0 : screenfac;
            };

            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force for (m(a),beta(a)) model (screening)\n";
            FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                      density_fifth_force,
                                                                      coupling,
                                                                      screening_function,
                                                                      norm_poisson_equation * std::pow(H0Box / a, 2));

            // Ensure that the large scales are behaving correctly
            // We set delta_fifth_force => A * (1-f) + B * f
            // i.e. we use the linear prediction on large sales and the
            // screened prediction on small scales
            if (screening_enforce_largescale_linear and screening_linear_scale_hmpc > 0.0) {

                if (FML::ThisTask == 0) {
                    std::cout << "Combining screened solution with linear solution\n";
                    std::cout << "We use linear solution for k < " << screening_linear_scale_hmpc << " h/Mpc\n";
                }

                // Make a spline of the low-pass filter we use
                const int npts = 1000;
                const double kBoxmin = M_PI;
                const double kBoxmax = M_PI * std::sqrt(NDIM) * density_fourier.get_nmesh();
                const double kcutBox = screening_linear_scale_hmpc / this->H0_hmpc * H0Box;
                std::vector<double> kBox(npts);
                std::vector<double> f(npts);
                for (int i = 0; i < npts; i++) {
                    kBox[i] = kBoxmin + (kBoxmax - kBoxmin) * i / double(npts - 1);
                    f[i] = std::exp(-0.5 * kBox[i] * kBox[i] / (kcutBox * kcutBox));
                }
                Spline f_spline(kBox, f, "Low-pass filter");

                // Combine the screened and the linear solution together
                auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] std::array<double, NDIM> kvec;
                    double kmag;
                    for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                        density_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        auto delta = density_fourier.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force_linear = delta * FML::GRID::FloatType(GeffOverG(a, kmag / H0Box) - 1.0);
                        FML::GRID::FloatType filter = f_spline(kmag);
                        auto value = delta_fifth_force_linear * filter + delta_fifth_force * FML::GRID::FloatType(1.0 - filter);
                        density_fifth_force.set_fourier_from_index(fourier_index, value);
                    }
                }
            }

        } else {

            // No screening - linear evolution
            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force (m(a),beta(a)) models (linear)\n";
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
        if (this->force_use_finite_difference_force) {
          // Use a by default a 4 point formula (using phi(i+/-2), phi(i+/-1) to compute DPhi)
          // This requires 2 boundary cells (stencil_order=2,4,6 implemented so far)
          const int stencil_order = this->force_finite_difference_stencil_order;
          const int nboundary_cells = stencil_order/2;

          FFTWGrid<NDIM> potential_real;
          FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fifth_force,
              potential_real,
              norm_poisson_equation,
              nboundary_cells);

          FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
              force_real,
              density_assignment_method_used,
              stencil_order);

        } else {
          // Computes gravitational force using fourier-methods
          FML::NBODY::compute_force_from_density_fourier<NDIM>(
              density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
        }
    }
    
    virtual void init() override { 
      // Here we need to compute and spline phi(a)/Mpl needed for the screening method
      
      // Integrand for computing the phi_over_Mpl_of_a_spline = 9 * OmegaM0 * Int_0^a beta(a) / [a^3(m/H0)^2] dloga
      double OmegaM0 = this->cosmo->get_OmegaM();
      auto integrand = [=](double a){
        return 9.0 * OmegaM0 * beta_of_a(a) / (a*a*a * mass_over_H0_squared(a));
      };

      // Make an a-array, evaluate integral over it and make the spline
      // This part needs to be tested (how best to evaluate this integral). Might want to do compute logphi maybe? ...
      const double amin = 0.001;
      const double amax = 1.0;
      const int npts    = 10000;
      std::vector<double> a_array(npts, amin), loga_array(npts, std::log(amin)), phi_array(npts, 0.0);
      for(size_t i = 1; i < npts; i++){
        loga_array[i] = std::log(amin) + std::log(amax/amin) * i / double(npts-1);
        a_array[i]    = std::exp(loga_array[i]);
        phi_array[i]  = phi_array[i-1] + integrand(a_array[i]) * (loga_array[i] - loga_array[i-1]);
      }
      phi_over_Mpl_of_a_spline = Spline(a_array, phi_array, "phi(a)/Mpl spline for m(a),beta(a) models");

      // If we use the power-law model beta(a) = beta0 * a^n and m(a) = m0 * H0 * a^m
      // then we should ensure that n-4-2m > -1 for the integral to not have an uncurable divergence at a=0
      FML::assert_mpi(mbeta_params[1] - 4.0 - 2.0 * mbeta_params[3] > -1.0, 
          "Error in m(a) = m0H0a^m beta(a) = beta0 a^n models: n-4-2m <= -1.0 => phi(a) integral diverges");

      // Important: call base-call initializer in the end
      GravityModel<NDIM>::init();
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        mbeta_params = param.get<std::vector<double>>("gravity_model_mbeta_params");
        use_screening_method = param.get<bool>("gravity_model_screening");
        if (use_screening_method) {
            screening_enforce_largescale_linear = param.get<bool>("gravity_model_screening_enforce_largescale_linear");
            screening_linear_scale_hmpc = param.get<double>("gravity_model_screening_linear_scale_hmpc");
            screening_efficiency = param.get<double>("gravity_model_screening_efficiency", 1.0);
        }
        this->scaledependent_growth = true;
    }
};

#endif

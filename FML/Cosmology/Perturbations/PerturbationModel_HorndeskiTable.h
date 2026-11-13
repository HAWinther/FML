#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_HORNDESKITABLE_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_HORNDESKITABLE_H

// =====================================================================================================================
// A reduced Horndeski theory (K(phi, X), G3(phi, X), G4(phi)) from the table of
// SimulationCode/scripts/horndeski_frontend.py (the model horndeski_table; BackgroundParameters::model_file): the
// background of BackgroundModel_HorndeskiTable.h and the perturbations of the alpha parameterization
// (PerturbedFieldEquation_HorndeskiAlpha.h, alpha_T = 0, no quasi-static approximation; stability checked) with the
// alphas the frontend derives from the Lagrangian (Bellini & Sawicki 2014, arXiv:1404.3713):
//   M*^2 = 2 G4,  alpha_M = dln M*^2/dlna,  H M*^2 alpha_B = 2 dphi/dt (X G3_X - G4_phi),
//   H^2 M*^2 alpha_K = 2 X (K_X + 2 X K_XX - 2 G3_phi - 2 X G3_phiX) + 12 dphi/dt X H (G3_X + X G3_XX)
// and Rx (the matter terms, computed symbolically by the frontend: exact where the scalar is negligible). Outside the
// table: before it the alphas go to zero as a (M*^2 and its derivative-free terms as at its start; Rx from the
// background), after it the values at its end. The same table gives SimulationCode's force (gravity_model =
// Horndeski, QuasiStaticGravity_Horndeski.h) and, with this model, the initial conditions of its runs.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_HorndeskiTable.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_Horndeski.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// The alphas of a horndeski_frontend.py table
    class TabulatedHorndeskiAlphas : public AlphaFunctions {
      public:
        explicit TabulatedHorndeskiAlphas(const BACKGROUND::HorndeskiTable & t) : filename(t.filename) {
            if (not t.has_alphas())
                throw std::invalid_argument("[horndeski_table] [" + t.filename +
                                            "] has no alphas (made by an older horndeski_frontend.py: run it again)");
            std::vector<double> x;
            for (double a : t.a())
                x.push_back(std::log(a));
            x_lo = x.front();
            x_hi = x.back();
            const char * names[8] = {"M2", "dM2", "aK", "aB", "aM", "daKdx", "daBdx", "Rx"};
            for (int i = 0; i < 8; i++)
                splines[i].create(x, t.col[10 + i], std::string("horndeski_table ") + names[i]);
        }
        std::string info() const override { return "tabulated (" + filename + ")"; }
        Alphas operator()(double x, const BACKGROUND::ComponentBackground & /*bg*/) const override {
            Alphas al;
            const double xc = std::min(std::max(x, x_lo), x_hi);
            // before the table the alphas go to zero as a
            const double f = x < x_lo ? std::exp(x - x_lo) : 1.0;
            al.M2 = splines[0](xc);
            al.dM2 = splines[1](xc);
            al.aK = f * splines[2](xc);
            al.aB = f * splines[3](xc);
            al.aM = f * splines[4](xc);
            al.daKdx = x < x_lo ? al.aK : (x > x_hi ? 0.0 : splines[5](xc));
            al.daBdx = x < x_lo ? al.aB : (x > x_hi ? 0.0 : splines[6](xc));
            if (x >= x_lo)
                al.Rx = splines[7](xc);
            return al;
        }

      private:
        std::string filename;
        double x_lo{0.0}, x_hi{0.0};
        FML::INTERPOLATION::SPLINE::GSLSpline splines[8];
    };

    /// The model of the Einstein-Boltzmann solver: the table of p.model_file (background and alphas)
    inline std::shared_ptr<const PerturbationModel_Horndeski>
    make_horndeski_table_perturbations(BACKGROUND::ComponentBackgroundPtr bg, bool check_stability = true) {
        const auto t = BACKGROUND::read_horndeski_table(bg->parameters().model_file);
        return std::make_shared<const PerturbationModel_Horndeski>(std::move(bg),
                                                                   std::make_shared<TabulatedHorndeskiAlphas>(t),
                                                                   check_stability);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif

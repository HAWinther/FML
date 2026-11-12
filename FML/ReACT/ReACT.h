#ifndef FML_REACT_REACT_H
#define FML_REACT_REACT_H

#include <FML/Spline/Spline.h>
#include <array>
#include <vector>

namespace FML::COSMOLOGY {

    // Interface to the official ACTio-ReACTio v2 C++ engine. Enable USE_REACT to link the backend.
    // All arrays at this interface use h/Mpc and (Mpc/h)^3; z is increasing and arrays are [z][k].
    class ReACT {
      public:
        enum class Model { GR = 1, HuSawicki = 2, NormalDGP = 3, Quintessence = 4, CPL = 5 };
        struct Cosmology {
            double h = 0.67, n_s = 0.965;
            double Omega_m = 0.317, Omega_b = 0.05, Omega_nu = 0.0;
            // No A_s or k_pivot: with power-spectrum input upstream uses them (and n_s) only to convert P to a transfer
            // function and back, so they cancel. FML passes fixed values inside upstream's allowed range.
        };
        struct Options {
            Model model = Model::GR;
            // f(R): [0]=|fR0| (n=1); nDGP: [0]=Omega_rc; quintessence: [0]=w;
            // CPL: [0]=w0, [1]=wa. Other entries must remain zero in this interface.
            std::array<double, 20> parameters{};
            int mass_loop = 30;
            bool verbose = false;
        };
        using Grid = std::vector<std::vector<double>>;
        struct Input {
            Cosmology cosmology;
            std::vector<double> k, z;
            // Target spectra at each requested redshift, not z=0 LCDM seeds.
            Grid total, cold;
            // LCDM CDM+baryon spectrum with the same cosmological/primordial parameters, including
            // the same neutrinos, at the same redshifts. This is auxiliary input for halo concentrations;
            // it is NOT the matched pseudo spectrum used in P_NL = R * P_NL,pseudo.
            Grid lcdm_cold;
        };
        struct Result {
            std::vector<double> k, z;
            Grid reaction, linear, pseudo, nonlinear;
            double sigma8_at_lowest_z{};
        };

        static bool available();
        static void validate(const Input & input, const Options & options);
        static Result compute(const Input & input, const Options & options);

        // Immutable reaction interpolation table, safe to evaluate concurrently after construction.
        explicit ReACT(const Result & result);
        double reaction(double x, double k_hmpc) const; // x=ln(a); throws outside the redshift range
        double x_min() const { return xmin; }
        double x_max() const { return xmax; }

      private:
        double xmin{}, xmax{}, lkmin{}, lkmax{};
        FML::INTERPOLATION::SPLINE::GSLSpline2D log_reaction;
    };

} // namespace FML::COSMOLOGY
#endif

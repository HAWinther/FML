#ifndef FML_COSMOLOGY_MODELS_STANDARDSPECIES_H
#define FML_COSMOLOGY_MODELS_STANDARDSPECIES_H

// =============================================================================
// The species of LCDM and its standard extensions:
//   PowerLawSpecies         rho = Omega_0 a^{-n}, smooth: baryons (n = 3) and photons (4) (their perturbations
//                           are in the core of Perturbations), cosmological constant (0)
//   ColdDarkMatter          n = 3, with perturbations
//   MasslessNeutrinos       n = 4, with perturbations
//   MassiveNeutrinoSpecies  one massive neutrino species (Fermi-Dirac momentum integrals)
//   CPLDarkEnergy           w(a) = w0 + wa (1 - a), smooth (background only)
// Perturbations (in StandardSpecies.cpp):
//   CDM                     delta_cdm, v_cdm
//   massless neutrinos      Boltzmann hierarchy Nu_0..Nu_lmax_nu
//   massive neutrinos       momentum-resolved hierarchy Psi_ell(q), then the CLASS fluid approximation
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <string>

namespace FML::COSMOLOGY {

    class PowerLawSpecies : public Species {
      public:
        PowerLawSpecies(std::string name, Role role, double Omega0, double n)
            : name_(std::move(name)), role_(role), Omega0(Omega0), n(n) {}
        std::string name() const override { return name_; }
        Role role() const override { return role_; }
        double rho(double x) const override { return Omega0 * std::exp(-n * x); }
        double p(double x) const override { return (n / 3.0 - 1.0) * rho(x); }
        double drhodx(double x) const override { return -n * rho(x); }
        double ddrhoddx(double x) const override { return n * n * rho(x); }
        std::string info() const override;
        double Omega_today() const { return Omega0; }

      private:
        std::string name_;
        Role role_;
        double Omega0, n;
    };

    class ColdDarkMatter : public PowerLawSpecies {
      public:
        explicit ColdDarkMatter(double Omega0) : PowerLawSpecies("cdm", Role::CDM, Omega0, 3.0) {}
        std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations & pert) const override;
    };

    class MasslessNeutrinos : public PowerLawSpecies {
      public:
        explicit MasslessNeutrinos(double Omega0)
            : PowerLawSpecies("massless_nu", Role::MasslessNeutrinos, Omega0, 4.0) {}
        bool free_streaming() const override { return true; }
        // nullptr if Omega_nu = 0 or PerturbationOptions::neutrinos = false
        std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations & pert) const override;
    };

    // rho = C a^-4 I_rho(M a), p = C a^-4 I_p(M a), with M = m c^2 / (k_B T_ncdm,0) and
    // C = Omega_gamma (15/pi^4) (T_ncdm/TCMB)^4 (MassiveNeutrinos.h)
    class MassiveNeutrinoSpecies : public Species {
      public:
        explicit MassiveNeutrinoSpecies(const CosmologyParameters & params);
        std::string name() const override { return "ncdm"; }
        Role role() const override { return Role::MassiveNeutrinos; }
        double rho(double x) const override;
        double p(double x) const override;
        double drhodx(double x) const override { return -3.0 * (rho(x) + p(x)); }
        double ddrhoddx(double x) const override { return -3.0 * (drhodx(x) + dpdx(x)); }
        bool free_streaming() const override { return true; }
        std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations & pert) const override;
        std::string info() const override;

        double dpdx(double x) const;
        double pseudo_p(double x) const;                           // pseudo-pressure, dp/dx = -(5p - pseudo_p)
        double M_of_x(double x) const { return M0 * std::exp(x); } // m / T at x
        double M_today() const { return M0; }
        double prefactor() const { return C; } // rho a^4 / I_rho(M)  [rho_c0]

      private:
        double M0, C;
    };

    // Dark energy with w(a) = w0 + wa (1 - a):
    //   rho / rho_c0 = Omega_DE exp(-3 (1 + w0 + wa) x - 3 wa (1 - e^x)),  d ln rho / dx = -3 (1 + w)
    // Smooth (no perturbations), which is only consistent for w = -1.
    class CPLDarkEnergy : public Species {
      public:
        CPLDarkEnergy(double Omega0, double w0, double wa) : Omega0(Omega0), w0(w0), wa(wa) {}
        std::string name() const override { return "cpl"; }
        Role role() const override { return Role::DarkEnergy; }
        double w_of_x(double x) const { return w0 + wa * (1.0 - std::exp(x)); }
        double rho(double x) const override {
            return Omega0 * std::exp(-3.0 * (1.0 + w0 + wa) * x - 3.0 * wa * (1.0 - std::exp(x)));
        }
        double p(double x) const override { return w_of_x(x) * rho(x); }
        double drhodx(double x) const override { return -3.0 * (1.0 + w_of_x(x)) * rho(x); }
        double ddrhoddx(double x) const override {
            const double L = -3.0 * (1.0 + w_of_x(x)), dL = 3.0 * wa * std::exp(x);
            return (L * L + dL) * rho(x);
        }
        std::string info() const override;

      private:
        double Omega0, w0, wa;
    };

} // namespace FML::COSMOLOGY

#endif

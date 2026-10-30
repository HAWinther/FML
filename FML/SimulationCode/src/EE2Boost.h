#ifndef EE2BOOST_HEADER
#define EE2BOOST_HEADER

#include <FML/Emulator/EuclidEmulator2.h>
#include <FML/Spline/Spline.h>

#include <array>
#include <memory>

//========================================================================
/// The EuclidEmulator2 boost P_nl / P_linear with the emulator built once per set of cosmological parameters.
/// Building the emulator reads its data file and sets up 15 2D splines (~0.2 s); the boost at a given redshift is
/// cheap. Without the cache this was done for every call, i.e. at every step with pofk_every_step (a third of the
/// run time of a 64^3 COLA run). The results are the same as building the emulator for every call.
//========================================================================
class EE2BoostCache {
  public:
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    Spline boost(double Omegab,
                 double OmegaM,
                 double Mnu_eV,
                 double ns,
                 double h,
                 double w0,
                 double wa,
                 double As,
                 double redshift) const {
        namespace EE2 = FML::EMULATOR::EUCLIDEMULATOR2;
        const std::array<double, 8> key{Omegab, OmegaM, Mnu_eV, ns, h, w0, wa, As};
        if (state == State::not_built or key != parameters) {
            parameters = key;
            emulator.reset();
            try {
                EE2::Cosmology ee2cosmo(Omegab, OmegaM, Mnu_eV, ns, h, w0, wa, As);
                if (ee2cosmo.is_good_to_use()) {
                    emulator = std::make_shared<EE2::EuclidEmulator>(ee2cosmo);
                    state = State::built;
                } else {
                    state = State::out_of_range;
                }
            } catch (...) {
                state = State::failed;
            }
        }
        if (state == State::out_of_range)
            return Spline{"P/Plinear from EuclidEmulator2 uninitialized"};
        if (state == State::failed)
            return Spline();
        try {
            auto result = emulator->compute_boost(redshift);
            return Spline(result.first, result.second, "P/Plinear from EuclidEmulator2");
        } catch (...) {
            return Spline();
        }
    }

  private:
    enum class State { not_built, built, out_of_range, failed };
    mutable State state{State::not_built};
    mutable std::array<double, 8> parameters{};
    mutable std::shared_ptr<FML::EMULATOR::EUCLIDEMULATOR2::EuclidEmulator> emulator;
};

#endif

#ifndef FML_MATCHED_SPECTRA_HEADER
#define FML_MATCHED_SPECTRA_HEADER
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/EnsembleStatistics/ControlVariates.h>
namespace FML::ENSEMBLE {
    struct MatchedSpectra {
        Vector k, modes, target, control, cross, correlation;
    };
    /// Collective raw auto/cross spectra. Input grids must already be Fourier-space
    /// density contrasts in the same box; no assignment or particle-noise corrections.
    /// kmin/kmax are the existing FML bin-center limits in physical inverse-length units.
    inline MatchedSpectra matched_spectra(const FML::GRID::FFTWGrid<3>& target, const FML::GRID::FFTWGrid<3>& control,
                                          double boxsize, double kmin, double kmax, int bins) {
        int valid = boxsize > 0 && std::isfinite(boxsize * boxsize * boxsize) && boxsize * boxsize * boxsize > 0 &&
                    kmin >= 0 && kmax > kmin && std::isfinite(kmax * boxsize) && bins >= 2 && target.get_nmesh() > 0 &&
                    target.get_nmesh() == control.get_nmesh() && target.get_local_nx() == control.get_local_nx() &&
                    target.get_local_x_start() == control.get_local_x_start() &&
                    !const_cast<FML::GRID::FFTWGrid<3>&>(target).get_grid_status_real() &&
                    !const_cast<FML::GRID::FFTWGrid<3>&>(control).get_grid_status_real();
        FML::MinOverTasks(&valid);
        detail::require(valid, "Invalid matched Fourier-grid spectra inputs");
        Count schema = target.get_nmesh();
        detail::word(schema, bins);
        detail::values(schema, {boxsize, kmin, kmax});
        detail::same(schema);
        using Binning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>;
        Binning yy(kmin * boxsize, kmax * boxsize, bins, Binning::LINEAR_SPACING), cc = yy, yc = yy;
        FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(target, yy);
        FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(control, cc);
        FML::CORRELATIONFUNCTIONS::bin_up_cross_power_spectrum(target, control, yc);
        yy.scale(boxsize);
        cc.scale(boxsize);
        yc.scale(boxsize);
        MatchedSpectra out;
        out.k = yy.kbin;
        out.modes = yy.count;
        out.target = yy.pofk;
        out.control = cc.pofk;
        out.cross = yc.pofk;
        out.correlation.resize(yy.pofk.size());
        for (size_t i = 0; i < out.k.size(); ++i)
            out.correlation[i] = out.target[i] > 0 && out.control[i] > 0
                                     ? out.cross[i] / (std::sqrt(out.target[i]) * std::sqrt(out.control[i]))
                                     : detail::nan();
        return out;
    }
} // namespace FML::ENSEMBLE
#endif

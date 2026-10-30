#ifndef SIMCODE_SNAPSHOT_POWER_SPECTRUM_H
#define SIMCODE_SNAPSHOT_POWER_SPECTRUM_H
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include "SimulationParameters.h"

namespace SIMCODE {
    /// The same estimator for live single-species outputs and temporary analysis snapshots.
    template <int NDIM, class P>
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> measure_snapshot_power_spectrum(P * particles,
                                                                                          size_t count,
                                                                                          size_t total,
                                                                                          const PofkSettings & settings,
                                                                                          double boxsize) {
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> result(settings.nmesh / 2);
        result.subtract_shotnoise = settings.subtract_shotnoise;
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM, P>(settings.nmesh,
                                                                   particles,
                                                                   count,
                                                                   total,
                                                                   result,
                                                                   settings.density_assignment_method,
                                                                   settings.interlacing);
        result.scale(boxsize);
        return result;
    }
} // namespace SIMCODE
#endif

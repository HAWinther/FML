#ifndef FML_TIME_STEP_TYPES_H
#define FML_TIME_STEP_TYPES_H
namespace FML::NBODY {
    enum class TimestepMethod { Quinn, Tassev, FastPM, BullFrog };
    enum class ScaleFactorSpacing { Linear, Logarithmic, Powerlaw, GrowthFactor };
    struct TimestepOptions {
        TimestepMethod method{TimestepMethod::Quinn};
        ScaleFactorSpacing spacing{ScaleFactorSpacing::Linear};
        double cola_nLPT{-2.5};
        double spacing_power{1.0};
    };
} // namespace FML::NBODY
#endif

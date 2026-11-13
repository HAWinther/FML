#ifndef FML_FIELD_LEVEL_SURVEY_CHECKPOINT_H
#define FML_FIELD_LEVEL_SURVEY_CHECKPOINT_H
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/SurveyLikelihood.h>
namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT {
    // Same byte encoding/checksum as the standard codec, without retaining
    // complete voxel inputs in an Encoder. The outer target stores this digest.
    class SurveyInputHash {
        Codec::Encoder block_;
        std::uint64_t hash_ = 14695981039346656037ULL;
        void flush() {
            for (auto byte : block_.bytes) {
                hash_ ^= byte;
                hash_ *= 1099511628211ULL;
            }
            block_.bytes.clear();
        }
        void bounded() {
            if (block_.bytes.size() >= 4096)
                flush();
        }

      public:
        SurveyInputHash() { block_.bytes.reserve(4224); }
        void integer(std::uint64_t value) {
            block_.integer(value);
            bounded();
        }
        void real(double value) {
            block_.real(value);
            bounded();
        }
        void text(const std::string & value) {
            integer(value.size());
            for (unsigned char byte : value) {
                block_.bytes.push_back(byte);
                bounded();
            }
        }
        void vector(const Field & values) {
            integer(values.size());
            for (double value : values)
                real(value);
        }
        std::uint64_t finish() {
            flush();
            return hash_;
        }
    };
    template <class Encoder, class Forward>
    void append_survey_window(Encoder & e, const CartesianSurveyObservation<Forward> & observation) {
        e.text("FML Cartesian survey voxel observation 1");
        const auto & window = observation.window();
        for (double x : window.origin)
            e.real(x);
        e.integer(window.observed.size());
        for (auto flag : window.observed)
            e.integer(flag);
        e.vector(window.selection);
    }
    template <class Forward>
    struct LikelihoodIdentity<SurveyGaussianLikelihood<Forward>> {
        static void append(Codec::Encoder & e, const SurveyGaussianLikelihood<Forward> & likelihood) {
            e.text("FML selected voxel Gaussian likelihood 1");
            SurveyInputHash inputs;
            append_survey_window(inputs, likelihood.observation());
            const auto & mask = likelihood.observation().window().observed;
            inputs.integer(mask.size());
            for (size_t q = 0; q < mask.size(); ++q) {
                // Masked data/variance are absent from the target. Canonicalize
                // ignored values, including arbitrary NaN payloads from input IO.
                inputs.real(mask[q] ? likelihood.data()[q] : 0.);
                inputs.real(mask[q] ? likelihood.relative_variance()[q] : 0.);
            }
            e.integer(inputs.finish());
        }
    };
    template <class Forward>
    struct LikelihoodIdentity<SurveyPoissonLikelihood<Forward>> {
        static void append(Codec::Encoder & e, const SurveyPoissonLikelihood<Forward> & likelihood) {
            e.text("FML selected voxel exponential Poisson likelihood 1");
            SurveyInputHash inputs;
            append_survey_window(inputs, likelihood.observation());
            inputs.real(likelihood.offset());
            const auto & mask = likelihood.observation().window().observed;
            inputs.integer(mask.size());
            for (size_t q = 0; q < mask.size(); ++q) {
                inputs.integer(mask[q] ? likelihood.counts()[q] : 0);
                inputs.real(mask[q] ? likelihood.exposure()[q] : 0.);
            }
            e.integer(inputs.finish());
        }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT
#endif

#ifndef FML_VALIDATION_FROZEN_LOW_RANK_METRIC_H
#define FML_VALIDATION_FROZEN_LOW_RANK_METRIC_H
#include <FML/FieldLevel/Framework.h>
#include <fstream>
namespace GN {
    // Serial validation operator: M_beta=I+beta U diag(lambda) U^T.
    // Storage O(rank*dimension); no dense dimension-squared matrix or state-dependent mass.
    struct Metric {
        size_t dimension = 0;
        std::vector<double> eigenvalues;
        std::vector<FML::FIELDLEVEL::Field> vectors;
        static double dot(const std::vector<double> & a, const std::vector<double> & b) {
            if (a.size() != b.size())
                throw std::invalid_argument("Metric vector dimensions differ");
            double result = 0;
            for (size_t j = 0; j < a.size(); ++j)
                result += a[j] * b[j];
            return result;
        }
        void validate() const {
            if (FML::NTasks != 1 || dimension < 1 || vectors.size() != eigenvalues.size() || vectors.size() > 32 ||
                vectors.size() > dimension)
                throw std::invalid_argument("Invalid bounded serial low-rank metric");
            for (size_t i = 0; i < vectors.size(); ++i) {
                if (vectors[i].size() != dimension || !(eigenvalues[i] > 0) || !std::isfinite(eigenvalues[i]))
                    throw std::invalid_argument("Invalid metric eigenpair");
                for (double x : vectors[i])
                    if (!std::isfinite(x))
                        throw std::invalid_argument("Nonfinite metric basis");
                for (size_t j = 0; j <= i; ++j)
                    if (std::abs(dot(vectors[i], vectors[j]) - (i == j ? 1. : 0.)) > 1e-9)
                        throw std::invalid_argument("Metric basis is not orthonormal");
            }
        }
        static Metric read(const std::string & path) {
            std::ifstream in(path);
            std::string tag;
            size_t rank;
            Metric m;
            if (!(in >> tag >> m.dimension >> rank) || tag != "FML_FROZEN_GN_1" || m.dimension > 64 || rank > 32)
                throw std::invalid_argument("Invalid bounded metric file");
            m.eigenvalues.resize(rank);
            m.vectors.resize(rank, std::vector<double>(m.dimension));
            for (size_t i = 0; i < rank; ++i) {
                if (!(in >> m.eigenvalues[i]))
                    throw std::invalid_argument("Missing eigenvalue");
                for (auto & x : m.vectors[i])
                    if (!(in >> x))
                        throw std::invalid_argument("Missing eigenvector");
            }
            std::string extra;
            if (in >> extra)
                throw std::invalid_argument("Extra metric fields");
            m.validate();
            return m;
        }
        FML::FIELDLEVEL::Field apply(const FML::FIELDLEVEL::Field & input, double beta, double exponent) const {
            if (input.size() != dimension || !std::isfinite(beta) || beta < 0 || beta > 1 || !std::isfinite(exponent))
                throw std::invalid_argument("Invalid metric action");
            for (double x : input)
                if (!std::isfinite(x))
                    throw std::invalid_argument("Nonfinite metric input");
            auto result = input;
            for (size_t i = 0; i < vectors.size(); ++i) {
                const double coefficient =
                    std::expm1(exponent * std::log1p(beta * eigenvalues[i])) * dot(input, vectors[i]);
                if (!std::isfinite(coefficient))
                    throw std::invalid_argument("Unrepresentable metric action");
                for (size_t j = 0; j < dimension; ++j)
                    result[j] += coefficient * vectors[i][j];
            }
            for (double x : result)
                if (!std::isfinite(x))
                    throw std::invalid_argument("Nonfinite metric result");
            return result;
        }
    };
} // namespace GN
#endif

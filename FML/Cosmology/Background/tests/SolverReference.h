#ifndef FML_COSMOLOGY_BACKGROUND_TESTS_SOLVERREFERENCE_H
#define FML_COSMOLOGY_BACKGROUND_TESTS_SOLVERREFERENCE_H

// The reference of the comparisons of test_background, test_models and test_scalar_fields with the Einstein-Boltzmann
// solver: the backgrounds of the solver's old model classes (its Species and Gravity, Cosmology/Models, removed after
// the solver moved onto this module), stored in data/solver_background.txt, solver_models.txt and
// solver_scalar_fields.txt. The tables were written at FML commit 1bebec4 by a generator (not kept, the old classes are
// gone) that built each model of the tests as an explicit Model of BackgroundCosmology (Models::build with the same
// parameters, solve(-24, 2, 20000)) and wrote, at the x of the test, E, dlogE/dx (dHp/dx / Hp - 1), the Omega_i(x),
// eta and chi in Mpc/h (SI / (Mpc/h)), t in Gyr and D_A in Mpc/h, with 12 significant digits (see the file headers).

#include <FML/tests/ReferenceData.h>

#include <string>
#include <vector>

/// One x of a stored solver background
struct SolverRow {
    double x, E, dlogEdx, Omega_b, Omega_cdm, Omega_gamma, Omega_nu, Omega_ncdm, Omega_de, Omega_k;
    double eta, t, chi, D_A; ///< Mpc/h, Gyr, Mpc/h, Mpc/h (chi and D_A are 0 for |x| <= 0.05)
};

/// The block `label` of data/<file> (REFERENCE_DATA_DIR is set by the Makefile)
inline std::vector<SolverRow> solver_table(const std::string & file, const std::string & label) {
    const FML::TESTS::ReferenceData data(std::string(REFERENCE_DATA_DIR) + "/" + file);
    auto block = data.block(label);
    std::vector<SolverRow> rows;
    while (not block.done()) {
        const auto r = block.numbers("x");
        if (r.size() != 14)
            throw std::runtime_error("[solver_table] " + file + " [" + label + "]: expected 14 columns");
        rows.push_back({r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13]});
    }
    return rows;
}

#endif

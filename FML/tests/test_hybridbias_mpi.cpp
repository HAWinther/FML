//==========================================================================================================
// The LPT predictions of the hybrid bias spectra with the k-values shared out over the MPI tasks
// (ZeldovichOptions::mpi_parallel, as SimulationCode calls them): the result must not depend on how the k-values
// are distributed. Each task gets every NTasks-th k-value (as HybridBiasPredictionGrid in SimulationCode) and the
// result is compared with task 0 computing all of them alone.
//
// Without mpi_parallel each task chose its q-grid (qmax = 16 pi / kmin) from its OWN k-values, so a task without
// the small k got a q-grid too short for its k-values (P_11 wrong by percents at k ~ 0.1-0.3 with 64 tasks). The
// size of that error with this number of tasks is printed for information.
//
// Run with several tasks (e.g. mpirun -np 4: the fourth task gets only the larger k-values); with one task the
// checks are trivially true.
//==========================================================================================================
#include <FML/ComputePowerSpectra/HybridBiasOneLoop.h>
#include <FML/ComputePowerSpectra/HybridBiasOneLoopMG.h>
#include <FML/ComputePowerSpectra/HybridBiasZeldovich.h>
#include <FML/Global/Global.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <FML/tests/TestHelpers.h>

using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
using FML::TESTS::check;
using FML::TESTS::num;

namespace {
    double pofk_bbks(double k) {
        const double Gamma = 0.21, q = k / Gamma;
        const double T =
            std::log(1.0 + 2.34 * q) / (2.34 * q) *
            std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) + std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
        return 2.0e6 * k * T * T;
    }

    // The k-values of a box of size L as SimulationCode evaluates the predictions: a log table from k_f to 16 k_f
    // (there with 40 points; here 3, so that the tasks beyond the third get only larger k-values already with 4
    // tasks) and then the bin centres (spacing k_f) up to kmax
    std::vector<double> simulation_kvalues(double boxsize, double kmax) {
        const double kf = 2.0 * M_PI / boxsize;
        std::vector<double> k{kf, 4.0 * kf, 16.0 * kf};
        for (double kk = 16.5 * kf; kk < kmax; kk += kf)
            k.push_back(kk);
        return k;
    }

    // The k-values of this task (every NTasks-th, as HybridBiasPredictionGrid)
    std::vector<double> local_part(const std::vector<double> & k) {
        std::vector<double> local;
        for (size_t j = 0; j < k.size(); j++)
            if (int(j % FML::NTasks) == FML::ThisTask)
                local.push_back(k[j]);
        return local;
    }

    // Put the local results back at their place in the full list (result[pair][k]) on every task
    std::vector<std::vector<double>> gather(const std::vector<std::vector<double>> & local, size_t npairs, size_t nk) {
        std::vector<std::vector<double>> all(npairs, std::vector<double>(nk, 0.0));
        for (size_t p = 0; p < npairs; p++) {
            size_t i = 0;
            for (size_t j = 0; j < nk; j++)
                if (int(j % FML::NTasks) == FML::ThisTask)
                    all[p][j] = local.empty() ? 0.0 : local[p][i++];
            FML::SumArrayOverTasks(all[p].data(), nk);
        }
        return all;
    }

    // Largest |a - b| / max|b| over the spectra (on task 0, where b is)
    double max_relative_difference(const std::vector<std::vector<double>> & a,
                                   const std::vector<std::vector<double>> & b) {
        double worst = 0.0;
        for (size_t p = 0; p < b.size(); p++) {
            double scale = 0.0;
            for (double x : b[p])
                scale = std::max(scale, std::abs(x));
            if (scale == 0.0)
                continue;
            for (size_t j = 0; j < b[p].size(); j++)
                worst = std::max(worst, std::abs(a[p][j] - b[p][j]) / scale);
        }
        return worst;
    }
} // namespace

int main() {
    const double kcut = 0.3, boxsize = 500.0;
    std::function<double(double)> P = [&](double k) { return (k < 1e-4 or k > kcut) ? 0.0 : pofk_bbks(k); };
    std::function<double(double)> W = [&](double k) { return k < kcut ? 1.0 : 0.0; };
    ZeldovichOptions options;
    options.kmin_integration = 1e-4;
    options.kmax_integration = kcut;
    auto parallel = options;
    parallel.mpi_parallel = true;

    const auto kall = simulation_kvalues(boxsize, kcut);
    const auto klocal = local_part(kall);
    const size_t nk = kall.size();
    if (FML::ThisTask == 0)
        std::cout << "LPT predictions with the k-values shared out over " << FML::NTasks << " task(s), " << nk
                  << " k-values\n";

    // A function of the k-values returning result[pair][k], computed three ways: shared out with mpi_parallel,
    // shared out without it (the old way; only printed, and skipped if show_old is false) and on task 0 alone (the
    // reference)
    auto compare = [&](const std::string & name,
                       const std::function<std::vector<std::vector<double>>(const std::vector<double> &,
                                                                             const ZeldovichOptions &)> & predict,
                       bool show_old = true) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::vector<double>> reference;
        if (FML::ThisTask == 0)
            reference = predict(kall, options);
        const size_t npairs = FML::ThisTask == 0 ? reference.size() : 0;
        size_t np = npairs;
        FML::MaxOverTasks(&np);
        const auto shared = gather(predict(klocal, parallel), np, nk);
        double diff = FML::ThisTask == 0 ? max_relative_difference(shared, reference) : 0.0;
        FML::MaxOverTasks(&diff);
        std::string old_text;
        if (show_old) {
            const auto old = gather(predict(klocal, options), np, nk);
            double diff_old = FML::ThisTask == 0 ? max_relative_difference(old, reference) : 0.0;
            FML::MaxOverTasks(&diff_old);
            old_text = "; without mpi_parallel: " + num(diff_old, 3);
        }
        check(diff <= 1e-12,
              name + ": k-values shared out over the tasks = one task computing all (" + num(diff, 3) + old_text + ")");

        // A task without k-values must still take part (no early return): all k-values on task 0
        const std::vector<double> only_task0 = FML::ThisTask == 0 ? kall : std::vector<double>{};
        const auto result = predict(only_task0, parallel);
        double diff0 = FML::ThisTask == 0 ? max_relative_difference(result, reference) : 0.0;
        FML::MaxOverTasks(&diff0);
        check(diff0 <= 1e-12, name + ": tasks without k-values take part (" + num(diff0, 3) + ")");
        if (FML::ThisTask == 0)
            std::cout << "   (" << name << ": "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n";
    };

    compare("Zel'dovich", [&](const std::vector<double> & k, const ZeldovichOptions & o) {
        return zeldovich_hybrid_bias_spectra(P, W, W, k, o);
    });
    compare("Zel'dovich in redshift space (f = 0.8, l = 0, 2, 4, 4 nodes in mu)",
            [&](const std::vector<double> & k, const ZeldovichOptions & o) {
                const auto m = zeldovich_hybrid_bias_multipoles(P, W, W, k, 0.8, 4, o, 4);
                std::vector<std::vector<double>> flat; // [pair * 5 + l][k]
                for (auto & pair : m)
                    for (auto & ell : pair)
                        flat.push_back(ell);
                if (k.empty()) // as the other predictions: one (empty) list per spectrum and multipole
                    flat.assign(15 * 5, {});
                return flat;
            });
    compare("one-loop (EdS)", [&](const std::vector<double> & k, const ZeldovichOptions & o) {
        return oneloop_hybrid_bias_spectra(P, W, W, k, o);
    });
    compare("one-loop (general kernels, LCDM model)", [&](const std::vector<double> & k, const ZeldovichOptions & o) {
        return oneloop_hybrid_bias_spectra_mg(mgmodel_gr(0.3), {1.0}, {P}, W, W, k, o)[0];
    }, false);
    return FML::TESTS::summary("hybrid bias predictions with MPI");
}

#ifndef FML_LINEAR_INPUT_H
#define FML_LINEAR_INPUT_H
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Spline/Spline.h>
#include <FML/Global/Global.h>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
namespace FML::INITIALCONDITIONS {
    using DVector = std::vector<double>;
    /// P(k), P_primordial(k) and T(k) of baryons+CDM at the initial redshift.
    /// Units: k in h/Mpc, P in (Mpc/h)^3 and T in Mpc^2
    struct LinearPowerSpectrumTable {
        DVector k;
        DVector power;
        DVector power_primordial;
        DVector transfer;

        void resize(size_t n) {
            k.resize(n);
            power.resize(n);
            power_primordial.resize(n);
            transfer.resize(n);
        }

        /// Multiply the power-spectra by factor (and the transfer function by sqrt(factor))
        void scale(double factor) {
            for (auto & p : power)
                p *= factor;
            for (auto & p : power_primordial)
                p *= factor;
            for (auto & t : transfer)
                t *= std::sqrt(factor);
        }

        void print_sample(const std::string & description) const {
            if (FML::ThisTask != 0)
                return;
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# " << description << "\n";
            std::cout << "# Found n: " << k.size() << " k-values (sample of it below):\n";
            std::cout << "#=====================================================\n";
            std::cout << "# k (h/Mpc)      P(k,zini)  (Mpc/h)^3    T(k,zini)  (Mpc)^2\n";
            const size_t every = std::max(size_t(1), k.size() / 10);
            for (size_t i = 0; i < k.size(); i += every) {
                std::cout << std::setw(15) << k[i] << " ";
                std::cout << std::setw(15) << power[i] << " ";
                std::cout << std::setw(15) << transfer[i] << "\n";
            }
        }
    };


    enum class LinearInputType { PowerSpectrum, TransferFunction, TransferInfoFile };
    struct LinearInput {
        LinearInputType type;
        std::string filename;
        double a_input, h;
    };
    /// P(k) and single-T(k) inputs need no species provider or Boltzmann solver.
    inline LinearPowerSpectrumTable prepare_linear_input(const LinearInput & input,
                                                         const std::function<double(double)> & growth_ratio,
                                                         const std::function<double(double)> & primordial_power,
                                                         const TransferProvider * provider = nullptr) {
        if (!(input.a_input > 0) || !std::isfinite(input.a_input) || !(input.h > 0) || !std::isfinite(input.h) ||
            !growth_ratio || !primordial_power)
            throw std::invalid_argument("[LinearInput] Invalid input conventions or missing growth/primordial power");
        // Read a two column ascii file with a single header line
        auto read_two_columns = [&]() {
            const int ncols = 2;
            std::vector<int> cols_to_keep{0, 1};
            const int nheaderlines = 1;
            return FML::FILEUTILS::read_regular_ascii(input.filename, ncols, cols_to_keep, nheaderlines);
        };

        LinearPowerSpectrumTable table;
        switch (input.type) {
            case LinearInputType::PowerSpectrum: {
                // Fileformat: [k (h/Mpc)  P(k)  (Mpc/h)^3]
                auto pofkdata = read_two_columns();
                table.resize(pofkdata.size());
                for (size_t i = 0; i < table.k.size(); i++) {
                    table.k[i] = pofkdata[i][0];
                    const double k_hmpc = table.k[i];
                    const double k_mpc = table.k[i] * input.h;
                    const double DinioverDinput = growth_ratio(k_hmpc);
                    table.power_primordial[i] = primordial_power(k_hmpc);
                    table.power[i] = DinioverDinput * DinioverDinput * pofkdata[i][1];
                    table.transfer[i] = std::sqrt(table.power[i] / table.power_primordial[i]) / (k_mpc * k_mpc);
                }
                break;
            }
            case LinearInputType::TransferFunction: {
                // Fileformat: [k (h/Mpc)  T(k)  (Mpc)^2]
                auto tofkdata = read_two_columns();
                table.resize(tofkdata.size());
                for (size_t i = 0; i < table.k.size(); i++) {
                    table.k[i] = tofkdata[i][0];
                    const double k_hmpc = table.k[i];
                    const double k_mpc = table.k[i] * input.h;
                    const double DinioverDinput = growth_ratio(k_hmpc);
                    table.power_primordial[i] = primordial_power(k_hmpc);
                    table.transfer[i] = DinioverDinput * tofkdata[i][1];
                    table.power[i] = table.power_primordial[i] * std::pow(table.transfer[i] * k_mpc * k_mpc, 2);
                }
                break;
            }
            case LinearInputType::TransferInfoFile: {
                if (!provider)
                    throw std::invalid_argument("[LinearInput] Transfer provider is required");
                provider->require_ic_density_gauge();
                if (std::abs(provider->h() / input.h - 1.0) > 1e-10)
                    throw std::invalid_argument("[LinearInput] Provider and IC wavenumber units use different h");
                // For consistency we generate the arrays at the input redshift and use growth-factors
                // to bring them to the initial time (this way we can pick backscaling or not by
                // selecting ic_input_redshift)
                const int nk = 1000;
                const double kmin = provider->range().kmin_hmpc;
                const double kmax = provider->range().kmax_hmpc;
                table.resize(nk);
                for (size_t i = 0; i < table.k.size(); i++) {
                    // From kmin to kmax for every provider (with i / nk the table stopped 1.6% short of kmax)
                    const double k_hmpc = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1));
                    table.k[i] = k_hmpc;
                    table.transfer[i] = provider->density_over_k_mpc2(Species::CDMBaryon, table.k[i], input.a_input);
                    table.power[i] = provider->power(Species::CDMBaryon, table.k[i], input.a_input);
                    const double DinioverDinput = growth_ratio(k_hmpc);
                    table.power_primordial[i] = primordial_power(k_hmpc);
                    table.transfer[i] *= DinioverDinput;
                    table.power[i] *= DinioverDinput * DinioverDinput;
                }
                break;
            }
        }

        if (table.k.size() < 3)
            throw std::invalid_argument("[LinearInput] Need at least three input samples");
        for (size_t i = 0; i < table.k.size(); i++) {
            if (!(table.k[i] > 0) || !std::isfinite(table.k[i]) || (i && table.k[i] <= table.k[i - 1]) ||
                !(table.power[i] >= 0) || !std::isfinite(table.power[i]) || !(table.power_primordial[i] > 0) ||
                !std::isfinite(table.power_primordial[i]) || !std::isfinite(table.transfer[i]))
                throw std::invalid_argument("[LinearInput] Invalid k ordering, power or transfer normalization");
        }
        return table;
    }
} // namespace FML::INITIALCONDITIONS
#endif

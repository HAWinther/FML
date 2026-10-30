#ifndef NBODYGAUGE_HEADER
#define NBODYGAUGE_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/Global/Global.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
// Running the simulation in the N-body gauge (Fidler et al. 2015, 2016)
//
// In the N-body gauge the CDM+baryon particles follow the Newtonian equations of
// motion exactly to linear order in GR if the force is sourced by
//
//     rho_cb (delta_cb + delta_extra)
//
// where delta_extra(k,z) is the (linear) contribution from photons, massless and
// massive neutrinos and a metric term. We read T_extra(k,z) = -delta_extra/k^2
// made by scripts/make_nbody_gauge_transfer.py (with CLASS) and add
//
//     delta_extra(k,a) = delta_cb(k,a_ini) * T_extra(k,a) / T_cb(k,a_ini)
//
// to the density field every time we compute the force. Here delta_cb(k,a_ini) is the
// density field of the particles at the initial time and T_cb the (N-body gauge)
// transfer functions used to make the initial conditions.
//=============================================================================

namespace SIMCODE {

    class NBodyGaugeSource {
      public:
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        using DVector2D = FML::INTERPOLATION::SPLINE::DVector2D;

        NBodyGaugeSource() = default;

        /// Read the table T_extra(k,z) (see make_nbody_gauge_transfer.py for the format)
        void read(const std::string & filename) {
            std::ifstream fp(filename);
            if (not fp)
                throw std::runtime_error("[NBodyGaugeSource] Cannot open [" + filename + "]");

            // Comment lines (the cosmology is in a line starting with '# cosmology')
            std::string line;
            std::vector<std::string> datalines;
            while (std::getline(fp, line)) {
                if (line.rfind("# cosmology", 0) == 0) {
                    std::istringstream ss(line.substr(11));
                    std::string item;
                    while (ss >> item) {
                        auto pos = item.find('=');
                        if (pos != std::string::npos)
                            cosmology[item.substr(0, pos)] = std::stod(item.substr(pos + 1));
                    }
                } else if (not line.empty() and line[0] != '#') {
                    datalines.push_back(line);
                }
            }
            if (datalines.size() < 3)
                throw std::runtime_error("[NBodyGaugeSource] [" + filename + "] is not a valid file");

            int nk = 0, nz = 0;
            std::istringstream(datalines[0]) >> nk >> nz;
            if (nk < 4 or nz < 4 or datalines.size() != size_t(3 + nk))
                throw std::runtime_error("[NBodyGaugeSource] [" + filename + "] has the wrong number of lines");
            auto read_numbers = [&](const std::string & s, size_t n) {
                std::istringstream ss(s);
                DVector v(n);
                for (auto & x : v)
                    if (not(ss >> x))
                        throw std::runtime_error("[NBodyGaugeSource] [" + filename + "] has a line that is too short");
                return v;
            };
            z = read_numbers(datalines[1], nz);
            HoverH0 = read_numbers(datalines[2], nz);
            DVector logk(nk);
            DVector2D Textra(nz, DVector(nk));
            for (int ik = 0; ik < nk; ik++) {
                auto row = read_numbers(datalines[3 + ik], nz + 1);
                logk[ik] = std::log(row[0]);
                for (int iz = 0; iz < nz; iz++)
                    Textra[iz][ik] = row[iz + 1];
            }
            for (int iz = 1; iz < nz; iz++)
                if (not(z[iz] > z[iz - 1]))
                    throw std::runtime_error("[NBodyGaugeSource] The redshifts in [" + filename +
                                             "] must be increasing");
            kmin_hmpc = std::exp(logk.front());
            kmax_hmpc = std::exp(logk.back());
            Textra_spline.create(z, logk, Textra, "T_extra(z, log k)");
            filename_read = filename;
        }

        /// Check that the table was made for the cosmology we run with (throws if not) and
        /// that the expansion history agrees
        void check_cosmology(const Cosmology & cosmo, double zini, double boxsize, int nmesh) const {
            const std::map<std::string, double> ours{{"h", cosmo.get_h()},
                                                     {"Omegab", cosmo.get_Omegab()},
                                                     {"OmegaCDM", cosmo.get_OmegaCDM()},
                                                     {"OmegaMNu", cosmo.get_OmegaMNu()},
                                                     {"Neff", cosmo.get_Neff()},
                                                     {"TCMB", cosmo.get_TCMB_kelvin()},
                                                     {"ns", cosmo.get_ns()}};
            for (auto & [name, value] : ours) {
                auto it = cosmology.find(name);
                if (it == cosmology.end())
                    throw std::runtime_error("[NBodyGaugeSource] The file [" + filename_read + "] has no " + name);
                if (std::fabs(it->second - value) > 1e-6 * std::max(1.0, std::fabs(value)))
                    throw std::runtime_error("[NBodyGaugeSource] The file [" + filename_read + "] was made for " +
                                             name + " = " + std::to_string(it->second) + " but we have " +
                                             std::to_string(value) + ". Make it again with the same cosmology");
            }
            if (zini > z.back() or 0.0 < z.front() - 1e-6)
                throw std::runtime_error(
                    "[NBodyGaugeSource] The file [" + filename_read + "] covers z = " + std::to_string(z.front()) +
                    " - " + std::to_string(z.back()) + " but the simulation runs from z = " + std::to_string(zini));

            // The expansion history (CLASS vs ours). A large difference means the background is not the same
            double maxdiff = 0.0;
            for (size_t i = 0; i < z.size(); i++)
                maxdiff = std::max(maxdiff, std::fabs(cosmo.HoverH0_of_a(1.0 / (1.0 + z[i])) / HoverH0[i] - 1.0));
            if (FML::ThisTask == 0) {
                std::cout << "# N-body gauge source: H(z) of the simulation and of CLASS agree to " << maxdiff
                          << " (max rel. difference)\n";
                const double kfund = 2.0 * M_PI / boxsize;
                const double knyq = M_PI * nmesh / boxsize * std::sqrt(3.0);
                if (kfund < kmin_hmpc or knyq > kmax_hmpc)
                    std::cout << "# N-body gauge source: WARNING the table covers k = " << kmin_hmpc << " - "
                              << kmax_hmpc << " h/Mpc while the box has k = " << kfund << " - " << knyq
                              << " h/Mpc (we use the closest value of T_extra outside the table)\n";
            }
            if (maxdiff > 1e-3)
                throw std::runtime_error("[NBodyGaugeSource] H(z) of CLASS and of the simulation differ by " +
                                         std::to_string(maxdiff) + " > 1e-3. Is the cosmology the same?");
        }

        /// delta_extra(k,a) / delta_cb(k,a_ini) as a spline in log(k) [k in h/Mpc] at the scale factor a
        Spline ratio_spline(double a, double aini, const FML::FILEUTILS::LinearTransferData & transferdata) const {
            const double zz = 1.0 / a - 1.0;
            const int nk = 512;
            DVector logk(nk), ratio(nk);
            for (int i = 0; i < nk; i++) {
                logk[i] = std::log(kmin_hmpc) + std::log(kmax_hmpc / kmin_hmpc) * i / double(nk - 1);
                const double k = std::exp(logk[i]);
                ratio[i] = Textra_spline(zz, logk[i]) / transferdata.get_cdm_baryon_transfer_function(k, aini);
            }
            return Spline(logk, ratio, "delta_extra(k,a) / delta_cb(k,aini)");
        }

        double get_kmin_hmpc() const { return kmin_hmpc; }
        double get_kmax_hmpc() const { return kmax_hmpc; }

      private:
        std::string filename_read;
        std::map<std::string, double> cosmology;
        DVector z;
        DVector HoverH0;
        double kmin_hmpc{0.0};
        double kmax_hmpc{0.0};
        Spline2D Textra_spline{"T_extra(z, log k) not yet read"};
    };

} // namespace SIMCODE

#endif

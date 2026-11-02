#include "TransferInput.h"

#include <FML/TransferProviders/MakeTransferProvider.h>
#include <FML/Global/Global.h>
#include <FML/TransferProviders/TabulatedTransferProvider.h>
#ifdef USE_INTERNAL_COSMOLOGY
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/InitialConditions/InternalTransferProvider.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SIMCODE {

    namespace {
        bool matches(double a, double b, double rtol) {
            return std::abs(a - b) <= rtol * std::max({std::abs(a), std::abs(b), 1e-30});
        }

        using FML::INITIALCONDITIONS::TransferTables;

#ifdef USE_MPI
        /// The tables of task 0 on all tasks
        void broadcast(TransferTables & t) {
            auto bcast_vector = [](std::vector<double> & v) {
                long n = long(v.size());
                MPI_Bcast(&n, 1, MPI_LONG, 0, FML::communicator);
                v.resize(n);
                if (n > 0)
                    MPI_Bcast(v.data(), int(n), MPI_DOUBLE, 0, FML::communicator);
            };
            auto bcast_table = [&](TransferTables::DVector2D & table) {
                long n = long(table.size());
                MPI_Bcast(&n, 1, MPI_LONG, 0, FML::communicator);
                table.resize(n);
                for (auto & row : table)
                    bcast_vector(row);
            };
            bcast_vector(t.a);
            bcast_vector(t.k_hmpc);
            double scalars[] = {t.h, t.A_s, t.n_s, t.k_pivot_mpc};
            MPI_Bcast(scalars, 4, MPI_DOUBLE, 0, FML::communicator);
            t.h = scalars[0], t.A_s = scalars[1], t.n_s = scalars[2], t.k_pivot_mpc = scalars[3];
            int gauges[] = {int(t.density_gauge), int(t.velocity_gauge), int(t.range_policy)};
            MPI_Bcast(gauges, 3, MPI_INT, 0, FML::communicator);
            t.density_gauge = FML::INITIALCONDITIONS::Gauge(gauges[0]);
            t.velocity_gauge = FML::INITIALCONDITIONS::Gauge(gauges[1]);
            t.range_policy = FML::INITIALCONDITIONS::RangePolicy(gauges[2]);
            for (int s = 0; s < TransferTables::nspecies; s++) {
                bcast_table(t.density[s]);
                bcast_table(t.velocity[s]);
            }
        }
#endif

#ifdef USE_INTERNAL_COSMOLOGY
        /// The settings of the solver: those of the .ini file r.filename (or the defaults) with the cosmology of the
        /// simulation (its background parameters, model and model parameters, the primordial spectrum) and a k- and
        /// a-range that covers the request
        FML::COSMOLOGY::RunSettings internal_settings(const TransferInputRequest & r) {
            using FML::COSMOLOGY::Constants::Mpc;
            const std::string msg = "[ic_type_of_input = internal] ";
            if (r.model.empty())
                throw std::invalid_argument(msg + "the cosmology_model has no model of the solver (Tabulated)");
            auto settings = FML::COSMOLOGY::ParameterFile::read(r.filename);
            const auto & b = r.background;
            auto & c = settings.cosmo;
            c.h = b.h;
            c.OmegaB = b.Omega_b;
            c.OmegaCDM = b.Omega_cdm;
            c.OmegaK = b.Omega_k;
            c.TCMB = b.T_CMB;
            c.Neff = b.N_eff;
            c.m_nu = 0.0;
            c.deg_ncdm = 1;
            if (not b.m_nu_eV.empty()) {
                // N degenerate species of mass m (the solver has one mass)
                for (double m : b.m_nu_eV)
                    if (m != b.m_nu_eV.front())
                        throw std::invalid_argument(msg + "the neutrino masses must be degenerate");
                c.m_nu = b.m_nu_eV.front();
                c.deg_ncdm = int(b.m_nu_eV.size());
                c.T_ncdm_over_TCMB = b.T_ncdm_over_TCMB;
                // All the neutrinos massive (Neff = N_ncdm up to roundoff)
                if (c.N_ncdm() > c.Neff and matches(c.N_ncdm(), c.Neff, 1e-12))
                    c.Neff = c.N_ncdm();
            }
            c.model = r.model;
            c.model_params = r.model_parameters;
            c.A_s = r.As;
            c.n_s = r.ns;
            c.k_pivot = r.kpivot_mpc;
            c.alpha_s = 0.0;

            // k: the IC grid with a margin, from 1e-4 h/Mpc (the growth factors) to at least 5 h/Mpc (sigma8; above
            // kmax the tables are extrapolated with a power law). 15 points per decade, 50 in [0.01, 0.6] h/Mpc where
            // the BAO are (the transfer functions interpolated to 1e-4; 25 s with massive neutrinos and kmax = 5
            // h/Mpc on 2 cores). The cost grows with kmax (the photon and neutrino hierarchies until today)
            const double kmin = std::min(1e-4, 0.5 * r.required.kmin_hmpc);
            const double kmax = std::max(5.0, 1.5 * r.required.kmax_hmpc);
            std::vector<double> k{kmin};
            while (k.back() < kmax) {
                const double per_decade = k.back() < 0.01 or k.back() > 0.6 ? 15.0 : 50.0;
                k.push_back(k.back() * std::pow(10.0, 1.0 / per_decade));
            }
            if (k.size() > 2 and kmax - k[k.size() - 2] < 0.5 * (k[k.size() - 2] - k[k.size() - 3]))
                k.erase(k.end() - 2);
            k.back() = kmax;
            const double hmpc_to_si = b.h / Mpc;
            for (double & q : k)
                q *= hmpc_to_si;
            settings.pert.k_values = k;
            settings.pert.k_min = k.front();
            settings.pert.k_max = k.back();
            settings.pert.n_k = int(k.size());
            // a: past a = 1 (the growth rates at a = 1 are then not at the end of the table)
            settings.pert.x_end = std::log(std::max(1.1, 1.1 * r.required.amax));
            settings.bg_x_end = std::max(settings.bg_x_end, settings.pert.x_end);
            if (not(std::exp(settings.pert.x_start) < 0.5 * r.required.amin))
                throw std::invalid_argument(msg + "the initial redshift is earlier than pert_x_start of the solver");
            return settings;
        }

        /// The tables of the solver (on this task)
        TransferTables internal_tables(const TransferInputRequest & r) {
            using namespace FML::INITIALCONDITIONS;
            using FML::COSMOLOGY::Constants::Mpc;
            const auto t0 = std::chrono::steady_clock::now();
            const auto settings = internal_settings(r);
            const InternalTransferProvider solver(settings, Gauge::SynchronousCDM);

            // The k of the solver and the scale factors from half the earliest requested to the end
            // (with curvature the solver's k = sqrt(q^2 - K) differ from these q: the k inside its range)
            const auto range = solver.range();
            std::vector<double> k_hmpc{range.kmin_hmpc};
            for (double q : settings.pert.k_values) {
                const double k = q * Mpc / settings.cosmo.h;
                if (k > k_hmpc.back() * (1.0 + 1e-10) and k < range.kmax_hmpc * (1.0 - 1e-10))
                    k_hmpc.push_back(k);
            }
            k_hmpc.push_back(range.kmax_hmpc);
            const int nk = int(k_hmpc.size());
            const int na = 128;
            const double alow = std::max(range.amin, 0.5 * r.required.amin), ahigh = range.amax;
            std::vector<double> a(na);
            for (int i = 0; i < na; i++)
                a[i] = std::exp(std::log(alow) + std::log(ahigh / alow) * i / double(na - 1));
            a.front() = alow;
            a.back() = ahigh;
            auto tables = tabulate_transfers(solver,
                                             a,
                                             k_hmpc,
                                             r.As,
                                             r.ns,
                                             r.kpivot_mpc,
                                             RangePolicy::SourceExtrapolation,
                                             "InternalTransfers");
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto & c = settings.cosmo;
            std::cout << "\n#=====================================================\n"
                      << "# Transfer functions from the internal Einstein-Boltzmann solver\n"
                      << "# Model: " << c.model << (r.filename.empty() ? "" : " (solver settings: " + r.filename + ")")
                      << "\n"
                      << "# Massive neutrinos: "
                      << (c.m_nu > 0.0 ? std::to_string(c.deg_ncdm) + " x " + std::to_string(c.m_nu) + " eV" : "none")
                      << "\n"
                      << "# k = " << k_hmpc.front() << " - " << k_hmpc.back() << " h/Mpc (" << nk
                      << "), tables for z = " << 1.0 / ahigh - 1.0 << " - " << 1.0 / alow - 1.0 << " (" << na
                      << "), " << seconds << " s\n"
                      << "#=====================================================\n";
            return tables;
        }
#endif

        /// internal: FML's Einstein-Boltzmann solver run with the cosmology of the simulation: its background
        ///   parameters (the massive neutrinos as N degenerate species), cosmology_model with its parameters and the
        ///   primordial spectrum. ic_input_filename: an optional .ini of the solver with its precision, recombination
        ///   and reionization settings (its cosmology is replaced; its k-range is widened if needed). The solver runs on
        ///   task 0 and its tables (CDM-synchronous densities, Newtonian velocities; all species it has) are sent to
        ///   the other tasks. They are extrapolated outside their range as the CAMB and CLASS tables.
        std::shared_ptr<TransferProvider> internal_transfers(const TransferInputRequest & r) {
#ifdef USE_INTERNAL_COSMOLOGY
            TransferTables tables;
            std::string error;
            if (FML::ThisTask == 0) {
                try {
                    tables = internal_tables(r);
                } catch (const std::exception & e) {
                    error = e.what();
                }
            }
#ifdef USE_MPI
            int failed = error.empty() ? 0 : 1;
            MPI_Bcast(&failed, 1, MPI_INT, 0, FML::communicator);
            if (failed) {
                if (FML::ThisTask == 0)
                    std::cerr << "[ic_type_of_input = internal] " << error << "\n";
                throw std::runtime_error("[ic_type_of_input = internal] the solver failed on task 0: " +
                                         (error.empty() ? std::string("see the output of task 0") : error));
            }
            broadcast(tables);
#else
            if (not error.empty())
                throw std::runtime_error("[ic_type_of_input = internal] " + error);
#endif
            return std::make_shared<FML::INITIALCONDITIONS::TransferTablesProvider>(tables);
#else
            (void)r;
            throw std::runtime_error("ic_type_of_input = internal needs a SimulationCode build with "
                                     "USE_INTERNAL_COSMOLOGY=true");
#endif
        }
    } // namespace

    //=============================================================================
    /// camb: CAMB transfer files (the columns read by name from the header, or by position), with the
    ///   weights of CDM+baryons from the cosmology of the simulation. The densities are CDM-synchronous.
    /// class: native CLASS output (format = class, output = mTk, vTk, write_background = yes) with the root
    ///   ic_input_filename, run in the gauge ic_input_gauge (Newtonian is converted to CDM-synchronous). h and
    ///   whether there are massive neutrinos are checked against the simulation.
    /// internal: FML's Einstein-Boltzmann solver run with the cosmology of the simulation (see internal_transfers
    ///   above; needs a build with USE_INTERNAL_COSMOLOGY=true).
    /// The tables are extrapolated outside their range as the old CAMB reader did (RangePolicy::SourceExtrapolation:
    /// a power law above k_max, the end values elsewhere)
    //=============================================================================
    std::shared_ptr<TransferProvider> load_transfers(const TransferInputRequest & r) {
        using namespace FML::INITIALCONDITIONS;
        TransferSourceSettings s;
        s.h = r.h;
        s.Omegab = r.Omegab;
        s.OmegaCDM = r.OmegaCDM;
        s.OmegaMNu = r.OmegaMNu;
        s.A_s = r.As;
        s.n_s = r.ns;
        s.k_pivot_mpc = r.kpivot_mpc;
        s.output_gauge = Gauge::SynchronousCDM;
        s.range_policy = RangePolicy::SourceExtrapolation;

        if (canonical_input_type(r.type) == "camb") {
            s.source = "camb";
            s.camb_infofile = r.filename;
            return make_transfer_provider(s);
        }

        if (r.type == "class") {
            if (r.gauge != "synchronous" and r.gauge != "newtonian")
                throw std::invalid_argument("ic_input_gauge must be synchronous or newtonian (the gauge of the CLASS "
                                            "run), not [" + r.gauge + "]");
            s.source = "class";
            s.class_root = r.filename;
            s.input_gauge = r.gauge == "newtonian" ? Gauge::Newtonian : Gauge::SynchronousCDM;
            auto provider = make_transfer_provider(s);
            if (not matches(provider->h(), r.h, 1e-4))
                throw std::invalid_argument("The CLASS run [" + r.filename + "] has h = " +
                                            std::to_string(provider->h()) + ", the simulation " + std::to_string(r.h));
            if (provider->has_density(Species::MassiveNeutrino) != (r.OmegaMNu > 0.0))
                throw std::invalid_argument("The CLASS run [" + r.filename + "] " +
                                            (r.OmegaMNu > 0.0 ? "has no" : "has") +
                                            " massive neutrinos, the simulation has cosmology_OmegaMNu = " +
                                            std::to_string(r.OmegaMNu));
            return provider;
        }

        if (r.type == "internal")
            return internal_transfers(r);

        throw std::invalid_argument("[load_transfers] Unknown ic_type_of_input [" + r.type +
                                    "] (camb, class or internal)");
    }

} // namespace SIMCODE

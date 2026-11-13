#include "TransferInput.h"

#include <FML/TransferProviders/MakeTransferProvider.h>
#include <FML/Global/Global.h>
#include <FML/TransferProviders/TabulatedTransferProvider.h>
#include <FML/TransferProviders/TransferTablesFile.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/InitialConditions/InternalTransferProvider.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
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
            bcast_table(t.nbody_gauge_source);
        }
#endif

        /// The parameters of the solver that the simulation sets, with what sets them (an error in the .ini file and
        /// in ic_internal_solver_parameters). Model parameters: those of the model of the simulation (the parameters
        /// of another model are unknown names for the solver)
        std::string set_by_simulation(const std::string & name, const std::string & model) {
            static const std::map<std::string, std::string> fixed{
                {"preset", "the cosmology_* parameters"},
                {"h", "cosmology_h"},
                {"OmegaB", "cosmology_Omegab"},
                {"OmegaCDM", "cosmology_OmegaCDM"},
                {"OmegaK", "cosmology_OmegaK"},
                {"TCMB", "cosmology_TCMB_kelvin"},
                {"Neff", "cosmology_Neffective"},
                {"m_nu", "cosmology_OmegaMNu or cosmology_neutrino_masses_eV"},
                {"deg_ncdm", "cosmology_OmegaMNu or cosmology_neutrino_masses_eV"},
                {"m_nu_list", "cosmology_OmegaMNu or cosmology_neutrino_masses_eV"},
                {"T_ncdm_over_TCMB", "the neutrino background of the simulation"},
                {"model", "cosmology_model"},
                {"model_file", "cosmology_horndeski_filename"},
                {"A_s", "cosmology_As"},
                {"n_s", "cosmology_ns"},
                {"k_pivot", "cosmology_kpivot_mpc"},
                {"alpha_s", "the primordial spectrum of the simulation, which has no running"},
                {"pert_k_min", "the k-range of the IC grid"},
                {"pert_k_max", "the k-range of the IC grid"},
                {"pert_n_k", "the k-range of the IC grid"},
                {"pert_dk_max", "the k-range of the IC grid"}};
            const auto it = fixed.find(name);
            if (it != fixed.end())
                return it->second;
            const auto model_parameters = FML::COSMOLOGY::ParameterFile::model_parameter_names(model);
            if (std::find(model_parameters.begin(), model_parameters.end(), name) != model_parameters.end())
                return "the parameters of cosmology_model";
            return "";
        }

        /// The settings of the solver: those of the .ini file r.filename (or the defaults) and of
        /// ic_internal_solver_parameters, with the cosmology of the simulation (its background parameters, model and
        /// model parameters, the primordial spectrum) and a k- and a-range that covers the request
        FML::COSMOLOGY::RunSettings internal_settings(const TransferInputRequest & r) {
            using FML::COSMOLOGY::Constants::Mpc;
            // (internal_transfers prefixes the errors with [ic_type_of_input = internal])
            if (r.model.empty())
                throw std::invalid_argument("the cosmology_model has no model of the solver (Tabulated)");
            const std::string where = "ic_internal_solver_parameters";
            for (const auto & e : FML::COSMOLOGY::ParameterFile::entries(r.filename, r.solver_parameters, where)) {
                const auto by = set_by_simulation(e.name, r.model);
                if (not by.empty())
                    throw std::invalid_argument(e.where + ": the solver parameter " + e.name +
                                                " is set by the simulation (" + by + "): remove it");
            }
            auto settings = FML::COSMOLOGY::ParameterFile::read(r.filename, r.solver_parameters, where);
            const auto & b = r.background;
            auto & c = settings.cosmo;
            c.h = b.h;
            c.OmegaB = b.Omega_b;
            c.OmegaCDM = b.Omega_cdm;
            c.OmegaK = b.Omega_k;
            c.TCMB = b.T_CMB;
            c.Neff = b.N_eff;
            // The massive species (equal masses are solved as one)
            c.m_nu = 0.0;
            c.deg_ncdm = 1;
            c.m_nu_list = b.m_nu_eV;
            if (not b.m_nu_eV.empty()) {
                c.T_ncdm_over_TCMB = b.T_ncdm_over_TCMB;
                // All the neutrinos massive (Neff = N_ncdm up to roundoff)
                if (c.N_ncdm() > c.Neff and matches(c.N_ncdm(), c.Neff, 1e-12))
                    c.Neff = c.N_ncdm();
            }
            c.model = r.model;
            c.model_params = r.model_parameters;
            c.model_file = b.model_file;
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
                throw std::invalid_argument("the initial redshift is earlier than pert_x_start of the solver");
            return settings;
        }

        /// The tables of the solver (on this task)
        TransferTables internal_tables(const TransferInputRequest & r) {
            using namespace FML::INITIALCONDITIONS;
            using FML::COSMOLOGY::Constants::Mpc;
            const auto t0 = std::chrono::steady_clock::now();
            const auto settings = internal_settings(r);
            const InternalTransferProvider solver(
                settings, r.nbody_gauge ? Gauge::NBody : Gauge::SynchronousCDM, r.axion_ic);

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
            // The axion of ic_axion_fraction must be dark matter (a w = 0 fluid) from the start of the tables
            if (r.axion_ic and not(solver.axion_fluid_scale_factor() <= alow))
                throw std::invalid_argument(
                    "ic_axion_log10_m = " + std::to_string(r.model_parameters.at("ax_log10_m")) +
                    ": the axion is a matter-like fluid only from a = " +
                    std::to_string(solver.axion_fluid_scale_factor()) + ", after the start of the tables (a = " +
                    std::to_string(alow) + ", half the initial scale factor): a heavier axion or a later start");
            auto tables = tabulate_transfers(solver,
                                             a,
                                             k_hmpc,
                                             r.As,
                                             r.ns,
                                             r.kpivot_mpc,
                                             RangePolicy::SourceExtrapolation,
                                             "InternalTransfers");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto & c = settings.cosmo;
            std::cout << "\n#=====================================================\n"
                      << "# Transfer functions from the internal Einstein-Boltzmann solver\n"
                      << "# Model: " << c.model << (r.filename.empty() ? "" : " (solver settings: " + r.filename + ")")
                      << (r.axion_ic ? " (ultralight axion in the IC only: m = 10^" +
                                           std::to_string(r.model_parameters.at("ax_log10_m")) +
                                           " eV, ax_fraction = " + std::to_string(r.model_parameters.at("ax_fraction")) +
                                           "; CDM = CDM + axion)" :
                                       "")
                      << "\n"
                      << "# Density gauge: "
                      << (r.nbody_gauge ? "N-body (with the N-body gauge source)" : "CDM-synchronous") << "\n"
                      << "# Solver parameters: " <<
                [&] {
                    std::string o;
                    for (const auto & p : r.solver_parameters)
                        o += (o.empty() ? "" : ", ") + p;
                    return o.empty() ? std::string("none") : o;
                }() << "\n"
                      << "# Massive neutrinos: " <<
                [&] {
                    std::string o;
                    for (double m : c.neutrino_masses())
                        o += (o.empty() ? "" : ", ") + std::to_string(m);
                    return o.empty() ? std::string("none") : o + " eV";
                }() << "\n"
                      << "# k = " << k_hmpc.front() << " - " << k_hmpc.back() << " h/Mpc (" << nk
                      << "), tables for z = " << 1.0 / ahigh - 1.0 << " - " << 1.0 / alow - 1.0 << " (" << na << "), "
                      << seconds << " s\n"
                      << "#=====================================================\n";
            return tables;
        }

        /// The cosmology of the request as metadata of a tables file (the values with 17 digits): written with the
        /// tables and compared when they are read
        std::vector<std::pair<std::string, std::string>> cosmology_metadata(const TransferInputRequest & r) {
            const auto exact = [](double x) {
                std::ostringstream o;
                o.precision(17);
                o << x;
                return o.str();
            };
            const auto & b = r.background;
            std::string masses;
            for (double m : b.m_nu_eV)
                masses += (masses.empty() ? "" : " ") + exact(m);
            std::vector<std::pair<std::string, std::string>> m{
                {"cosmology_model", r.model},
                {"cosmology_h", exact(b.h)},
                {"cosmology_Omega_b", exact(b.Omega_b)},
                {"cosmology_Omega_cdm", exact(b.Omega_cdm)},
                {"cosmology_Omega_k", exact(b.Omega_k)},
                {"cosmology_T_CMB", exact(b.T_CMB)},
                {"cosmology_N_eff", exact(b.N_eff)},
                {"cosmology_m_nu_eV", masses.empty() ? "none" : masses},
                {"cosmology_T_ncdm_over_TCMB", exact(b.T_ncdm_over_TCMB)}};
            for (const auto & [name, value] : r.model_parameters)
                m.push_back({"cosmology_model_" + name, exact(value)});
            return m;
        }

        /// tables: the file ic_input_filename, checked against the cosmology and the gauge of the simulation and the
        ///   redshifts it needs
        std::shared_ptr<TransferProvider> tables_transfers(const TransferInputRequest & r) {
            using namespace FML::INITIALCONDITIONS;
            const std::string msg = "[ic_type_of_input = tables] [" + r.filename + "] ";
            const std::string remake = " (make the tables for this parameterfile with ./nbody --write-transfers "
                                       "parameterfile.lua file, with ic_type_of_input = internal in it)";
            auto file = read_transfer_tables(r.filename);
            // The cosmology (the background, model and its parameters, the primordial spectrum)
            for (const auto & [key, value] : cosmology_metadata(r)) {
                if (not file.contains(key))
                    throw std::runtime_error(msg + "The file has no " + key + remake);
                if (file.get(key) != value) {
                    // The same number written differently (an older writer): compare the values
                    std::istringstream a(file.get(key)), b(value);
                    double x, y;
                    bool same = true, any = false;
                    while (true) {
                        const bool ga = bool(a >> x), gb = bool(b >> y);
                        if (ga != gb or (ga and not matches(x, y, 1e-12))) {
                            same = false;
                            break;
                        }
                        if (not ga)
                            break;
                        any = true;
                    }
                    if (not same or not any)
                        throw std::runtime_error(msg + "Made for " + key + " = " + file.get(key) +
                                                 ", the simulation has " + value + remake);
                }
            }
            for (const auto & [key, value] : file.metadata)
                if (key.rfind("cosmology_model_", 0) == 0 and
                    r.model_parameters.count(key.substr(std::string("cosmology_model_").size())) == 0)
                    throw std::runtime_error(msg + "Made with the model parameter " + key +
                                             ", which the simulation does not have" + remake);
            const auto & t = file.tables;
            if (not matches(t.A_s, r.As, 1e-12) or not matches(t.n_s, r.ns, 1e-12) or
                not matches(t.k_pivot_mpc, r.kpivot_mpc, 1e-12))
                throw std::runtime_error(msg + "Made for another primordial spectrum (A_s, n_s, k_pivot)" + remake);
            // The gauge
            if (r.nbody_gauge and (t.density_gauge != Gauge::NBody or t.nbody_gauge_source.empty()))
                throw std::runtime_error(msg +
                                         "force_nbody_gauge = true needs N-body gauge tables with the source "
                                         "(made with force_nbody_gauge = true)" +
                                         remake);
            if (not r.nbody_gauge and t.density_gauge != Gauge::SynchronousCDM)
                throw std::runtime_error(msg +
                                         "The tables are in the N-body gauge: use them with force_nbody_gauge "
                                         "= true, or make CDM-synchronous ones" +
                                         remake);
            // The redshifts: up to the initial (and input) redshift
            if (t.a.front() > r.required.amin * (1.0 + 1e-10))
                throw std::runtime_error(msg + "The tables start at z = " + std::to_string(1.0 / t.a.front() - 1.0) +
                                         ", the simulation needs z = " + std::to_string(1.0 / r.required.amin - 1.0) +
                                         remake);
            if (FML::ThisTask == 0) {
                std::cout << "\n#=====================================================\n"
                          << "# Transfer functions from the tables " << r.filename << "\n"
                          << "# Model: " << r.model << ", density gauge: "
                          << (t.density_gauge == Gauge::NBody ? "N-body (with the N-body gauge source)" :
                                                                "CDM-synchronous")
                          << "\n"
                          << "# k = " << t.k_hmpc.front() << " - " << t.k_hmpc.back() << " h/Mpc (" << t.k_hmpc.size()
                          << "), z = " << 1.0 / t.a.back() - 1.0 << " - " << 1.0 / t.a.front() - 1.0 << " ("
                          << t.a.size() << ")\n"
                          << "#=====================================================\n";
            }
            return std::make_shared<TransferTablesProvider>(t);
        }

        /// internal: FML's Einstein-Boltzmann solver run with the cosmology of the simulation: its background
        ///   parameters (its massive neutrino species; equal masses solved as one), cosmology_model with its parameters
        ///   and the primordial spectrum. ic_input_filename: an optional .ini of the solver with its precision,
        ///   recombination and reionization settings (its cosmology is replaced; its k-range is widened if needed). The
        ///   solver runs on task 0 and its tables (CDM-synchronous densities, Newtonian velocities; all species it has)
        ///   are sent to the other tasks. They are extrapolated outside their range as the CAMB and CLASS tables.
        std::shared_ptr<TransferProvider> internal_transfers(const TransferInputRequest & r) {
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
        }
    } // namespace

    //=============================================================================
    /// camb: CAMB transfer files (the columns read by name from the header, or by position), with the
    ///   weights of CDM+baryons from the cosmology of the simulation. The densities are CDM-synchronous.
    /// class: native CLASS output (format = class, output = mTk, vTk, write_background = yes) with the root
    ///   ic_input_filename, run in the gauge ic_input_gauge (Newtonian is converted to CDM-synchronous). h and
    ///   whether there are massive neutrinos are checked against the simulation.
    /// internal: FML's Einstein-Boltzmann solver run with the cosmology of the simulation (see internal_transfers
    ///   above).
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
                                            "run), not [" +
                                            r.gauge + "]");
            s.source = "class";
            s.class_root = r.filename;
            s.input_gauge = r.gauge == "newtonian" ? Gauge::Newtonian : Gauge::SynchronousCDM;
            auto provider = make_transfer_provider(s);
            if (not matches(provider->h(), r.h, 1e-4))
                throw std::invalid_argument("The CLASS run [" + r.filename + "] has h = " +
                                            std::to_string(provider->h()) + ", the simulation " + std::to_string(r.h));
            if (provider->has_density(Species::MassiveNeutrino) != (r.OmegaMNu > 0.0))
                throw std::invalid_argument(
                    "The CLASS run [" + r.filename + "] " + (r.OmegaMNu > 0.0 ? "has no" : "has") +
                    " massive neutrinos, the simulation has cosmology_OmegaMNu = " + std::to_string(r.OmegaMNu));
            return provider;
        }

        if (r.type == "internal")
            return internal_transfers(r);

        if (r.type == "tables")
            return tables_transfers(r);

        throw std::invalid_argument("[load_transfers] Unknown ic_type_of_input [" + r.type +
                                    "] (camb, class, internal or tables)");
    }

    //=============================================================================
    /// The solver parameters a SimulationCode user sets (with their defaults): those of the sections Recombination,
    /// Background and Perturbations that the simulation does not set (set_by_simulation: the cosmology and the k-grid).
    /// Not written: the cosmology (set by the simulation), isocurvature (an error: adiabatic IC only), the tensors, the
    /// CMB spectra / line of sight and the output (unused: only the perturbations are solved) and
    /// store_source_components (CMB sources only). Reading the file gives the solver's defaults
    //=============================================================================
    void write_internal_solver_defaults(std::ostream & os) {
        using FML::COSMOLOGY::ParameterFile::describe;
        const auto defaults = FML::COSMOLOGY::ParameterFile::read("");
        std::vector<FML::COSMOLOGY::ParameterFile::Description> params;
        for (const auto & p : describe(defaults))
            if ((p.section == "Recombination" or p.section == "Background" or p.section == "Perturbations") and
                p.name != "store_source_components" and set_by_simulation(p.name, defaults.cosmo.model).empty())
                params.push_back(p);
        size_t w = 0;
        for (const auto & p : params)
            w = std::max(w, p.name.size() + 3 + p.value.size());
        os << "# Settings of FML's Einstein-Boltzmann solver for SimulationCode (ic_type_of_input = \"internal\"):\n"
           << "#   ic_input_filename = \"internal_solver_default.ini\" (or a copy of it)\n"
           << "# These are the solver's defaults (the same as ic_input_filename = \"\"): change the ones you need and "
              "remove\n"
           << "# the rest. ic_internal_solver_parameters = {\"name=value\", ...} in the parameter file is applied "
              "after this.\n"
           << "# The cosmology, the primordial spectrum, the model and its parameters and the k-grid (pert_k_*) are "
              "set by\n"
           << "# the simulation and are an error here; the tensors, CMB spectra and output settings are not used (see\n"
           << "# Cosmology/EinsteinBoltzmannSolver/params/default.ini for all the parameters).\n"
           << "# bg_x_end is raised past the end of the perturbations (a = 1.1) if needed.\n"
           << "# Generated with ./nbody --write-solver-defaults internal_solver_default.ini\n"
           << "#\n"
           << "# precision = default    # or fast: a faster set of precision parameters (explicit ones override it)\n";
        std::string section;
        for (const auto & p : params) {
            if (p.section != section) {
                section = p.section;
                os << "\n# --- " << section << " " << std::string(70 - section.size(), '-') << "\n";
            }
            os << std::left << std::setw(int(w)) << (p.name + " = " + p.value) << "  # " << p.description << "\n";
        }
    }

    //=============================================================================
    /// The tables of the internal solver (as for ic_type_of_input = internal) written to filename on task 0, with
    /// the cosmology (see cosmology_metadata) and the settings of the solver
    //=============================================================================
    void write_internal_transfer_tables(const TransferInputRequest & r, const std::string & filename) {
        if (r.type != "internal")
            throw std::invalid_argument("--write-transfers needs ic_type_of_input = internal in the parameterfile "
                                        "(it has " +
                                        r.type + ")");
        std::string error;
        if (FML::ThisTask == 0) {
            try {
                const auto tables = internal_tables(r);
                auto metadata = cosmology_metadata(r);
                std::string solver_parameters;
                for (const auto & p : r.solver_parameters)
                    solver_parameters += (solver_parameters.empty() ? "" : " ") + p;
                metadata.insert(metadata.begin(),
                                {{"written_by",
                                  "FML SimulationCode: nbody --write-transfers (the internal "
                                  "Einstein-Boltzmann solver)"},
                                 {"solver_ini_file", r.filename.empty() ? "none" : r.filename},
                                 {"solver_parameters", solver_parameters.empty() ? "none" : solver_parameters}});
                FML::INITIALCONDITIONS::write_transfer_tables(filename, tables, metadata);
                std::cout << "Wrote the transfer tables to " << filename << "\n";
            } catch (const std::exception & e) {
                error = e.what();
            }
        }
#ifdef USE_MPI
        int failed = error.empty() ? 0 : 1;
        MPI_Bcast(&failed, 1, MPI_INT, 0, FML::communicator);
        if (failed)
            throw std::runtime_error("[--write-transfers] " +
                                     (error.empty() ? std::string("failed on task 0") : error));
#else
        if (not error.empty())
            throw std::runtime_error("[--write-transfers] " + error);
#endif
    }

} // namespace SIMCODE

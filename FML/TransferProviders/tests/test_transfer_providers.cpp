//=====================================================================================================================
// Tests of the CLASS provider, the gauge conversion and make_transfer_provider against reference runs of CLASS (in
// the synchronous and the Newtonian gauge) and CAMB with the same cosmology (make_test_data.sh writes them), and with
// USE_INTERNAL_COSMOLOGY of source = internal (FML's Einstein-Boltzmann solver) against CLASS, with and without a
// massive neutrino (two solves, a few minutes).
//
//   ./test_transfer_providers [data folder (default: data)]
//
// Without the data folder the test prints SKIP and exits 0 (the reference codes are not part of FML).
//=====================================================================================================================

#include <FML/TransferProviders/MakeTransferProvider.h>
#include <FML/Units/Units.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::INITIALCONDITIONS;
using FML::TESTS::check;
using FML::TESTS::num;

namespace {
    const Species all_species[] = {
        Species::CDM, Species::Baryon, Species::CDMBaryon, Species::MassiveNeutrino, Species::TotalMatter};
    const char * name_of(Species s) {
        switch (s) {
            case Species::CDM:
                return "CDM";
            case Species::Baryon:
                return "baryons";
            case Species::CDMBaryon:
                return "CDM+baryons";
            case Species::MassiveNeutrino:
                return "massive neutrinos";
            case Species::TotalMatter:
                return "total matter";
        }
        return "?";
    }
    const double redshifts[] = {49, 20, 10, 5, 2, 1, 0};

    template <class F>
    bool throws(F && f) {
        try {
            f();
        } catch (std::exception &) {
            return true;
        }
        return false;
    }
} // namespace

int main(int argc, char ** argv) {
    const std::string data = argc > 1 ? argv[1] : "data";
    if (not std::filesystem::exists(data + "/class_synchronous_00_background.dat")) {
        if (FML::ThisTask == 0)
            std::cout << "SKIP test_transfer_providers: no reference data in [" << data
                      << "] (make it with CLASS=/path/to/class ./make_test_data.sh)\n";
        return 0;
    }

    TransferSourceSettings cls;
    cls.source = "class";
    cls.A_s = 2.1e-9;
    cls.n_s = 0.965;
    cls.range_policy = RangePolicy::Reject;
    auto class_settings = [&](const std::string & gauge, Gauge in, Gauge out) {
        auto s = cls;
        s.class_root = data + "/class_" + gauge + "_00_";
        s.input_gauge = in;
        s.output_gauge = out;
        return s;
    };
    auto sync = make_transfer_provider(class_settings("synchronous", Gauge::SynchronousCDM, Gauge::SynchronousCDM));
    auto newt_to_sync = make_transfer_provider(class_settings("newtonian", Gauge::Newtonian, Gauge::SynchronousCDM));
    auto newt = std::make_shared<CLASSTransferProvider>([&] {
        CLASSTransferProvider::Settings s;
        auto files = find_class_output_files(data + "/class_newtonian_00_");
        s.transfer_files = files.first;
        s.background_file = files.second;
        s.gauge = s.output_gauge = Gauge::Newtonian;
        s.A_s = cls.A_s;
        s.n_s = cls.n_s;
        return s;
    }());

    TransferSourceSettings cmb;
    cmb.source = "camb";
    cmb.camb_infofile = data + "/camb_transferinfo.txt";
    cmb.A_s = cls.A_s;
    cmb.n_s = cls.n_s;
    cmb.range_policy = RangePolicy::Reject;
    {
        std::ifstream f(data + "/camb_parameters.txt");
        f >> cmb.Omegab >> cmb.OmegaCDM >> cmb.OmegaMNu >> cmb.h;
    }
    auto camb = make_transfer_provider(cmb);

    // Reading
    check(find_class_output_files(data + "/class_synchronous_00_").first.size() == 40,
          "find_class_output_files finds the 40 transfer files");
    check(std::abs(sync->h() - 0.67) < 1e-6, "h from the CLASS background: " + num(sync->h()));
    check(sync->density_gauge() == Gauge::SynchronousCDM and newt_to_sync->density_gauge() == Gauge::SynchronousCDM and
              newt_to_sync->velocity_gauge() == Gauge::Newtonian,
          "gauges: converted densities are CDM-synchronous, the velocities stay Newtonian");

    // 1. Newtonian run converted to the CDM-synchronous gauge = synchronous run. At
    // k = 1e-4 h/Mpc the conversion cancels two terms 1e5 times larger than the result
    for (auto s : all_species) {
        double max_large = 0.0, max_small = 0.0, max_unconverted = 0.0;
        for (double z : redshifts)
            for (double k : {1e-3, 1e-2, 0.1, 1.0}) {
                const double a = 1.0 / (1.0 + z), ref = sync->density_transfer(s, k, a);
                max_small = std::max(max_small, std::abs(newt_to_sync->density_transfer(s, k, a) / ref - 1.0));
            }
        for (double z : redshifts) {
            const double a = 1.0 / (1.0 + z), ref = sync->density_transfer(s, 1e-4, a);
            max_large = std::max(max_large, std::abs(newt_to_sync->density_transfer(s, 1e-4, a) / ref - 1.0));
            max_unconverted = std::max(max_unconverted, std::abs(newt->density_transfer(s, 1e-4, a) / ref - 1.0));
        }
        // Massive neutrinos: CLASS's neutrino fluid approximation is not gauge-independent at high k
        const double tol = s == Species::MassiveNeutrino ? 6e-2 : 2e-3;
        check(max_small < tol,
              std::string("Newtonian -> CDM-synchronous ") + name_of(s) + ", 1e-3 <= k <= 1: max rel. diff " +
                  num(max_small, 2));
        check(max_large < 2e-2,
              std::string("Newtonian -> CDM-synchronous ") + name_of(s) + ", k = 1e-4: max rel. diff " +
                  num(max_large, 2) + " (unconverted: " + num(max_unconverted, 2) + ")");
        check(max_unconverted > 10.0, std::string("the conversion matters on large scales for ") + name_of(s));
    }

    // 2. Growth rates of the converted densities = those of the synchronous run
    {
        double max_large = 0.0, max_small = 0.0;
        for (double z : {49.0, 20.0, 10.0, 2.0, 0.0}) {
            const double a = 1.0 / (1.0 + z);
            const auto f = [&](auto & p, double k) { return p->density_growth_rate(Species::CDMBaryon, k, a); };
            max_large = std::max(max_large, std::abs(f(newt_to_sync, 1e-4) / f(sync, 1e-4) - 1.0));
            for (double k : {1e-3, 1e-2, 0.1})
                max_small = std::max(max_small, std::abs(f(newt_to_sync, k) / f(sync, k) - 1.0));
        }
        check(max_small < 1e-3, "growth rate f_cb of the converted densities, 1e-3 <= k <= 0.1: " + num(max_small, 2));
        check(max_large < 2e-2, "growth rate f_cb of the converted densities, k = 1e-4: " + num(max_large, 2));
    }

    // 3. A GaugeView on the Newtonian provider (the generic conversion) gives the same densities, up to the
    // interpolation error (the provider converts on its (a, k)-grid and then interpolates, the view interpolates and
    // then converts, and the conversion factor depends on a and k)
    {
        GaugeView view(newt, Gauge::SynchronousCDM, newt->background().gauge_background());
        double max_diff = 0.0;
        for (auto s : all_species)
            for (double z : redshifts)
                for (double k : {1e-3, 1e-2, 0.1, 1.0}) {
                    const double a = 1.0 / (1.0 + z);
                    max_diff = std::max(max_diff, std::abs(view.density_transfer(s, k, a) /
                                                               newt_to_sync->density_transfer(s, k, a) -
                                                           1.0));
                }
        check(max_diff < 1e-3, "GaugeView = conversion in the CLASS provider: " +
                                   num(max_diff, 2));
    }

    // 4. CLASS (synchronous) and CAMB agree (densities, P(k), the Newtonian CDM velocity). Not massive neutrinos at
    // k >~ k_fs, where the two codes' neutrino approximations differ by tens of per cent
    for (auto s : {Species::CDM, Species::Baryon, Species::CDMBaryon, Species::TotalMatter}) {
        double max_diff = 0.0;
        for (double z : redshifts)
            for (double k : {1e-3, 1e-2, 0.1, 1.0}) {
                const double a = 1.0 / (1.0 + z);
                max_diff = std::max(max_diff,
                                    std::abs(sync->density_transfer(s, k, a) / camb->density_transfer(s, k, a) - 1.0));
            }
        check(max_diff < 2e-3, std::string("CLASS vs CAMB ") + name_of(s) + ": max rel. diff " + num(max_diff, 2));
    }
    {
        double pk = 0.0, vel = 0.0;
        for (double k : {1e-3, 1e-2, 0.1, 1.0})
            pk = std::max(pk, std::abs(sync->power(Species::CDMBaryon, k, 1.0) /
                                           camb->power(Species::CDMBaryon, k, 1.0) -
                                       1.0));
        for (double z : {49.0, 10.0, 0.0})
            for (double k : {1e-3, 1e-2, 0.1}) {
                const double a = 1.0 / (1.0 + z);
                vel = std::max(vel, std::abs(newt->velocity_divergence_transfer(Species::CDM, k, a) /
                                                 camb->velocity_divergence_transfer(Species::CDM, k, a) -
                                             1.0));
            }
        check(pk < 2e-3, "CLASS vs CAMB P_cb(k, z = 0): " + num(pk, 2));
        // Both providers take the growth rate from a 4-point Lagrange polynomial in log(a) through the tables
        double growth = 0.0;
        for (double z : {49.0, 20.0, 10.0, 2.0, 0.0})
            for (double k : {1e-3, 1e-2, 0.1, 1.0}) {
                const double a = 1.0 / (1.0 + z);
                growth = std::max(growth, std::abs(sync->density_growth_rate(Species::CDMBaryon, k, a) /
                                                       camb->density_growth_rate(Species::CDMBaryon, k, a) -
                                                   1.0));
            }
        check(growth < 2e-3, "CLASS vs CAMB growth rate f_cb: " + num(growth, 2));
        check(vel < 1e-3, "CLASS vs CAMB Newtonian theta_cdm / aH: " + num(vel, 2));
        double nu = 0.0;
        for (double z : redshifts)
            nu = std::max(nu, std::abs(sync->density_transfer(Species::MassiveNeutrino, 1e-3, 1.0 / (1.0 + z)) /
                                           camb->density_transfer(Species::MassiveNeutrino, 1e-3, 1.0 / (1.0 + z)) -
                                       1.0));
        check(nu < 2e-3, "CLASS vs CAMB massive neutrinos at k = 1e-3: " + num(nu, 2));
    }

    // 4b. The CAMB reader: the columns by position (files without the header line: 13 columns, or the first 7 as old
    // CAMB versions and CLASS's format = camb write) give the same as by name, and the files are found next to the
    // info file when its folder does not exist
    {
        namespace fs = std::filesystem;
        const fs::path tmp = fs::temp_directory_path() / ("fml_test_camb_" + std::to_string(FML::ThisTask));
        fs::create_directories(tmp);
        const auto write_copy = [&](const std::string & label, int ncol) {
            std::ifstream info(data + "/camb_transferinfo.txt");
            std::string folder;
            int n;
            info >> folder >> n;
            const std::string infofile = (tmp / ("info_" + label + ".txt")).string();
            std::ofstream out(infofile);
            out << "/no/such/folder " << n << "\n";
            for (int i = 0; i < n; i++) {
                std::string name, z;
                info >> name >> z;
                std::ifstream in(data + "/" + name);
                std::ofstream copy(tmp / (label + "_" + name));
                for (std::string line; std::getline(in, line);) {
                    if (line.empty() or line[0] == '#')
                        continue;
                    std::istringstream ss(line);
                    std::string token;
                    for (int c = 0; c < ncol and ss >> token; c++)
                        copy << token << (c + 1 < ncol ? " " : "\n");
                }
                out << label + "_" + name << " " << z << "\n";
            }
            out.close();
            auto s = cmb;
            s.camb_infofile = infofile;
            return make_transfer_provider(s);
        };
        auto positional = write_copy("p13", 13);
        auto old_format = write_copy("p7", 7);
        double diff = 0.0;
        for (auto s : all_species)
            for (double z : redshifts)
                for (double k : {1e-3, 0.1, 1.0}) {
                    const double a = 1.0 / (1.0 + z), ref = camb->density_transfer(s, k, a);
                    diff = std::max(diff, std::abs(positional->density_transfer(s, k, a) / ref - 1.0));
                    diff = std::max(diff, std::abs(old_format->density_transfer(s, k, a) / ref - 1.0));
                }
        check(diff < 1e-12, "CAMB columns by position (13 and 7 columns) = by name: " + num(diff, 2));
        check(positional->has_velocity(Species::CDM) and not old_format->has_velocity(Species::CDM),
              "CAMB velocities only with the velocity columns");
        fs::remove_all(tmp);
    }

    // 5. The primordial normalization
    {
        const double before = sync->power(Species::CDMBaryon, 0.1, 1.0);
        sync->rescale_primordial_power(2.0);
        check(std::abs(sync->power(Species::CDMBaryon, 0.1, 1.0) / before - 2.0) < 1e-12,
              "rescale_primordial_power scales P(k)");
        sync->rescale_primordial_power(0.5);
    }

    // 6. Wrong declarations and unsupported conversions are errors
    check(throws([&] { make_transfer_provider(class_settings("newtonian", Gauge::SynchronousCDM, Gauge::SynchronousCDM)); }),
          "a Newtonian run declared synchronous is an error");
    check(throws([&] { make_transfer_provider(class_settings("synchronous", Gauge::Newtonian, Gauge::Newtonian)); }),
          "a synchronous run declared Newtonian is an error");
    check(throws([&] { make_transfer_provider(class_settings("synchronous", Gauge::SynchronousCDM, Gauge::Newtonian)); }),
          "synchronous -> Newtonian is an error");
    check(throws([&] {
              auto s = cmb;
              s.input_gauge = Gauge::Newtonian;
              make_transfer_provider(s);
          }),
          "CAMB input declared Newtonian is an error");
    check(throws([&] {
              auto s = cmb;
              s.source = "cmbfast";
              make_transfer_provider(s);
          }),
          "an unknown source is an error");
#ifndef USE_INTERNAL_COSMOLOGY
    check(throws([&] {
              auto s = cmb;
              s.source = "internal";
              make_transfer_provider(s);
          }),
          "source = internal without USE_INTERNAL_COSMOLOGY is an error");
#endif
    check(throws([&] { sync->density_transfer(Species::CDM, 1e-3, 1.0 / 100.0); }),
          "a query outside the range is an error (RangePolicy::Reject)");

#ifdef USE_INTERNAL_COSMOLOGY
    // 7.-8. FML's Einstein-Boltzmann solver (source = internal) against CLASS for the same cosmology, without (7) and
    // with (8) a massive neutrino. The solver runs once per cosmology: the Newtonian view shares the solved state
    {
        FML::COSMOLOGY::Utils::verbose = false;
        const auto max_rel = [&](auto & p, auto & q, Species s, auto && ks, auto && quantity) {
            double m = 0.0;
            for (double z : {49.0, 10.0, 2.0, 0.0})
                for (double k : ks) {
                    const double a = 1.0 / (1.0 + z);
                    m = std::max(m, std::abs(quantity(p, s, k, a) / quantity(q, s, k, a) - 1.0));
                }
            return m;
        };
        const auto density = [](auto & p, Species s, double k, double a) { return p->density_transfer(s, k, a); };
        const auto growth = [](auto & p, Species s, double k, double a) { return p->density_growth_rate(s, k, a); };
        const auto velocity = [](auto & p, Species s, double k, double a) {
            return p->velocity_divergence_transfer(s, k, a);
        };
        const std::vector<double> ks{1e-3, 1e-2, 0.1, 1.0}, ks_large{1e-4, 1e-3}, ks_nu{1e-3, 1e-2}, ks_nu_vel{1e-3, 3e-3};

        // The internal provider in both gauges, solved once
        const auto make_internal = [&](double OmegaMNu) {
            TransferSourceSettings in;
            in.source = "internal";
            in.h = 0.67;
            in.Omegab = 0.0224 / (0.67 * 0.67);
            in.OmegaCDM = 0.119 / (0.67 * 0.67);
            in.OmegaMNu = OmegaMNu;
            in.A_s = cls.A_s;
            in.n_s = cls.n_s;
            in.range_policy = RangePolicy::Reject;
            in.internal_settings.cosmo.Yp = 0.245;
            auto sync = make_transfer_provider(in);
            auto newt = std::make_shared<InternalTransferProvider>(
                std::dynamic_pointer_cast<InternalTransferProvider>(sync)->with_density_gauge(Gauge::Newtonian));
            return std::make_pair(sync, std::shared_ptr<TransferProvider>(newt));
        };
        const auto class_run = [&](const std::string & root, Gauge gauge) {
            auto s = class_settings("synchronous", gauge, gauge);
            s.class_root = data + "/" + root;
            return make_transfer_provider(s);
        };
        const auto compare = [&](const std::string & label, auto & internal_sync, auto & internal_newt, auto & class_sync,
                                 auto & class_newt, const std::vector<Species> & species) {
            check(internal_sync->density_gauge() == Gauge::SynchronousCDM and
                      internal_newt->density_gauge() == Gauge::Newtonian,
                  label + ": the requested density gauges");
            for (auto s : species) {
                const double sync_small = max_rel(internal_sync, class_sync, s, ks, density);
                const double sync_large = max_rel(internal_sync, class_sync, s, ks_large, density);
                const double newt_all = max_rel(internal_newt, class_newt, s, ks, density);
                std::cout << "# " << label << " vs CLASS " << name_of(s) << ": synchronous " << num(sync_small, 2)
                          << " (k = 1e-4, 1e-3: " << num(sync_large, 2) << "), Newtonian " << num(newt_all, 2) << "\n";
                check(sync_small < 5e-3, label + " vs CLASS, CDM-synchronous " + name_of(s) + ", 1e-3 <= k <= 1: " +
                                             num(sync_small, 2));
                check(sync_large < 5e-3, label + " vs CLASS, CDM-synchronous " + name_of(s) + ", k = 1e-4, 1e-3: " +
                                             num(sync_large, 2));
                check(newt_all < 5e-3,
                      label + " vs CLASS, Newtonian " + name_of(s) + ", 1e-3 <= k <= 1: " + num(newt_all, 2));
            }
            const double f_cb = max_rel(internal_sync, class_sync, Species::CDMBaryon, ks, growth);
            const double v_cdm = max_rel(internal_newt, class_newt, Species::CDM, ks, velocity);
            std::cout << "# " << label << " vs CLASS: f_cb " << num(f_cb, 2) << ", Newtonian theta_cdm " << num(v_cdm, 2)
                      << "\n";
            check(f_cb < 1e-3, label + " vs CLASS growth rate f_cb: " + num(f_cb, 2));
            check(v_cdm < 5e-3, label + " vs CLASS Newtonian theta_cdm / aH: " + num(v_cdm, 2));
        };

        // 7. Massless neutrinos
        {
            auto [internal_sync, internal_newt] = make_internal(0.0);
            auto class_sync = class_run("class_massless_synchronous_00_", Gauge::SynchronousCDM);
            auto class_newt = class_run("class_massless_newtonian_00_", Gauge::Newtonian);
            compare("internal", internal_sync, internal_newt, class_sync, class_newt,
                    {Species::CDM, Species::Baryon, Species::CDMBaryon, Species::TotalMatter});
            check(not internal_sync->has_density(Species::MassiveNeutrino),
                  "internal without massive neutrinos has no MassiveNeutrino");

            // The solver's own CDM-synchronous view = a GaugeView on its Newtonian view (the background from CLASS)
            GaugeView view(internal_newt, Gauge::SynchronousCDM,
                           std::dynamic_pointer_cast<CLASSTransferProvider>(class_sync)->background().gauge_background());
            double conversion = 0.0;
            for (auto s : {Species::CDM, Species::Baryon, Species::CDMBaryon})
                for (double z : {49.0, 10.0, 0.0})
                    for (double k : {1e-4, 1e-3, 1e-2, 0.1, 1.0}) {
                        const double a = 1.0 / (1.0 + z);
                        conversion = std::max(conversion, std::abs(view.density_transfer(s, k, a) /
                                                                       internal_sync->density_transfer(s, k, a) -
                                                                   1.0));
                    }
            check(conversion < 1e-3, "internal: GaugeView(Newtonian view) = the solver's own CDM-synchronous view: " +
                                         num(conversion, 2));
        }

        // 8. One massive neutrino of 0.1 eV (the CLASS runs above, with CLASS's default neutrino fluid approximation,
        // which the solver also uses). OmegaMNu from the CLASS background
        {
            const auto & bg = std::dynamic_pointer_cast<const CLASSTransferProvider>(sync)->background();
            const double H0_mpc = bg.h() / FML::UTILS::SI::c_over_H0_mpch, OmegaMNu = bg.rho_ncdm(0, 1.0) / (H0_mpc * H0_mpc);
            auto [internal_sync, internal_newt] = make_internal(OmegaMNu);
            auto class_newt = class_run("class_newtonian_00_", Gauge::Newtonian);
            compare("internal (m_nu = 0.1 eV)", internal_sync, internal_newt, sync, class_newt,
                    {Species::CDM, Species::Baryon, Species::CDMBaryon, Species::TotalMatter});

            // The neutrinos themselves on large scales, and their velocity. Around the switch to the neutrino fluid
            // approximation (k tau = 31) the two codes' theta_nu differ (by 6e-3 at k = 0.01, z = 49 and 40% at
            // k = 0.03, z = 49; 1e-3 or better at z <= 2), so the velocity is checked at k <= 3e-3
            const double nu_sync = max_rel(internal_sync, sync, Species::MassiveNeutrino, ks_nu, density);
            const double nu_newt = max_rel(internal_newt, class_newt, Species::MassiveNeutrino, ks_nu, density);
            const double nu_vel = max_rel(internal_newt, class_newt, Species::MassiveNeutrino, ks_nu_vel, velocity);
            const double tot_vel = max_rel(internal_newt, class_newt, Species::TotalMatter, ks, velocity);
            std::cout << "# internal (m_nu = 0.1 eV) vs CLASS massive neutrinos, k <= 0.01: synchronous "
                      << num(nu_sync, 2) << ", Newtonian " << num(nu_newt, 2) << ", velocity (k <= 3e-3) " << num(nu_vel, 2)
                      << "; total matter velocity " << num(tot_vel, 2) << "\n";
            for (double k : {0.1, 1.0})
                std::cout << "#   k = " << k << ": delta_nu internal / CLASS - 1 at z = 0: "
                          << num(internal_sync->density_transfer(Species::MassiveNeutrino, k, 1.0) /
                                         sync->density_transfer(Species::MassiveNeutrino, k, 1.0) -
                                     1.0,
                                 2)
                          << "\n";
            check(nu_sync < 5e-3, "internal vs CLASS, CDM-synchronous massive neutrinos, k <= 0.01: " + num(nu_sync, 2));
            check(nu_newt < 5e-3, "internal vs CLASS, Newtonian massive neutrinos, k <= 0.01: " + num(nu_newt, 2));
            check(nu_vel < 5e-3, "internal vs CLASS, massive neutrino theta / aH, k <= 3e-3: " + num(nu_vel, 2));
            check(tot_vel < 5e-3, "internal vs CLASS, total matter theta / aH, 1e-3 <= k <= 1: " + num(tot_vel, 2));
        }

        check(throws([&] {
                  TransferSourceSettings s;
                  s.source = "internal";
                  s.h = 0.67;
                  s.Omegab = 0.05;
                  s.OmegaCDM = 0.26;
                  s.A_s = 2e-9;
                  s.OmegaMNu = -0.001;
                  make_transfer_provider(s);
              }),
              "internal with OmegaMNu < 0 is an error");
    }
#endif

    return FML::TESTS::summary("test_transfer_providers");
}

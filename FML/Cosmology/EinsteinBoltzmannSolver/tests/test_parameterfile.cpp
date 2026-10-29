// =============================================================================
// Tests for the parameter file reader/writer (ParameterFile.h). Run with `make test`.
// =============================================================================

#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cstdio>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace FML::COSMOLOGY;

using Test::check;
using Test::check_rel;

namespace {
    std::string slurp(const std::string & f) {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    bool throws(const std::function<void()> & f) {
        try {
            f();
        } catch (const std::exception &) {
            return true;
        }
        return false;
    }
} // namespace

int main() {
    const std::string dir = std::filesystem::temp_directory_path().string() + "/test_parameterfile";
    std::filesystem::create_directories(dir);
    using Constants::Mpc;

    std::cout << "Defaults and round trip\n";
    {
        const RunSettings d = ParameterFile::read("");
        check_rel("defaults = fiducial cosmology (h)", d.cosmo.h, CosmologyParameters::fiducial().h, 0.0);
        check_rel("defaults = PerturbationOptions defaults (k_max)", d.pert.k_max, PerturbationOptions().k_max, 0.0);
        ParameterFile::write(d, dir + "/a.ini");
        const RunSettings r = ParameterFile::read(dir + "/a.ini");
        ParameterFile::write(r, dir + "/b.ini");
        check("write -> read -> write is identical", slurp(dir + "/a.ini") == slurp(dir + "/b.ini"));
        check("every parameter appears in the written file", [&] {
            const std::string txt = slurp(dir + "/a.ini");
            for (const auto & n : ParameterFile::names())
                if (txt.find("\n" + n + " = ") == std::string::npos)
                    return false;
            return true;
        }());
    }

    std::cout << "Reading values\n";
    {
        std::ofstream(dir + "/c.ini") << "# comment line\n"
                                         "h = 0.7        # trailing comment\n"
                                         "\n"
                                         "pert_k_max = 1.0\n"
                                         "lensing = yes\n"
                                         "method_full = rk8pd\n"
                                         "transfer_ells = 10, 20,30\n"
                                         "angular_z = 0.5, 1\n"
                                         "ells =\n"
                                         "output_dir = out/run 1\n";
        const RunSettings s = ParameterFile::read(dir + "/c.ini", {"m_nu=0.1", "h=0.68"});
        check_rel("double with trailing comment, overridden on the command line", s.cosmo.h, 0.68, 0.0);
        check_rel("units: pert_k_max in 1/Mpc", s.pert.k_max * Mpc, 1.0, 1e-15);
        check("boolean 'yes'", s.ps.lensing);
        check("ODE method by name", s.pert.method_full == gsl_odeiv2_step_rk8pd);
        check("integer list", s.transfer_ells.size() == 3 && s.transfer_ells[2] == 30);
        check("double list", s.angular_z.size() == 2 && s.angular_z[1] == 1.0);
        check("empty list", s.ps.ells.empty());
        check("string with a space", s.output_dir == "out/run 1");
        check_rel("command-line override of an unset parameter", s.cosmo.m_nu, 0.1, 0.0);
    }

    std::cout << "Presets\n";
    {
        std::ofstream(dir + "/d.ini") << "OmegaK = 0.01\npreset = planck2018\n";
        const RunSettings s = ParameterFile::read(dir + "/d.ini");
        check_rel("preset planck2018 (applied first)", s.cosmo.h, CosmologyParameters::Planck2018().h, 0.0);
        check_rel("... other parameters still applied", s.cosmo.OmegaK, 0.01, 0.0);
        const RunSettings b = ParameterFile::read("", {"preset=planck2018_baseline"});
        check_rel("preset planck2018_baseline: m_nu = 0.06 eV", b.cosmo.m_nu, 0.06, 0.0);
        std::ofstream(dir + "/f.ini") << "lmax_nu = 20\nprecision = fast\n";
        const RunSettings f = ParameterFile::read(dir + "/f.ini", {"ell_step=20"});
        check("precision = fast sets its parameters",
              f.pert.dk_max > 0.0 && f.pert.n_k == 60 && f.ps.n_k_per_oscillation == 2.0);
        check("... explicit precision parameters override it (also earlier in the file)",
              f.pert.lmax_nu == 20 && f.ps.ell_step == 20);
        check("precision = default changes nothing",
              ParameterFile::read("", {"precision=default"}).pert.n_k == PerturbationOptions().n_k);
        check("unknown precision preset is an error", throws([] { ParameterFile::read("", {"precision=ultra"}); }));
    }

    std::cout << "Errors\n";
    {
        std::ofstream(dir + "/e1.ini") << "hh = 0.7\n";
        check("unknown parameter is an error", throws([&] { ParameterFile::read(dir + "/e1.ini"); }));
        std::ofstream(dir + "/e2.ini") << "h = abc\n";
        check("non-numeric value is an error", throws([&] { ParameterFile::read(dir + "/e2.ini"); }));
        std::ofstream(dir + "/e3.ini") << "pert_n_k = 10.5\n";
        check("non-integer for an integer is an error", throws([&] { ParameterFile::read(dir + "/e3.ini"); }));
        check("z_reion < 0 is an error", throws([&] { ParameterFile::read("", {"z_reion=-1"}); }));
        check("z_He_reion > 50 is an error", throws([&] { ParameterFile::read("", {"z_He_reion=60"}); }));
        check("... but not without reionization",
              !throws([&] { ParameterFile::read("", {"z_reion=-1", "reionization=false"}); }));
        check("alpha_s with an isocurvature mode is an error",
              throws([&] { ParameterFile::read("", {"alpha_s=-0.01", "f_cdi=0.1"}); }));
        check("nan is an error", throws([&] { ParameterFile::read("", {"h=nan"}); }));
        check("inf is an error", throws([&] { ParameterFile::read("", {"A_s=inf"}); }));
        check("1e400 is an error", throws([&] { ParameterFile::read("", {"h=1e400"}); }));
        check("integer beyond the int range is an error", throws([&] { ParameterFile::read("", {"ell_max=1e10"}); }));
        std::ofstream(dir + "/e4.ini") << "lensing = maybe\n";
        check("bad boolean is an error", throws([&] { ParameterFile::read(dir + "/e4.ini"); }));
        std::ofstream(dir + "/e5.ini") << "method_tc = euler\n";
        check("unknown ODE method is an error", throws([&] { ParameterFile::read(dir + "/e5.ini"); }));
        std::ofstream(dir + "/e6.ini") << "h = 0.7\nh = 0.8\n";
        check("parameter set twice in the file is an error", throws([&] { ParameterFile::read(dir + "/e6.ini"); }));
        std::ofstream(dir + "/e7.ini") << "OmegaB = -0.1\n";
        check("unphysical cosmology is an error (validate)", throws([&] { ParameterFile::read(dir + "/e7.ini"); }));
        check("missing file is an error", throws([&] { ParameterFile::read(dir + "/does_not_exist.ini"); }));
    }

    std::cout << "Example parameter files in params/\n";
    for (const auto & f : {"params/default.ini",
                           "params/planck2018.ini",
                           "params/neutrinos.ini",
                           "params/twopoint.ini",
                           "params/curvature.ini",
                           "params/musigma.ini",
                           "params/dgp.ini",
                           "params/quintessence.ini",
                           "params/jbd.ini",
                           "params/fr.ini",
                           "params/coupled_quintessence.ini",
                           "params/dark_scattering.ini",
                           "params/cubic_galileon.ini",
                           "params/horndeski.ini",
                           "params/ede.ini",
                           "params/ide.ini",
                           "params/tensors.ini"}) {
        if (!std::filesystem::exists(f))
            continue; // tests may run from another directory
        check(std::string("reads ") + f, !throws([&] { ParameterFile::read(f); }));
    }
    std::cout << "Model parameters\n";
    {
        const RunSettings m = ParameterFile::read("", {"model=musigma", "mu0=0.2"});
        check_rel("model parameter is read", m.cosmo.model_param("mu0"), 0.2, 0.0);
        Test::check_close("unset model parameter takes the default", m.cosmo.model_param("Sigma0"), 0.0, 0.0);
        check("parameter of another model is an error",
              throws([] { ParameterFile::read("", {"model=musigma", "bd_omega=100"}); }));
        check("model parameter without its model is an error", throws([] { ParameterFile::read("", {"mu0=0.2"}); }));
        ParameterFile::write(m, dir + "/m.ini");
        const RunSettings m2 = ParameterFile::read(dir + "/m.ini");
        check("written file keeps the model and its parameters",
              m2.cosmo.model == "musigma" && m2.cosmo.model_param("mu0") == 0.2);
    }
    std::filesystem::remove_all(dir);
    return Test::summary();
}

// =============================================================================
// Tests for the parameter file reader/writer (ParameterFile.h). Run with `make test`.
// =============================================================================

#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>

#include <cstdio>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using FML::TESTS::throws;

using namespace FML::COSMOLOGY;

using FML::TESTS::check;
using FML::TESTS::check_rel;

namespace {
    std::string slurp(const std::string & f) {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
} // namespace

int main() {
    const std::string dir = std::filesystem::temp_directory_path().string() + "/test_parameterfile";
    std::filesystem::create_directories(dir);
    using Constants::Mpc;

    std::cout << "Defaults and round trip\n";
    {
        const RunSettings d = ParameterFile::read("");
        check_rel(d.cosmo.h, CosmologyParameters::fiducial().h, 0.0, "defaults = fiducial cosmology (h)");
        check_rel(d.pert.k_max, PerturbationOptions().k_max, 0.0, "defaults = PerturbationOptions defaults (k_max)");
        ParameterFile::write(d, dir + "/a.ini");
        const RunSettings r = ParameterFile::read(dir + "/a.ini");
        ParameterFile::write(r, dir + "/b.ini");
        check(slurp(dir + "/a.ini") == slurp(dir + "/b.ini"), "write -> read -> write is identical");
        check(
            [&] {
                const std::string txt = slurp(dir + "/a.ini");
                for (const auto & n : ParameterFile::names())
                    if (txt.find("\n" + n + " = ") == std::string::npos)
                        return false;
                return true;
            }(),
            "every parameter appears in the written file");
    }

    std::cout << "Reading values\n";
    {
        const auto s = ParameterFile::read("", {"nonlinear=react", "react_n_z=8", "react_n_k=128",
                                               "react_mass_loop=20", "react_z_max=2"});
        check(s.ps.nonlinear == PowerSpectrumOptions::Nonlinear::ReACT && s.ps.react_n_z == 8 &&
              s.ps.react_n_k == 128 && s.ps.react_mass_loop == 20 && s.ps.react_z_max == 2,
              "ReACT method and sampling settings parsed");
        ParameterFile::write(s, dir + "/react.ini");
        const auto r = ParameterFile::read(dir + "/react.ini");
        check(r.ps.nonlinear == s.ps.nonlinear && r.ps.react_n_z == s.ps.react_n_z &&
              r.ps.react_n_k == s.ps.react_n_k && r.ps.react_mass_loop == s.ps.react_mass_loop &&
              r.ps.react_z_max == s.ps.react_z_max, "ReACT settings survive a parameter-file round trip");
    }
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
        check_rel(s.cosmo.h, 0.68, 0.0, "double with trailing comment, overridden on the command line");
        check_rel(s.pert.k_max * Mpc, 1.0, 1e-15, "units: pert_k_max in 1/Mpc");
        check(s.ps.lensing, "boolean 'yes'");
        check(s.pert.method_full == gsl_odeiv2_step_rk8pd, "ODE method by name");
        check(s.transfer_ells.size() == 3 && s.transfer_ells[2] == 30, "integer list");
        check(s.angular_z.size() == 2 && s.angular_z[1] == 1.0, "double list");
        check(s.ps.ells.empty(), "empty list");
        check(s.output_dir == "out/run 1", "string with a space");
        check_rel(s.cosmo.m_nu, 0.1, 0.0, "command-line override of an unset parameter");
    }

    std::cout << "Presets\n";
    {
        std::ofstream(dir + "/d.ini") << "OmegaK = 0.01\npreset = planck2018\n";
        const RunSettings s = ParameterFile::read(dir + "/d.ini");
        check_rel(s.cosmo.h, CosmologyParameters::Planck2018().h, 0.0, "preset planck2018 (applied first)");
        check_rel(s.cosmo.OmegaK, 0.01, 0.0, "... other parameters still applied");
        const RunSettings b = ParameterFile::read("", {"preset=planck2018_baseline"});
        check_rel(b.cosmo.m_nu, 0.06, 0.0, "preset planck2018_baseline: m_nu = 0.06 eV");
        std::ofstream(dir + "/f.ini") << "lmax_nu = 20\nprecision = fast\n";
        const RunSettings f = ParameterFile::read(dir + "/f.ini", {"ell_step=20"});
        check(f.pert.dk_max > 0.0 && f.pert.n_k == 60 && f.ps.n_k_per_oscillation == 2.0,
              "precision = fast sets its parameters");
        check(f.pert.lmax_nu == 20 && f.ps.ell_step == 20,
              "... explicit precision parameters override it (also earlier in the file)");
        check(ParameterFile::read("", {"precision=default"}).pert.n_k == PerturbationOptions().n_k,
              "precision = default changes nothing");
        check(throws([] { ParameterFile::read("", {"precision=ultra"}); }), "unknown precision preset is an error");
    }

    std::cout << "Errors\n";
    {
        std::ofstream(dir + "/e1.ini") << "hh = 0.7\n";
        check(throws([&] { ParameterFile::read(dir + "/e1.ini"); }), "unknown parameter is an error");
        std::ofstream(dir + "/e2.ini") << "h = abc\n";
        check(throws([&] { ParameterFile::read(dir + "/e2.ini"); }), "non-numeric value is an error");
        std::ofstream(dir + "/e3.ini") << "pert_n_k = 10.5\n";
        check(throws([&] { ParameterFile::read(dir + "/e3.ini"); }), "non-integer for an integer is an error");
        check(throws([&] { ParameterFile::read("", {"z_reion=-1"}); }), "z_reion < 0 is an error");
        check(throws([&] { ParameterFile::read("", {"z_He_reion=60"}); }), "z_He_reion > 50 is an error");
        check(!throws([&] { ParameterFile::read("", {"z_reion=-1", "reionization=false"}); }),
              "... but not without reionization");
        check(throws([&] { ParameterFile::read("", {"alpha_s=-0.01", "f_cdi=0.1"}); }),
              "alpha_s with an isocurvature mode is an error");
        check(throws([&] { ParameterFile::read("", {"h=nan"}); }), "nan is an error");
        check(throws([&] { ParameterFile::read("", {"A_s=inf"}); }), "inf is an error");
        check(throws([&] { ParameterFile::read("", {"h=1e400"}); }), "1e400 is an error");
        check(throws([&] { ParameterFile::read("", {"ell_max=1e10"}); }), "integer beyond the int range is an error");
        std::ofstream(dir + "/e4.ini") << "lensing = maybe\n";
        check(throws([&] { ParameterFile::read(dir + "/e4.ini"); }), "bad boolean is an error");
        std::ofstream(dir + "/e5.ini") << "method_tc = euler\n";
        check(throws([&] { ParameterFile::read(dir + "/e5.ini"); }), "unknown ODE method is an error");
        std::ofstream(dir + "/e6.ini") << "h = 0.7\nh = 0.8\n";
        check(throws([&] { ParameterFile::read(dir + "/e6.ini"); }), "parameter set twice in the file is an error");
        std::ofstream(dir + "/e7.ini") << "OmegaB = -0.1\n";
        check(throws([&] { ParameterFile::read(dir + "/e7.ini"); }), "unphysical cosmology is an error (validate)");
        check(throws([&] { ParameterFile::read(dir + "/does_not_exist.ini"); }), "missing file is an error");
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
        check(!throws([&] { ParameterFile::read(f); }), std::string("reads ") + f);
    }
    std::cout << "Model parameters\n";
    {
        const RunSettings m = ParameterFile::read("", {"model=musigma", "mu0=0.2"});
        check_rel(m.cosmo.model_param("mu0"), 0.2, 0.0, "model parameter is read");
        FML::TESTS::check_close(m.cosmo.model_param("Sigma0"), 0.0, 0.0, "unset model parameter takes the default");
        check(throws([] { ParameterFile::read("", {"model=musigma", "bd_omega=100"}); }),
              "parameter of another model is an error");
        check(throws([] { ParameterFile::read("", {"mu0=0.2"}); }), "model parameter without its model is an error");
        ParameterFile::write(m, dir + "/m.ini");
        const RunSettings m2 = ParameterFile::read(dir + "/m.ini");
        check(m2.cosmo.model == "musigma" && m2.cosmo.model_param("mu0") == 0.2,
              "written file keeps the model and its parameters");
    }
    std::filesystem::remove_all(dir);
    return FML::TESTS::summary("EBS parameterfile");
}

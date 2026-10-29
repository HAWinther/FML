#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::Mpc;

    namespace {

        // ---------------------------------------------------------------------------
        // Conversions
        // ---------------------------------------------------------------------------
        std::string trim(const std::string & s) {
            const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? "" : s.substr(a, b - a + 1);
        }
        std::string lower(std::string s) {
            for (auto & ch : s)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return s;
        }
        // A finite number (stod also accepts nan and inf, and throws out_of_range for 1e400)
        double to_double(const std::string & v) {
            size_t pos = 0;
            double d;
            try {
                d = std::stod(v, &pos);
            } catch (const std::out_of_range &) {
                throw std::invalid_argument("number out of range: '" + v + "'");
            }
            if (trim(v.substr(pos)) != "")
                throw std::invalid_argument("not a number: '" + v + "'");
            if (!std::isfinite(d))
                throw std::invalid_argument("not a finite number: '" + v + "'");
            return d;
        }
        int to_int(const std::string & v) {
            const double d = to_double(v);
            // range check first: the cast of a double outside the int range is undefined
            if (!(std::fabs(d) <= double(std::numeric_limits<int>::max())) || d != std::floor(d))
                throw std::invalid_argument("not an integer: '" + v + "'");
            return static_cast<int>(d);
        }
        bool to_bool(const std::string & v) {
            const std::string l = lower(trim(v));
            if (l == "true" || l == "yes" || l == "on" || l == "1")
                return true;
            if (l == "false" || l == "no" || l == "off" || l == "0")
                return false;
            throw std::invalid_argument("not a boolean: '" + v + "'");
        }
        std::vector<std::string> split(const std::string & v) {
            std::vector<std::string> out;
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ','))
                if (trim(item) != "")
                    out.push_back(trim(item));
            return out;
        }
        std::string fmt(double d) {
            std::ostringstream os;
            os << std::setprecision(10) << d;
            return os.str();
        }

        // GSL steppers by name (implicit ones get a finite-difference jacobian)
        const std::map<std::string, ODEStepper> & methods() {
            static const std::map<std::string, ODEStepper> m = {{"rk2", gsl_odeiv2_step_rk2},
                                                                {"rk4", gsl_odeiv2_step_rk4},
                                                                {"rkf45", gsl_odeiv2_step_rkf45},
                                                                {"rkck", gsl_odeiv2_step_rkck},
                                                                {"rk8pd", gsl_odeiv2_step_rk8pd},
                                                                {"rk1imp", gsl_odeiv2_step_rk1imp},
                                                                {"rk2imp", gsl_odeiv2_step_rk2imp},
                                                                {"rk4imp", gsl_odeiv2_step_rk4imp},
                                                                {"bsimp", gsl_odeiv2_step_bsimp},
                                                                {"msadams", gsl_odeiv2_step_msadams},
                                                                {"msbdf", gsl_odeiv2_step_msbdf}};
            return m;
        }
        const std::map<std::string, PowerSpectrumOptions::Nonlinear> & nonlinear_methods() {
            using N = PowerSpectrumOptions::Nonlinear;
            static const std::map<std::string, N> m = {{"none", N::None}, {"halofit", N::Halofit}};
            return m;
        }
        const std::map<std::string, Perturbations::SourceTerm> & source_terms() {
            using S = Perturbations::SourceTerm;
            static const std::map<std::string, S> m = {{"all", S::All},
                                                       {"sw", S::SW},
                                                       {"isw", S::ISW},
                                                       {"doppler", S::Doppler},
                                                       {"quadrupole", S::Quadrupole}};
            return m;
        }
        template <class E>
        std::string enum_name(const std::map<std::string, E> & m, E e) {
            for (const auto & [k, v] : m)
                if (v == e)
                    return k;
            return "?";
        }
        template <class E>
        E enum_value(const std::map<std::string, E> & m, const std::string & v) {
            const auto it = m.find(lower(trim(v)));
            if (it == m.end()) {
                std::string opts;
                for (const auto & kv : m)
                    opts += (opts.empty() ? "" : ", ") + kv.first;
                throw std::invalid_argument("'" + v + "' is not one of: " + opts);
            }
            return it->second;
        }

        // ---------------------------------------------------------------------------
        // Parameter table
        // ---------------------------------------------------------------------------
        struct Param {
            std::string section, name, description;
            std::function<void(RunSettings &, const std::string &)> set;
            std::function<std::string(const RunSettings &)> get;
        };
        using S = RunSettings;
        template <class T>
        using Ref = T & (*)(S &);

        // value in the file = stored value / unit
        Param D(const char * sec, const char * name, const char * desc, Ref<double> r, double unit = 1.0) {
            return {sec,
                    name,
                    desc,
                    [r, unit](S & s, const std::string & v) { r(s) = to_double(v) * unit; },
                    [r, unit](const S & s) { return fmt(r(const_cast<S &>(s)) / unit); }};
        }
        Param I(const char * sec, const char * name, const char * desc, Ref<int> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) { r(s) = to_int(v); },
                    [r](const S & s) { return std::to_string(r(const_cast<S &>(s))); }};
        }
        Param B(const char * sec, const char * name, const char * desc, Ref<bool> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) { r(s) = to_bool(v); },
                    [r](const S & s) { return std::string(r(const_cast<S &>(s)) ? "true" : "false"); }};
        }
        Param Str(const char * sec, const char * name, const char * desc, Ref<std::string> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) { r(s) = trim(v); },
                    [r](const S & s) { return r(const_cast<S &>(s)); }};
        }
        Param Meth(const char * sec, const char * name, const char * desc, Ref<ODEStepper> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) { r(s) = enum_value(methods(), v); },
                    [r](const S & s) { return enum_name(methods(), r(const_cast<S &>(s))); }};
        }
        Param DL(const char * sec, const char * name, const char * desc, Ref<std::vector<double>> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) {
                        r(s).clear();
                        for (const auto & t : split(v))
                            r(s).push_back(to_double(t));
                    },
                    [r](const S & s) {
                        std::string o;
                        for (double d : r(const_cast<S &>(s)))
                            o += (o.empty() ? "" : ", ") + fmt(d);
                        return o;
                    }};
        }
        Param IL(const char * sec, const char * name, const char * desc, Ref<std::vector<int>> r) {
            return {sec,
                    name,
                    desc,
                    [r](S & s, const std::string & v) {
                        r(s).clear();
                        for (const auto & t : split(v))
                            r(s).push_back(to_int(t));
                    },
                    [r](const S & s) {
                        std::string o;
                        for (int i : r(const_cast<S &>(s)))
                            o += (o.empty() ? "" : ", ") + std::to_string(i);
                        return o;
                    }};
        }

        std::string model_description() {
            std::string d = "model (Model.h):";
            for (const auto & m : Models::registry())
                d += " " + m.name + " = " + m.description + ";";
            return d + " its parameters follow";
        }

#define REF(T, expr) [](S & s) -> T & { return s.expr; }

        const std::vector<Param> & table() {
            static const std::vector<Param> t = {
                // --- Cosmology ---------------------------------------------------------
                D("Cosmology", "h", "H0 / (100 km/s/Mpc)", REF(double, cosmo.h)),
                D("Cosmology", "OmegaB", "baryon density today", REF(double, cosmo.OmegaB)),
                D("Cosmology", "OmegaCDM", "cold dark matter density today", REF(double, cosmo.OmegaCDM)),
                D("Cosmology",
                  "OmegaK",
                  "curvature today (0 = flat; > 0 open, < 0 closed). Omega_Lambda follows from closure",
                  REF(double, cosmo.OmegaK)),
                D("Cosmology", "TCMB", "CMB temperature today [K]", REF(double, cosmo.TCMB)),
                D("Cosmology",
                  "Neff",
                  "effective number of neutrinos, massless + massive (0 = no neutrinos)",
                  REF(double, cosmo.Neff)),
                D("Cosmology",
                  "m_nu",
                  "mass of the single massive neutrino species [eV] (0 = all massless)",
                  REF(double, cosmo.m_nu)),
                D("Cosmology",
                  "T_ncdm_over_TCMB",
                  "temperature of the massive species / TCMB (CLASS default)",
                  REF(double, cosmo.T_ncdm_over_TCMB)),
                {"Cosmology",
                 "model",
                 model_description(),
                 [](S & s, const std::string & v) { s.cosmo.model = trim(v); },
                 [](const S & s) { return s.cosmo.model; }},
                D("Cosmology", "A_s", "primordial curvature power at k_pivot", REF(double, cosmo.A_s)),
                D("Cosmology", "n_s", "scalar spectral index", REF(double, cosmo.n_s)),
                D("Cosmology", "k_pivot", "pivot scale [1/Mpc]", REF(double, cosmo.k_pivot)),
                D("Cosmology",
                  "f_cdi",
                  "CDM isocurvature: amplitude ratio, P = f^2 A_s (k/k_pivot)^(n - 1) (0 = off)",
                  REF(double, cosmo.iso_f[0])),
                D("Cosmology", "n_cdi", "CDM isocurvature: spectral index", REF(double, cosmo.iso_n[0])),
                D("Cosmology",
                  "c_ad_cdi",
                  "CDM isocurvature: correlation with the adiabatic mode (CLASS convention)",
                  REF(double, cosmo.iso_c[0])),
                D("Cosmology",
                  "f_bi",
                  "baryon isocurvature: amplitude ratio, P = f^2 A_s (k/k_pivot)^(n - 1) (0 = off)",
                  REF(double, cosmo.iso_f[1])),
                D("Cosmology", "n_bi", "baryon isocurvature: spectral index", REF(double, cosmo.iso_n[1])),
                D("Cosmology",
                  "c_ad_bi",
                  "baryon isocurvature: correlation with the adiabatic mode (CLASS convention)",
                  REF(double, cosmo.iso_c[1])),
                D("Cosmology",
                  "f_nid",
                  "neutrino density isocurvature: amplitude ratio, P = f^2 A_s (k/k_pivot)^(n - 1) (0 = off)",
                  REF(double, cosmo.iso_f[2])),
                D("Cosmology", "n_nid", "neutrino density isocurvature: spectral index", REF(double, cosmo.iso_n[2])),
                D("Cosmology",
                  "c_ad_nid",
                  "neutrino density isocurvature: correlation with the adiabatic mode (CLASS convention)",
                  REF(double, cosmo.iso_c[2])),
                D("Cosmology",
                  "f_niv",
                  "neutrino velocity isocurvature: amplitude ratio, P = f^2 A_s (k/k_pivot)^(n - 1) (0 = off)",
                  REF(double, cosmo.iso_f[3])),
                D("Cosmology", "n_niv", "neutrino velocity isocurvature: spectral index", REF(double, cosmo.iso_n[3])),
                D("Cosmology",
                  "c_ad_niv",
                  "neutrino velocity isocurvature: correlation with the adiabatic mode (CLASS convention)",
                  REF(double, cosmo.iso_c[3])),
                D("Cosmology",
                  "r",
                  "tensor-to-scalar ratio at k_pivot (> 0: tensor modes, flat universe only)",
                  REF(double, cosmo.r)),
                B("Cosmology",
                  "n_t_consistency",
                  "tensor tilt from the consistency relation n_t = -r/8 (2 - r/8 - n_s)",
                  REF(bool, cosmo.n_t_consistency)),
                D("Cosmology", "n_t", "tensor tilt (used if n_t_consistency = false)", REF(double, cosmo.n_t)),
                // --- Recombination and reionization -------------------------------------
                D("Recombination", "Yp", "primordial helium mass fraction", REF(double, cosmo.Yp)),
                B("Recombination",
                  "Yp_bbn",
                  "Yp from standard BBN for omega_b and Neff (PArthENoPE table, as CLASS YHe = BBN)",
                  REF(bool, cosmo.Yp_bbn)),
                B("Recombination",
                  "reionization",
                  "tanh reionization (CAMB/CLASS reio_camb)",
                  REF(bool, cosmo.reionization)),
                D("Recombination",
                  "z_reion",
                  "hydrogen (+ first helium) reionization redshift",
                  REF(double, cosmo.z_reion)),
                D("Recombination", "delta_z_reion", "width of hydrogen reionization", REF(double, cosmo.delta_z_reion)),
                D("Recombination", "z_He_reion", "second helium reionization redshift", REF(double, cosmo.z_He_reion)),
                D("Recombination",
                  "delta_z_He_reion",
                  "width of second helium reionization",
                  REF(double, cosmo.delta_z_He_reion)),
                {"Recombination",
                 "recombination",
                 "recombination model: recfast (RECFAST 1.5, H + He) or peebles (Peebles + Saha)",
                 [](S & s, const std::string & v) {
                     const auto t = trim(v);
                     if (t == "recfast")
                         s.rec.model = RecombinationOptions::Model::Recfast;
                     else if (t == "peebles")
                         s.rec.model = RecombinationOptions::Model::Peebles;
                     else
                         throw std::invalid_argument("recombination must be recfast or peebles, not '" + t + "'");
                 },
                 [](const S & s) {
                     return std::string(s.rec.model == RecombinationOptions::Model::Recfast ? "recfast" : "peebles");
                 }},
                B("Recombination", "rec_helium", "helium ionization", REF(bool, rec.helium)),
                D("Recombination",
                  "rec_recfast_fudge_H",
                  "RECFAST: hydrogen fudge factor",
                  REF(double, rec.recfast_fudge_H)),
                D("Recombination",
                  "rec_recfast_fudge_He",
                  "RECFAST: helium fudge factor",
                  REF(double, rec.recfast_fudge_He)),
                D("Recombination",
                  "rec_peebles_fudge",
                  "Peebles: factor on the case-B recombination rate",
                  REF(double, rec.peebles_fudge)),
                D("Recombination",
                  "rec_Xe_saha_limit",
                  "Peebles: switch from Saha to Peebles when x_H drops below this",
                  REF(double, rec.Xe_saha_limit)),
                B("Recombination",
                  "rec_evolve_Tb",
                  "evolve the baryon temperature (else T_b = T_gamma)",
                  REF(bool, rec.evolve_Tb)),
                D("Recombination",
                  "rec_x_start",
                  "start of the recombination grid (x = ln a)",
                  REF(double, rec.x_start)),
                I("Recombination", "rec_n_points", "points in the recombination grid", REF(int, rec.n_points)),
                // --- Background --------------------------------------------------------
                D("Background", "bg_x_start", "start of the background ODE", REF(double, bg_x_start)),
                D("Background", "bg_x_end", "end of the background ODE", REF(double, bg_x_end)),
                I("Background", "bg_n_points", "points of the eta(x), t(x) splines", REF(int, bg_n_points)),
                // --- Perturbations -----------------------------------------------------
                B("Perturbations", "polarization", "photon polarization hierarchy", REF(bool, pert.polarization)),
                B("Perturbations",
                  "neutrinos",
                  "massless neutrino hierarchy (only active if Neff > 0)",
                  REF(bool, pert.neutrinos)),
                D("Perturbations",
                  "pert_k_min",
                  "smallest k (q in a curved universe) [1/Mpc]",
                  REF(double, pert.k_min),
                  1.0 / Mpc),
                D("Perturbations", "pert_k_max", "largest k [1/Mpc]", REF(double, pert.k_max), 1.0 / Mpc),
                I("Perturbations", "pert_n_k", "number of k-modes (log spaced)", REF(int, pert.n_k)),
                D("Perturbations",
                  "pert_dk_max",
                  "if > 0: log-linear k-grid, largest spacing [1/Mpc] (n_k sets the log spacing)",
                  REF(double, pert.dk_max),
                  1.0 / Mpc),
                D("Perturbations",
                  "pert_x_start",
                  "initial time of the perturbations (x = ln a)",
                  REF(double, pert.x_start)),
                I("Perturbations", "pert_n_x_early", "stored x-points in [x_start, -8]", REF(int, pert.n_x_early)),
                I("Perturbations", "pert_n_x_recomb", "stored x-points in [-8, -6]", REF(int, pert.n_x_recomb)),
                I("Perturbations", "pert_n_x_late", "stored x-points in [-6, 0]", REF(int, pert.n_x_late)),
                I("Perturbations",
                  "lmax",
                  "highest photon multipole (temperature and polarization)",
                  REF(int, pert.lmax)),
                I("Perturbations", "lmax_nu", "highest massless-neutrino multipole", REF(int, pert.lmax_nu)),
                I("Perturbations", "lmax_ncdm", "highest massive-neutrino multipole", REF(int, pert.lmax_ncdm)),
                I("Perturbations", "n_q_ncdm", "momentum nodes for the massive neutrinos", REF(int, pert.n_q_ncdm)),
                D("Perturbations",
                  "ncdm_fluid_trigger",
                  "k eta above which massive neutrinos are a fluid (1e30 = never)",
                  REF(double, pert.ncdm_fluid_trigger)),
                D("Perturbations", "tc_tau_min", "end tight coupling when |tau'| < this", REF(double, pert.tc_tau_min)),
                D("Perturbations",
                  "tc_k_over_tau",
                  "... or when ck/(Hp |tau'|) > this",
                  REF(double, pert.tc_k_over_tau)),
                D("Perturbations", "tc_z_max", "... or at the latest at this redshift", REF(double, pert.tc_z_max)),
                D("Perturbations", "ode_abserr", "absolute ODE tolerance", REF(double, pert.ode_abserr)),
                D("Perturbations", "ode_relerr", "relative ODE tolerance", REF(double, pert.ode_relerr)),
                Meth("Perturbations",
                     "method_tc",
                     "ODE method in tight coupling (rk8pd, rkf45, rkck, msbdf, bsimp, ...)",
                     REF(ODEStepper, pert.method_tc)),
                Meth("Perturbations",
                     "method_full",
                     "ODE method after tight coupling (stiff: msbdf)",
                     REF(ODEStepper, pert.method_full)),
                Meth("Perturbations",
                     "method_ncdm",
                     "ODE method with the massive-neutrino hierarchy",
                     REF(ODEStepper, pert.method_ncdm)),
                B("Perturbations",
                  "store_source_components",
                  "also store the SW, ISW, Doppler, quadrupole terms of S_T",
                  REF(bool, pert.store_source_components)),
                // --- Tensors (r > 0) ----------------------------------------------------
                D("Tensors",
                  "tensor_k_max",
                  "largest k of the tensor modes [1/Mpc]",
                  REF(double, tensor.k_max),
                  1.0 / Mpc),
                D("Tensors",
                  "tensor_dk_max",
                  "largest k-spacing of the tensor modes above k_fine [1/Mpc]",
                  REF(double, tensor.dk_max),
                  1.0 / Mpc),
                D("Tensors",
                  "tensor_k_fine",
                  "below this k the tensor mode spacing resolves 2 pi/eta0 [1/Mpc]",
                  REF(double, tensor.k_fine),
                  1.0 / Mpc),
                D("Tensors",
                  "tensor_n_k_per_oscillation",
                  "tensor modes per 2 pi/eta0 below k_fine",
                  REF(double, tensor.n_k_per_oscillation)),
                I("Tensors",
                  "tensor_lmax",
                  "highest photon multipole of the tensor hierarchies",
                  REF(int, tensor.lmax)),
                I("Tensors",
                  "tensor_lmax_nu",
                  "highest neutrino multipole of the tensor hierarchy",
                  REF(int, tensor.lmax_nu)),
                B("Tensors",
                  "tensor_neutrinos",
                  "neutrino anisotropic stress in the tensor equation",
                  REF(bool, tensor.neutrinos)),
                D("Tensors",
                  "tensor_ode_relerr",
                  "relative ODE tolerance of the tensor modes",
                  REF(double, tensor.ode_relerr)),
                D("Tensors",
                  "tensor_ode_abserr",
                  "absolute ODE tolerance of the tensor modes",
                  REF(double, tensor.ode_abserr)),
                I("Tensors", "tensor_ell_max", "tensor C_ell up to this ell (zero above)", REF(int, tensor.ell_max)),
                // --- Power spectra -----------------------------------------------------
                I("PowerSpectrum", "ell_max", "highest multipole of the CMB spectra", REF(int, ps.ell_max)),
                IL("PowerSpectrum",
                   "ells",
                   "multipoles where C_ell is computed (empty = default list up to ell_max)",
                   REF(std::vector<int>, ps.ells)),
                I("PowerSpectrum",
                  "ell_step",
                  "spacing of the default ell list above ell = 100",
                  REF(int, ps.ell_step)),
                B("PowerSpectrum", "lensing", "lensing potential and lensed TT, EE, TE, BB", REF(bool, ps.lensing)),
                I("PowerSpectrum",
                  "lensing_delta_ell",
                  "extra multipoles for lensing (-1 = max(500, ell_max/3))",
                  REF(int, ps.lensing_delta_ell)),
                I("PowerSpectrum",
                  "lensing_limber_ell",
                  "C_ell^phiphi from Limber above this ell",
                  REF(int, ps.lensing_limber_ell)),
                D("PowerSpectrum",
                  "los_k_min",
                  "smallest k of the LOS integration [1/Mpc] (0 = perturbation range)",
                  REF(double, ps.k_min),
                  1.0 / Mpc),
                D("PowerSpectrum",
                  "los_k_max",
                  "largest k of the LOS integration [1/Mpc] (0 = perturbation range)",
                  REF(double, ps.k_max),
                  1.0 / Mpc),
                D("PowerSpectrum",
                  "n_k_per_oscillation",
                  "LOS k-points per 2 pi / eta0",
                  REF(double, ps.n_k_per_oscillation)),
                D("PowerSpectrum", "dlnk_max", "largest LOS spacing in ln k", REF(double, ps.dlnk_max)),
                D("PowerSpectrum", "x_start_los", "start of the LOS integral", REF(double, ps.x_start_los)),
                D("PowerSpectrum",
                  "n_eta_per_oscillation",
                  "LOS x-points per Bessel oscillation at k_max",
                  REF(double, ps.n_eta_per_oscillation)),
                D("PowerSpectrum", "dx_recomb", "LOS x-spacing around recombination", REF(double, ps.dx_recomb)),
                D("PowerSpectrum",
                  "x_recomb_min",
                  "start of the fine recombination window",
                  REF(double, ps.x_recomb_min)),
                D("PowerSpectrum",
                  "x_recomb_max",
                  "end of the fine recombination window",
                  REF(double, ps.x_recomb_max)),
                D("PowerSpectrum",
                  "bessel_points_per_2pi",
                  "points per 2 pi in the Bessel tables",
                  REF(double, ps.bessel_points_per_2pi)),
                B("PowerSpectrum",
                  "use_radial_functions",
                  "use the curved-space radial functions also when flat (test)",
                  REF(bool, ps.use_radial_functions)),
                D("PowerSpectrum",
                  "flat_approximation_nu",
                  "curved: flat approximation of Phi^nu_ell above this nu (1e30 = off)",
                  REF(double, ps.flat_approximation_nu)),
                D("PowerSpectrum",
                  "dx_max_subsampled",
                  "curved: largest x-spacing of the subsampled LOS grid",
                  REF(double, ps.dx_max_subsampled)),
                {"PowerSpectrum",
                 "source_term",
                 "temperature source to integrate: all, sw, isw, doppler, quadrupole "
                 "(anything but all needs store_source_components = true)",
                 [](S & s, const std::string & v) { s.ps.source_term = enum_value(source_terms(), v); },
                 [](const S & s) { return enum_name(source_terms(), s.ps.source_term); }},
                {"PowerSpectrum",
                 "nonlinear",
                 "nonlinear matter power for lensing and P(k): none or halofit (Takahashi 2012)",
                 [](S & s, const std::string & v) { s.ps.nonlinear = enum_value(nonlinear_methods(), v); },
                 [](const S & s) { return enum_name(nonlinear_methods(), s.ps.nonlinear); }},
                // --- Output ------------------------------------------------------------
                Str("Output", "output_dir", "directory for all output files", REF(std::string, output_dir)),
                B("Output", "write_background", "background.txt", REF(bool, write_background)),
                B("Output", "write_recombination", "recombination.txt", REF(bool, write_recombination)),
                DL("Output",
                   "perturbation_k_output",
                   "k [1/Mpc] for perturbations_k<k>.txt (empty = none; closed universe: nearest allowed mode)",
                   REF(std::vector<double>, perturbation_k_output)),
                B("Output",
                  "compute_cells",
                  "LOS integration: cells.txt (and cells_lensed.txt); false = stop after the perturbations",
                  REF(bool, compute_cells)),
                B("Output", "write_matter_power", "matter_power.txt (k, P_m, P_cb)", REF(bool, write_matter_power)),
                DL("Output",
                   "matter_power_z",
                   "redshifts for the matter power spectrum (z = 0: matter_power.txt, "
                   "else matter_power_z<z>.txt)",
                   REF(std::vector<double>, matter_power_z)),
                IL("Output",
                   "transfer_ells",
                   "ells for transfer.txt (empty = none)",
                   REF(std::vector<int>, transfer_ells)),
                B("Output",
                  "cmb_correlation",
                  "cmb_correlation_unlensed.txt (and _lensed.txt with lensing): C^TT, xi_+, xi_-, xi_X(theta)",
                  REF(bool, cmb_correlation)),
                D("Output",
                  "cmb_correlation_beam_fwhm",
                  "Gaussian beam FWHM [deg]; > 0 also writes cmb_correlation_*_beam<fwhm>deg.txt",
                  REF(double, cmb_correlation_beam_fwhm)),
                B("Output",
                  "matter_correlation",
                  "matter_correlation.txt: xi(r) and sigma_8 at z = 0",
                  REF(bool, matter_correlation)),
                D("Output",
                  "matter_correlation_smoothing",
                  "Gaussian smoothing of xi(r) [Mpc/h]",
                  REF(double, matter_correlation_smoothing)),
                DL("Output",
                   "angular_z",
                   "centres of Gaussian n(z) for angular_cl_z<z>.txt and angular_w_z<z>.txt (empty = none)",
                   REF(std::vector<double>, angular_z)),
                D("Output", "angular_sigma_z", "width of the Gaussian n(z)", REF(double, angular_sigma_z)),
                D("Output", "angular_bias", "linear bias of the tracer", REF(double, angular_bias)),
                I("Output", "angular_ell_max", "highest ell in angular_cl_z<z>.txt", REF(int, angular_ell_max)),
            };
            return t;
        }
#undef REF

        void apply(RunSettings & s, const std::string & name, const std::string & value, const std::string & where) {
            for (const auto & p : table())
                if (p.name == name) {
                    try {
                        p.set(s, value);
                    } catch (const std::exception & e) {
                        throw std::invalid_argument(where + ": parameter " + name + ": " + e.what());
                    }
                    return;
                }
            // Parameters of the selected model (the model is set before everything else, see read())
            for (const auto & p : Models::find(s.cosmo.model).parameters)
                if (p.name == name) {
                    try {
                        s.cosmo.model_params[name] = to_double(value);
                    } catch (const std::exception & e) {
                        throw std::invalid_argument(where + ": parameter " + name + ": " + e.what());
                    }
                    return;
                }
            for (const auto & m : Models::registry())
                for (const auto & p : m.parameters)
                    if (p.name == name)
                        throw std::invalid_argument(where + ": parameter '" + name + "' belongs to model " + m.name +
                                                    ", but model = " + s.cosmo.model);
            throw std::invalid_argument(where + ": unknown parameter '" + name + "'");
        }

        void apply_preset(RunSettings & s, const std::string & v, const std::string & where) {
            const std::string l = lower(trim(v));
            if (l == "fiducial")
                s.cosmo = CosmologyParameters::fiducial();
            else if (l == "planck2018")
                s.cosmo = CosmologyParameters::Planck2018();
            else if (l == "planck2018_baseline")
                s.cosmo = CosmologyParameters::Planck2018Baseline();
            else
                throw std::invalid_argument(
                    where + ": preset must be fiducial, planck2018 or planck2018_baseline, not '" + v + "'");
        }
    } // namespace

    namespace ParameterFile {

        std::vector<std::pair<std::string, std::string>> precision_preset(const std::string & name) {
            const std::string l = lower(trim(name));
            if (l == "default")
                return {};
            // fast: tuned on lensed TT, EE, TE to ell = 2500 (Planck 2018 LCDM) against a high-precision run; about 5x
            // faster than the defaults with dchi2 ~ 0.2 for Planck-like noise (python/precision_scan.py, README)
            if (l == "fast")
                return {{"pert_n_k", "60"},
                        {"pert_dk_max", "0.004"}, // log-linear k-grid: 112 modes instead of 400
                        {"pert_n_x_early", "100"},
                        {"pert_n_x_recomb", "300"},
                        {"pert_n_x_late", "200"},
                        {"lmax_nu", "16"},
                        {"ode_abserr", "1e-6"},
                        {"n_k_per_oscillation", "2"},
                        {"n_eta_per_oscillation", "3"},
                        {"dx_recomb", "0.006"},
                        {"bessel_points_per_2pi", "16"}};
            throw std::invalid_argument("precision must be default or fast, not '" + name + "'");
        }

        RunSettings read(const std::string & filename, const std::vector<std::string> & overrides) {
            // Collect (name, value, location); the preset is applied first, then everything else in order
            struct Entry {
                std::string name, value, where;
            };
            std::vector<Entry> entries;
            if (!filename.empty()) {
                std::ifstream fp(filename);
                if (!fp)
                    throw std::runtime_error("ParameterFile: cannot open " + filename);
                std::string line;
                int n = 0;
                while (std::getline(fp, line)) {
                    n++;
                    const auto hash = line.find('#');
                    if (hash != std::string::npos)
                        line = line.substr(0, hash);
                    if (trim(line).empty())
                        continue;
                    const auto eq = line.find('=');
                    const std::string where = filename + ":" + std::to_string(n);
                    if (eq == std::string::npos)
                        throw std::invalid_argument(where + ": expected 'name = value'");
                    entries.push_back({trim(line.substr(0, eq)), trim(line.substr(eq + 1)), where});
                }
            }
            for (const auto & o : overrides) {
                const auto eq = o.find('=');
                if (eq == std::string::npos)
                    throw std::invalid_argument("command line: expected name=value, got '" + o + "'");
                entries.push_back({trim(o.substr(0, eq)), trim(o.substr(eq + 1)), "command line"});
            }
            // The preset first, then the model (its parameters are only known once it is set), then the rest in order
            RunSettings s;
            std::map<std::string, std::string> seen;
            for (const auto & e : entries)
                if (e.name == "preset")
                    apply_preset(s, e.value, e.where);
            // Precision preset: a set of precision parameters, applied before (and overridable by) the explicit ones
            for (const auto & e : entries)
                if (e.name == "precision") {
                    try {
                        for (const auto & [n, v] : precision_preset(e.value))
                            apply(s, n, v, e.where + " (precision = " + e.value + ")");
                    } catch (const std::invalid_argument & ex) {
                        throw std::invalid_argument(e.where + ": " + ex.what());
                    }
                }
            for (const auto & e : entries)
                if (e.name == "model") {
                    try {
                        Models::find(trim(e.value));
                    } catch (const std::exception & ex) {
                        throw std::invalid_argument(e.where + ": " + ex.what());
                    }
                    apply(s, e.name, e.value, e.where);
                }
            for (const auto & e : entries) {
                if (e.name == "preset" || e.name == "precision")
                    continue;
                if (seen.count(e.name) && e.where != "command line" && seen[e.name] != "command line")
                    throw std::invalid_argument(e.where + ": parameter " + e.name + " already set at " + seen[e.name]);
                seen[e.name] = e.where;
                apply(s, e.name, e.value, e.where);
            }
            s.cosmo.validate();
            return s;
        }

        void write(const RunSettings & s, const std::string & filename) {
            std::ofstream fp(filename);
            if (!fp)
                throw std::runtime_error("ParameterFile::write: cannot open " + filename);
            fp << "# Parameter file for the Einstein-Boltzmann solver: ./cmb <this file> [name=value ...]\n"
                  "# Every parameter is listed with its value; any of them can be left out (default) or overridden on "
                  "the\n"
                  "# command line. Units: k in 1/Mpc, angles in degrees, x = ln a. Booleans: true/false.\n"
                  "#\n"
                  "# preset = fiducial      # or planck2018, planck2018_baseline: loads that parameter set before the "
                  "cosmological parameters below\n"
                  "# precision = default    # or fast: precision parameters for parameter fits (lensed TT/EE/TE to ell "
                  "= 2500\n"
                  "#                        # with dchi2 < 1 for Planck, ~5x faster); explicit precision parameters "
                  "override it\n";
            std::string section;
            size_t w = 0;
            for (const auto & p : table())
                w = std::max(w, p.name.size() + 3 + p.get(s).size());
            w = std::min<size_t>(w, 40);
            for (const auto & p : table()) {
                if (p.section != section) {
                    section = p.section;
                    fp << "\n# --- " << section << " " << std::string(70 - section.size(), '-') << "\n";
                }
                const std::string kv = p.name + " = " + p.get(s);
                fp << std::left << std::setw(static_cast<int>(w)) << kv << "  # " << p.description << "\n";
                if (p.name == "model")
                    for (const auto & mp : Models::find(s.cosmo.model).parameters) {
                        const std::string mkv = mp.name + " = " + fmt(s.cosmo.model_param(mp.name));
                        fp << std::left << std::setw(static_cast<int>(w)) << mkv << "  # " << mp.description << "\n";
                    }
            }
        }

        std::vector<std::string> names() {
            std::vector<std::string> n;
            for (const auto & p : table())
                n.push_back(p.name);
            return n;
        }

        std::vector<std::string> model_parameter_names(const std::string & model) {
            std::vector<std::string> n;
            for (const auto & p : Models::find(model).parameters)
                n.push_back(p.name);
            return n;
        }
    } // namespace ParameterFile

} // namespace FML::COSMOLOGY

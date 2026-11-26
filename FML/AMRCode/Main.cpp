//==========================================================================================
//
// AMRNBody: a simple AMR N-body code for LCDM, symmetron, dilaton and coupled quintessence / the Fading Dark Sector
// (see AMRNBody.h and README.md)
//
//   1. 2LPT initial conditions (FML NBodyInitialConditions) from a linear P(k) at z = 0
//      (a CAMB-format file, or Eisenstein & Hu no-wiggle normalized to sigma8)
//   2. Coarse steps uniform in log(a) (plus the output times). In each: refine on particle
//      count, then kick-drift-kick with block time steps by level and partial Poisson solves
//   3. Outputs: P(k) (and optionally GADGET snapshots) at the output redshifts
//
// Run: mpirun -np 4 ./amrnbody parameterfile.txt
//
//==========================================================================================

#include "AMRNBody.h"

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/NBody/NBody.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/Visualization/Visualization.h>
#include <FML/InitialConditions/SpeciesIC.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>
#include <FML/TransferProviders/CLASSTransferProvider.h>
#include <FML/TransferProviders/TransferTablesFile.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <map>
#include <set>
#include <sstream>

using namespace FML::NBODY::AMR;

//==========================================================================================
// Parameters: "key = value" lines, # starts a comment
//==========================================================================================
class ParameterFile {
    std::map<std::string, std::string> values;
    mutable std::set<std::string> used;

  public:
    /// "include = file" reads the keys of another file there (relative to this file's folder); a key set later
    /// (after the include, or in the including file) replaces an earlier value
    explicit ParameterFile(const std::string & filename, int depth = 0) { read(filename, depth); }
    void read(const std::string & filename, int depth) {
        assert_mpi(depth < 10, "ParameterFile: too many nested includes");
        std::ifstream fp(filename);
        assert_mpi(fp.good(), ("Cannot open parameter file " + filename).c_str());
        const auto slash = filename.find_last_of('/');
        const std::string folder = slash == std::string::npos ? "" : filename.substr(0, slash + 1);
        std::string line;
        while (std::getline(fp, line)) {
            line = line.substr(0, line.find('#'));
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            auto trim = [](std::string s) {
                s.erase(0, s.find_first_not_of(" \t\r"));
                s.erase(s.find_last_not_of(" \t\r") + 1);
                return s;
            };
            const auto key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
            if (key == "include") {
                read(value.size() and value[0] == '/' ? value : folder + value, depth + 1);
                continue;
            }
            values[key] = value;
        }
    }
    /// Keys in the file that were never read (typos): a warning on task 0
    void warn_unused() const {
        for (auto & kv : values)
            if (not used.count(kv.first) and FML::ThisTask == 0)
                std::cout << "Warning: [" << kv.first << "] in the parameter file is not a parameter (typo?)\n";
    }
    std::string get(const std::string & key, const std::string & def) const {
        used.insert(key);
        auto it = values.find(key);
        return it == values.end() ? def : it->second;
    }
    double get(const std::string & key, double def) const {
        used.insert(key);
        auto it = values.find(key);
        return it == values.end() ? def : std::stod(it->second);
    }
    int get(const std::string & key, int def) const { return int(get(key, double(def))); }
    bool get(const std::string & key, bool def) const {
        used.insert(key);
        auto it = values.find(key);
        return it == values.end() ? def : (it->second == "true" or it->second == "1" or it->second == "yes");
    }
    std::vector<double> get_list(const std::string & key) const {
        std::vector<double> v;
        std::stringstream ss(get(key, std::string("")));
        std::string item;
        while (std::getline(ss, item, ','))
            if (item.find_first_not_of(" \t") != std::string::npos)
                v.push_back(std::stod(item));
        return v;
    }
};

//==========================================================================================
// Linear matter power spectrum at z = 0 (k in h/Mpc, P in (Mpc/h)^3)
//==========================================================================================
class LinearPower {
    std::vector<double> logk, logP;
    double OmegaM, OmegaB, h, ns, A{1.0};
    bool from_file{false};

    // Eisenstein & Hu (1998) no-wiggle transfer function, k in h/Mpc
    double T_EH(double k) const {
        const double om = OmegaM * h * h, ob = OmegaB * h * h, fb = OmegaB / OmegaM, theta = 2.7255 / 2.7;
        const double s = 44.5 * std::log(9.83 / om) / std::sqrt(1.0 + 10.0 * std::pow(ob, 0.75)); // Mpc
        const double alpha = 1.0 - 0.328 * std::log(431.0 * om) * fb + 0.38 * std::log(22.3 * om) * fb * fb;
        const double kMpc = k * h;
        const double Gamma = OmegaM * h * (alpha + (1.0 - alpha) / (1.0 + std::pow(0.43 * kMpc * s, 4)));
        const double q = k * theta * theta / Gamma;
        const double L0 = std::log(2.0 * M_E + 1.8 * q), C0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
        return L0 / (L0 + C0 * q * q);
    }
    double sigma_R(double R) const {
        // sigma^2 = int dlogk k^3 P(k) / (2 pi^2) W(kR)^2
        double sum = 0.0;
        const int n = 4000;
        const double lk1 = std::log(1e-5), lk2 = std::log(1e3), dlk = (lk2 - lk1) / n;
        for (int i = 0; i <= n; i++) {
            const double k = std::exp(lk1 + i * dlk), x = k * R;
            const double W = x < 1e-4 ? 1.0 : 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
            sum += (i == 0 or i == n ? 0.5 : 1.0) * k * k * k * (*this)(k) * W * W;
        }
        return std::sqrt(sum * dlk / (2.0 * M_PI * M_PI));
    }

  public:
    LinearPower(const std::string & filename, double OmegaM_, double OmegaB_, double h_, double ns_, double sigma8)
        : OmegaM(OmegaM_), OmegaB(OmegaB_), h(h_), ns(ns_) {
        if (not filename.empty()) {
            std::ifstream fp(filename);
            assert_mpi(fp.good(), ("Cannot open power spectrum file " + filename).c_str());
            std::string line;
            while (std::getline(fp, line)) {
                if (line.empty() or line[0] == '#')
                    continue;
                std::stringstream ss(line);
                double k, P;
                if (ss >> k >> P and k > 0 and P > 0) {
                    logk.push_back(std::log(k));
                    logP.push_back(std::log(P));
                }
            }
            assert_mpi(logk.size() > 2, "Too few points in the power spectrum file");
            from_file = true;
        }
        if (sigma8 > 0.0)
            A *= std::pow(sigma8 / sigma_R(8.0), 2);
    }
    double operator()(double k) const {
        if (k <= 0.0)
            return 0.0;
        if (not from_file) {
            const double T = T_EH(k);
            return A * std::pow(k, ns) * T * T;
        }
        // Power-law extrapolation outside the table
        const double lk = std::log(k);
        size_t i = std::upper_bound(logk.begin(), logk.end(), lk) - logk.begin();
        i = std::min(std::max(i, size_t(1)), logk.size() - 1);
        const double w = (lk - logk[i - 1]) / (logk[i] - logk[i - 1]);
        return A * std::exp(logP[i - 1] + w * (logP[i] - logP[i - 1]));
    }
};

//==========================================================================================
// Outputs
//==========================================================================================
/// A particle with a mass (the CDM+baryon P(k) of CDM and baryon particles: masses f_c(a), f_b(a))
struct MassParticle {
    double Pos[3];
    double Vel[3];
    double mass;
    constexpr int get_ndim() const { return 3; }
    double * get_pos() { return Pos; }
    double * get_vel() { return Vel; }
    double get_mass() const { return mass; }
};

/// P(k) of the particles with weight(p) > 0 (weight nullptr: all, equal masses; else the weights are the masses) and
/// the linear P(k) plin(k) at a
template <class SP>
void power_spectrum_of(const std::string & filename,
                       FML::PARTICLE::MPIParticles<AMRParticle> & part,
                       int Ngrid,
                       double box,
                       double a,
                       const std::function<double(const AMRParticle &)> & weight,
                       const std::function<double(double)> & plin) {
    // Copy the positions to particles on the slab domain (the domain the FFTW grids use)
    std::vector<SP> tmp;
    tmp.reserve(part.get_npart());
    double sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < part.get_npart(); i++) {
        const double w = weight ? weight(part[i]) : 1.0;
        if (w == 0.0)
            continue;
        SP sp;
        for (int d = 0; d < 3; d++) {
            sp.Pos[d] = part[i].Pos[d];
            sp.Vel[d] = part[i].Vel[d];
        }
        if constexpr (std::is_same_v<SP, MassParticle>)
            sp.mass = w;
        sum1 += w;
        sum2 += w * w;
        tmp.push_back(sp);
    }
    FML::SumOverTasks(&sum1);
    FML::SumOverTasks(&sum2);
    long long ntot = (long long)tmp.size();
    FML::SumOverTasks(&ntot);
    FML::PARTICLE::MPIParticles<SP> slab;
    slab.create(tmp, size_t(1.5 * double(ntot) / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);

    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(Ngrid / 2);
    // No shot-noise subtraction: a lattice (the initial conditions) has none. The Poisson value is in the header
    pofk.subtract_shotnoise = false;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3>(Ngrid,
                                                         slab.get_particles_ptr(),
                                                         slab.get_npart(),
                                                         slab.get_npart_total(),
                                                         pofk,
                                                         "CIC",
                                                         true);
    pofk.scale(box);
    if (FML::ThisTask == 0) {
        std::ofstream fp(filename);
        fp << "# a = " << a << "  z = " << 1.0 / a - 1.0 << "  Poisson shot noise V sum(m^2) / sum(m)^2 = "
           << box * box * box * sum2 / (sum1 * sum1) << " (not subtracted)\n"
           << "# k (h/Mpc)   P(k) ((Mpc/h)^3)   P_linear(k)\n";
        for (int i = 0; i < pofk.n; i++)
            if (pofk.count[i] > 0)
                fp << std::setw(15) << pofk.kbin[i] << " " << std::setw(15) << pofk.pofk[i] << " " << std::setw(15)
                   << plin(pofk.kbin[i]) << "\n";
    }
}

/// P(k) of all the particles (one set of CDM+baryon particles), linear: the input P(k) times growth^2 (growth: D(a) /
/// D(1) of the input P(k) at z = 0)
void output_power_spectrum(const std::string & filename,
                           FML::PARTICLE::MPIParticles<AMRParticle> & part,
                           int Ngrid,
                           double box,
                           double a,
                           const LinearPower & pk,
                           double growth) {
    power_spectrum_of<SimpleParticle<3>>(filename, part, Ngrid, box, a, nullptr, [&](double k) {
        return pk(k) * growth * growth;
    });
}

/// CDM and baryon particles: P(k) of the CDM (pofk_cdm_), the baryons (pofk_baryon_) and the CDM+baryons (pofk_,
/// weighted by f_c(a), f_b(a)), linear theory from the transfer functions
void output_power_spectra_two_species(const std::string & outdir,
                                      const std::string & label,
                                      const std::string & zs,
                                      FML::PARTICLE::MPIParticles<AMRParticle> & part,
                                      int Ngrid,
                                      double box,
                                      double a,
                                      double fc,
                                      const FML::INITIALCONDITIONS::TransferProvider & tr) {
    using S = FML::INITIALCONDITIONS::Species;
    auto plin = [&](S s) { return [&tr, s, a](double k) { return tr.power(s, k, a); }; };
    power_spectrum_of<SimpleParticle<3>>(outdir + "/pofk_cdm_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                         [](const AMRParticle & p) { return p.species == 0 ? 1.0 : 0.0; },
                                         plin(S::CDM));
    power_spectrum_of<SimpleParticle<3>>(outdir + "/pofk_baryon_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                         [](const AMRParticle & p) { return p.species == 1 ? 1.0 : 0.0; },
                                         plin(S::Baryon));
    power_spectrum_of<MassParticle>(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                    [fc](const AMRParticle & p) { return p.species == 0 ? fc : 1.0 - fc; },
                                    plin(S::CDMBaryon));
}

/// species: -1 all the particles, else only those of that species (OmegaM: the density of what is written)
void output_gadget(const std::string & filename,
                   FML::PARTICLE::MPIParticles<AMRParticle> & part,
                   double a,
                   double box,
                   double OmegaM,
                   double h,
                   int species = -1,
                   double OmegaL = -1.0) {
    FML::FILEUTILS::GADGET::GadgetWriter gw;
    // Velocities: v_code = a^2 dx/dt / (H0 B) -> GADGET's sqrt(a) dx/dt in km/s
    const double vel_norm = 100.0 * box / std::pow(a, 1.5);
    std::vector<AMRParticle> sel;
    if (species >= 0)
        for (size_t i = 0; i < part.get_npart(); i++)
            if (part[i].species == species)
                sel.push_back(part[i]);
    long long nsel = (long long)sel.size();
    FML::SumOverTasks(&nsel);
    gw.write_gadget_single(filename + "." + std::to_string(FML::ThisTask),
                           species >= 0 ? sel.data() : part.get_particles_ptr(),
                           species >= 0 ? sel.size() : part.get_npart(),
                           species >= 0 ? size_t(nsel) : part.get_npart_total(),
                           FML::NTasks,
                           a,
                           box,
                           OmegaM,
                           OmegaL >= 0.0 ? OmegaL : 1.0 - OmegaM,
                           h,
                           box,
                           vel_norm);
}

//==========================================================================================
// CDM and baryons as two sets of particles: the transfer functions and the initial conditions
//==========================================================================================

/// The transfer functions: tables (a file of TransferTablesFile.h, checked against OmegaB, OmegaM, h, the model and its
/// parameters), camb (a transfer info file) or class (the root of a run: <root>z*_tk.dat, <root>background.dat)
std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> make_transfers(const std::string & source,
                                                                         const std::string & file,
                                                                         const std::string & class_gauge,
                                                                         const AMRNBodyParameters & par,
                                                                         double h,
                                                                         double n_s,
                                                                         double A_s,
                                                                         double k_pivot_mpc) {
    using namespace FML::INITIALCONDITIONS;
    assert_mpi(not file.empty(), "particle_baryons: set transfer_file");
    const std::string msg = "[transfer_source = " + source + "] [" + file + "] ";
    if (source == "tables") {
        auto tf = read_transfer_tables(file);
        auto check = [&](const std::string & key, double value) {
            if (not tf.contains(key)) {
                if (FML::ThisTask == 0)
                    std::cout << msg << "warning: the file has no " << key << " (not checked)\n";
                return;
            }
            const double v = std::stod(tf.get(key));
            if (std::abs(v - value) > 1e-6 * std::max(1.0, std::abs(value)))
                throw std::runtime_error(msg + "Made for " + key + " = " + tf.get(key) + ", the simulation has " +
                                         std::to_string(value));
        };
        check("cosmology_Omega_b", par.OmegaB);
        check("cosmology_h", h);
        // The matter of the simulation is CDM + baryons (+ massive neutrinos in the tables: not supported)
        if (tf.contains("cosmology_Omega_cdm"))
            check("cosmology_Omega_cdm", par.OmegaM - par.OmegaB);
        if (tf.contains("cosmology_m_nu_eV") and tf.get("cosmology_m_nu_eV") != "none")
            throw std::runtime_error(msg + "Tables with massive neutrinos are not supported");
        // The model: LCDM, or coupled quintessence / the Fading Dark Sector with the same parameters
        const std::string model = tf.get("cosmology_model");
        const std::string want = par.gravity_model == "FadingDarkSector"      ? "fading_dark_sector"
                                 : par.gravity_model == "CoupledQuintessence" ? "coupled_quintessence"
                                                                               : "lcdm";
        if (not model.empty() and model != want and not(want == "lcdm" and model == "LCDM"))
            throw std::runtime_error(msg + "Made for cosmology_model = " + model + ", the simulation has " + want);
        if (want == "fading_dark_sector") {
            check("cosmology_model_fds_c", par.fds_c);
            check("cosmology_model_fds_cprime", par.fds_cprime);
        }
        if (want == "coupled_quintessence") {
            check("cosmology_model_cq_lambda", par.cq_lambda);
            check("cosmology_model_cq_beta", par.cq_beta);
        }
        if (FML::ThisTask == 0 and tf.contains("cosmology_T_CMB") and std::stod(tf.get("cosmology_T_CMB")) > 0.0 and
            par.OmegaR == 0.0)
            std::cout << "# Note: the transfer tables include radiation, the simulation's background does not (OmegaR = 0)\n";
        if (FML::ThisTask == 0)
            std::cout << "# Background radiation OmegaR = " << par.OmegaR << "\n";
        return std::make_shared<TransferTablesFileProvider>(file);
    }
    if (source == "camb") {
        CAMBFileTransferProvider::Settings cs;
        cs.infofile = file;
        cs.h = h;
        cs.Omegab = par.OmegaB;
        cs.OmegaCDM = par.OmegaM - par.OmegaB;
        cs.A_s = A_s;
        cs.n_s = n_s;
        cs.k_pivot_mpc = k_pivot_mpc;
        return std::make_shared<CAMBFileTransferProvider>(cs);
    }
    if (source == "class") {
        assert_mpi(class_gauge == "synchronous" or class_gauge == "newtonian", "class_gauge must be synchronous or newtonian");
        CLASSTransferProvider::Settings cs;
        auto files = find_class_output_files(file);
        cs.transfer_files = files.first;
        cs.background_file = files.second;
        cs.gauge = class_gauge == "synchronous" ? Gauge::SynchronousCDM : Gauge::Newtonian;
        cs.output_gauge = Gauge::SynchronousCDM;
        cs.A_s = A_s;
        cs.n_s = n_s;
        cs.k_pivot_mpc = k_pivot_mpc;
        cs.range_policy = RangePolicy::SourceExtrapolation;
        return std::make_shared<CLASSTransferProvider>(cs);
    }
    throw std::invalid_argument("transfer_source must be tables, camb or class");
}

/// CDM (species 0) and baryon (species 1) particles, Npart_1D^3 of each from the same lattice, as SimulationCode's
/// particle_baryons (relative mode in the displacements): one Gaussian field delta_cb(k, zini) with the CDM+baryon
/// P(k) of the transfer functions, delta_i = (T_i / T_cb) delta_cb, the 2LPT term of delta_cb for both, and the 1LPT
/// velocities f_i = f_cb + dlog(T_i / T_cb)/dlna with f_cb the growth rate of the simulation
/// (FML::INITIALCONDITIONS::two_species_ic_splines, FML::NBODY::generate_species_from_delta_cb). offset_lattice: the
/// baryons start from the lattice offset by half a particle spacing in each direction. The baryon IDs follow
/// the CDM IDs. The same seed gives the same delta_cb phases as one set of particles
void two_species_initial_conditions(FML::PARTICLE::MPIParticles<AMRParticle> & part,
                                    const FML::INITIALCONDITIONS::TransferProvider & tr,
                                    const Cosmology & cosmo,
                                    FML::RANDOM::RandomGenerator * rng,
                                    int Npart_1D,
                                    int Nmesh,
                                    bool fix_amplitude,
                                    int LPT_order,
                                    double box,
                                    double zini,
                                    bool offset_lattice) {
    using namespace FML::INITIALCONDITIONS;
    const double aini = 1.0 / (1.0 + zini);
    const double fc = cosmo.cdm_fraction(aini);
    const auto splines =
        two_species_ic_splines(tr, [&](double, double a) { return cosmo.f1(a); }, zini, fc, 1.0 - fc);

    // delta_cb(k, zini) (as NBodyInitialConditions makes it)
    auto nextra = FML::INTERPOLATION::get_extra_slices("CIC");
    FML::GRID::FFTWGrid<3> delta(Nmesh, nextra.first, nextra.second);
    delta.set_grid_status_real(false);
    std::function<double(double)> Pofk_over_volume = [&](double kBox) {
        if (kBox == 0.0)
            return 0.0;
        return tr.power(Species::CDMBaryon, kBox / box, aini) / (box * box * box);
    };
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(delta, rng, Pofk_over_volume, fix_amplitude);

    const double kmin = tr.range().kmin_hmpc, kmax = tr.range().kmax_hmpc;
    auto logk = [&](double kBox) { return std::log(std::min(std::max(kBox / box, kmin), kmax)); };
    const double vnorm = aini * aini * cosmo.E(aini);
    const std::vector<double> velocity_norms{vnorm * cosmo.f1(aini), vnorm * cosmo.f2(aini)};
    const long long N = (long long)Npart_1D * Npart_1D * Npart_1D;
    FML::PARTICLE::MPIParticles<AMRParticle> species[2];
    const FML::INTERPOLATION::SPLINE::Spline * ratio[2] = {&splines.ratio_cdm, &splines.ratio_baryon};
    const FML::INTERPOLATION::SPLINE::Spline * growth[2] = {&splines.f_cdm, &splines.f_baryon};
    for (int s = 0; s < 2; s++) {
        std::function<double(double)> r = [&, s](double kBox) { return (*ratio[s])(logk(kBox)); };
        std::function<double(double)> v1 = [&, s](double kBox) { return vnorm * (*growth[s])(logk(kBox)); };
        FML::NBODY::generate_species_from_delta_cb<3, AMRParticle>(
            species[s], delta, r, v1, velocity_norms, LPT_order, Npart_1D, 1.5, box, zini,
            (s == 1 and offset_lattice) ? 0.5 / double(Npart_1D) : 0.0, s * N);
        for (auto & p : species[s])
            p.species = s;
    }
    // One set of particles (both species are on the same slab domain here)
    std::vector<AMRParticle> all;
    all.reserve(species[0].get_npart() + species[1].get_npart());
    for (int s = 0; s < 2; s++)
        for (size_t i = 0; i < species[s].get_npart(); i++)
            all.push_back(species[s][i]);
    species[0].free();
    species[1].free();
    part.create(all, size_t(1.5 * 2.0 * double(N) / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);
}

//==========================================================================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        if (FML::ThisTask == 0)
            std::cout << "Usage: mpirun -np N " << argv[0] << " parameterfile.txt\n";
        return 1;
    }
    ParameterFile pf(argv[1]);

    // Cosmology and power spectrum
    const double OmegaM = pf.get("OmegaM", 0.3);
    const double OmegaB = pf.get("OmegaB", 0.049);
    const double h = pf.get("h", 0.67);
    const double ns = pf.get("n_s", 0.96);
    const double sigma8 = pf.get("sigma8", 0.81);
    const std::string pk_file = pf.get("pk_file", std::string(""));
    const double box = pf.get("box", 200.0);

    // Simulation
    AMRNBodyParameters par;
    par.OmegaM = OmegaM;
    par.levelmin = pf.get("levelmin", 6);
    par.levelmax = pf.get("levelmax", 10);
    par.n_refine = pf.get("n_refine", 8.0);
    par.cic = pf.get("cic", true);
    par.gradient_order = pf.get("gradient_order", 4);
    par.subcycling = pf.get("subcycling", true);
    par.cfl = pf.get("cfl", 0.5);
    par.mg_epsilon = pf.get("mg_epsilon", 1e-6);
    par.mg_maxsteps = pf.get("mg_maxsteps", 50);
    par.boxsize = box;
    par.gravity_model = pf.get("gravity_model", std::string("LCDM"));
    par.symmetron_assb = pf.get("symmetron_assb", 0.5);
    par.symmetron_beta = pf.get("symmetron_beta", 1.0);
    par.symmetron_L_mpch = pf.get("symmetron_L_mpch", 1.0);
    par.symmetron_N = pf.get("symmetron_N", 2.0);
    par.symmetron_M = pf.get("symmetron_M", 4.0);
    par.symmetron_epsilon = pf.get("symmetron_epsilon", 1e-6);
    par.dilaton_A2 = pf.get("dilaton_A2", 5e5);
    par.dilaton_beta0 = pf.get("dilaton_beta0", 0.5);
    par.dilaton_r = pf.get("dilaton_r", 1.0);
    par.dilaton_xi = pf.get("dilaton_xi", 1e-3);
    par.dilaton_epsilon = pf.get("dilaton_epsilon", 1e-6);
    par.dilaton_start_redshift = pf.get("dilaton_start_redshift", 4.0);
    par.OmegaB = OmegaB;
    if (par.gravity_model == "CoupledQuintessence") {
        par.cq_lambda = pf.get("cq_lambda", 0.5);
        par.cq_beta = pf.get("cq_beta", 0.05);
    }
    if (par.gravity_model == "FadingDarkSector") {
        par.fds_c = pf.get("fds_c", -0.85);
        par.fds_cprime = pf.get("fds_cprime", 0.05);
    }
    // Transfer functions (transfer_file set): the initial conditions from the transfer functions at zini
    // (transfer_source = tables: a file of FML's transfer tables, e.g. from SimulationCode's nbody --write-transfers
    // with the Einstein-Boltzmann solver; camb: a CAMB transfer info file; class: the root of a CLASS run with mTk).
    // The amplitude is that of the transfer functions (A_s), not pk_file / sigma8. Needed for CDM and baryons as two
    // sets of particles (particle_baryons, Npart_1D^3 of each, from the CDM and baryon transfer functions); with one
    // set of particles the CDM+baryon P(k, zini) of the transfer functions (the same field as with particle_baryons)
    par.baryons = pf.get("particle_baryons", false);
    std::string transfer_source, class_gauge;
    const std::string transfer_file = pf.get("transfer_file", std::string(""));
    assert_mpi(not par.baryons or not transfer_file.empty(), "particle_baryons = true needs transfer_file");
    double A_s = 0.0, k_pivot_mpc = 0.05;
    // The baryons start from the same lattice as the CDM (same) or one offset by half a particle spacing (offset)
    const std::string baryon_lattice = par.baryons ? pf.get("baryon_lattice", std::string("same")) : "same";
    assert_mpi(baryon_lattice == "same" or baryon_lattice == "offset", "baryon_lattice must be same or offset");
    if (not transfer_file.empty()) {
        transfer_source = pf.get("transfer_source", std::string("tables"));
        if (transfer_source != "tables") {
            A_s = pf.get("A_s", 2.1e-9);
            k_pivot_mpc = pf.get("k_pivot_mpc", 0.05);
        }
        if (transfer_source == "class")
            class_gauge = pf.get("class_gauge", std::string("synchronous"));
    }
    // The radiation of the background (photons + massless neutrinos). Default 0; with transfer_source = tables the
    // radiation of the tables (T_CMB, N_eff), so that the growth of the simulation is that of the transfer functions
    par.OmegaR = pf.get("OmegaR", -1.0);
    if (par.OmegaR < 0.0) {
        par.OmegaR = 0.0;
        if (transfer_source == "tables") {
            const auto tf = FML::INITIALCONDITIONS::read_transfer_tables(transfer_file);
            if (tf.contains("cosmology_T_CMB") and tf.contains("cosmology_N_eff") and tf.contains("cosmology_h")) {
                FML::COSMOLOGY::BACKGROUND::BackgroundParameters bp;
                bp.h = std::stod(tf.get("cosmology_h"));
                bp.T_CMB = std::stod(tf.get("cosmology_T_CMB"));
                bp.N_eff = std::stod(tf.get("cosmology_N_eff"));
                par.OmegaR = bp.Omega_gamma() + bp.Omega_nu_massless();
            }
        }
    }
    // Coupled quintessence / FDS: the input P(k) is the linear P(k) at z = 0 of the model (ic_growth = model, the
    // default), or of LCDM with the same OmegaM (ic_growth = LCDM: the same initial amplitude as an LCDM run)
    const std::string ic_growth = (par.coupled_quintessence() and transfer_file.empty())
                                      ? pf.get("ic_growth", std::string("model"))
                                      : "model";
    assert_mpi(ic_growth == "model" or ic_growth == "LCDM", "ic_growth must be model or LCDM");
    const int Npart_1D = pf.get("Npart_1D", 64);
    const int Nmesh_ic = pf.get("Nmesh_ic", Npart_1D);
    const int LPT_order = pf.get("LPT_order", 2);
    const unsigned int seed = (unsigned int)pf.get("seed", 1234);
    // GSL (gsl_rng_ranlxd1, as N-GenIC/2LPTic/L-PICOLA: the same IC for the same seed) or MT19937 (std::mt19937)
    const std::string random_generator = pf.get("random_generator", std::string("GSL"));
    const bool fix_amplitude = pf.get("fix_amplitude", false);
    const double zini = pf.get("zini", 49.0);
    const int nsteps = pf.get("nsteps", 50);
    auto zout = pf.get_list("output_redshifts");
    const std::string outdir = pf.get("output_folder", std::string("."));
    const std::string label = pf.get("label", std::string("amrnbody"));
    const int pofk_ngrid = pf.get("pofk_ngrid", 2 * Npart_1D);
    const bool gadget = pf.get("output_gadget", false);
    // A PNG of the particles in a slab through the middle of the box at every output
    const bool images = pf.get("output_images", false);
    const int images_npix = pf.get("images_npix", 1024);
    const double images_thickness = pf.get("images_slab_thickness", 0.1);

    const Cosmology cosmo = make_cosmology(par);
    Cosmology lcdm(OmegaM, par.OmegaR);
    lcdm.OmegaB = OmegaB;
    const double sigma8_rescale = pf.get("sigma8_rescale", -1.0);
    LinearPower pk(pk_file, OmegaM, OmegaB, h, ns, pk_file.empty() ? sigma8 : sigma8_rescale);
    pf.warn_unused();

    // The transfer functions (CDM and baryons as two sets of particles)
    std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> transfers;
    if (not transfer_file.empty())
        transfers = make_transfers(transfer_source, transfer_file, class_gauge, par, h, ns, A_s, k_pivot_mpc);

    //======================================================================================
    // Initial conditions (made on the FFTW slab domain, then moved to the grid's domain)
    //======================================================================================
    const double aini = 1.0 / (1.0 + zini);
    const Cosmology & ic_cosmo = ic_growth == "LCDM" ? lcdm : cosmo;
    const double growth_ini = ic_cosmo.D1(aini) / ic_cosmo.D1(1.0);
    // D(a) / D(1) of the input P(k): the model's growth from the initial amplitude
    auto growth_of_a = [&](double a) { return growth_ini * cosmo.D1(a) / cosmo.D1(aini); };
    if (par.coupled_quintessence() and FML::ThisTask == 0) {
        std::cout << "\n" << par.gravity_model << ": lambda = " << cosmo.cq_lambda << ", beta = " << cosmo.cq_beta
                  << " (OmegaM = " << OmegaM << ", OmegaB = " << OmegaB << ")\n";
        for (double a : {aini, 0.25, 0.5, 1.0})
            std::cout << "  a = " << std::setw(8) << a << ": E / E_LCDM = " << std::setw(10) << cosmo.E(a) / lcdm.E(a)
                      << "  dphi/dlna = " << std::setw(11) << cosmo.dphidlna(a) << "  G_eff rho_m a^3 / (G OmegaM) = "
                      << std::setw(10) << cosmo.poisson_factor(a) << "  Gamma = " << std::setw(11) << cosmo.friction(a)
                      << "  D / D_LCDM (equal at zini) = " << cosmo.D1(a) / cosmo.D1(aini) * lcdm.D1(aini) / lcdm.D1(a) << "\n";
        std::cout << "  input P(k) back-scaled with the growth of " << (ic_growth == "LCDM" ? "LCDM" : "the model")
                  << ": D(zini) / D(0) = " << growth_ini << "\n";
    }
    FML::PARTICLE::MPIParticles<AMRParticle> part;
    {
        std::unique_ptr<FML::RANDOM::RandomGenerator> rng;
        if (random_generator == "GSL")
            rng = std::make_unique<FML::RANDOM::GSLRandomGenerator>();
        else if (random_generator == "MT19937")
            rng = std::make_unique<FML::RANDOM::RandomGenerator>();
        else
            throw std::invalid_argument("random_generator must be GSL or MT19937");
        rng->set_seed(seed);
        auto Pofk_over_volume = [&](double kBox) {
            const double k = kBox / box;
            if (transfers)
                return transfers->power(FML::INITIALCONDITIONS::Species::CDMBaryon, k, aini) / (box * box * box);
            return pk(k) * growth_ini * growth_ini / (box * box * box);
        };
        auto one = [](double) { return 1.0; };
        // Velocities v = a^2 E f_n Psi_n (Psi_n the nLPT displacement at aini, box units)
        const double vnorm = aini * aini * cosmo.E(aini);
        std::vector<double> velocity_norms{vnorm * cosmo.f1(aini), vnorm * cosmo.f2(aini)};
        if (par.baryons)
            two_species_initial_conditions(part, *transfers, cosmo, rng.get(), Npart_1D, Nmesh_ic, fix_amplitude,
                                           LPT_order, box, zini, baryon_lattice == "offset");
        else
            FML::NBODY::NBodyInitialConditions<3, AMRParticle>(part,
                                                               Npart_1D,
                                                               1.5,
                                                               Nmesh_ic,
                                                               fix_amplitude,
                                                               rng.get(),
                                                               one,
                                                               Pofk_over_volume,
                                                               LPT_order,
                                                               "gaussian",
                                                               0.0,
                                                               box,
                                                               zini,
                                                               velocity_norms);
        for (size_t i = 0; i < part.get_npart(); i++)
            for (int d = 0; d < 3; d++)
                part[i].Acc[d] = 0.0;
    }
    if (FML::ThisTask == 0 and transfers)
        std::cout << "\n" << (par.baryons ? "CDM and baryons as two sets of particles (" + std::to_string(Npart_1D) + "^3 of each)"
                                          : std::string("CDM+baryons as one set of particles"))
                  << ", initial conditions from the transfer functions: " << transfer_source << " (" << transfer_file << ")\n";
    if (FML::ThisTask == 0)
        std::cout << "\nAMRNBody (" << par.gravity_model << "): " << Npart_1D << "^3 particles, box " << box << " Mpc/h, levels " << par.levelmin
                  << "-" << par.levelmax << ", refine above " << par.n_refine << " particles, z = " << zini
                  << " -> " << (zout.empty() ? 0.0 : *std::min_element(zout.begin(), zout.end())) << "\n";

    //======================================================================================
    // Time steps: uniform in log(a), plus the output times
    //======================================================================================
    if (zout.empty())
        zout.push_back(0.0);
    std::sort(zout.begin(), zout.end(), std::greater<double>());
    const double aend = 1.0 / (1.0 + zout.back());
    std::vector<double> times;
    for (int i = 0; i <= nsteps; i++)
        times.push_back(aini * std::pow(aend / aini, double(i) / nsteps));
    for (auto z : zout)
        times.push_back(1.0 / (1.0 + z));
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end(), [](double x, double y) { return std::abs(x / y - 1.0) < 1e-6; }),
                times.end());

    AMRNBodySimulation sim(par, part, aini);
    auto write_pofk = [&](const std::string & zs, double a) {
        if (par.baryons)
            output_power_spectra_two_species(outdir, label, zs, part, pofk_ngrid, box, a, cosmo.cdm_fraction(a), *transfers);
        else if (transfers)
            power_spectrum_of<SimpleParticle<3>>(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, pofk_ngrid, box, a,
                                                 nullptr, [&](double k) {
                                                     return transfers->power(FML::INITIALCONDITIONS::Species::CDMBaryon, k, a);
                                                 });
        else
            output_power_spectrum(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, pofk_ngrid, box, a, pk, growth_of_a(a));
    };
    write_pofk(std::to_string(zini).substr(0, 6), aini);

    size_t iout = 0;
    for (size_t istep = 1; istep < times.size(); istep++) {
        const auto t0 = std::chrono::steady_clock::now();
        sim.step(times[istep]);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (FML::ThisTask == 0)
            std::cout << "\nStep " << istep << " / " << times.size() - 1 << ": a = " << sim.a << " (z = " << 1.0 / sim.a - 1.0
                      << "), " << dt << " s\n";
        sim.print_status(std::cout);

        while (iout < zout.size() and std::abs(sim.a * (1.0 + zout[iout]) - 1.0) < 1e-6) {
            char zs[32];
            std::snprintf(zs, sizeof(zs), "%.3f", zout[iout]);
            write_pofk(zs, sim.a);
            if (gadget and par.baryons) {
                // One file set per species (the CDM mass today; with coupled quintessence it changes with time)
                output_gadget(outdir + "/gadget_cdm_" + label + "_z" + zs, part, sim.a, box, OmegaM - OmegaB, h, 0, 1.0 - OmegaM);
                output_gadget(outdir + "/gadget_baryon_" + label + "_z" + zs, part, sim.a, box, OmegaB, h, 1, 1.0 - OmegaM);
            } else if (gadget)
                output_gadget(outdir + "/gadget_" + label + "_z" + zs, part, sim.a, box, OmegaM, h);
            if (images) {
                const auto image = FML::VISUALIZATION::project_particles(
                    part.get_particles_ptr(), part.get_npart(), images_npix, 2, 0.5 - 0.5 * images_thickness, images_thickness);
                if (FML::ThisTask == 0)
                    FML::VISUALIZATION::write_png(outdir + "/image_density_" + label + "_z" + zs + ".png",
                                                  image,
                                                  "magma",
                                                  FML::VISUALIZATION::ColorScale::automatic_log(1.0, 99.9));
            }
            if (FML::ThisTask == 0)
                std::cout << "  output at z = " << zs << "\n";
            iout++;
        }
    }
    return 0;
}

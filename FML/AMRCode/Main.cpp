//==========================================================================================
//
// AMRNBody: a simple AMR N-body code for LCDM (see AMRNBody.h and README.md)
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
void output_power_spectrum(const std::string & filename,
                           FML::PARTICLE::MPIParticles<AMRParticle> & part,
                           int Ngrid,
                           double box,
                           double a,
                           const LinearPower & pk,
                           const LCDM & cosmo) {
    // Copy the positions to particles on the slab domain (the domain the FFTW grids use)
    using SP = SimpleParticle<3>;
    std::vector<SP> tmp(part.get_npart());
    for (size_t i = 0; i < part.get_npart(); i++)
        for (int d = 0; d < 3; d++) {
            tmp[i].Pos[d] = part[i].Pos[d];
            tmp[i].Vel[d] = part[i].Vel[d];
        }
    FML::PARTICLE::MPIParticles<SP> slab;
    slab.create(tmp, size_t(1.5 * double(part.get_npart_total()) / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);

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
        const double growth2 = std::pow(cosmo.D1(a) / cosmo.D1(1.0), 2);
        fp << "# a = " << a << "  z = " << 1.0 / a - 1.0 << "  Poisson shot noise V/N = "
           << box * box * box / double(part.get_npart_total()) << " (not subtracted)\n"
           << "# k (h/Mpc)   P(k) ((Mpc/h)^3)   P_linear(k)\n";
        for (int i = 0; i < pofk.n; i++)
            if (pofk.count[i] > 0)
                fp << std::setw(15) << pofk.kbin[i] << " " << std::setw(15) << pofk.pofk[i] << " " << std::setw(15)
                   << pk(pofk.kbin[i]) * growth2 << "\n";
    }
}

void output_gadget(const std::string & filename,
                   FML::PARTICLE::MPIParticles<AMRParticle> & part,
                   double a,
                   double box,
                   double OmegaM,
                   double h) {
    FML::FILEUTILS::GADGET::GadgetWriter gw;
    // Velocities: v_code = a^2 dx/dt / (H0 B) -> GADGET's sqrt(a) dx/dt in km/s
    const double vel_norm = 100.0 * box / std::pow(a, 1.5);
    gw.write_gadget_single(filename + "." + std::to_string(FML::ThisTask),
                           part.get_particles_ptr(),
                           part.get_npart(),
                           part.get_npart_total(),
                           FML::NTasks,
                           a,
                           box,
                           OmegaM,
                           1.0 - OmegaM,
                           h,
                           box,
                           vel_norm);
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

    LCDM cosmo(OmegaM);
    const double sigma8_rescale = pf.get("sigma8_rescale", -1.0);
    LinearPower pk(pk_file, OmegaM, OmegaB, h, ns, pk_file.empty() ? sigma8 : sigma8_rescale);
    pf.warn_unused();

    //======================================================================================
    // Initial conditions (made on the FFTW slab domain, then moved to the grid's domain)
    //======================================================================================
    const double aini = 1.0 / (1.0 + zini);
    const double growth_ini = cosmo.D1(aini) / cosmo.D1(1.0);
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
            return pk(k) * growth_ini * growth_ini / (box * box * box);
        };
        auto one = [](double) { return 1.0; };
        // Velocities v = a^2 E f_n Psi_n (Psi_n the nLPT displacement at aini, box units)
        const double vnorm = aini * aini * cosmo.E(aini);
        std::vector<double> velocity_norms{vnorm * cosmo.f1(aini), vnorm * cosmo.f2(aini)};
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
    output_power_spectrum(outdir + "/pofk_" + label + "_z" + std::to_string(zini).substr(0, 6) + ".txt", part, pofk_ngrid, box, aini, pk, cosmo);

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
            output_power_spectrum(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, pofk_ngrid, box, sim.a, pk, cosmo);
            if (gadget)
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

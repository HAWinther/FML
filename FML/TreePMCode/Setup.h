#ifndef FML_TREEPMCODE_SETUP_HEADER
#define FML_TREEPMCODE_SETUP_HEADER

//==========================================================================================
//
// TreePMCode around the time stepping (as AMRCode's Main.cpp, kept separate): the parameters of a run (the
// cosmology and model, the initial conditions, the time steps and the outputs) read from the parameter file, the
// initial conditions (1LPT/2LPT from a linear P(k) at z = 0 or from transfer functions; CDM and baryons as two sets of
// particles), the time steps and the outputs (P(k) of every species, GADGET files, images). The particle type P needs
// Pos[3], Vel[3], species (0: CDM+baryons or CDM, 1: baryons) and what FML's IC and GADGET writer need (get_pos,
// get_vel, get_id, set_id).
//
//==========================================================================================

#include "Cosmology.h"
#include "LinearPower.h"
#include "ParameterFile.h"

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/InitialConditions/SpeciesIC.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>
#include <FML/TransferProviders/CLASSTransferProvider.h>
#include <FML/TransferProviders/TransferTablesFile.h>
#include <FML/Visualization/Visualization.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace FML::NBODY::TREEPM {

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
    template <class SP, class P>
    void power_spectrum_of(const std::string & filename,
                           FML::PARTICLE::MPIParticles<P> & part,
                           int Ngrid,
                           double box,
                           double a,
                           const std::function<double(const P &)> & weight,
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
    template <class P>
    void output_power_spectrum(const std::string & filename,
                               FML::PARTICLE::MPIParticles<P> & part,
                               int Ngrid,
                               double box,
                               double a,
                               const LinearPower & pk,
                               double growth) {
        power_spectrum_of<::SimpleParticle<3>, P>(filename, part, Ngrid, box, a, nullptr, [&](double k) {
            return pk(k) * growth * growth;
        });
    }

    /// CDM and baryon particles: P(k) of the CDM (pofk_cdm_), the baryons (pofk_baryon_) and the CDM+baryons (pofk_,
    /// weighted by f_c(a), f_b(a)), linear theory from the transfer functions
    template <class P>
    void output_power_spectra_two_species(const std::string & outdir,
                                          const std::string & label,
                                          const std::string & zs,
                                          FML::PARTICLE::MPIParticles<P> & part,
                                          int Ngrid,
                                          double box,
                                          double a,
                                          double fc,
                                          const FML::INITIALCONDITIONS::TransferProvider & tr) {
        using S = FML::INITIALCONDITIONS::Species;
        auto plin = [&](S s) { return [&tr, s, a](double k) { return tr.power(s, k, a); }; };
        power_spectrum_of<::SimpleParticle<3>, P>(outdir + "/pofk_cdm_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                             [](const P & p) { return p.species == 0 ? 1.0 : 0.0; },
                                             plin(S::CDM));
        power_spectrum_of<::SimpleParticle<3>, P>(outdir + "/pofk_baryon_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                             [](const P & p) { return p.species == 1 ? 1.0 : 0.0; },
                                             plin(S::Baryon));
        power_spectrum_of<MassParticle, P>(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, Ngrid, box, a,
                                        [fc](const P & p) { return p.species == 0 ? fc : 1.0 - fc; },
                                        plin(S::CDMBaryon));
    }

    /// species: -1 all the particles, else only those of that species (OmegaM: the density of what is written)
    template <class P>
    void output_gadget(const std::string & filename,
                       FML::PARTICLE::MPIParticles<P> & part,
                       double a,
                       double box,
                       double OmegaM,
                       double h,
                       int species = -1,
                       double OmegaL = -1.0) {
        FML::FILEUTILS::GADGET::GadgetWriter gw;
        // Velocities: v_code = a^2 dx/dt / (H0 B) -> GADGET's sqrt(a) dx/dt in km/s
        const double vel_norm = 100.0 * box / std::pow(a, 1.5);
        std::vector<P> sel;
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
    inline std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> make_transfers(const std::string & source,
                                                                             const std::string & file,
                                                                             const std::string & class_gauge,
                                                                             const ModelParameters & par,
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
            // The model: LCDM, or coupled quintessence / the Fading Dark Sector / covariant dark scattering with the same
            // parameters (ds_cs2, the sound speed of the dark energy, only enters the transfer functions)
            const std::string model = tf.get("cosmology_model");
            const std::string want = par.gravity_model == "FadingDarkSector"          ? "fading_dark_sector"
                                     : par.gravity_model == "CoupledQuintessence"     ? "coupled_quintessence"
                                     : par.gravity_model == "CovariantDarkScattering" ? "covariant_dark_scattering"
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
            if (want == "covariant_dark_scattering") {
                check("cosmology_model_ds_w", par.ds_w);
                check("cosmology_model_cds_alpha", par.cds_alpha);
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
    template <class P>
    void two_species_initial_conditions(FML::PARTICLE::MPIParticles<P> & part,
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
        FML::PARTICLE::MPIParticles<P> species[2];
        const FML::INTERPOLATION::SPLINE::Spline * ratio[2] = {&splines.ratio_cdm, &splines.ratio_baryon};
        const FML::INTERPOLATION::SPLINE::Spline * growth[2] = {&splines.f_cdm, &splines.f_baryon};
        for (int s = 0; s < 2; s++) {
            std::function<double(double)> r = [&, s](double kBox) { return (*ratio[s])(logk(kBox)); };
            std::function<double(double)> v1 = [&, s](double kBox) { return vnorm * (*growth[s])(logk(kBox)); };
            FML::NBODY::generate_species_from_delta_cb<3, P>(
                species[s], delta, r, v1, velocity_norms, LPT_order, Npart_1D, 1.5, box, zini,
                (s == 1 and offset_lattice) ? 0.5 / double(Npart_1D) : 0.0, s * N);
            for (auto & p : species[s])
                p.species = s;
        }
        // One set of particles (both species are on the same slab domain here)
        std::vector<P> all;
        all.reserve(species[0].get_npart() + species[1].get_npart());
        for (int s = 0; s < 2; s++)
            for (size_t i = 0; i < species[s].get_npart(); i++)
                all.push_back(species[s][i]);
        species[0].free();
        species[1].free();
        part.create(all, size_t(1.5 * 2.0 * double(N) / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);
    }


    //==========================================================================================
    /// The parameters of a run that the codes share (read_run_settings): the cosmology and the input P(k) or transfer
    /// functions, the initial conditions, the time steps and the outputs
    //==========================================================================================
    struct RunSettings {
        double OmegaM{0.3}, OmegaB{0.049}, h{0.67}, ns{0.96}, sigma8{0.81}, sigma8_rescale{-1.0};
        std::string pk_file;
        double box{200.0};
        int Npart_1D{64}, Nmesh_ic{64}, LPT_order{2};
        unsigned int seed{1234};
        std::string random_generator{"GSL"};
        bool fix_amplitude{false};
        double zini{49.0};
        int nsteps{50};
        std::vector<double> zout;
        std::string outdir{"."}, label;
        int pofk_ngrid{128};
        bool gadget{false}, images{false};
        int images_npix{1024};
        double images_thickness{0.1};
        std::string transfer_source, transfer_file, class_gauge, baryon_lattice{"same"}, ic_growth{"model"};
        double A_s{0.0}, k_pivot_mpc{0.05};
    };

    /// Read the shared parameters, and the model into model (gravity_model and its parameters, OmegaM, OmegaB,
    /// OmegaR, particle_baryons). A code reads its own parameters too, then calls pf.warn_unused()
    inline RunSettings read_run_settings(const ParameterFile & pf, ModelParameters & model, const std::string & default_label) {
        RunSettings r;
        // Cosmology and power spectrum
        r.OmegaM = pf.get("OmegaM", 0.3);
        r.OmegaB = pf.get("OmegaB", 0.049);
        r.h = pf.get("h", 0.67);
        r.ns = pf.get("n_s", 0.96);
        r.sigma8 = pf.get("sigma8", 0.81);
        r.pk_file = pf.get("pk_file", std::string(""));
        r.box = pf.get("box", 200.0);
        model.OmegaM = r.OmegaM;
        model.OmegaB = r.OmegaB;
        model.gravity_model = pf.get("gravity_model", std::string("LCDM"));
        if (model.gravity_model == "CoupledQuintessence") {
            model.cq_lambda = pf.get("cq_lambda", 0.5);
            model.cq_beta = pf.get("cq_beta", 0.05);
        }
        if (model.gravity_model == "FadingDarkSector") {
            model.fds_c = pf.get("fds_c", -0.85);
            model.fds_cprime = pf.get("fds_cprime", 0.05);
        }
        if (model.covariant_dark_scattering()) {
            model.ds_w = pf.get("ds_w", -0.98);
            model.cds_alpha = pf.get("cds_alpha", 1.0);
        }
        // Transfer functions (transfer_file set): the initial conditions from the transfer functions at zini
        // (transfer_source = tables: a file of FML's transfer tables, e.g. from SimulationCode's nbody --write-transfers
        // with the Einstein-Boltzmann solver; camb: a CAMB transfer info file; class: the root of a CLASS run with mTk).
        // The amplitude is that of the transfer functions (A_s), not pk_file / sigma8. Needed for CDM and baryons as two
        // sets of particles (particle_baryons, Npart_1D^3 of each, from the CDM and baryon transfer functions); with one
        // set of particles the CDM+baryon P(k, zini) of the transfer functions (the same field as with particle_baryons)
        model.baryons = pf.get("particle_baryons", false);
        r.transfer_file = pf.get("transfer_file", std::string(""));
        assert_mpi(not model.baryons or not r.transfer_file.empty(), "particle_baryons = true needs transfer_file");
        // The baryons start from the same lattice as the CDM (same) or one offset by half a particle spacing (offset)
        r.baryon_lattice = model.baryons ? pf.get("baryon_lattice", std::string("same")) : "same";
        assert_mpi(r.baryon_lattice == "same" or r.baryon_lattice == "offset", "baryon_lattice must be same or offset");
        if (not r.transfer_file.empty()) {
            r.transfer_source = pf.get("transfer_source", std::string("tables"));
            if (r.transfer_source != "tables") {
                r.A_s = pf.get("A_s", 2.1e-9);
                r.k_pivot_mpc = pf.get("k_pivot_mpc", 0.05);
            }
            if (r.transfer_source == "class")
                r.class_gauge = pf.get("class_gauge", std::string("synchronous"));
        }
        // The radiation of the background (photons + massless neutrinos). Default 0; with transfer_source = tables the
        // radiation of the tables (T_CMB, N_eff), so that the growth of the simulation is that of the transfer functions
        model.OmegaR = pf.get("OmegaR", -1.0);
        if (model.OmegaR < 0.0) {
            model.OmegaR = 0.0;
            if (r.transfer_source == "tables") {
                const auto tf = FML::INITIALCONDITIONS::read_transfer_tables(r.transfer_file);
                if (tf.contains("cosmology_T_CMB") and tf.contains("cosmology_N_eff") and tf.contains("cosmology_h")) {
                    FML::COSMOLOGY::BACKGROUND::BackgroundParameters bp;
                    bp.h = std::stod(tf.get("cosmology_h"));
                    bp.T_CMB = std::stod(tf.get("cosmology_T_CMB"));
                    bp.N_eff = std::stod(tf.get("cosmology_N_eff"));
                    model.OmegaR = bp.Omega_gamma() + bp.Omega_nu_massless();
                }
            }
        }
        // Coupled quintessence / FDS / covariant dark scattering: the input P(k) is the linear P(k) at z = 0 of the model
        // (ic_growth = model, the default), or of LCDM with the same OmegaM (ic_growth = LCDM: the same initial amplitude
        // as an LCDM run)
        r.ic_growth = ((model.coupled_quintessence() or model.covariant_dark_scattering()) and r.transfer_file.empty())
                          ? pf.get("ic_growth", std::string("model"))
                          : "model";
        assert_mpi(r.ic_growth == "model" or r.ic_growth == "LCDM", "ic_growth must be model or LCDM");
        r.Npart_1D = pf.get("Npart_1D", 64);
        r.Nmesh_ic = pf.get("Nmesh_ic", r.Npart_1D);
        r.LPT_order = pf.get("LPT_order", 2);
        r.seed = (unsigned int)pf.get("seed", 1234);
        // GSL (gsl_rng_ranlxd1, as N-GenIC/2LPTic/L-PICOLA: the same IC for the same seed) or MT19937 (std::mt19937)
        r.random_generator = pf.get("random_generator", std::string("GSL"));
        r.fix_amplitude = pf.get("fix_amplitude", false);
        r.zini = pf.get("zini", 49.0);
        r.nsteps = pf.get("nsteps", 50);
        r.zout = pf.get_list("output_redshifts");
        r.outdir = pf.get("output_folder", std::string("."));
        r.label = pf.get("label", default_label);
        r.pofk_ngrid = pf.get("pofk_ngrid", 2 * r.Npart_1D);
        r.gadget = pf.get("output_gadget", false);
        // A PNG of the particles in a slab through the middle of the box at every output
        r.images = pf.get("output_images", false);
        r.images_npix = pf.get("images_npix", 1024);
        r.images_thickness = pf.get("images_slab_thickness", 0.1);
        r.sigma8_rescale = pf.get("sigma8_rescale", -1.0);
        return r;
    }

    //==========================================================================================
    /// What a run needs around its time stepping: the background (cosmo, and LCDM with the same OmegaM), the input
    /// P(k) or the transfer functions, the initial conditions, the time steps and the outputs
    //==========================================================================================
    class RunSetup {
      public:
        RunSettings s;
        ModelParameters model;
        Cosmology cosmo, lcdm;
        LinearPower pk;
        std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> transfers;
        double aini{0.02};
        double growth_ini{1.0}; // D(zini) / D(0) of the input P(k)

        RunSetup(const RunSettings & settings, const ModelParameters & model_)
            : s(settings), model(model_), cosmo(make_cosmology(model_)), lcdm(model_.OmegaM, model_.OmegaR),
              pk(settings.pk_file,
                 settings.OmegaM,
                 settings.OmegaB,
                 settings.h,
                 settings.ns,
                 settings.pk_file.empty() ? settings.sigma8 : settings.sigma8_rescale) {
            lcdm.OmegaB = s.OmegaB;
            if (not s.transfer_file.empty())
                transfers = make_transfers(s.transfer_source, s.transfer_file, s.class_gauge, model, s.h, s.ns, s.A_s, s.k_pivot_mpc);
            aini = 1.0 / (1.0 + s.zini);
            const Cosmology & ic_cosmo = s.ic_growth == "LCDM" ? lcdm : cosmo;
            growth_ini = ic_cosmo.D1(aini) / ic_cosmo.D1(1.0);
            // The outputs from the first to the last
            if (s.zout.empty())
                s.zout.push_back(0.0);
            std::sort(s.zout.begin(), s.zout.end(), std::greater<double>());
        }

        /// D(a) / D(1) of the input P(k): the model's growth from the initial amplitude
        double growth_of_a(double a) const { return growth_ini * cosmo.D1(a) / cosmo.D1(aini); }

        /// The model (CQ/FDS, CDS) on task 0
        void print_model() const {
            if (not((model.coupled_quintessence() or model.covariant_dark_scattering()) and FML::ThisTask == 0))
                return;
            if (model.coupled_quintessence())
                std::cout << "\n" << model.gravity_model << ": lambda = " << cosmo.cq_lambda << ", beta = " << cosmo.cq_beta;
            else
                std::cout << "\n" << model.gravity_model << ": w = " << cosmo.ds_w << ", alpha = " << cosmo.cds_alpha
                          << " (CDM drag alpha a^3 / (Omega_c0 E); the dark energy at rest)";
            std::cout << " (OmegaM = " << s.OmegaM << ", OmegaB = " << s.OmegaB << ")\n";
            for (double a : {aini, 0.25, 0.5, 1.0})
                std::cout << "  a = " << std::setw(8) << a << ": E / E_LCDM = " << std::setw(10) << cosmo.E(a) / lcdm.E(a)
                          << "  dphi/dlna = " << std::setw(11) << cosmo.dphidlna(a) << "  G_eff rho_m a^3 / (G OmegaM) = "
                          << std::setw(10) << cosmo.poisson_factor(a) << "  Gamma = " << std::setw(11) << cosmo.friction(a)
                          << "  D / D_LCDM (equal at zini) = " << cosmo.D1(a) / cosmo.D1(aini) * lcdm.D1(aini) / lcdm.D1(a)
                          << "\n";
            std::cout << "  input P(k) back-scaled with the growth of " << (s.ic_growth == "LCDM" ? "LCDM" : "the model")
                      << ": D(zini) / D(0) = " << growth_ini << "\n";
        }

        /// The initial conditions (on the FFTW slab domain): 1LPT/2LPT from the input P(k) back-scaled to zini or from
        /// the CDM+baryon P(k, zini) of the transfer functions; CDM and baryons from their transfer functions
        template <class P>
        void initial_conditions(FML::PARTICLE::MPIParticles<P> & part) const {
            std::unique_ptr<FML::RANDOM::RandomGenerator> rng;
            if (s.random_generator == "GSL")
                rng = std::make_unique<FML::RANDOM::GSLRandomGenerator>();
            else if (s.random_generator == "MT19937")
                rng = std::make_unique<FML::RANDOM::RandomGenerator>();
            else
                throw std::invalid_argument("random_generator must be GSL or MT19937");
            rng->set_seed(s.seed);
            const double box = s.box;
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
            if (model.baryons)
                two_species_initial_conditions(part, *transfers, cosmo, rng.get(), s.Npart_1D, s.Nmesh_ic, s.fix_amplitude,
                                               s.LPT_order, box, s.zini, s.baryon_lattice == "offset");
            else
                FML::NBODY::NBodyInitialConditions<3, P>(part,
                                                         s.Npart_1D,
                                                         1.5,
                                                         s.Nmesh_ic,
                                                         s.fix_amplitude,
                                                         rng.get(),
                                                         one,
                                                         Pofk_over_volume,
                                                         s.LPT_order,
                                                         "gaussian",
                                                         0.0,
                                                         box,
                                                         s.zini,
                                                         velocity_norms);
            if (FML::ThisTask == 0 and transfers)
                std::cout << "\n"
                          << (model.baryons ? "CDM and baryons as two sets of particles (" + std::to_string(s.Npart_1D) + "^3 of each)"
                                            : std::string("CDM+baryons as one set of particles"))
                          << ", initial conditions from the transfer functions: " << s.transfer_source << " ("
                          << s.transfer_file << ")\n";
        }

        /// The coarse steps: uniform in log(a) from aini to the last output (nsteps), plus the output times
        std::vector<double> step_times() const {
            const double aend = 1.0 / (1.0 + s.zout.back());
            std::vector<double> times;
            for (int i = 0; i <= s.nsteps; i++)
                times.push_back(aini * std::pow(aend / aini, double(i) / s.nsteps));
            for (auto z : s.zout)
                times.push_back(1.0 / (1.0 + z));
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end(), [](double x, double y) { return std::abs(x / y - 1.0) < 1e-6; }),
                        times.end());
            return times;
        }

        /// P(k) at a (file suffix zs): of all the particles, or of the CDM, the baryons and the CDM+baryons
        template <class P>
        void write_power_spectra(FML::PARTICLE::MPIParticles<P> & part, const std::string & zs, double a) const {
            const auto & outdir = s.outdir;
            const auto & label = s.label;
            if (model.baryons)
                output_power_spectra_two_species(outdir, label, zs, part, s.pofk_ngrid, s.box, a, cosmo.cdm_fraction(a), *transfers);
            else if (transfers)
                power_spectrum_of<::SimpleParticle<3>, P>(
                    outdir + "/pofk_" + label + "_z" + zs + ".txt", part, s.pofk_ngrid, s.box, a, nullptr, [&](double k) {
                        return transfers->power(FML::INITIALCONDITIONS::Species::CDMBaryon, k, a);
                    });
            else
                output_power_spectrum(outdir + "/pofk_" + label + "_z" + zs + ".txt", part, s.pofk_ngrid, s.box, a, pk, growth_of_a(a));
        }

        /// The outputs at a (P(k), GADGET files, an image) for the output redshift string zs
        template <class P>
        void write_outputs(FML::PARTICLE::MPIParticles<P> & part, const std::string & zs, double a) const {
            const auto & outdir = s.outdir;
            const auto & label = s.label;
            write_power_spectra(part, zs, a);
            if (s.gadget and model.baryons) {
                // One file set per species (the CDM mass today; with coupled quintessence it changes with time)
                output_gadget(outdir + "/gadget_cdm_" + label + "_z" + zs, part, a, s.box, s.OmegaM - s.OmegaB, s.h, 0, 1.0 - s.OmegaM);
                output_gadget(outdir + "/gadget_baryon_" + label + "_z" + zs, part, a, s.box, s.OmegaB, s.h, 1, 1.0 - s.OmegaM);
            } else if (s.gadget)
                output_gadget(outdir + "/gadget_" + label + "_z" + zs, part, a, s.box, s.OmegaM, s.h);
            if (s.images) {
                const auto image = FML::VISUALIZATION::project_particles(
                    part.get_particles_ptr(), part.get_npart(), s.images_npix, 2, 0.5 - 0.5 * s.images_thickness, s.images_thickness);
                if (FML::ThisTask == 0)
                    FML::VISUALIZATION::write_png(outdir + "/image_density_" + label + "_z" + zs + ".png",
                                                  image,
                                                  "magma",
                                                  FML::VISUALIZATION::ColorScale::automatic_log(1.0, 99.9));
            }
        }
    };

} // namespace FML::NBODY::TREEPM
#endif

//==========================================================================================
//
// TreePMCode: a simple TreePM N-body code (GADGET-style) for CDM, or CDM and baryons (see TreePM.h and README.md)
//
//   1. 2LPT initial conditions from a linear P(k) at z = 0 (a CAMB-format file, or Eisenstein & Hu normalized to
//      sigma8) or from transfer functions (CDM and baryons as two sets of particles)
//   2. Steps uniform in log(a) (plus the output times): kick-drift-kick, with the domain rebuilt every step
//   3. Outputs: P(k) (and optionally GADGET snapshots and images) at the output redshifts
//
// Run: mpirun -np 4 ./treepm parameterfile.txt
//
//==========================================================================================

#include "Setup.h"
#include "TreePM.h"

#include <chrono>
#include <cstdio>

using namespace FML::NBODY::TREEPM;

int main(int argc, char ** argv) {
    if (argc < 2) {
        if (FML::ThisTask == 0)
            std::cout << "Usage: mpirun -np N " << argv[0] << " parameterfile.txt\n";
        return 1;
    }
    ParameterFile pf(argv[1]);

    // The cosmology, the model, the initial conditions, the steps and the outputs (Setup.h), then the force
    TreePMParameters par;
    const auto settings = read_run_settings(pf, par, "treepm");
    par.Npart_1D = settings.Npart_1D;
    par.boxsize = settings.box;
    par.Nmesh = pf.get("Nmesh", 2 * settings.Npart_1D);
    par.assignment = pf.get("assignment", std::string("CIC"));
    par.deconvolve = pf.get("deconvolve", true);
    par.asmth = pf.get("asmth", 1.25);
    par.softening = pf.get("softening", 1.0 / 30.0);
    par.cutoff_tolerance = pf.get("cutoff_tolerance", 1e-4);
    par.backend = pf.get("backend", std::string("tree"));
    par.opening_angle = pf.get("opening_angle", 0.5);
    par.leaf_size = pf.get("leaf_size", 8);
    par.opening_criterion = pf.get("opening_criterion", std::string("geometric"));
    assert_mpi(par.opening_criterion == "geometric" or par.opening_criterion == "relative",
               "opening_criterion must be geometric or relative");
    par.relative_tolerance = pf.get("relative_tolerance", 0.005);
    par.hilbert_bits = pf.get("hilbert_bits", 21);
    // Time steps: fixed (nsteps uniform in log a, plus the outputs) or gadget (GADGET-2: the PM step from
    // max_rms_displacement (MaxRMSDisplacementFac) and max_dlna (MaxSizeTimestep, in log a), individual short-range
    // steps sqrt(2 eta eps / |a|) (eta: ErrTolIntAccuracy) in at most max_bins power-of-two bins of the PM step)
    par.timestep_method = pf.get("timestep_method", std::string("fixed"));
    if (par.gadget()) {
        par.eta = pf.get("eta", 0.025);
        par.max_rms_displacement = pf.get("max_rms_displacement", 0.25);
        par.max_dlna = pf.get("max_dlna", 0.025);
        par.max_bins = pf.get("max_bins", 20);
    }
    pf.warn_unused();

    const RunSetup setup(settings, par);
    const auto & s = setup.s;
    setup.print_model();

    // Initial conditions (made on the FFTW slab domain, then moved to the Peano-Hilbert domain)
    FML::PARTICLE::MPIParticles<TreePMParticle> part;
    setup.initial_conditions(part);
    for (size_t i = 0; i < part.get_npart(); i++)
        for (int d = 0; d < 3; d++)
            part[i].Acc[d] = 0.0;
    if (FML::ThisTask == 0) {
        const double cell = s.box / par.Nmesh;
        std::cout << "\nTreePM (" << par.gravity_model << "): " << s.Npart_1D << "^3 particles"
                  << (par.baryons ? " of each species" : "") << ", box " << s.box << " Mpc/h, PM grid " << par.Nmesh
                  << "^3, split r_s = " << par.asmth * cell << " Mpc/h, softening (Plummer) "
                  << par.softening * s.box / s.Npart_1D << " Mpc/h, " << par.backend << ", z = " << s.zini << " -> "
                  << s.zout.back() << "\n";
    }

    const auto times = setup.step_times();
    TreePMSimulation sim(par, part, setup.aini);
    setup.write_power_spectra(part, std::to_string(s.zini).substr(0, 6), setup.aini);

    size_t iout = 0;
    for (size_t istep = 1; par.gadget() or istep < times.size(); istep++) {
        // The next step: the list (fixed), or the PM step of GADGET-2 up to the next output (gadget)
        double anew;
        std::string limit;
        if (par.gadget()) {
            if (iout >= s.zout.size())
                break;
            const double aout = 1.0 / (1.0 + s.zout[iout]);
            anew = sim.a * std::exp(sim.gadget_pm_timestep(&limit));
            if (anew >= aout * (1.0 - 1e-8)) {
                anew = aout;
                limit += ", then the output";
            }
        } else {
            anew = times[istep];
        }
        const double aold = sim.a;
        const auto t0 = std::chrono::steady_clock::now();
        sim.step(anew);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (FML::ThisTask == 0) {
            if (par.gadget())
                std::cout << "\nPM step " << istep << ": a = " << sim.a << " (z = " << 1.0 / sim.a - 1.0
                          << "), dlog(a) = " << std::log(anew / aold) << " [" << limit << "], " << dt << " s\n";
            else
                std::cout << "\nStep " << istep << " / " << times.size() - 1 << ": a = " << sim.a << " (z = " << 1.0 / sim.a - 1.0
                          << "), " << dt << " s\n";
        }
        sim.print_status(std::cout);

        while (iout < s.zout.size() and std::abs(sim.a * (1.0 + s.zout[iout]) - 1.0) < 1e-6) {
            char zs[32];
            std::snprintf(zs, sizeof(zs), "%.3f", s.zout[iout]);
            setup.write_outputs(part, zs, sim.a);
            if (FML::ThisTask == 0)
                std::cout << "  output at z = " << zs << "\n";
            iout++;
        }
    }
    return 0;
}

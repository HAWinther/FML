#include "../src/Models.h"
#include "../src/Particles.h"
#include <FML/tests/TestHelpers.h>
#include <iostream>
using FML::TESTS::check;
using FML::TESTS::throws;
using ParameterMap = FML::UTILS::ParameterMap;
using Model = GravityModel<3>;
using Grid = FML::GRID::FFTWGrid<3>;
ParameterMap parameters() {
    ParameterMap p;
    p["cosmology_OmegaCDM"] = .25;
    p["cosmology_Omegab"] = .05;
    p["cosmology_OmegaMNu"] = 0.;
    p["cosmology_neutrino_masses_eV"] = std::vector<double>{};
    p["cosmology_OmegaK"] = 0.;
    p["cosmology_h"] = .7;
    p["cosmology_As"] = 2.1e-9;
    p["cosmology_ns"] = .96;
    p["cosmology_kpivot_mpc"] = .05;
    p["cosmology_Neffective"] = 3.046;
    p["cosmology_TCMB_kelvin"] = 2.7255;
    p["ic_initial_redshift"] = 19.;
    p["ic_type_of_input"] = std::string("powerspectrum");
    p["force_use_finite_difference_force"] = false;
    p["force_finite_difference_stencil_order"] = 4;
    p["gravity_model_screening"] = false;
    p["gravity_model_screening_enforce_largescale_linear"] = false;
    p["gravity_model_screening_linear_scale_hmpc"] = .1;
    p["gravity_model_screening_efficiency"] = 1.;
    p["gravity_model_fofr_fofr0"] = 1e-5;
    p["gravity_model_fofr_nfofr"] = 1.;
    p["gravity_model_fofr_exact_solution"] = false;
    p["gravity_model_dgp_rcH0overc"] = 1.;
    p["gravity_model_dgp_exact_solution"] = false;
    p["gravity_model_symmetron_assb"] = .5;
    p["gravity_model_symmetron_beta"] = .7;
    p["gravity_model_symmetron_L_mpch"] = 2.;
    p["gravity_model_symmetron_exact_solution"] = false;
    p["gravity_model_mbeta_params"] = std::vector<double>{.4, 0., 100., -2.};
    p["gravity_model_mu_parametrization"] = std::string("BZ");
    p["gravity_model_mu_BZ_params"] = std::vector<double>{4. / 3., 8., 4.};
    return p;
}
// Only the pre-existing Geff interface is overridden: the new extension must work without any new override.
class FutureModel : public Model {
  public:
    explicit FutureModel(std::shared_ptr<Cosmology> c) : Model(c, "future") {}
    double GeffOverG(double a, double k = 0) const override {
        const double x = k * .002;
        const double ratio = x > 1 ? 1 / (1 + 1 / x / x) : x * x / (1 + x * x);
        return 1 + .4 * a * ratio + .15 * a * std::exp(-x * x);
    }
};
void check_modes(Model & model, bool generic) {
    Grid density(8);
    density.set_grid_status_real(false);
    for (auto i : density.get_fourier_range())
        density.set_fourier_from_index(i, 1.);
    SIMCODE::ShortRangeSettings settings;
    settings.enabled = true;
    settings.split_scale = 2.5;
    settings.softening_support = .1;
    settings.table_points = 0;
    FML::NBODY::ForceOptions options;
    std::array<Grid, 3> force;
    const double a = .8, box = 200., H0Box = model.H0_hmpc * box;
    const auto pair = model.short_range_interaction(a);
    model.compute_force_for_step(a, H0Box, density, "CIC", force, options, settings);
    bool unchanged = true;
    for (auto i : density.get_fourier_range())
        unchanged &= density.get_fourier_from_index(i) == FML::GRID::ComplexType(1.);
    check(unchanged, "model split preserves original Fourier density: " + model.get_name());
    for (auto & f : force)
        f.fftw_r2c();
    double error = 0;
    for (auto i : density.get_fourier_range()) {
        const auto k = density.get_fourier_wavevector_from_index(i);
        double k2 = 0;
        for (int d = 0; d < 3; ++d)
            k2 += k[d] * k[d];
        const double rs = settings.split_scale / box, gaussian = std::exp(-k2 * rs * rs);
        double multiplier;
        if (generic)
            multiplier = model.GeffOverG(a, std::sqrt(k2) / H0Box) - pair.muinf * (1 - gaussian);
        else if (pair.range_mpch == 0)
            multiplier = pair.muinf * gaussian;
        else {
            const double mass = box / pair.range_mpch;
            multiplier = gaussian * (pair.mu0 + (pair.muinf - pair.mu0) * k2 / (k2 + mass * mass) *
                                                    std::exp(-mass * mass * rs * rs));
        }
        for (int d = 0; d < 3; ++d) {
            const double kd = std::abs(k[d]) >= 8 * M_PI - 1e-12 ? 0 : k[d];
            const std::complex<double> expected =
                k2 == 0 ? std::complex<double>(0) :
                          std::complex<double>(0, -kd * 1.5 * model.get_cosmo()->get_OmegaM() * a * multiplier / k2);
            const auto actual = std::complex<double>(force[d].get_fourier_from_index(i));
            error = std::max(error, std::abs(actual - expected));
        }
    }
    FML::MaxOverTasks(&error);
    if (FML::ThisTask == 0)
        std::cout << "MODE_ERROR " << model.get_name() << " " << error << "\n";
    check(error < 2e-7, "matched long-force Fourier coefficients (no duplicate Geff): " + model.get_name());
}
// Independent Newtonian erfc force checks the model-owned kick normalization and sign.
void check_particle_kick(std::shared_ptr<Cosmology> c) {
    struct Constant : Model {
        explicit Constant(std::shared_ptr<Cosmology> c) : Model(c, "constant") {}
        double GeffOverG(double a, double = 0) const override { return 1 + .4 * a; }
    } model(c);
    using P = ParticleWithMass<ParticleBasic<3>>;
    std::vector<P> all(2);
    for (int i = 0; i < 2; ++i) {
        all[i].id = i + 1;
        all[i].mass = i + 1;
        for (int d = 0; d < 3; ++d) {
            all[i].pos[d] = d == 0 ? .49 + .02 * i : .5;
            all[i].vel[d] = 0;
        }
    }
    FML::PARTICLE::MPIParticles<P> particles;
    particles.create(all, 34, [](P & p) { return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain; });
    SIMCODE::ShortRangeSettings settings{true, "cell", 2.5, .1, 1e-8, .3, 0};
    const double a = .8, box = 200, dt = .07, r = .02, x = r / (2 * settings.split_scale / box);
    const double window = std::erfc(x) + 2 * x / std::sqrt(M_PI) * std::exp(-x * x);
    const double factor = dt * 1.5 * c->get_OmegaM() * a * model.GeffOverG(a) * window / (4 * M_PI * 3 * r * r);
    model.kick_short_range(particles, settings, a, box, dt);
    bool ok = true;
    auto * p = particles.get_particles_ptr();
    for (size_t i = 0; i < particles.get_npart(); ++i) {
        const double expected = p[i].id == 1 ? 2 * factor : -factor;
        ok &= std::abs(p[i].vel[0] - expected) < 1e-11;
        ok &= p[i].vel[1] == 0 && p[i].vel[2] == 0;
    }
    check(ok, "inherited model-owned particle kick matches independent erfc force and global mass normalization");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    auto p = parameters();
    auto c = SIMCODE::make_cosmology("LCDM");
    FML::TESTS::quietly([&] {
        c->read_parameters(p);
        c->init();
    });
    check_particle_kick(c);
    FutureModel future(c);
    check(future.GeffOverG_highk(.8) == future.GeffOverG(.8, Model::short_range_fiducial_koverH0),
          "new model inherits large finite-k fiducial from GeffOverG");
    check(future.short_range_comoving_range(.8) == 0, "new model inherits explicit infinite-range convention");
    check_modes(future, true);
    for (const std::string name : {"GR", "f(R)", "DGP", "Symmetron", "mbeta", "mu"}) {
        auto g = SIMCODE::make_gravity_model<3>(name, c);
        FML::TESTS::quietly([&] {
            g->read_parameters(p);
            g->init();
        });
        bool matches = true;
        for (double a : {.3, .5, .8, 1.}) {
            auto pair = g->short_range_interaction(a);
            for (double k : {0., .001, .01, .1, 1., 100.}) {
                const double truth = g->GeffOverG(a, k / g->H0_hmpc),
                             actual = pair.fourier_multiplier(k * k * 200. * 200., 200.);
                matches &= std::abs(actual - truth) < 3e-12 * std::max(1., std::abs(truth));
            }
            const double high = g->GeffOverG(a, 1e10);
            matches &= std::abs(pair.muinf - high) < 1e-9 * std::max(1., std::abs(high));
        }
        check(matches, "analytic limits/range reproduce existing response across epochs and k: " + name);
        check_modes(*g, false);
        if (name == "f(R)") {
            auto pair = g->short_range_interaction(.8);
            check(std::abs(pair.muinf / pair.mu0 - 4. / 3.) < 1e-14 && pair.range_mpch > 0,
                  "f(R) retains background prefactor and one-third Yukawa enhancement");
            auto kernel = pair.kernel(SIMCODE::ShortRangeSettings{true, "cell", 2.5, .1, 1e-6, .3, 0}, 200.);
            const double r = .02, x = r / (pair.range_mpch / 200.);
            const double exact = (pair.mu0 + (pair.muinf - pair.mu0) * (1 + x) * std::exp(-x)) / (r * r * r);
            check(std::abs(kernel.full_coefficient(r) - exact) < 1e-10 * std::abs(exact),
                  "Yukawa force differentiates potential including (1+mr)");
        }
    }
    auto j = p;
    j["cosmology_JBD_wBD"] = 100.;
    j["cosmology_JBD_GeffG_today"] = 1.;
    j["cosmology_JBD_density_parameter_definition"] = std::string("hi-class");
    auto jc = SIMCODE::make_cosmology("JBD");
    auto jg = SIMCODE::make_gravity_model<3>("JBD", jc);
    FML::TESTS::quietly([&] {
        jc->read_parameters(j);
        jc->init();
        jg->read_parameters(j);
        jg->init();
    });
    check(jg->GeffOverG_highk(.8) == jg->GeffOverG(.8, 0) && jg->short_range_comoving_range(.8) == 0,
          "JBD uses its exact scale-independent coupling and infinite range");
    check_modes(*jg, false);
    struct Bad : FutureModel {
        using FutureModel::FutureModel;
        double GeffOverG_highk(double a) const override {
            return FML::ThisTask == 0 ? NAN : FutureModel::GeffOverG_highk(a);
        }
    } bad(c);
    check(throws([&] { bad.checked_short_range_interaction(.8); }), "one-rank invalid model limit fails collectively");
    auto screened = p;
    screened["gravity_model_dgp_smoothing_filter"] = std::string("gaussian");
    screened["gravity_model_dgp_smoothing_scale_over_boxsize"] = 0.;
    screened["gravity_model_screening"] = true;
    auto sg = SIMCODE::make_gravity_model<3>("DGP", c);
    sg->read_parameters(screened);
    check(throws([&] { sg->validate_short_range_support(); }),
          "screened forces remain explicitly outside the linear pair extension");
    return FML::TESTS::summary("GravityModel short-range interaction");
}

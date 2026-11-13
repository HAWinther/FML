#include <FML/Hydrodynamics/FiniteVolume.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace H = FML::HYDRODYNAMICS;
namespace C = FML::COMMUNICATION;
namespace FS = std::filesystem;

template <int D>
int experiment(const FS::path & output, const std::string & mode, int n, double end, int order, H::FluxMethod method) {
    H::Options<D> o;
    o.shape.fill(n);
    o.gamma = 1.4;
    o.order = order;
    o.flux = method;
    const bool adv = mode == "advect", sound = mode == "sound", blast = mode == "blast";
    if (adv) {
        if constexpr (D > 1)
            o.shape[1] = n / 2;
        if constexpr (D > 2)
            o.shape[2] = n / 2;
    }
    if (!adv && !sound && !blast)
        for (auto & bc : o.boundaries)
            bc = {H::Boundary::Outflow, H::Boundary::Outflow};
    H::Vector<D> velocity{};
    for (int d = 0; d < D; ++d)
        velocity[d] = d == 0 ? .7 : d == 1 ? .3 : .2;
    const double amplitude = sound ? 1e-4 : .2;
    double sinc = 1;
    for (int d = 0; d < (adv ? D : 1); ++d)
        sinc *= std::sin(M_PI / o.shape[d]) / (M_PI / o.shape[d]);
    auto initial = [&](const H::Vector<D> & x) {
        H::Primitive<D> q;
        if (adv || sound) {
            double phase = x[0];
            if (adv)
                for (int d = 1; d < D; ++d)
                    phase += x[d];
            double wave = amplitude * sinc * std::sin(2 * M_PI * phase);
            q.density = 1 + wave;
            q.velocity = adv ? velocity : H::Vector<D>{};
            q.pressure = sound ? (1 + o.gamma * wave) / o.gamma : 1;
            if (sound)
                q.velocity[0] = wave;
        } else if (blast) {
            double r2 = 0;
            for (auto y : x)
                r2 += (y - .5) * (y - .5);
            q.pressure = r2 < .12 * .12 ? 10 : .1;
        } else if (mode == "sod") {
            q.density = x[0] < .5 ? 1 : .125;
            q.pressure = x[0] < .5 ? 1 : .1;
        } else if (mode == "lax") {
            q.density = x[0] < .5 ? .445 : .5;
            q.pressure = x[0] < .5 ? 3.528 : .571;
            q.velocity[0] = x[0] < .5 ? .698 : 0;
        } else if (mode == "strong") {
            q.pressure = x[0] < .5 ? 1000 : .01;
        } else if (mode == "rarefaction") {
            q.pressure = .4;
            q.velocity[0] = x[0] < .5 ? -2 : 2;
        } else if (mode == "vacuum") {
            q.pressure = 1;
            q.velocity[0] = x[0] < .5 ? -8 : 8;
        } else
            throw std::invalid_argument("Unknown initial condition");
        return q;
    };
    H::FiniteVolume<D> solver(o, initial);
    C::collective_try(
        [&] {
            if (FML::ThisTask == 0) {
                if (FS::exists(output))
                    throw std::invalid_argument("Use a fresh output directory");
                FS::create_directories(output);
            }
        },
        "Hydro output directory");
    std::ofstream history;
    C::collective_try(
        [&] {
            if (FML::ThisTask == 0) {
                history.open(output / "history.tsv");
                history << std::setprecision(17)
                        << "# time mass total_energy thermal_energy momentum[D] signed_energy_repair "
                           "absolute_energy_repair low_order_stages retries\n";
                if (!history)
                    throw std::runtime_error("History open failed");
            }
        },
        "Hydro history");
    auto before = solver.diagnostics();
    auto record = [&] {
        const auto d = solver.diagnostics();
        if (FML::ThisTask == 0) {
            const auto & s = solver.statistics();
            history << solver.time() << ' ' << d.mass << ' ' << d.energy << ' ' << d.thermal;
            for (auto v : d.momentum)
                history << ' ' << v;
            history << ' ' << s.energy_repair << ' ' << s.absolute_energy_repair << ' ' << s.low_order_stages << ' '
                    << s.rejected_attempts << '\n';
        }
    };
    record();
    int steps = 0;
    while (solver.time() < end) {
        const double remaining = end - solver.time();
        if (remaining <= 8 * std::numeric_limits<double>::epsilon() * end)
            break;
        solver.advance(remaining);
        if (++steps % 10 == 0)
            record();
        if (steps > 200000)
            throw std::runtime_error("Step limit");
    }
    record();
    auto fields = solver.primitive_cells();
    auto after = solver.diagnostics();
    double rho_error = 0, v_error = 0, minrho = std::numeric_limits<double>::infinity(), minpressure = minrho;
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto x = solver.layout().center(i);
        const auto & q = fields[i];
        minrho = std::min(minrho, q.density);
        minpressure = std::min(minpressure, q.pressure);
        if (adv || sound) {
            double phase = x[0], speed = 1;
            if (adv) {
                speed = velocity[0];
                for (int d = 1; d < D; ++d) {
                    phase += x[d];
                    speed += velocity[d];
                }
            }
            const double wave = amplitude * sinc * std::sin(2 * M_PI * (phase - speed * solver.time()));
            rho_error += std::abs(q.density - 1 - wave) * solver.layout().cell_volume() / amplitude;
            if (sound)
                v_error += std::abs(q.velocity[0] - wave) * solver.layout().cell_volume() / amplitude;
        }
    }
    FML::SumOverTasks(&rho_error);
    FML::SumOverTasks(&v_error);
    FML::MinOverTasks(&minrho);
    FML::MinOverTasks(&minpressure);
    auto s = solver.statistics();
    C::collective_try(
        [&] {
            std::ofstream file(output / ("cells.rank" + std::to_string(FML::ThisTask) + ".tsv"));
            file << std::setprecision(17)
                 << "# global_index center[D] rho velocity[D] pressure U[rho,momentum[D],total_energy,aux_internal]\n";
            for (size_t i = 0; i < fields.size(); ++i) {
                file << solver.layout().global_index(i);
                for (auto x : solver.layout().center(i))
                    file << ' ' << x;
                file << ' ' << fields[i].density;
                for (auto x : fields[i].velocity)
                    file << ' ' << x;
                file << ' ' << fields[i].pressure;
                for (auto x : solver.cells()[i])
                    file << ' ' << x;
                file << '\n';
            }
            file.close();
            if (!file)
                throw std::runtime_error("Cell output failed");
            if (FML::ThisTask == 0) {
                history.close();
                if (!history)
                    throw std::runtime_error("History output failed");
                std::ofstream meta(output / "meta.tsv");
                meta << std::setprecision(17) << "dimension " << D << "\nn " << n << "\ngamma " << o.gamma << "\norder "
                     << o.order << "\nflux " << int(o.flux) << "\nsteps " << steps << "\ntime " << solver.time()
                     << "\namplitude " << amplitude << "\nsinc " << sinc << "\ndensity_l1_over_amplitude " << rho_error
                     << "\nvelocity_l1_over_amplitude " << v_error << "\ninitial_mass " << before.mass
                     << "\nfinal_mass " << after.mass << "\ninitial_energy " << before.energy << "\nfinal_energy "
                     << after.energy << "\nminimum_density " << minrho << "\nminimum_pressure " << minpressure
                     << "\nlow_order_stages " << s.low_order_stages << "\nretries " << s.rejected_attempts
                     << "\nriemann_fallbacks " << s.riemann_fallbacks << "\nenergy_repair " << s.energy_repair << '\n';
                for (int d = 0; d < D; ++d)
                    meta << "shape" << d << ' ' << o.shape[d] << '\n';
                meta.close();
                if (!meta)
                    throw std::runtime_error("Metadata output failed");
                std::cout << "HYDRO " << mode << " D=" << D << " n=" << n << " steps=" << steps
                          << " rho_L1=" << rho_error << " velocity_L1=" << v_error << " minrho=" << minrho
                          << " minp=" << minpressure << " energy_change=" << after.energy - before.energy
                          << " fallback=" << s.riemann_fallbacks << " retries=" << s.rejected_attempts << '\n';
            }
        },
        "Hydro output files");
    return 0;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(17);
    try {
        if (argc < 5 || argc > 7)
            throw std::invalid_argument("Usage: hydro_example OUTPUT "
                                        "advect1d|advect2d|advect3d|sound|sod|lax|strong|rarefaction|vacuum|blast2d|"
                                        "blast3d N END_TIME [ORDER] [rusanov|hlle|hllc]");
        const FS::path output = argv[1];
        std::string mode = argv[2];
        int n = std::stoi(argv[3]), order = argc > 5 ? std::stoi(argv[5]) : 2;
        double end = std::stod(argv[4]);
        std::string flux = argc > 6 ? argv[6] : "hllc";
        if (n < 8 || n > 1000000 || !std::isfinite(end) || end <= 0)
            throw std::invalid_argument("Invalid problem controls");
        H::FluxMethod method = flux == "hllc" ? H::FluxMethod::HLLC :
                               flux == "hlle" ? H::FluxMethod::HLLE :
                                                H::FluxMethod::Rusanov;
        if (flux != "hllc" && flux != "hlle" && flux != "rusanov")
            throw std::invalid_argument("Unknown Riemann solver");
        if (mode == "advect2d" || mode == "blast2d") {
            if (n > 1024)
                throw std::invalid_argument("2D example size cap");
            return experiment<2>(output, mode == "advect2d" ? "advect" : "blast", n, end, order, method);
        }
        if (mode == "advect3d" || mode == "blast3d") {
            if (n > 128)
                throw std::invalid_argument("3D example size cap");
            return experiment<3>(output, mode == "advect3d" ? "advect" : "blast", n, end, order, method);
        }
        return experiment<1>(output, mode == "advect1d" ? "advect" : mode, n, end, order, method);
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
}

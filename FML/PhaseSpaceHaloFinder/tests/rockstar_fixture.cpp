#include <FML/PhaseSpaceHaloFinder/PhaseSpaceHaloFinder.h>
#include <FML/PhaseSpaceHaloFinder/tests/Plummer.h>
#include <FML/Units/Units.h>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    if (argc != 2)
        return 1;
    const std::string directory = argv[1];
    using namespace FML::PHASE_SPACE_HALOS;
    namespace SI = FML::UTILS::SI;
    const double G = SI::G * SI::Msun / (SI::Mpc * std::pow(SI::km / SI::s, 2)), total_mass = .3 * 2.77536627e11;
    auto p = phase_test::plummer(4000);
    for (auto & x : p) {
        x.mass *= total_mass;
        for (auto & v : x.velocity)
            v *= std::sqrt(G * total_mass);
    }
    std::ofstream rock(directory + "/plummer.ascii"), fml(directory + "/plummer.0.txt");
    if (!rock || !fml)
        return 2;
    rock << std::setprecision(17);
    fml << std::setprecision(17);
    for (const auto & x : p) {
        for (auto v : x.position)
            rock << v << ' ';
        for (auto v : x.velocity)
            rock << v << ' ';
        rock << x.id << '\n';
        fml << x.id;
        for (auto v : x.position)
            fml << ' ' << v;
        for (auto v : x.velocity)
            fml << ' ' << v;
        fml << ' ' << x.mass << '\n';
    }
    Options o;
    o.gravitational_constant = G;
    o.softening_support = .0005;
    o.potential_opening_angle = .2;
    auto c = find(p, o);
    std::cout << std::setprecision(17) << "G=" << G << " Mtotal=" << total_mass
              << " particle_mass=" << total_mass / p.size() << " halos=" << c.halos.size() << '\n';
    std::ofstream summary(directory + "/fml_summary.txt");
    summary << std::setprecision(17) << "# id mass_exclusive_bound vmax x y z np\n";
    for (const auto & h : c.halos)
        summary << h.id << ' ' << h.bound_mass << ' ' << h.vmax << ' ' << h.position[0] << ' ' << h.position[1] << ' '
                << h.position[2] << ' ' << h.particle_ids.size() << '\n';
    for (const auto & h : c.halos)
        std::cout << "FML id=" << h.id << " mass=" << h.bound_mass << " vmax=" << h.vmax
                  << " np=" << h.particle_ids.size() << " x=" << h.position[0] << " y=" << h.position[1]
                  << " z=" << h.position[2] << '\n';
}

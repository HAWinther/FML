#include <FML/PhaseSpaceHaloFinder/Distributed.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
int main(int argc, char ** argv) {
    using namespace FML::PHASE_SPACE_HALOS;
    if (argc != 7) {
        if (FML::ThisTask == 0)
            std::cerr
                << "Usage: phase_halos input_prefix output_prefix boxsize softening_support G position_to_physical\n"
                << "Read input_prefix.rank.txt: id x y z vx vy vz mass; peculiar velocities and already migrated x "
                   "slabs.\n";
        return 1;
    }
    Options o;
    std::vector<Particle> particles;
    bool valid = true;
    try {
        o.boxsize = std::stod(argv[3]);
        o.softening_support = std::stod(argv[4]);
        o.gravitational_constant = std::stod(argv[5]);
        o.position_to_physical = std::stod(argv[6]);
        o.validate();
        std::ifstream in(std::string(argv[1]) + "." + std::to_string(FML::ThisTask) + ".txt");
        if (!in)
            throw std::runtime_error("Cannot open particle input");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream row(line);
            Particle p;
            if (!(row >> p.id >> p.position[0] >> p.position[1] >> p.position[2] >> p.velocity[0] >> p.velocity[1] >>
                  p.velocity[2] >> p.mass))
                throw std::runtime_error("Invalid input row");
            std::string extra;
            if (row >> extra)
                throw std::runtime_error("Trailing input columns");
            particles.push_back(p);
        }
    } catch (const std::exception & e) {
        std::cerr << "rank " << FML::ThisTask << ": " << e.what() << '\n';
        valid = false;
    }
    int failed = !valid;
    FML::MaxOverTasks(&failed);
    if (failed)
        return 1;
    try {
        const auto c = find_distributed(particles, o);
        const auto name = std::string(argv[2]) + "." + std::to_string(FML::ThisTask);
        std::ofstream halos(name + ".halos.txt"), members(name + ".members.txt");
        if (!halos || !members)
            throw std::runtime_error("Cannot open output");
        halos << std::setprecision(17)
              << "# id fof_id host_id exclusive_bound_mass inclusive_bound_mass bound_radius vmax rmax x y z vx_core "
                 "vy_core vz_core\n";
        halos << "# lengths/masses/velocities follow input; bound_radius is not R200; G=" << o.gravitational_constant
              << " physical_length_factor=" << o.position_to_physical << '\n';
        for (const auto & h : c.halos) {
            halos << h.id << ' ' << h.fof_id << ' ' << h.host << ' ' << h.bound_mass << ' ' << h.inclusive_bound_mass
                  << ' ' << h.bound_radius << ' ' << h.vmax << ' ' << h.rmax;
            for (auto x : h.position)
                halos << ' ' << x;
            for (auto v : h.velocity)
                halos << ' ' << v;
            halos << '\n';
        }
        members << "# particle_id halo_id fof_id; no_halo=" << no_halo << '\n';
        for (const auto & m : c.assignments)
            members << m.particle << ' ' << m.halo << ' ' << m.fof << '\n';
        halos.close();
        members.close();
        if (!halos || !members)
            throw std::runtime_error("Output write failed");
    } catch (const std::exception & e) {
        std::cerr << "rank " << FML::ThisTask << ": " << e.what() << '\n';
        failed = 1;
    }
    FML::MaxOverTasks(&failed);
    return failed;
}

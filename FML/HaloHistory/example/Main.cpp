#include <FML/Global/Global.h>
#include <FML/HaloHistory/CatalogueIO.h>
#include <iostream>
#include <sstream>
using namespace FML::HALO_HISTORY;
Halo make_halo(ID id, std::vector<ID> members) {
    Halo h;
    h.id = id;
    h.mass = 1e12;
    h.radius = .2;
    h.vmax = 150;
    h.position = {.2, .3, .4};
    h.particles = std::move(members);
    return h;
}
int main() {
    Snapshot first;
    first.id = 10;
    first.scale_factor = .5;
    first.boxsize = 100;
    first.metadata.mass_definition = "M200m";
    first.metadata.id_namespace = "example-run";
    first.halos = {make_halo(1, {100, 101, 102, 103}), make_halo(2, {200, 201})};
    Snapshot second = first;
    second.id = 20;
    second.scale_factor = .6;
    second.halos = {make_halo(50, {100, 101, 102, 103, 200, 201})};
    auto history = build({first, second});
    if (FML::ThisTask == 0) {
        std::cout << "# source_snapshot source_halo target_snapshot target_halo Nshared merit skipped\n";
        for (const auto& link : history.links)
            std::cout << link.source.snapshot << ' ' << link.source.halo << ' ' << link.target.snapshot << ' '
                      << link.target.halo << ' ' << link.shared << ' ' << link.merit << ' ' << link.skipped_snapshots
                      << '\n';
    }
    std::stringstream interchange;
    write_catalogue(interchange, second);
    auto imported = read_catalogue(interchange);
    // This deliberately explicit toy dispersion is not a mass-dispersion model.
    auto hod = to_hod(imported, 5, [](const Halo&) { return 150.; });
    if (FML::ThisTask == 0)
        std::cout << "# imported hosts for HOD: " << hod.size() << "; snapshot-relative halo ID=" << hod[0].id << '\n';
}

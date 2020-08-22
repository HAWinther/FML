#include <FML/FileUtils/FileUtils.h>
#include <FML/Survey/GalaxiesToBox.h>
#include <fstream>

// RA, DEC is in degrees
struct Galaxy {
    double RA;
    double DEC;
    double z;
    double weight;
    Galaxy() {}
    Galaxy(double _RA, double _DEC, double _z, double _weight) : RA(_RA), DEC(_DEC), z(_z), weight(_weight) {}
    double get_RA() const { return RA; }
    double get_DEC() const { return DEC; }
    double get_z() const { return z; }
    double get_weight() const { return weight; }
};

// Galaxy in Cartesian coordinates
struct Particle {
    double Pos[3];
    double * get_pos() { return Pos; }
    int get_ndim() { return 3; }
};

using namespace FML;

int main() {

    // Read ascii file with [RA, DEC, z, w]
    const std::string filename = "../../../TestData/Randoms.txt";
    const int ncols = 3;
    const int nskip_header = 0;
    const std::vector<int> cols_to_keep{0, 1, 2};
    auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header, 3412304);

    // Make galaxies
    std::vector<Galaxy> galaxies;
    for (auto & c : data) {
        Galaxy g(c[0], c[1], c[2], 1.0);
        galaxies.push_back(g);
    }
    std::cout << "Read: " << galaxies.size() << " galaxies\n";

    // The Hubble function H/c needed to compute the comoving distance
    std::function<double(double)> hubble_over_c_of_z = [&](double z) -> double {
        return 1.0 / 2997.92458 * sqrt(0.315 * (1 + z) * (1 + z) * (1 + z) + 0.685);
    };

    // Galaxies to box options (shift/scale the positions to [0,1))
    const bool shiftPositions = true;
    const bool scalePositions = true;
    const bool verbose = true;

    // Convert galaxies to cartesian coordinates
    std::vector<Particle> particles;
    double boxsize;
    FML::SURVEY::GalaxiesToBox<Galaxy, Particle>(galaxies.data(),
                                                 galaxies.size(),
                                                 particles,
                                                 hubble_over_c_of_z,
                                                 boxsize,
                                                 shiftPositions,
                                                 scalePositions,
                                                 verbose);

    // Output positions
    // for(size_t i = 0; i < galaxies.size(); i++)
    //  std::cout << particles[i].Pos[0] << " " << particles[i].Pos[1] << " " << particles[i].Pos[2] << "\n";
}

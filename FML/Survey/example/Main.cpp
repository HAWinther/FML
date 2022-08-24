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
    constexpr int get_ndim() const { return 3; }
};

using namespace FML;

int main() {

    // Read ascii file with [DEC, RA, z, w]
    const std::string filename = "../../../TestData/ExampleSurveyData/Randoms_DEC_RA_z.txt";

    // Make galaxies
    std::vector<Galaxy> galaxies;
    auto data = FML::FILEUTILS::loadtxt(filename);
    for (auto & row : data) {
      Galaxy g;
      g.DEC = row[0];
      g.RA = row[1];
      g.z = row[2];
      if(row.size() > 3)
        g.weight = row[3];
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
    std::vector<double> observer_position;

    // Convert galaxies to cartesian coordinates
    std::vector<Particle> particles;
    double boxsize;
    FML::SURVEY::GalaxiesToBox<Galaxy, Particle>(galaxies.data(),
                                                 galaxies.size(),
                                                 particles,
                                                 hubble_over_c_of_z,
                                                 boxsize,
                                                 observer_position,
                                                 shiftPositions,
                                                 scalePositions,
                                                 verbose);

    // Output positions
    // for(size_t i = 0; i < galaxies.size(); i++)
    //  std::cout << particles[i].Pos[0] << " " << particles[i].Pos[1] << " " << particles[i].Pos[2] << "\n";
}

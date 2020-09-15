#include <FML/Global/Global.h>
#include <FML/PairCounting/PairCount.h>
#include <FML/ParticlesInBoxes/ParticlesInBoxes.h>
#include <FML/Survey/GalaxiesToBox.h>

#include <fstream>
#include <vector>

using namespace FML;
using namespace FML::CORRELATIONFUNCTIONS;

// Number of dimensions we are working in (N<=3)
const int N = 3;

// Particle struct
struct Particle {
    double Pos[N];
    double * get_pos() { return Pos; }
    int get_ndim() { return N; }
};

struct Galaxy {
    double RA;
    double DEC;
    double z;
    double get_RA() const { return RA; }
    double get_DEC() const { return DEC; }
    double get_z() const { return z; }
};

void readData(std::string filename, std::vector<Galaxy> & galaxies) {
    std::ifstream fp(filename.c_str());
    while (1) {
        double RA, DEC, z;
        fp >> DEC;
        if (fp.eof())
            break;
        fp >> RA;
        fp >> z;
        // fp >> w;

        Galaxy g;
        g.RA = RA;
        g.DEC = DEC;
        g.z = z;
        galaxies.push_back(g);
    }
}

void TestSurvey() {

    // Read data
    std::vector<Galaxy> galaxies;
    readData("../../../TestData/Galaxies.txt", galaxies);

    std::vector<Galaxy> randoms;
    readData("../../../TestData/Randoms.txt", randoms);

    // The Hubble function H/c needed to compute the comoving distance
    std::function<double(double)> hubble_over_c_of_z = [&](double z) -> double {
        return 1.0 / 2997.92458 * sqrt(0.315 * (1 + z) * (1 + z) * (1 + z) + 0.685);
    };

    // Options
    const bool shiftPositions = true;
    const bool scalePositions = true;
    const bool verbose = true;
    double boxsize;

    // Convert to cartesian coordinates
    std::vector<Particle> galaxies_xyz;
    std::vector<Particle> randoms_xyz;
    FML::SURVEY::GalaxiesRandomsToBox(galaxies.data(),
                                      galaxies.size(),
                                      randoms.data(),
                                      randoms.size(),
                                      galaxies_xyz,
                                      randoms_xyz,
                                      hubble_over_c_of_z,
                                      boxsize,
                                      shiftPositions,
                                      scalePositions,
                                      verbose);

    // Options
    const int nbins = 30;
    const double rmax = 10.0 / boxsize;
    CorrelationFunctionSurvey(galaxies_xyz, randoms_xyz, nbins, rmax, boxsize, verbose);
}

void TestPeriodic() {

    // Make particles
    const int npart = 1000;
    std::vector<Particle> particles(npart);
    std::ofstream fp("part.txt");
    const int NDIM = particles[0].get_ndim();
    for (int i = 0; i < npart; i++) {
        auto Pos = particles[i].get_pos();
        for (int j = 0; j < NDIM; j++) {
            Pos[j] = FML::uniform_random();
            fp << Pos[j] << " ";
        }
        fp << "\n";
    }
    fp.close();

    // Options
    const int nbins = 10;
    const double rmax = 0.1;
    const bool verbose = true;

    // Compute pair counts directly
    // const bool periodic = true;
    // auto res_auto = AutoPairCount(particles, nbins, rmax, periodic, verbose);
    // auto r          = res_auto.r;
    // auto DD_auto    = res_auto.paircount;
    // auto res_cross  = CrossPairCount(particles, particles, nbins, rmax, periodic, verbose);
    // auto DD_cross   = res_cross.paircount;

    // Output (Cross should agree with Auto in all bins except the first)
    // for(size_t i = 0; i < r.size(); i++){
    //  std::cout << r[i] << " " << DD_auto[i] << " " << DD_cross[i] << "\n";
    // }

    // This now works
    CorrelationFunctionSimulationBox(particles, nbins, rmax, verbose);
}

int main() {
    TestSurvey();

    // TestPeriodic(); exit(1);
}

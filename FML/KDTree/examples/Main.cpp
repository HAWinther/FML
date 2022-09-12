#include <FML/FileUtils/FileUtils.h>
#include <FML/KDTree/KDTree.h>
#include <functional>
#include <iostream>
#include <vector>

//================================================================
// 
// Examples of how to use the generic binner that uses the kdtree
// We have voids and galaxies from a simulation (periodic box)
// We compute the average profile of the galaxies around the voids
//
// For examples of using the kdtree directly see Test.cpp
//
//================================================================

using DVector = FML::KDTREE::DVector;

// An example galaxy
struct Galaxy {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; };
};

// An example void
struct Void {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; };
};

int main() {

    // Settings
    const double boxsize = 1000.0;
    const double rmin = 0.0;
    const double rmax = 100.0 / boxsize;
    const int nbins = 25;
    const bool periodic_box = true;
    const std::string filename1 = "../../../TestData/voids_realspace.txt";
    const std::string filename2 = "../../../TestData/galaxies_redshiftspace.txt";

    // Read the data
    auto data1 = FML::FILEUTILS::loadtxt(filename1);
    auto data2 = FML::FILEUTILS::loadtxt(filename2);

    // Assign the voids (position in [0,1])
    std::vector<Void> tracer1;
    tracer1.reserve(data1.size());
    for (auto line : data1) {
        Void p;
        for (int idim = 0; idim < 3; idim++)
            p.pos[idim] = line[idim] / boxsize;
        tracer1.push_back(p);
    }

    // Assign the galaxies (position in [0,1])
    std::vector<Galaxy> tracer2;
    tracer2.reserve(data2.size());
    for (auto line : data2) {
        Galaxy p;
        for (int idim = 0; idim < 3; idim++)
            p.pos[idim] = line[idim] / boxsize;
        tracer2.push_back(p);
    }

    // Perform the binning
    DVector rbin_center;
    std::vector<DVector> individual_density_profiles;
    FML::KDTREE::DensityProfileBinner(tracer1.data(),
                                      tracer1.size(),
                                      tracer2.data(),
                                      tracer2.size(),
                                      rmin,
                                      rmax,
                                      nbins,
                                      periodic_box,
                                      rbin_center,
                                      individual_density_profiles);

    // Compute the average (the stacked profile) and std of the profiles
    DVector mean_density(nbins, 0.0);
    DVector std_density(nbins, 0.0);
    for (int j = 0; j < nbins; j++) {
        for (size_t i = 0; i < individual_density_profiles.size(); i++) {
            mean_density[j] += individual_density_profiles[i][j];
            std_density[j] += individual_density_profiles[i][j] * individual_density_profiles[i][j];
        }
        mean_density[j] /= individual_density_profiles.size();
        std_density[j] /= individual_density_profiles.size();
        std_density[j] = std::sqrt(std_density[j] - mean_density[j] * mean_density[j]);
        std_density[j] /= std::sqrt(individual_density_profiles.size());
    }

    // Output to screen
    if (FML::ThisTask == 0) {
        std::cout << "# r (Mpc/h)            rho/rho_mean       standard_error\n";
        for (int i = 0; i < nbins; i++) {
            std::cout << std::setw(15) << rbin_center[i] * boxsize << "  ";
            std::cout << std::setw(15) << mean_density[i] << "  ";
            std::cout << std::setw(15) << std_density[i] << "  ";
            std::cout << "\n";
        }
    }
}

#include <FML/ExampleData/ExampleData.h>
#include <FML/KDTree/KDTree.h>
#include <functional>
#include <iostream>
#include <vector>

//================================================================
//
// Examples of how to use the generic binner that uses the kdtree
// We have voids and galaxies in a periodic box (a synthetic toy box)
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
    // The data: a toy clustered box with voids; the galaxies are its particles in redshift space (to use your own
    // data, read it e.g. with FML::FILEUTILS::loadtxt)
    const auto data = FML::EXAMPLEDATA::clustered_box(100000, boxsize, 100);

    // Assign the voids (position in [0,1])
    std::vector<Void> tracer1;
    for (auto & v : data.voids) {
        Void p;
        for (int idim = 0; idim < 3; idim++)
            p.pos[idim] = v.pos[idim] / boxsize;
        tracer1.push_back(p);
    }

    // Assign the galaxies (position in [0,1])
    std::vector<Galaxy> tracer2;
    for (auto & g : FML::EXAMPLEDATA::redshift_space(data)) {
        Galaxy p;
        for (int idim = 0; idim < 3; idim++)
            p.pos[idim] = g.pos[idim] / boxsize;
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

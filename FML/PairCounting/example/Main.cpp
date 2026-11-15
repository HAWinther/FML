#include <FML/PairCounting/PairCount.h>
#include <FML/Spline/Spline.h>
#include <FML/Survey/GalaxiesToBox.h>
#include <FML/ExampleData/ExampleData.h>
#include <FML/Units/Units.h>
#include <fstream>

using DVector = FML::INTERPOLATION::SPLINE::DVector;
using DVector2D = FML::INTERPOLATION::SPLINE::DVector2D;
using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
using Spline = FML::INTERPOLATION::SPLINE::Spline;

//================================
// A simple 3D particle
//================================
struct Particle {
    double Pos[3];
    double * get_pos() { return Pos; }
    constexpr int get_ndim() const { return 3; }
};

//================================
// A position on the sky
//================================
struct Galaxy {
    double RA;
    double DEC;
    double z;
    double weight{1.0};
    double get_RA() const { return RA; }
    double get_DEC() const { return DEC; }
    double get_z() const { return z; }
    double get_weight() const { return weight; }
};

//===============================================
// Compute the cross correlation function
// for survey data (randoms)
//===============================================
void ExampleRadialCorrelationFunctionSurvey() {

    //===============================================
    // Galaxies (clustered) and randoms [DEC, RA, z] in a patch of the sky (synthetic; to use your own, read them
    // e.g. with FML::FILEUTILS::loadtxt)
    //===============================================
    const auto catalogue = FML::EXAMPLEDATA::survey_catalogue();
    auto to_galaxies = [](const std::vector<FML::EXAMPLEDATA::SurveyObject> & objects) {
        std::vector<Galaxy> result;
        for (auto & o : objects) {
            Galaxy g;
            g.DEC = o.DEC;
            g.RA = o.RA;
            g.z = o.z;
            result.push_back(g);
        }
        return result;
    };
    std::vector<Galaxy> galaxies = to_galaxies(catalogue.galaxies);
    std::vector<Galaxy> randoms = to_galaxies(catalogue.randoms);

    //===============================================
    // The Hubble function H/c needed to compute the comoving distance
    //===============================================
    const double OmegaM = 0.315;
    const double OmegaLambda = 1.0 - OmegaM;
    const double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
    std::function<double(double)> hubble_over_c_of_z = [&](double z) -> double {
        return H0_hmpc * sqrt(OmegaM * (1 + z) * (1 + z) * (1 + z) + OmegaLambda);
    };

    //===============================================
    // Convert to cartesian coordinates
    // Shift and scale so particles are inside [0,1)^3
    // Observer position (0,0,0) gets shifted/scales
    // appropriately
    // Boxsize in physical coordinates is returned
    //===============================================
    const bool shiftPositions = true;
    const bool scalePositions = true;
    const bool verbose = true;
    double boxsize;
    std::vector<Particle> galaxies_xyz;
    std::vector<Particle> randoms_xyz;
    std::vector<double> observer_position;
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
                                      observer_position,
                                      verbose);

    //===============================================
    // Correlation function settings
    //===============================================
    std::string estimator = "LZ";
    const int nbins = 30;
    const double rmin = 0.0;
    const double rmax = 10.0 / boxsize;
    bool periodic_box = false;
    std::vector<double> r_array{};

    //===============================================
    // If there are other quantities we want binned up alongside the pairs then add this here
    // As an example we bin up the number of pairs and the mean value of r in each bin
    //===============================================
    int nextratobin = 2;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra_quantities_D1D2_array;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra_quantities_D1R2_array;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra_quantities_R1D2_array;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra_quantities_R1R2_array;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::ExtraQuantitiesToBinFunction<Particle, Particle>
        extra_stuff_to_bin_function = [&]([[maybe_unused]] const double * dr,
                                          [[maybe_unused]] double r,
                                          [[maybe_unused]] double mu,
                                          [[maybe_unused]] const Particle & part1,
                                          [[maybe_unused]] const Particle & part2,
                                          [[maybe_unused]] double * storage) {
            // Sum up the number of pairs
            storage[0] += 1.0;
            // Sum up r
            storage[1] += r;
        };

    //===============================================
    // Compute the paircounts (and other binning)
    //===============================================
    DVector paircounts_D1D2_array, paircounts_D1R2_array, paircounts_R1D2_array, paircounts_R1R2_array;
    DVector corr_func_array;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::RadialCorrelationFunctionSurvey(galaxies_xyz.data(),
                                                                           galaxies_xyz.size(),
                                                                           randoms_xyz.data(),
                                                                           randoms_xyz.size(),
                                                                           galaxies_xyz.data(),
                                                                           galaxies_xyz.size(),
                                                                           randoms_xyz.data(),
                                                                           randoms_xyz.size(),
                                                                           rmin,
                                                                           rmax,
                                                                           nbins,
                                                                           r_array,
                                                                           paircounts_D1D2_array,
                                                                           paircounts_D1R2_array,
                                                                           paircounts_R1D2_array,
                                                                           paircounts_R1R2_array,
                                                                           corr_func_array,
                                                                           observer_position,
                                                                           periodic_box,
                                                                           verbose,
                                                                           nextratobin,
                                                                           extra_quantities_D1D2_array,
                                                                           extra_quantities_D1R2_array,
                                                                           extra_quantities_R1D2_array,
                                                                           extra_quantities_R1R2_array,
                                                                           extra_stuff_to_bin_function);

    if (FML::ThisTask == 0) {
        std::cout << "\n# Correlationfunction with randoms using the " + estimator + " estimator\n";
        std::cout << "#         r          xi_r             Pairs\n";
        for (int j = 0; j < nbins; j++) {
            std::cout << std::setw(15) << r_array[j] * boxsize << " ";
            std::cout << std::setw(15) << corr_func_array[j] << " ";
            std::cout << std::setw(15) << paircounts_D1D2_array[j] << " ";
            std::cout << std::setw(15) << paircounts_D1R2_array[j] << " ";
            std::cout << std::setw(15) << paircounts_R1D2_array[j] << " ";
            std::cout << std::setw(15) << paircounts_R1R2_array[j] << " ";
            if (nextratobin > 0) {
                // Print the mean r in each bin
                std::cout << std::setw(15)
                          << extra_quantities_D1D2_array[j][1] / extra_quantities_D1D2_array[j][0] * boxsize << " ";
                std::cout << std::setw(15)
                          << extra_quantities_D1R2_array[j][1] / extra_quantities_D1R2_array[j][0] * boxsize << " ";
                std::cout << std::setw(15)
                          << extra_quantities_R1D2_array[j][1] / extra_quantities_R1D2_array[j][0] * boxsize << " ";
                std::cout << std::setw(15)
                          << extra_quantities_R1R2_array[j][1] / extra_quantities_R1R2_array[j][0] * boxsize << " ";
            }
            std::cout << "\n";
        }
    }

    //===============================================
    // Comparison of the different estimators we have
    //===============================================
    if (FML::ThisTask == 0) {
        std::cout << "\n# Comparison of different estimators\n";
        std::cout << "#         r                 LZ               DP              HE              PH             HA\n";
        for (int j = 0; j < nbins; j++) {
            auto corr_LZ = FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(paircounts_D1D2_array[j],
                                                                                               paircounts_D1R2_array[j],
                                                                                               paircounts_R1D2_array[j],
                                                                                               paircounts_R1R2_array[j],
                                                                                               "LZ");
            auto corr_DP = FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(paircounts_D1D2_array[j],
                                                                                               paircounts_D1R2_array[j],
                                                                                               paircounts_R1D2_array[j],
                                                                                               paircounts_R1R2_array[j],
                                                                                               "DP");
            auto corr_HE = FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(paircounts_D1D2_array[j],
                                                                                               paircounts_D1R2_array[j],
                                                                                               paircounts_R1D2_array[j],
                                                                                               paircounts_R1R2_array[j],
                                                                                               "HEW");
            auto corr_PH = FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(paircounts_D1D2_array[j],
                                                                                               paircounts_D1R2_array[j],
                                                                                               paircounts_R1D2_array[j],
                                                                                               paircounts_R1R2_array[j],
                                                                                               "PH");
            auto corr_HA = FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(paircounts_D1D2_array[j],
                                                                                               paircounts_D1R2_array[j],
                                                                                               paircounts_R1D2_array[j],
                                                                                               paircounts_R1R2_array[j],
                                                                                               "HAM");

            std::cout << std::setw(15) << r_array[j] * boxsize << " ";
            std::cout << std::setw(15) << corr_LZ << " ";
            std::cout << std::setw(15) << corr_DP << " ";
            std::cout << std::setw(15) << corr_HE << " ";
            std::cout << std::setw(15) << corr_PH << " ";
            std::cout << std::setw(15) << corr_HA << " ";
            std::cout << "\n";
        }
    }
}

//===============================================
// Compute the radial correlation function in a
// periodic box
//===============================================
void ExamplePeriodicBox() {

    //===============================================
    // Make some random particles
    //===============================================
    const int npart = 10000;
    std::vector<Particle> particles(npart);
    const int NDIM = particles[0].get_ndim();
    for (int i = 0; i < npart; i++) {
        auto Pos = particles[i].get_pos();
        for (int j = 0; j < NDIM; j++) {
            Pos[j] = FML::uniform_random();
        }
    }

    //===============================================
    // Correlation function settings
    //===============================================
    const int nbins = 10;
    const double rmin = 0.0;
    const double rmax = 0.1;
    const bool verbose = true;
    const bool periodic_box = true;

    //===============================================
    // Compute everything
    // Normalized paircounts is DD/tot_num_pairs
    //===============================================
    std::vector<double> r_array, paircounts_array, corr_func_array;
    bool normalize_paircounts = true;
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::RadialCorrelationFunctionBox<Particle>(particles.data(),
                                                                                  particles.size(),
                                                                                  rmin,
                                                                                  rmax,
                                                                                  nbins,
                                                                                  r_array,
                                                                                  paircounts_array,
                                                                                  normalize_paircounts,
                                                                                  corr_func_array,
                                                                                  periodic_box,
                                                                                  verbose);

    if (FML::ThisTask == 0) {
        std::cout << "\n# Correlationfunction from a simulation box\n";
        std::cout << "#         r          xi_r             Pairs\n";
        for (int j = 0; j < nbins; j++) {
            std::cout << std::setw(15) << r_array[j] << " ";
            std::cout << std::setw(15) << corr_func_array[j] << " ";
            std::cout << std::setw(15) << paircounts_array[j] << " ";
            std::cout << "\n";
        }
    }
}

//===============================================
// This method computes the correlation function xi(r,mu)
// and the resulting multipoles for a fixed line of sight direction
// (Input below have RSD added to the z-direction and
// consequently the observer positions is (0,0,-infty))
//===============================================
void ExampleCorrelationFunctionMultipoles() {

    //===============================================
    // Catalogues: a toy clustered box; the galaxies are its particles in redshift space (along z) and the voids
    // the centers of its voids (synthetic; to use your own, read them e.g. with FML::FILEUTILS::loadtxt)
    //===============================================
    const double box = 1000.0;
    const auto data = FML::EXAMPLEDATA::clustered_box(100000, box, 100);
    const bool periodic = true;
    const bool verbose = true;

    //===============================================
    // Binning specifications
    //===============================================
    const double rmin = 0.0;
    const double rmax = 120.0 / box;
    const int nrbins = 30;
    const double mumin = -1.0;
    const double mumax = 1.0;
    const int nmubins = 101;
    const bool normalize_paircounts = true;

    //===============================================
    // Observer position (this is the same as a fixed LOS along the z axis)
    //===============================================
    std::vector<double> observer_position{0, 0, -FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::effective_infinity};

    //===============================================
    // Results arrays
    //===============================================
    DVector2D paircounts_array, corr_func_array;
    std::vector<double> r_array, mu_array;

    //===============================================
    // The galaxies and voids with positions in [0,1)
    //===============================================
    std::vector<Particle> particles1;
    for (auto & g : FML::EXAMPLEDATA::redshift_space(data)) {
        Particle p;
        for (int idim = 0; idim < 3; idim++)
            p.Pos[idim] = g.pos[idim] / box;
        particles1.push_back(p);
    }
    std::vector<Particle> particles2;
    for (auto & v : data.voids) {
        Particle p;
        for (int idim = 0; idim < 3; idim++)
            p.Pos[idim] = v.pos[idim] / box;
        particles2.push_back(p);
    }

    //===============================================
    // Compute paircounts
    //===============================================
    Particle * part1 = particles1.data();
    Particle * part2 = particles2.data();
    auto npart1 = particles1.size();
    auto npart2 = particles2.size();
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::AngularCorrelationFunctionBox(part1,
                                                                         npart1,
                                                                         part2,
                                                                         npart2,
                                                                         rmin,
                                                                         rmax,
                                                                         nrbins,
                                                                         mumin,
                                                                         mumax,
                                                                         nmubins,
                                                                         r_array,
                                                                         mu_array,
                                                                         paircounts_array,
                                                                         normalize_paircounts,
                                                                         corr_func_array,
                                                                         observer_position,
                                                                         periodic,
                                                                         verbose);

    //===============================================
    // Spline up xi(mu,r)
    //===============================================
    Spline2D xi_mu_r_spline = Spline2D(mu_array, r_array, corr_func_array, "xi(mu,r)");

    //===============================================
    // Compute multipoles
    //===============================================
    std::vector<DVector> multipoles(5);
    FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::FromAngularCorrelationToMultipoles(r_array, xi_mu_r_spline, multipoles);

    //===============================================
    // Spline up multipoles
    //===============================================
    Spline xi0r = Spline(r_array, multipoles[0]);
    Spline xi2r = Spline(r_array, multipoles[2]);
    Spline xi4r = Spline(r_array, multipoles[4]);

    if (FML::ThisTask == 0) {
        std::cout << "\n# Correlation function multipoles\n";
        std::cout << "#   r (Mpc/h)       xi0(r)        xi2(r)       xi4(r)\n";
        for (int i = 0; i < nrbins; i++) {
            std::cout << std::setw(15) << r_array[i] * box << "  ";
            std::cout << std::setw(15) << xi0r(r_array[i]) << "  ";
            std::cout << std::setw(15) << xi2r(r_array[i]) << "  ";
            std::cout << std::setw(15) << xi4r(r_array[i]) << "  ";
            std::cout << "\n";
        }
    }
}

int main() {

    ExamplePeriodicBox();

    ExampleRadialCorrelationFunctionSurvey();

    ExampleCorrelationFunctionMultipoles();
}

#include <FML/RamsesUtils/RamsesUtils.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/PowerSpectrumBinning.h>
#include <FML/MPIParticles/MPIParticles.h>

// Simple particle type that only stores positions
template <int NDIM>
class RamsesParticle {
  private:
    double x[NDIM];

  public:
    int get_ndim() { return NDIM; }
    double * get_pos() { return x; } // only store positions
};

// snapdir: a ramses snapshot directory path (e.g. path/to/ramses/stuff/output_00001/)
void compute_power_spectrum(std::string snapdir, int level=-1, std::vector<std::string> format={}, std::string density_assignment="PCS", bool subtract_shotnoise=false, bool verbose=false) {
    const int NDIM = 3;

    // Ensure path ends with trailing /
    if (snapdir.back() != '/') {
        snapdir += "/";
    }

    // Set up Ramses reader with custom format, if provided
    FML::FILEUTILS::RAMSES::RamsesReader reader(snapdir, 1.0, true, verbose);
    if (not format.empty()) {
        reader.set_file_format(format); // default: POS, VEL, MASS, ID, LEVEL, FAMILY, TAG
    }

    // Read particles into MPI-able container
    std::vector<RamsesParticle<NDIM>> parts;
    reader.read_ramses(parts);
    FML::PARTICLE::MPIParticles<RamsesParticle<NDIM>> mpiparts;
    mpiparts.move_from(std::move(parts));

    if (level == -1) {
        level = reader.get_levelmin(); // default to minimum level (as in a non-AMR simulation)
    }
    int Nmesh = 1 << level; // == 2^levelmin
    int Nbins = Nmesh / 2;

    if (FML::ThisTask == 0) {
        std::cout << "Computing P(k) with " << Nbins << " k-bins "
                  << "on " << Nmesh << "^" << NDIM << " mesh "
                  << "using " << density_assignment << " density assignment "
                  << (subtract_shotnoise ? "with" : "without") << " subtracting shot noise\n";
    }

    // Compute power spectrum
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Nbins);
    pofk.subtract_shotnoise = subtract_shotnoise;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM>(Nmesh, mpiparts.get_particles_ptr(), mpiparts.get_npart(), mpiparts.get_npart_total(), pofk, density_assignment, true);
    pofk.scale(reader.get_boxsize()); // scale to box

    // Output to file
    std::string pkpath = snapdir + "pofk_fml.dat";
    std::ofstream pkfile(pkpath);
    pkfile << "#" << std::setw(15) << "k/(h/Mpc)" << " " << std::setw(15) << "P/(Mpc/h)^3" << "\n";
    for (int i = 0; i < pofk.n; i++) {
        pkfile << " " << std::setw(15) << pofk.kbin[i] << " " << std::setw(15) << pofk.pofk[i] << "\n";
    }
    pkfile.close();
    if (FML::ThisTask == 0) {
        std::cout << "Wrote P(k) to " << pkpath << "\n";
    }
}

int main(int argc, char *argv[]) {
    int level = -1;
    bool subtract_shotnoise = false;
    std::string density_assignment = "PCS";
    std::vector<std::string> format;
    bool verbose = false;
    bool user_is_stupid = true; // always assume the user is stupid!

    for (int argi = 1; argi < argc; argi++) {
        std::string arg = argv[argi];
        if (arg.starts_with("--level=")) {
            level = std::stoi(arg.substr(8));
        } else if (arg.starts_with("--format=")) { // e.g. --format=POS,VEL,MASS,ID
            std::string formatstr = arg.substr(9); // e.g. POS,VEL,MASS,ID
            while (formatstr != "") {
                int i = formatstr.find(",");
                format.push_back(formatstr.substr(0, i)); // push e.g. POS
                formatstr = formatstr.erase(0, i).erase(0, 1); // remove e.g. POS, (erase(0, i+1) fails on last element with i==-1)
            }
        } else if (arg.starts_with("--density-assignment=")) { // e.g. --density-assignment=PCS
            density_assignment = arg.substr(21); // e.g. PCS
        } else if (arg == "--subtract-shotnoise") {
            subtract_shotnoise = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            user_is_stupid = true;
            break;
        } else { // argument is a snapshot directory path to process
            compute_power_spectrum(arg, level, format, density_assignment, subtract_shotnoise, verbose);
            user_is_stupid = false; // user figured out how the program works
        }
    }

    // Help stupid users
    if (user_is_stupid and FML::ThisTask == 0) {
        std::cout << "SYNTAX:\n"
                  << "ramses2pk\n"
                  << "[--help] [--verbose]\n"
                  << "[--level=LEVEL]               (default: detected levelmin)\n"
                  << "[--density-assignment=METHOD] (default: PCS)\n"
                  << "[--subtract-shotnoise]        (default: off)\n"
                  << "[--format=FORMAT]             (default: detected or POS,VEL,MASS,ID,LEVEL,FAMILY,TAG)\n"
                  << "ramses/simulation/snapshot/   (usually .../output_00123/)\n";
    }
}

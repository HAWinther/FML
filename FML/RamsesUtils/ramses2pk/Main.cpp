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
void compute_power_spectrum(std::string snapdir, std::vector<std::string> format={}, std::string density_assignment="PCS", bool subtract_shotnoise=false, bool verbose=false) {
    const int NDIM = 3;

    // ensure path ends with trailing /
    if (snapdir.back() != '/') {
        snapdir += "/";
    }

    // Set up Ramses reader
    FML::FILEUTILS::RAMSES::RamsesReader reader(snapdir, 1.0, true, verbose);

    // Set custom format, if provided
    if (not format.empty()) {
        reader.set_file_format(format); // default: POS, VEL, MASS, ID, LEVEL, FAMILY, TAG
    }

    // Read particles
    std::vector<RamsesParticle<NDIM>> part;
    reader.read_ramses(part);

    int Npart = reader.get_npart();
    int Npart1D = int(round(pow(Npart, 1.0 / NDIM))); // Npart == Npart1D^NDIM
    assert(int(round(pow(Npart1D, NDIM))) == Npart);
    int Ncell1D = 1 << reader.get_levelmin(); // == 2^n
    double Lh = reader.get_boxsize(); // == L / (Mpc/h) == L*h / Mpc

    FML::PARTICLE::MPIParticles<RamsesParticle<NDIM>> parts; // TODO: does not work with mpirun -np 16
    parts.create(part.data(), Npart, Npart, 0.0, 1.0, false); // TODO: Npart_total? what is x [0.0, 1.0] domain?

    // Compute power spectrum
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Ncell1D / 2); // store resulting P(k) // TODO: why divide by 2?
    pofk.subtract_shotnoise = subtract_shotnoise; // same as in COLA by default // TODO: do this in COLA sims or not? true gives negative P(k)
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM>(Ncell1D, parts.get_particles_ptr(), parts.get_npart(), parts.get_npart_total(), pofk, density_assignment, true);
    pofk.scale(Lh); // scale to box

    // Output to file
    std::string pkpath = snapdir + "pofk_fml.dat";
    std::ofstream pkfile(pkpath);
    pkfile << "#" << std::setw(15) << "k/(h/Mpc)" << " " << std::setw(15) << "P/(Mpc/h)^3" << "\n";
    for (int i = 0; i < pofk.n; i++) {
        pkfile << " " << std::setw(15) << pofk.kbin[i] << " " << std::setw(15) << pofk.pofk[i] << "\n";
    }
    pkfile.close();
    std::cout << "Wrote P(k) to " << pkpath << "\n";
}

int main(int argc, char *argv[]) {
    bool subtract_shotnoise = false;
    std::string density_assignment = "PCS";
    std::vector<std::string> format;
    bool verbose = false;
    bool user_is_stupid = true; // always assume the user is stupid!

    for (int argi = 1; argi < argc; argi++) {
        std::string arg = argv[argi];
        if (arg.starts_with("--format=")) { // e.g. --format=POS,VEL,MASS,ID
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
            compute_power_spectrum(arg, format, density_assignment, subtract_shotnoise, verbose);
            user_is_stupid = false; // user figured out how the program works
        }
    }

    // Help stupid users
    if (user_is_stupid) {
        std::cout << "SYNTAX:\n"
                  << "ramses2pk [--help] [--verbose]\n"
                  << "          [--subtract-shotnoise]\n"
                  << "          [--density-assignment=METHOD] (default: PCS)\n"
                  << "          [--format=FORMAT]             (override detected format, default: POS,VEL,MASS,ID,LEVEL,FAMILY,TAG)\n"
                  << "          path/to/ramses/snapshot/directory/like/output_00123/\n";
    }
}

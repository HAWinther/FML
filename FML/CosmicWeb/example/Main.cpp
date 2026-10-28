#include <FML/CosmicWeb/CosmicWeb.h>
#include <FML/CosmicWeb/ORIGAMI.h>
#ifdef USE_CGAL
#include <FML/CosmicWeb/ORIGAMIHalos.h>
#endif
#include <FML/MPIParticles/MPIParticles.h>
#include <random>

//=======================================================================================
// Example of the cosmic web classification:
//
// T-web : the number of eigenvalues of the tidal tensor (of the smoothed density field)
//         above a threshold: 0 = void, 1 = sheet, 2 = filament, 3 = knot (needs FFTW)
// ORIGAMI: the number of orthogonal axes of the initial lattice along which a particle
//         has been crossed by other particles: 0 = void, 1 = wall, 2 = filament, 3 = halo
// ORIGAMIHalos: group the ORIGAMI halo particles into halos on the Delaunay tessellation
//         (needs CGAL: compile with USE_CGAL = true)
// (V-web: the same as the T-web for the velocity shear, e.g. from the DTFE, see CosmicWeb.h)
//
// We make particles on a lattice and displace them with some plane waves strong enough
// to cause shell crossing (a crude Zel'dovich approximation). Positions are in [0,1).
//=======================================================================================

const int NDIM = 3;
struct Particle {
    double Pos[NDIM];
    long long int id;
    constexpr int get_ndim() const { return NDIM; }
    double * get_pos() { return Pos; }
    // ORIGAMI uses the ID to find the lattice point: id = (ix * N + iy) * N + iz
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
};

int main() {
    const int Nlattice = 64;
    const int Nmesh = 64;
    const double twopi = 2.0 * M_PI;

    // Displacement: a sum of plane waves with random directions and phases
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const int nwaves = 12;
    std::vector<std::array<double, 5>> waves; // kx, ky, kz, phase, amplitude
    for (int w = 0; w < nwaves; w++) {
        std::array<int, 3> n{int(uniform(rng) * 5) - 2, int(uniform(rng) * 5) - 2, int(uniform(rng) * 5) - 2};
        if (n[0] == 0 and n[1] == 0 and n[2] == 0)
            n[0] = 1;
        waves.push_back({double(n[0]), double(n[1]), double(n[2]), twopi * uniform(rng), 2.0 + 4.0 * uniform(rng)});
    }
    std::vector<Particle> partvec;
    for (int ix = 0; ix < Nlattice; ix++)
        for (int iy = 0; iy < Nlattice; iy++)
            for (int iz = 0; iz < Nlattice; iz++) {
                Particle p;
                const double q[3] = {double(ix) / Nlattice, double(iy) / Nlattice, double(iz) / Nlattice};
                for (int d = 0; d < NDIM; d++)
                    p.Pos[d] = q[d];
                for (auto & w : waves) {
                    const double k2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                    const double phase = twopi * (w[0] * q[0] + w[1] * q[1] + w[2] * q[2]) + w[3];
                    // Displacement along k: psi = A k / (2 pi k^2) sin(k.q) (gradient of a potential)
                    for (int d = 0; d < NDIM; d++)
                        p.Pos[d] += w[4] * w[d] / (twopi * k2) * std::sin(phase) / std::sqrt(double(nwaves));
                }
                for (int d = 0; d < NDIM; d++) {
                    p.Pos[d] -= std::floor(p.Pos[d]);
                    if (p.Pos[d] >= 1.0)
                        p.Pos[d] = 0.0;
                }
                p.id = ((long long int)(ix)*Nlattice + iy) * Nlattice + iz;
                partvec.push_back(p);
            }
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create(partvec.data(), partvec.size(), partvec.size(), FML::xmin_domain, FML::xmax_domain, true);

    // T-web: smoothing on 2 grid cells, threshold 0.2 (as is often used)
    FML::COSMICWEB::TWebOptions twebopt;
    twebopt.smoothing_scale = 2.0 / Nmesh;
    twebopt.lambda_threshold = 0.2;
    FML::COSMICWEB::CosmicWeb tweb;
    FML::COSMICWEB::TWeb(part.get_particles_ptr(), part.get_npart(), Nmesh, twebopt, tweb);

    // ORIGAMI morphology of every particle
    std::vector<int> morphology;
    FML::COSMICWEB::ORIGAMI(part.get_particles_ptr(), part.get_npart(), Nlattice, morphology);

#ifdef USE_CGAL
    // ORIGAMI halos: cores above 20 times the mean density here (Falck et al. use 200 for simulations)
    std::vector<long long int> halo_id;
    std::vector<FML::COSMICWEB::OrigamiHalo> halos;
    FML::COSMICWEB::OrigamiHaloOptions halo_options;
    halo_options.core_density_threshold = 20.0;
    FML::COSMICWEB::ORIGAMIHalos(part.get_particles_ptr(),
                                 part.get_npart(),
                                 Nlattice,
                                 morphology,
                                 halo_id,
                                 halos,
                                 halo_options);
    if (FML::ThisTask == 0) {
        std::cout << "The 5 largest ORIGAMI halos:\n";
        for (size_t h = 0; h < std::min(size_t(5), halos.size()); h++)
            std::cout << "  " << halos[h].npart << " particles at (" << halos[h].pos[0] << ", " << halos[h].pos[1]
                      << ", " << halos[h].pos[2] << ") max density " << halos[h].density_max << "\n";
    }
#endif

    // How the two compare: the T-web class of the cell of each particle vs its ORIGAMI morphology
    // (NB: with MPI the particles are inside the local domain and the grid slab matches it)
    std::array<std::array<long long int, 4>, 4> table{};
    for (size_t i = 0; i < part.get_npart(); i++) {
        const auto * pos = part[i].get_pos();
        const int ix = int(pos[0] * Nmesh) - int(tweb.classification.get_local_x_start());
        const int iy = int(pos[1] * Nmesh), iz = int(pos[2] * Nmesh);
        if (ix < 0 or ix >= tweb.classification.get_local_nx())
            continue;
        const int type = int(tweb.classification.get_real({ix, iy, iz}));
        table[morphology[i]][type]++;
    }
    for (auto & row : table)
        FML::SumArrayOverTasks(row.data(), 4);
    if (FML::ThisTask == 0) {
        std::cout << "\nNumber of particles by ORIGAMI morphology (rows) and T-web class (columns)\n";
        std::cout << "            void      sheet   filament      knot\n";
        const char * names[4] = {"void    ", "wall    ", "filament", "halo    "};
        for (int m = 0; m < 4; m++) {
            std::cout << names[m];
            for (int t = 0; t < 4; t++)
                std::cout << " " << std::setw(10) << table[m][t];
            std::cout << "\n";
        }
    }
}

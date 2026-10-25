#include <FML/ComputePowerSpectra/HybridBiasSpectra.h>
#include <FML/ComputePowerSpectra/HybridBiasZeldovich.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/RandomFields/GaussianRandomField.h>

//=======================================================================================
// Example of the hybrid bias (Lagrangian bias basis) spectra.
//
// We make a Gaussian linear field, displace particles from a lattice with the Zel'dovich
// approximation and compute the 15 cross spectra P_ab(k) of the operators
// 1, delta_L, delta_L^2, s^2, nabla^2 delta_L advected with the particles. With an N-body
// simulation you would use the snapshot particles and the linear field of the initial
// conditions (scaled to the redshift of the snapshot), see the COLASolver (hybridbias = true).
// Since the particles here follow the Zel'dovich approximation exactly we can compare the
// result with the Zel'dovich prediction of the spectra (HybridBiasZeldovich.h).
//=======================================================================================

using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
using Grid = FML::GRID::FFTWGrid<3>;

const int NDIM = 3;
struct Particle {
    double pos[NDIM];
    long long int id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return NDIM; }
    // The ID tells us the lattice point: id = (ix * N + iy) * N + iz
    long long int get_id() const { return id; }
    void set_id(long long int i) { id = i; }
};

int main() {
    const int Npart_1D = 64;
    const int Nmesh = 128;
    const double boxsize = 500.0; // Mpc/h

    //=======================================================================================
    // The linear density field delta_L(k) (here P(k) = A k^-1.5 in box units)
    //=======================================================================================
    Grid delta_linear(Npart_1D);
    FML::RANDOM::RandomGenerator rng;
    rng.set_seed(1234);
    std::function<double(double)> Pofk_of_kBox_over_volume = [](double kBox) {
        return kBox > 0.0 ? 1e-4 * std::pow(kBox / (2.0 * M_PI), -1.5) : 0.0;
    };
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(
        delta_linear, &rng, Pofk_of_kBox_over_volume, false);
    // We cut the linear field at |k| < k_cut (the Nyquist frequency of the lattice) so the operators are isotropic.
    // The displacement field has the same modes
    const double kcut = M_PI * Npart_1D; // In units of 1/box
    FML::GRID::multiply_fourier_by_function_of_k<NDIM>(delta_linear, [&](double k) { return k < kcut ? 1.0 : 0.0; });

    //=======================================================================================
    // Particles on a lattice (with IDs) displaced with the Zel'dovich approximation
    // Psi(k) = i k / k^2 delta_L(k)
    //=======================================================================================
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create_particle_grid(Npart_1D, 1.5, FML::xmin_domain, FML::xmax_domain);
    long long int npart_local = part.get_npart();
    auto npart_per_task = FML::GatherFromTasks(&npart_local);
    long long int id = 0;
    for (int t = 0; t < FML::ThisTask; t++)
        id += npart_per_task[t];
    for (auto & p : part)
        p.set_id(id++);

    std::array<Grid, NDIM> Psi;
    for (int idim = 0; idim < NDIM; idim++) {
        Psi[idim] = delta_linear;
        for (int islice = 0; islice < Psi[idim].get_local_nx(); islice++) {
            double kmag2;
            std::array<double, NDIM> kvec;
            for (auto && fourier_index : Psi[idim].get_fourier_range(islice, islice + 1)) {
                Psi[idim].get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                auto value = delta_linear.get_fourier_from_index(fourier_index);
                const double fac = kmag2 > 0.0 ? kvec[idim] / kmag2 : 0.0;
                Psi[idim].set_fourier_from_index(fourier_index, value * std::complex<FML::GRID::FloatType>(0, fac));
            }
        }
        Psi[idim].fftw_c2r();
    }
    // The lattice points are the grid points of Psi
    for (auto & p : part) {
        const long long int i = p.get_id();
        std::array<int, NDIM> coord{int(i / Npart_1D / Npart_1D), int((i / Npart_1D) % Npart_1D), int(i % Npart_1D)};
        coord[0] -= int(Psi[0].get_local_x_start());
        const auto index = Psi[0].get_index_real(coord);
        for (int idim = 0; idim < NDIM; idim++) {
            double x = p.pos[idim] + Psi[idim].get_real_from_index(index);
            x -= std::floor(x);
            p.pos[idim] = x >= 1.0 ? 0.0 : x;
        }
    }
    part.communicate_particles();

    //=======================================================================================
    // The hybrid bias spectra
    //=======================================================================================
    HybridBiasOptions options;
    options.kcut = kcut;
    options.density_assignment_method = "CIC";
    options.interlacing = true;
    HybridBiasSpectra<NDIM> spectra;
    compute_hybrid_bias_spectra<NDIM>(part.get_particles_ptr(),
                                      part.get_npart(),
                                      part.get_npart_total(),
                                      delta_linear,
                                      Npart_1D,
                                      Nmesh,
                                      options,
                                      spectra);
    spectra.scale(boxsize);
    spectra.write("hybridbias_spectra.txt");

    // The power spectrum of a tracer with b1 = 1.5, b2 = 0.5, bs = -0.2, bnabla = 0 (Lagrangian biases)
    auto Ptracer = spectra.tracer_power_spectrum({1.0, 1.5, 0.5, -0.2, 0.0});

    //=======================================================================================
    // The Zel'dovich prediction (in physical units: k in h/Mpc, P in (Mpc/h)^3)
    //=======================================================================================
    auto pofk = [&](double k) { return Pofk_of_kBox_over_volume(k * boxsize) * std::pow(boxsize, 3); };
    // The box has no modes below the fundamental mode (and P(k) = A k^-1.5 has an infinite displacement variance if we
    // integrate it to k = 0) so we also cut the theory there
    const double kfundamental = 2.0 * M_PI / boxsize;
    auto W = [&](double k) { return (k * boxsize < kcut and k >= kfundamental) ? 1.0 : 0.0; };
    ZeldovichOptions zoptions;
    zoptions.kmin_integration = kfundamental;
    zoptions.kmax_integration = kcut / boxsize;
    auto & P11 = spectra.get("1", "1");
    std::vector<double> kvalues;
    for (int i = 0; i < P11.n; i += 4)
        kvalues.push_back(P11.kbin[i]);
    auto zeldovich = zeldovich_hybrid_bias_spectra(pofk, W, W, kvalues, zoptions);

    if (FML::ThisTask == 0) {
        std::cout << "#  k (h/Mpc)    P_11  (Zel'dovich)    P_deltadelta  (Zel'dovich)    P_s2s2  (Zel'dovich)    "
                     "P_tracer\n";
        for (int i = 0, j = 0; i < P11.n; i += 4, j++)
            std::cout << P11.kbin[i] << "   " << P11.pofk[i] << " (" << zeldovich[spectra.index(0, 0)][j] << ")   "
                      << spectra.get("delta", "delta").pofk[i] << " (" << zeldovich[spectra.index(1, 1)][j] << ")   "
                      << spectra.get("s2", "s2").pofk[i] << " (" << zeldovich[spectra.index(3, 3)][j] << ")   "
                      << Ptracer[i] << "\n";
    }
}

// The Fourier RSD reconstruction (LPT/Reconstruction.h) on PencilGrid: the same particles (Zeldovich
// displaced lattice in redshift space) reconstructed on slabs and on every pencil decomposition end at the same
// positions (every particle), along every line of sight
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/Reconstruction.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;

struct ReconParticle {
    double pos[3];
    double s0[3]; // redshift-space position (the key of the particle)
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

// The particles: a lattice displaced by a smooth (band-limited) Zeldovich field with rms 0.5 cells, then by
// beta Psi along the line of sight
std::vector<ReconParticle> make_particles(int Nmesh, double beta, int los) {
    FML::GRID::FFTWGrid<3> delta(Nmesh, 1, 1);
    delta.set_grid_status_real(false);
    for (auto && index : delta.get_fourier_range()) {
        std::array<double, 3> k;
        double k2;
        delta.get_fourier_wavevector_and_norm2_by_index(index, k, k2);
        const auto m = delta.get_fourier_coord_from_index(index);
        const double phase = 0.7 * (m[0] + delta.get_local_x_start()) + 1.3 * m[1] + 0.4 * m[2];
        const double amp = k2 > 0 and k2 < std::pow(2 * M_PI * 3.5, 2) ? 1.0 / (1 + k2) : 0.0;
        delta.set_fourier_from_index(index, std::polar(amp, phase));
    }
    // (make the kz = 0 and Nyquist planes Hermitian by a round trip)
    delta.fftw_c2r();
    delta.fftw_r2c();
    FML::GRID::FFTWGrid<3> phi;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi);
    std::array<FML::GRID::FFTWGrid<3>, 3> Psi;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<3>(phi, Psi);
    double s2 = 0;
    for (int d = 0; d < 3; d++)
        for (auto && ri : Psi[d].get_real_range())
            s2 += std::pow(Psi[d].get_real_from_index(ri), 2);
    FML::SumOverTasks(&s2);
    const double scale = 0.5 / Nmesh / std::sqrt(s2 / (3.0 * std::pow(Nmesh, 3)));
    for (int d = 0; d < 3; d++) {
        for (auto && ri : Psi[d].get_real_range())
            Psi[d].set_real_from_index(ri, Psi[d].get_real_from_index(ri) * scale);
        Psi[d].communicate_boundaries();
    }
    FML::PARTICLE::MPIParticles<ReconParticle> lattice;
    lattice.create_particle_grid(Nmesh, 3.0);
    std::array<std::vector<FML::GRID::FloatType>, 3> Pq;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, ReconParticle>(
        Psi, lattice.get_particles_ptr(), lattice.get_npart(), Pq, "CIC");
    std::vector<ReconParticle> out(lattice.get_npart());
    auto * p = lattice.get_particles_ptr();
    for (size_t i = 0; i < out.size(); i++)
        for (int d = 0; d < 3; d++) {
            double s = p[i].pos[d] + (1.0 + (d == los ? beta : 0.0)) * Pq[d][i];
            s -= std::floor(s);
            out[i].pos[d] = out[i].s0[d] = s >= 1.0 ? 0.0 : s;
        }
    return out;
}

// All particles on every task, sorted by the redshift-space position
std::vector<std::array<double, 6>> gather(FML::PARTICLE::MPIParticles<ReconParticle> & part) {
    std::vector<double> local;
    for (auto & q : part)
        for (int d = 0; d < 3; d++)
            local.push_back(q.s0[d]), local.push_back(q.pos[d]);
    int n = int(local.size());
    auto counts = FML::GatherFromTasks(&n);
    std::vector<int> displs(FML::NTasks, 0);
    for (int t = 1; t < FML::NTasks; t++)
        displs[t] = displs[t - 1] + counts[t - 1];
    std::vector<double> all(displs.back() + counts.back());
    MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   FML::communicator);
    std::vector<std::array<double, 6>> result(all.size() / 6);
    for (size_t i = 0; i < result.size(); i++)
        for (int d = 0; d < 3; d++)
            result[i][d] = all[6 * i + 2 * d], result[i][3 + d] = all[6 * i + 2 * d + 1];
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    const int Nmesh = FML::NTasks % 3 == 0 ? 24 : 32;
    const double beta = 0.5;
    const double tolerance = sizeof(FML::GRID::FloatType) == 4 ? 1e-6 : 1e-12;
    for (int los = 0; los < 3; los++) {
        auto particles = make_particles(Nmesh, beta, los);
        std::vector<double> los_direction(3, 0.0);
        los_direction[los] = 1.0;
        auto run = [&](const FML::Domain & domain, auto grid_tag, const std::string & method) {
            FML::PARTICLE::MPIParticles<ReconParticle> part(domain);
            auto copy = particles;
            part.create(copy.data(), copy.size(), 3 * copy.size() + 1000, FML::PARTICLE::Input::Distributed);
            using G = typename decltype(grid_tag)::type;
            if constexpr (std::is_same_v<G, FML::GRID::FFTWGrid<3>>)
                FML::COSMOLOGY::LPT::RSDReconstructionFourierMethod<3, ReconParticle>(
                    part, method, los_direction, Nmesh, 3, beta, {"gaussian", 0.5 / Nmesh}, false);
            else
                FML::COSMOLOGY::LPT::RSDReconstructionFourierMethod<3, ReconParticle, FML::GRID::PencilGrid>(
                    part, method, los_direction, Nmesh, 3, beta, {"gaussian", 0.5 / Nmesh}, false);
            // Every particle is on the task that owns it
            bool owned = true;
            for (auto & q : part)
                owned = owned and domain.owner(q.pos) == FML::ThisTask;
            int ok = owned ? 1 : 0;
            FML::MinOverTasks(&ok);
            return std::make_pair(gather(part), ok == 1);
        };
        struct SlabTag {
            using type = FML::GRID::FFTWGrid<3>;
        };
        struct PencilTag {
            using type = FML::GRID::PencilGrid<3>;
        };
        for (std::string method : {"CIC", "NGP", "PCS"}) {
            const auto ref = run(FML::Domain::uniform_slab(FML::NTasks), SlabTag{}, method);
            // The reference removes the RSD: the shift is along the line of sight only and of the right size
            double removed = 0, perp = 0;
            for (auto & r : ref.first) {
                removed = std::max(removed, std::abs(std::remainder(r[los] - r[3 + los], 1.0)));
                for (int d = 0; d < 3; d++)
                    if (d != los)
                        perp = std::max(perp, std::abs(r[d] - r[3 + d]));
            }
            const std::string where = "los " + std::to_string(los) + " " + method + ": ";
            check(ref.second and perp == 0 and removed > 0.05 / Nmesh and removed < 3.0 / Nmesh,
                  where + "slabs: shift along the line of sight only (max " + num(removed * Nmesh, 3) + " cells)");
            for (int p0 = 1; p0 <= FML::NTasks; p0++) {
                const int p1 = FML::NTasks / p0;
                if (FML::NTasks % p0 or Nmesh % p0 or Nmesh % p1)
                    continue;
                const auto pen = run(FML::Domain::uniform_pencil(p0, p1), PencilTag{}, method);
                double d = pen.first.size() == ref.first.size() ? 0.0 : 1.0;
                for (size_t i = 0; d < 1 and i < ref.first.size(); i++)
                    for (int c = 0; c < 6; c++)
                        d = std::max(d, std::abs(std::remainder(ref.first[i][c] - pen.first[i][c], 1.0)));
                check(pen.second and d < tolerance,
                      where + "pencils " + std::to_string(p0) + "x" + std::to_string(p1) +
                          " = slabs for every particle (max diff " + num(d, 2) + "), particles owned");
            }
        }
    }
    return FML::TESTS::summary("RSD reconstruction on pencils");
}

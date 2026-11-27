// Particles and grids with different decompositions (Global/Domain.h, Interpolation/DistributedDeposit.h): P(k) of
// particles (compute_power_spectrum with the domain of the particles), the multipoles of MPIParticles
// (compute_power_spectrum_multipoles) and an N-body kick-drift-kick step
// (KickDriftKickNBodyStep) on an FFTWGrid give the same result for the particles on the slabs of the grid (deposited
// and kicked locally) and on a PeanoHilbert domain or pencils (the deposit and interpolation for any domain)
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/tests/TestHelpers.h>

#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;

namespace {
    struct Part {
        double pos[3];
        double vel[3];
        long long id;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        long long get_id() const { return id; }
        void set_id(long long i) { id = i; }
        constexpr int get_ndim() const { return 3; }
    };

    // Clumps and a uniform background (the same on every task count), this task's share
    std::vector<Part> my_particles() {
        std::mt19937 gen(99);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.02);
        std::vector<Part> mine;
        for (long long i = 0; i < 20000; i++) {
            Part p{};
            const bool clump = i % 3 == 0;
            const double c[3] = {0.3, 0.7, 0.5};
            for (int d = 0; d < 3; d++) {
                double x = clump ? c[d] + G(gen) : U(gen);
                x -= std::floor(x);
                p.pos[d] = x < 1.0 ? x : 0.0;
                p.vel[d] = 0.01 * (U(gen) - 0.5);
            }
            p.id = i;
            if (i % FML::NTasks == FML::ThisTask)
                mine.push_back(p);
        }
        return mine;
    }

    std::map<long long, std::array<double, 6>> state(FML::PARTICLE::MPIParticles<Part> & part) {
        std::vector<double> local;
        for (auto & p : part)
            local.insert(local.end(), {double(p.id), p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]});
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displ(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displ[i] = displ[i - 1] + counts[i - 1];
        std::vector<double> all(displ.back() + counts.back());
        MPI_Allgatherv(local.data(),
                       n,
                       MPI_DOUBLE,
                       all.data(),
                       counts.data(),
                       displ.data(),
                       MPI_DOUBLE,
                       FML::communicator);
        std::map<long long, std::array<double, 6>> s;
        for (size_t k = 0; k < all.size(); k += 7)
            s[(long long)all[k]] = {all[k + 1], all[k + 2], all[k + 3], all[k + 4], all[k + 5], all[k + 6]};
        return s;
    }
} // namespace

int main() {
    const int Ngrid = 16 * FML::NTasks;
    auto mine = my_particles();
    std::vector<std::pair<std::string, FML::Domain>> domains{
        {"PeanoHilbert", FML::Domain::uniform_peano_hilbert(FML::NTasks)}};
    for (int p0 = 1; p0 < FML::NTasks; p0++)
        if (FML::NTasks % p0 == 0)
            domains.push_back({"Pencil " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0),
                               FML::Domain::uniform_pencil(p0, FML::NTasks / p0)});

    // P(k) of the particles on an FFTWGrid: the slabs (local) vs the other domains (any domain)
    for (bool interlacing : {false, true}) {
        FML::PARTICLE::MPIParticles<Part> slabs;
        slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> ref(Ngrid / 2);
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3, Part>(Ngrid,
                                                                   slabs.get_particles_ptr(),
                                                                   slabs.get_npart(),
                                                                   slabs.get_npart_total(),
                                                                   ref,
                                                                   "CIC",
                                                                   interlacing,
                                                                   slabs.get_domain());
        for (auto & [name, domain] : domains) {
            FML::PARTICLE::MPIParticles<Part> part(domain);
            part.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(Ngrid / 2);
            FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3, Part>(Ngrid,
                                                                       part.get_particles_ptr(),
                                                                       part.get_npart(),
                                                                       part.get_npart_total(),
                                                                       pofk,
                                                                       "CIC",
                                                                       interlacing,
                                                                       part.get_domain());
            double diff = 0.0;
            for (int i = 0; i < ref.n; i++)
                if (ref.pofk[i] != 0.0)
                    diff = std::max(diff, std::abs(pofk.pofk[i] / ref.pofk[i] - 1.0));
            check(diff < 1e-10,
                  name + ": P(k)" + (interlacing ? " (interlacing)" : "") + " = on the slabs (max rel. diff " +
                      FML::TESTS::num(diff, 2) + ")");
        }
    }

    // The multipoles of P(k) in redshift space (MPIParticles: the grid follows the particles if it can)
    for (bool interlacing : {false, true}) {
        auto multipoles = [&](FML::PARTICLE::MPIParticles<Part> & part) {
            std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>> Pell(3, Ngrid / 2);
            FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles<3, Part>(Ngrid,
                                                                                  part,
                                                                                  5.0,
                                                                                  Pell,
                                                                                  "CIC",
                                                                                  interlacing);
            return Pell;
        };
        FML::PARTICLE::MPIParticles<Part> slabs;
        slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
        const auto ref = multipoles(slabs);
        for (auto & [name, domain] : domains) {
            FML::PARTICLE::MPIParticles<Part> part(domain);
            part.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
            const auto Pell = multipoles(part);
            double diff = 0.0;
            for (size_t ell = 0; ell < ref.size(); ell++)
                for (int i = 0; i < ref[ell].n; i++)
                    diff =
                        std::max(diff,
                                 std::abs(Pell[ell].pofk[i] - ref[ell].pofk[i]) / (1e-30 + std::abs(ref[0].pofk[i])));
            check(diff < 1e-10,
                  name + ": P_ell(k)" + (interlacing ? " (interlacing)" : "") + " = on the slabs (max diff / P_0 " +
                      FML::TESTS::num(diff, 2) + ")");
        }
    }

    // A kick-drift-kick step on an FFTWGrid: the same particles after the step
    {
        const auto nlr = FML::INTERPOLATION::get_extra_slices("CIC");
        FML::PARTICLE::MPIParticles<Part> slabs;
        slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
        FML::NBODY::KickDriftKickNBodyStep<3, Part, FML::GRID::FFTWGrid>(
            FML::GRID::FFTWGrid<3>(Ngrid, nlr.first, nlr.second),
            slabs,
            0.05,
            "CIC",
            1.5);
        const auto ref = state(slabs);
        for (auto & [name, domain] : domains) {
            FML::PARTICLE::MPIParticles<Part> part(domain);
            part.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
            FML::NBODY::KickDriftKickNBodyStep<3, Part, FML::GRID::FFTWGrid>(
                FML::GRID::FFTWGrid<3>(Ngrid, nlr.first, nlr.second),
                part,
                0.05,
                "CIC",
                1.5);
            const auto s = state(part);
            double dx = 0.0, dv = 0.0;
            bool same_particles = s.size() == ref.size() and part.get_domain() == domain;
            for (auto & [id, a] : ref) {
                auto it = s.find(id);
                if (it == s.end()) {
                    same_particles = false;
                    break;
                }
                for (int d = 0; d < 3; d++) {
                    double ddx = std::abs(it->second[d] - a[d]);
                    dx = std::max(dx, std::min(ddx, 1.0 - ddx));
                    dv = std::max(dv, std::abs(it->second[3 + d] - a[3 + d]));
                }
            }
            check(same_particles and dx < 1e-12 and dv < 1e-12,
                  name + ": KickDriftKickNBodyStep = on the slabs (max diff x " + FML::TESTS::num(dx, 2) + ", v " +
                      FML::TESTS::num(dv, 2) + ")");
        }
    }
    return FML::TESTS::summary("particles and grids on different domains");
}

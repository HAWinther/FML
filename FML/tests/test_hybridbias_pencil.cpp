// The hybrid bias spectra (Lagrangian operators on the lattice, the values at the particles from the
// owner of their lattice point, the advected fields) and the Zel'dovich field on pencils give the spectra of the
// slabs; a linear field file written on pencils is read on slabs (and the reverse)
#include <FML/ComputePowerSpectra/HybridBiasLinearFieldIO.h>
#include <FML/ComputePowerSpectra/HybridBiasSpectra.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace HB = FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;

namespace {
    struct Part {
        double pos[3];
        long long id;
        double * get_pos() { return pos; }
        long long get_id() const { return id; }
        void set_id(long long i) { id = i; }
        constexpr int get_ndim() const { return 3; }
    };
    int nlattice = 16; // (set in main: 12 if the number of tasks is a multiple of 3)

    double field(const std::array<double, 3> & x) {
        return 0.3 * std::sin(2 * M_PI * x[0]) * std::cos(4 * M_PI * x[1]) + 0.2 * std::cos(2 * M_PI * (x[2] + x[0])) +
               0.1 * std::sin(6 * M_PI * x[1] + 1.0);
    }
    // The lattice particles, displaced, every task makes all and keeps its share
    std::vector<Part> mine() {
        std::vector<Part> v;
        long long id = 0;
        for (int i = 0; i < nlattice; i++)
            for (int j = 0; j < nlattice; j++)
                for (int k = 0; k < nlattice; k++, id++) {
                    Part p;
                    const double q[3] = {double(i) / nlattice, double(j) / nlattice, double(k) / nlattice};
                    for (int d = 0; d < 3; d++) {
                        double x = q[d] + 0.02 * std::sin(2 * M_PI * (q[(d + 1) % 3] + 0.3 * d));
                        x -= std::floor(x);
                        p.pos[d] = x >= 1.0 ? 0.0 : x;
                    }
                    p.id = id;
                    if (id % FML::NTasks == FML::ThisTask)
                        v.push_back(p);
                }
        return v;
    }
    std::vector<double> flatten(const HB::HybridBiasSpectra<3> & s) {
        std::vector<double> r;
        for (auto & b : s.pofk)
            for (int i = 0; i < b.n; i++)
                r.push_back(b.pofk[i]);
        for (auto x : s.shotnoise)
            r.push_back(x);
        return r;
    }
} // namespace

int main() {
    nlattice = FML::NTasks % 3 == 0 ? 12 : 16;
    HB::HybridBiasOptions o;
    o.noperators = 5;
    const int nmesh = nlattice;
    auto v = mine();
    // Slabs
    FML::GRID::FFTWGrid<3> lin(nlattice);
    fill<3>(lin, field);
    lin.fftw_r2c();
    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    HB::HybridBiasSpectra<3> ref, ref_z;
    HB::compute_hybrid_bias_spectra<3, Part>(
        pslab.get_particles_ptr(), pslab.get_npart(), pslab.get_npart_total(), lin, nlattice, nmesh, o, ref);
    HB::compute_zeldovich_hybrid_bias_spectra<3>(lin, nlattice, nmesh, o, ref_z);
    const auto rs = flatten(ref), rz = flatten(ref_z);
    // A linear field file from the slabs
    auto lin_real = lin;
    lin_real.fftw_c2r();
    const std::string slabfile = "test_hybridbias_pencil_slab.dat";
    HB::write_linear_field<3>(slabfile, lin_real, 100.0, 49.0);
    const auto slab_values = global_real(lin_real);

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0 or nlattice % p0 or nlattice % (FML::NTasks / p0))
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        FML::GRID::PencilGrid<3> plin(nlattice, 0, 0, layout);
        fill<3>(plin, field);
        plin.fftw_r2c();
        const auto domain = FML::GRID::particle_domain_of(plin);
        FML::PARTICLE::MPIParticles<Part> p(domain);
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        HB::HybridBiasSpectra<3> s, z;
        HB::compute_hybrid_bias_spectra<3, Part>(
            p.get_particles_ptr(), p.get_npart(), p.get_npart_total(), plin, nlattice, nmesh, o, s);
        HB::compute_zeldovich_hybrid_bias_spectra<3>(plin, nlattice, nmesh, o, z);
        const double d1 = max_rel_diff(flatten(s), rs), d2 = max_rel_diff(flatten(z), rz);
        check(d1 < 1e-10, name + ": hybrid bias spectra (5 operators) = slabs (max diff " + num(d1, 2) + ")");
        check(d2 < 1e-10, name + ": Zel'dovich field spectra = slabs (max diff " + num(d2, 2) + ")");

        // Files: pencils -> slabs and slabs -> pencils
        auto preal = plin;
        preal.fftw_c2r();
        const std::string penfile = "test_hybridbias_pencil_pen.dat";
        HB::write_linear_field<3>(penfile, preal, 100.0, 49.0);
        HB::LinearFieldFileHeader h1, h2;
        const auto back_slab = HB::read_linear_field<3>(penfile, h1);
        const auto back_pen = HB::read_linear_field<3, FML::GRID::PencilGrid>(slabfile, h2, domain);
        const double d3 = max_rel_diff(global_real(back_slab), slab_values);
        const double d4 = max_rel_diff(global_real(back_pen), slab_values);
        check(d3 < 1e-6 and d4 < 1e-6 and h1.nmesh == nlattice and h1.redshift == 49.0,
              name + ": linear field file written on pencils read on slabs, and the reverse (max diff " +
                  num(std::max(d3, d4), 2) + ")");
        if (FML::ThisTask == 0)
            std::remove(penfile.c_str());
    }
    if (FML::ThisTask == 0)
        std::remove(slabfile.c_str());
    return FML::TESTS::summary("hybrid bias on pencils");
}

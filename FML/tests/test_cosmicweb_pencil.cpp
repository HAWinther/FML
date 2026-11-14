// The cosmic web (T-web from particles, V-web from shear grids; FFTWGrid or byte storage) on pencils
// gives the classification, the eigenvalues and the volume and mass fractions of the slabs
#include <FML/CosmicWeb/CosmicWeb.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace CW = FML::COSMICWEB;

namespace {
    struct Part {
        double pos[3];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
    };

    std::vector<Part> mine(int npart) {
        std::mt19937 gen(3);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.05);
        std::vector<Part> v;
        for (int i = 0; i < npart; i++) {
            Part p;
            for (int d = 0; d < 3; d++) {
                double x = i % 3 == 0 ? U(gen) : 0.3 + 0.2 * (i % 7 == 0) + G(gen);
                x -= std::floor(x);
                p.pos[d] = x >= 1.0 ? 0.0 : x;
            }
            if (i % FML::NTasks == FML::ThisTask)
                v.push_back(p);
        }
        return v;
    }

    template <class G>
    std::vector<double> flatten(const CW::CosmicWebT<G> & w, bool eigen) {
        std::vector<double> r;
        auto add = [&](const G & g) {
            auto x = global_real(g);
            r.insert(r.end(), x.begin(), x.end());
        };
        add(w.classification_as_grid());
        if (eigen)
            for (auto & g : w.eigenvalues)
                add(g);
        for (int k = 0; k < 4; k++) {
            r.push_back(w.volume_fraction[k]);
            r.push_back(w.mass_fraction[k]);
        }
        return r;
    }

    // A smooth symmetric "shear" field and its divergence
    double shear_value(int c, const std::array<double, 3> & x) {
        return std::sin(2 * M_PI * (x[0] + 0.3 * c)) * std::cos(2 * M_PI * x[1]) + 0.2 * c * std::sin(4 * M_PI * x[2]);
    }
} // namespace

int main() {
    const int nmesh = FML::NTasks % 3 == 0 ? 24 : 32, npart = 40000;
    auto v = mine(npart);
    CW::TWebOptions o;
    o.smoothing_scale = 0.03;
    o.store_eigenvalues = true;
    CW::TWebOptions ob = o;
    ob.classification_storage = CW::ClassificationStorage::Bytes;
    ob.store_eigenvalues = false;

    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    CW::CosmicWeb tw, twb;
    CW::TWeb(pslab.get_particles_ptr(), pslab.get_npart(), nmesh, o, tw);
    CW::TWeb(pslab.get_particles_ptr(), pslab.get_npart(), nmesh, ob, twb);
    const auto ref_t = flatten(tw, true), ref_tb = flatten(twb, false);
    std::array<FML::GRID::FFTWGrid<3>, 6> fshear;
    for (int c = 0; c < 6; c++) {
        fshear[c] = FML::GRID::FFTWGrid<3>(nmesh);
        fill<3>(fshear[c], [&](auto x) { return shear_value(c, x); });
    }
    FML::GRID::FFTWGrid<3> fdiv(nmesh);
    fill<3>(fdiv, [&](auto x) { return shear_value(7, x); });
    CW::CosmicWeb vw;
    CW::VWeb(fshear, fdiv, -0.7, 0.1, vw, true);
    const auto ref_v = flatten(vw, true);

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> proto(nmesh, 0, 0, layout);
        const auto domain = FML::GRID::particle_domain_of(proto);
        FML::PARTICLE::MPIParticles<Part> p(domain);
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        using PG = FML::GRID::PencilGrid<3>;
        CW::CosmicWebT<PG> ptw, ptwb, pvw;
        CW::TWeb<Part, FML::GRID::PencilGrid>(p.get_particles_ptr(), p.get_npart(), nmesh, o, ptw, domain);
        CW::TWeb<Part, FML::GRID::PencilGrid>(p.get_particles_ptr(), p.get_npart(), nmesh, ob, ptwb, domain);
        double d1 = max_rel_diff(flatten(ptw, true), ref_t);
        double d2 = max_rel_diff(flatten(ptwb, false), ref_tb);
        check(d1 < 1e-10 and d2 < 1e-10,
              name + ": T-web from particles (grid with eigenvalues; bytes) = slabs (max diff " +
                  num(std::max(d1, d2), 2) + ")");
        std::array<PG, 6> pshear;
        for (int c = 0; c < 6; c++) {
            pshear[c] = PG(nmesh, 0, 0, layout);
            fill<3>(pshear[c], [&](auto x) { return shear_value(c, x); });
        }
        PG pdiv(nmesh, 0, 0, layout);
        fill<3>(pdiv, [&](auto x) { return shear_value(7, x); });
        CW::VWeb<FML::GRID::PencilGrid>(pshear, pdiv, -0.7, 0.1, pvw, true);
        const double d3 = max_rel_diff(flatten(pvw, true), ref_v);
        check(d3 < 1e-12, name + ": V-web = slabs (max diff " + num(d3, 2) + ")");
    }
    return FML::TESTS::summary("cosmic web on pencils");
}

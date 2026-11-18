// Compare FML's native periodic Voronoi tessellation with CGAL's periodic Delaunay triangulations (2D and 3D): for
// points in general position the Delaunay edges, with their periodic offsets, must be identical. Also prints the
// timings. Build with make compare_cgal (CGAL >= 5, header only, with GMP/MPFR); run ./compare_cgal [N]
// [random|clustered]. Faces are kept down to zero area here (drop_zero_area_faces = false) since CGAL lists every
// Delaunay edge.
#include <FML/Experimental/Delaunay/PeriodicVoronoi.h>
#include <FML/Global/Global.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Periodic_2_triangulation_vertex_base_2.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Periodic_3_triangulation_ds_vertex_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using GT3 = CGAL::Periodic_3_Delaunay_triangulation_traits_3<K>;
using Vb3 = CGAL::Triangulation_vertex_base_3<GT3, CGAL::Periodic_3_triangulation_ds_vertex_base_3<>>;
using Vbi3 = CGAL::Triangulation_vertex_base_with_info_3<size_t, GT3, Vb3>;
using Cb3 = CGAL::Triangulation_cell_base_3<GT3, CGAL::Periodic_3_triangulation_ds_cell_base_3<>>;
using P3 = CGAL::Periodic_3_Delaunay_triangulation_3<GT3, CGAL::Triangulation_data_structure_3<Vbi3, Cb3>>;
using GT2 = CGAL::Periodic_2_Delaunay_triangulation_traits_2<K>;
using Vbi2 =
    CGAL::Triangulation_vertex_base_with_info_2<size_t, GT2, CGAL::Periodic_2_triangulation_vertex_base_2<GT2>>;
using Fb2 = CGAL::Periodic_2_triangulation_face_base_2<GT2>;
using P2 = CGAL::Periodic_2_Delaunay_triangulation_2<GT2, CGAL::Triangulation_data_structure_2<Vbi2, Fb2>>;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// (i, j, offset of j relative to i) with the same canonical orientation as PeriodicVoronoi's faces
template <int D>
static std::tuple<size_t, size_t, std::array<int, D>> canonical(size_t i, size_t j, std::array<int, D> o) {
    bool positive = false;
    for (int x : o)
        if (x != 0) {
            positive = x > 0;
            break;
        }
    if (j < i || (j == i && !positive)) {
        for (auto & x : o)
            x = -x;
        std::swap(i, j);
    }
    return {i, j, o};
}

template <int D>
static std::vector<std::array<double, D>> make_points(size_t n, const std::string & type) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> G(0, 0.02);
    std::vector<std::array<double, D>> p(n);
    const double q = std::ldexp(1.0, 40);
    for (size_t i = 0; i < n; i++)
        for (int d = 0; d < D; d++) {
            double x = (type == "clustered" && i % 2 == 0) ? 0.3 + G(rng) : U(rng);
            x -= std::floor(x);
            // Rounded as PeriodicVoronoi does, so that both triangulate the same points
            x = std::round(x * q) / q;
            p[i][d] = x >= 1.0 ? x - 1.0 : x;
        }
    return p;
}

template <int D>
static void compare(size_t n, const std::string & type) {
    const auto p = make_points<D>(n, type);
    typename FML::DELAUNAY::PeriodicVoronoi<D>::Options opt;
    opt.drop_zero_area_faces = false;
    const double t0 = now();
    FML::DELAUNAY::PeriodicVoronoi<D> v(p, 1.0, opt);
    const double t1 = now();

    std::set<std::tuple<size_t, size_t, std::array<int, D>>> cgal, ours;
    double t2, t3;
    if constexpr (D == 3) {
        std::vector<K::Point_3> pts;
        std::map<std::array<double, 3>, size_t> index;
        for (size_t i = 0; i < n; i++) {
            pts.emplace_back(p[i][0], p[i][1], p[i][2]);
            index[p[i]] = i;
        }
        t2 = now();
        P3 T(P3::Iso_cuboid(0, 0, 0, 1, 1, 1));
        T.insert(pts.begin(), pts.end(), true);
        t3 = now();
        for (auto vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
            vit->info() = index.at({vit->point().x(), vit->point().y(), vit->point().z()});
        for (auto e = T.edges_begin(); e != T.edges_end(); ++e) {
            const auto c = e->first;
            const auto oa = T.periodic_point(c, e->second).second, ob = T.periodic_point(c, e->third).second;
            cgal.insert(canonical<3>(c->vertex(e->second)->info(),
                                     c->vertex(e->third)->info(),
                                     {ob.x() - oa.x(), ob.y() - oa.y(), ob.z() - oa.z()}));
        }
    } else {
        std::vector<K::Point_2> pts;
        std::map<std::array<double, 2>, size_t> index;
        for (size_t i = 0; i < n; i++) {
            pts.emplace_back(p[i][0], p[i][1]);
            index[p[i]] = i;
        }
        t2 = now();
        P2 T(P2::Iso_rectangle(0, 0, 1, 1));
        T.insert(pts.begin(), pts.end(), true);
        t3 = now();
        for (auto vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
            vit->info() = index.at({vit->point().x(), vit->point().y()});
        for (auto e = T.edges_begin(); e != T.edges_end(); ++e) {
            const auto f = e->first;
            const int a = f->cw(e->second), b = f->ccw(e->second);
            const auto oa = T.periodic_point(f, a).second, ob = T.periodic_point(f, b).second;
            cgal.insert(canonical<2>(f->vertex(a)->info(), f->vertex(b)->info(), {ob.x() - oa.x(), ob.y() - oa.y()}));
        }
    }
    for (const auto & f : v.faces())
        ours.insert(canonical<D>(f.i, f.j, f.shift));
    size_t missing = 0, extra = 0;
    for (const auto & e : cgal)
        missing += ours.count(e) == 0;
    for (const auto & e : ours)
        extra += cgal.count(e) == 0;
    std::cout << D << "D " << type << " N = " << n << ": FML " << t1 - t0 << " s (triangulation "
              << v.statistics().time_triangulation << " s, geometry " << v.statistics().time_geometry
              << " s), CGAL triangulation " << t3 - t2 << " s; edges CGAL " << cgal.size() << ", FML " << ours.size()
              << ", missing " << missing << ", extra " << extra << (missing + extra == 0 ? "  [same]" : "  [DIFFERENT]")
              << "\n";
}

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    const size_t n = argc > 1 ? size_t(std::stoul(argv[1])) : 100000;
    const std::string type = argc > 2 ? argv[2] : "random";
    compare<2>(n, type);
    compare<3>(n, type);
}

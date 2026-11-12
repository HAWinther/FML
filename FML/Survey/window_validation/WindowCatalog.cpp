#include <FML/Survey/SurveyWindow.h>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    using namespace FML::SURVEY;
    Geometry g;
    std::vector<double> edges;
    std::vector<CatalogPoint> points;
    double alpha = 0, i2 = 0;
    int maxell = 8;
    bool self = true, valid = argc == 3;
    if (valid) {
        std::ifstream in(argv[1]);
        size_t ne = 0, np = 0;
        in >> g.nmesh >> g.boxsize;
        for (auto & v : g.minimum)
            in >> v;
        for (auto & v : g.observer)
            in >> v;
        in >> g.local_los;
        for (auto & v : g.fixed_los)
            in >> v;
        in >> alpha >> i2 >> maxell >> self >> ne;
        valid = bool(in) && ne >= 2 && ne <= 4096;
        if (valid) {
            edges.resize(ne);
            for (auto & v : edges)
                in >> v;
            in >> np;
            valid = bool(in) && np <= 5000000;
        }
        if (valid)
            for (size_t i = 0; i < np; ++i) {
                CatalogPoint p;
                for (auto & v : p.position)
                    in >> v;
                in >> p.weight >> p.nbar;
                if (int(i % FML::NTasks) == FML::ThisTask)
                    points.push_back(p);
            }
        valid = valid && bool(in);
    }
    require(valid, "Usage: survey_window_catalog input.txt output.csv (invalid input)");
    const auto q = estimate_window(points, g, edges, alpha, i2, maxell, self);
    if (!FML::ThisTask) {
        std::ofstream out(argv[2]);
        out << std::setprecision(17) << "ell,bin,s,nvectors,shell_volume,q,alpha,i2\n";
        for (size_t l = 0; l < q.ells.size(); ++l)
            for (size_t b = 0; b < q.s.size(); ++b)
                out << q.ells[l] << ',' << b << ',' << q.s[b] << ',' << q.nvectors[b] << ',' << q.shell_volume[b] << ','
                    << q.q[l][b] << ',' << alpha << ',' << i2 << '\n';
        valid = bool(out);
    }
    require(valid, "Could not write survey window output");
}

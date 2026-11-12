// Small validation driver; input/output format is documented in README.md.
#include <FML/Survey/SurveyPower.h>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    using namespace FML::SURVEY;
    Geometry g;
    PowerOptions o;
    std::vector<CatalogPoint> data, randoms;
    bool valid = argc == 3;
    if (valid) {
        std::ifstream in(argv[1]);
        size_t nb = 0, nd = 0, nr = 0;
        in >> g.nmesh >> g.boxsize;
        for (auto & v : g.minimum)
            in >> v;
        for (auto & v : g.observer)
            in >> v;
        in >> g.local_los;
        for (auto & v : g.fixed_los)
            in >> v;
        in >> o.i2 >> o.compensate_cic >> o.aliased_shotnoise >> nb;
        valid = bool(in) && nb >= 2 && nb <= 4096;
        if (valid) {
            o.edges.resize(nb);
            for (auto & v : o.edges)
                in >> v;
            in >> nd >> nr;
            valid = bool(in) && nd <= 5000000 && nr <= 5000000;
        }
        if (valid)
            for (int catalog = 0; catalog < 2; ++catalog) {
                const size_t n = catalog == 0 ? nd : nr;
                auto & points = catalog == 0 ? data : randoms;
                for (size_t i = 0; i < n; ++i) {
                    CatalogPoint p;
                    for (auto & v : p.position)
                        in >> v;
                    in >> p.weight >> p.nbar;
                    // Distinct, deliberately arbitrary input ownership, including empty ranks for small files.
                    if (int((i * 7 + catalog) % FML::NTasks) == FML::ThisTask)
                        points.push_back(p);
                }
                valid = valid && bool(in);
            }
    }
    require(valid, "Usage: survey_power_catalog input.txt output.csv (invalid/missing input)");
    const auto p = estimate_power(data, randoms, g, o);
    if (!FML::ThisTask) {
        std::ofstream out(argv[2]);
        out << std::setprecision(17) << "ell,bin,k,nmodes,raw,shotnoise,power,alpha,i2\n";
        for (size_t l = 0; l < p.ells.size(); ++l)
            for (size_t b = 0; b < p.k.size(); ++b)
                out << p.ells[l] << ',' << b << ',' << p.k[b] << ',' << p.nmodes[b] << ',' << p.raw[l][b] << ','
                    << p.shotnoise[l][b] << ',' << p.power[l][b] << ',' << p.alpha << ',' << p.i2 << '\n';
        valid = bool(out);
    }
    require(valid, "Could not write survey power validation output");
}

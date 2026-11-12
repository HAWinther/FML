#include <FML/Survey/SurveyWindow.h>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace FML::SURVEY;
void write_matrix(const WindowMatrix & m, const std::string & file) {
    std::ofstream out(file);
    out << std::setprecision(17) << "out_ell,out_bin,k,in_ell,in_bin,value\n";
    const size_t ni = m.input_edges.size() - 1, no = m.output_k.size();
    for (size_t l = 0; l < m.output_ells.size(); ++l)
        for (size_t b = 0; b < no; ++b)
            for (size_t j = 0; j < m.input_ells.size(); ++j)
                for (size_t a = 0; a < ni; ++a)
                    out << m.output_ells[l] << ',' << b << ',' << m.output_k[b] << ',' << m.input_ells[j] << ',' << a
                        << ',' << m.values[l * no + b][j * ni + a] << '\n';
    if (!out)
        throw std::runtime_error("Could not write matrix");
}
int main(int argc, char ** argv) {
    WindowMultipoles w;
    std::vector<double> kin, kout, bins, model;
    bool valid = argc == 3;
    if (valid) {
        std::ifstream in(argv[1]);
        size_t ns = 0, ni = 0, no = 0, ne = 0;
        in >> ns;
        valid = bool(in) && ns > 0 && ns <= 4096;
        if (valid) {
            w.ells = {0, 2, 4, 6, 8};
            w.edges.resize(ns + 1);
            w.s.resize(ns);
            w.nvectors.resize(ns);
            w.shell_volume.resize(ns);
            w.q.assign(5, std::vector<double>(ns));
            for (auto & v : w.edges)
                in >> v;
            for (size_t i = 0; i < ns; ++i) {
                in >> w.s[i] >> w.nvectors[i];
                for (auto & row : w.q)
                    in >> row[i];
                w.shell_volume[i] = 4 * M_PI / 3 * (std::pow(w.edges[i + 1], 3) - std::pow(w.edges[i], 3));
            }
            in >> ni;
            valid = bool(in) && ni >= 2 && ni <= 4096;
        }
        if (valid) {
            kin.resize(ni);
            for (auto & v : kin)
                in >> v;
            in >> no;
            valid = bool(in) && no > 0 && no <= 4096;
        }
        if (valid) {
            kout.resize(no);
            for (auto & v : kout)
                in >> v;
            in >> ne;
            valid = bool(in) && ne >= 2 && ne <= 4096;
        }
        if (valid) {
            bins.resize(ne);
            for (auto & v : bins)
                in >> v;
            model.resize(3 * (ni - 1));
            for (auto & v : model)
                in >> v;
            valid = bool(in);
        }
    }
    require(valid, "Usage: survey_window_matrix input.txt output_prefix (invalid input)");
    if (!FML::ThisTask)
        try {
            const std::string prefix = argv[2];
            std::ofstream predictions(prefix + "-prediction.csv");
            predictions << std::setprecision(17) << "kind,ic,ell,bin,k,power\n";
            for (bool ic : {false, true})
                for (bool binned : {false, true}) {
                    const auto m = binned ? make_binned_window_matrix(w, kin, bins, {0, 2, 4}, {0, 2, 4}, ic) :
                                            make_window_matrix(w, kin, kout, {0, 2, 4}, {0, 2, 4}, ic);
                    const std::string kind = binned ? "binned" : "point";
                    write_matrix(m, prefix + "-" + kind + "-ic" + std::to_string(int(ic)) + ".csv");
                    const auto p = m.apply(model);
                    for (size_t l = 0; l < 3; ++l)
                        for (size_t b = 0; b < m.output_k.size(); ++b)
                            predictions << kind << ',' << ic << ',' << m.output_ells[l] << ',' << b << ','
                                        << m.output_k[b] << ',' << p[l * m.output_k.size() + b] << '\n';
                }
            valid = bool(predictions);
        } catch (const std::exception & e) {
            std::cerr << e.what() << '\n';
            valid = false;
        }
    require(valid, "Survey window matrix validation failed");
}

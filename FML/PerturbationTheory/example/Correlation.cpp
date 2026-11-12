#include <FML/PerturbationTheory/CorrelationMultipoles.h>
#include <FML/PerturbationTheory/EPTRedshiftSpace.h>
#include <FML/Global/Global.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace FML::PERTURBATION;
using V = std::vector<double>;
void read(const std::string & file, V & k, V & p) {
    std::ifstream in(file);
    if (!in)
        throw std::runtime_error("Cannot open " + file);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        double x, y;
        std::istringstream row(line);
        if (!(row >> x >> y))
            throw std::runtime_error("Malformed spectrum");
        k.push_back(x);
        p.push_back(y);
    }
}
int main(int argc, char ** argv) {
    V k, p, knw, pnw;
    read(argc > 1 ? argv[1] : "../tests/reference/lptrsd_linear_pk.txt", k, p);
    read(argc > 2 ? argv[2] : "../tests/reference/rept_nowiggle_pk.txt", knw, pnw);
    if (k != knw)
        throw std::invalid_argument("P and Pnw must have identical k grids");
    EPTRedshiftSpace::Options model;
    model.kmin = .001;
    model.kmax = 1;
    model.nk = 512;
    EPTRedshiftSpace ept(k, p, pnw, model);
    EPTRedshiftSpace::Bias bias;
    bias.b1 = 1.8;
    bias.b2 = .2;
    bias.alpha0 = 2;
    const auto power = ept.multipoles(.75, bias);
    V s;
    for (int i = 0; i <= 80; ++i)
        s.push_back(20 + 2 * i);
    CorrelationOptions options;
    options.fft_samples = 8192;
    options.taper_fraction = .1;
    options.damping_radius = 4;
    CorrelationWorkspace workspace(power.k, s, options);
    const auto xi = workspace.evaluate_spectrum(power);
    if (!FML::ThisTask) {
        const std::string output = argc > 3 ? argv[3] : "correlation_multipoles.csv";
        std::ofstream out(output);
        out << "# IR EPT demo, finite k=.001..1, exp[-(4k)^2], log-end taper=.1; stochastic parameters zero\n";
        out << "s,xi0,xi2,xi4\n" << std::setprecision(17);
        for (size_t i = 0; i < s.size(); ++i)
            out << s[i] << ',' << xi.xi[0][i] << ',' << xi.xi[1][i] << ',' << xi.xi[2][i] << '\n';
        if (!out)
            throw std::runtime_error("Cannot write " + output);
        std::cout << "Wrote " << output << '\n';
    }
}
